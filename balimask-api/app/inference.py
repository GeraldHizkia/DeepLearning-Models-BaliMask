"""
inference.py — "Dapur" API: memuat model & menjalankan prediksi.

Alur satu gambar:
  bytes upload -> PIL.Image -> preprocessing (resize, normalize) -> tensor
  -> model(tensor) -> logits -> softmax -> probabilitas 13 kelas
"""
import hashlib
import threading
import time
from collections import Counter

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .architectures import build_model
from .config import CLASSES, DEMO_MODE, IMG_SIZE, LAZY_LOAD, MODEL_REGISTRY

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# Preprocessing HARUS sama dengan eval_transform di notebook
TRANSFORMS = {
    "resize": transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]),
    "resize_crop": transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 1.14)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]),
}


def _extract_state_dict(obj):
    """Menangani berbagai cara orang menyimpan checkpoint."""
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model", "ema"):
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break
    # Hapus awalan dari DataParallel / torch.compile
    return {k.removeprefix("module.").removeprefix("_orig_mod."): v for k, v in obj.items()}


class ModelManager:
    def __init__(self):
        self.entries = {m["id"]: dict(m) for m in MODEL_REGISTRY}
        self.models = {}          # id -> nn.Module yang sudah di-load
        self.errors = {}          # id -> pesan error saat load
        self._lock = threading.Lock()

    # ---------------- Loading ----------------
    def load_all(self):
        if LAZY_LOAD:
            return
        for mid in self.entries:
            self._ensure_loaded(mid)

    def _ensure_loaded(self, mid):
        if mid in self.models or mid in self.errors:
            return
        e = self.entries[mid]
        if not e["weight_path"].exists():
            return  # belum ada file bobot -> status "missing"
        with self._lock:
            if mid in self.models:
                return
            try:
                t0 = time.perf_counter()
                model = build_model(e["arch"], e["version"])
                ckpt = torch.load(e["weight_path"], map_location="cpu", weights_only=True)
                model.load_state_dict(_extract_state_dict(ckpt), strict=True)
                model.eval().to(DEVICE)
                self.models[mid] = model
                e["load_ms"] = round((time.perf_counter() - t0) * 1000)
                e["params_m"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 2)
                e["size_mb"] = round(e["weight_path"].stat().st_size / 1e6, 1)
                print(f"[OK]   {mid:32s} {e['params_m']:>7}M param  ({e['load_ms']} ms)")
            except Exception as ex:  # noqa: BLE001
                msg = str(ex).split("\n")[0][:300]
                self.errors[mid] = msg
                print(f"[GAGAL] {mid}: {msg}")

    def status(self, mid):
        if mid in self.models:
            return "ready"
        if mid in self.errors:
            return "error"
        if self.entries[mid]["weight_path"].exists():
            return "not_loaded"   # ada file, tapi lazy-load belum dipicu
        return "demo" if DEMO_MODE else "missing"

    def describe(self):
        out = []
        for mid, e in self.entries.items():
            out.append({
                "id": mid,
                "arch": e["arch"], "arch_label": e["arch_label"],
                "version": e["version"], "version_label": e["version_label"],
                "version_desc": e["version_desc"],
                "status": self.status(mid),
                "weight_file": str(e["weight_path"].relative_to(e["weight_path"].parents[1])),
                "params_m": e.get("params_m"), "size_mb": e.get("size_mb"),
                "error": self.errors.get(mid),
            })
        return out

    # ---------------- Prediksi ----------------
    def predict_all(self, image: Image.Image, top_k=3, only=None):
        ids = only or list(self.entries)
        tensors = {}  # cache tensor per jenis preprocessing (dipakai ulang antar model)
        img_hash = hashlib.md5(image.tobytes()).hexdigest()
        results = []

        for mid in ids:
            e = self.entries[mid]
            self._ensure_loaded(mid)
            base = {
                "id": mid, "arch": e["arch"], "arch_label": e["arch_label"],
                "version": e["version"], "version_label": e["version_label"],
            }
            st = self.status(mid)

            if st == "ready":
                pp = e["preprocess"]
                if pp not in tensors:
                    tensors[pp] = TRANSFORMS[pp](image).unsqueeze(0).to(DEVICE)
                t0 = time.perf_counter()
                with torch.inference_mode():
                    probs = torch.softmax(self.models[mid](tensors[pp]), dim=1)[0].cpu().numpy()
                ms = (time.perf_counter() - t0) * 1000
            elif st == "demo":
                probs, ms = self._fake_probs(img_hash, mid), 0.0
            else:
                results.append({**base, "status": st, "error": self.errors.get(mid)})
                continue

            order = np.argsort(probs)[::-1]
            results.append({
                **base,
                "status": "ok" if st == "ready" else "demo",
                "top1": {"label": CLASSES[order[0]], "prob": float(probs[order[0]])},
                "topk": [{"label": CLASSES[i], "prob": float(probs[i])} for i in order[:top_k]],
                "probs": {c: float(p) for c, p in zip(CLASSES, probs)},
                "inference_ms": round(ms, 1),
            })

        return results, self._summarize(results)

    @staticmethod
    def _summarize(results):
        # Ringkasan hanya dari model ASLI; prediksi demo dipakai hanya kalau tak ada model asli sama sekali
        done = [r for r in results if r["status"] == "ok"] or [r for r in results if r["status"] == "demo"]
        if not done:
            return None
        votes = Counter(r["top1"]["label"] for r in done)
        label, n = votes.most_common(1)[0]
        # Ensemble sederhana: rata-rata probabilitas semua model
        avg = {c: float(np.mean([r["probs"][c] for r in done])) for c in CLASSES}
        ens = sorted(avg.items(), key=lambda kv: kv[1], reverse=True)[:3]
        timed = [r for r in done if r["inference_ms"] > 0]
        return {
            "consensus_label": label,
            "votes": n,
            "total_models": len(done),
            "agreement": round(n / len(done), 3),
            "vote_breakdown": dict(votes.most_common()),
            "ensemble_top": [{"label": c, "prob": p} for c, p in ens],
            "most_confident": max(done, key=lambda r: r["top1"]["prob"])["id"],
            "fastest": min(timed, key=lambda r: r["inference_ms"])["id"] if timed else None,
        }

    @staticmethod
    def _fake_probs(img_hash, mid):
        """Prediksi palsu (DEMO). Konsisten untuk gambar & model yang sama."""
        seed = int(hashlib.md5((img_hash + mid).encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        base = int(img_hash[:4], 16) % len(CLASSES)   # semua model "condong" ke 1 kelas
        logits = rng.normal(0, 1.2, len(CLASSES))
        logits[base] += rng.uniform(1.0, 3.5)
        p = np.exp(logits - logits.max())
        return p / p.sum()


manager = ModelManager()
