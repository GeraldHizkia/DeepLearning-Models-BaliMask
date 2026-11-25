"""
curves.py - Membaca KURVA TRAINING (loss & akurasi per epoch) dari file notebook.

Kenapa dari notebook?
    Notebook training hanya menyimpan bobot (.pth). Riwayat per epoch cuma
    dicetak ke layar, misalnya:
        Epoch [03/25] Train Loss: 0.41 Acc: 88.10% | Val Loss: 0.52 Acc: 85.00% Macro-F1: 84.20%
    Teks itu ikut tersimpan di file .ipynb (bagian "outputs"). File ini membaca
    teks tersebut lalu mengubahnya menjadi angka untuk grafik garis.

Cara pakai:
    Taruh file .ipynb hasil training (yang sudah dijalankan, ada output-nya)
    di folder  kurva/  . Nama file menentukan versi & arsitektur:
        ...Canonical...      -> v2_clean
        ...Ultra_Robust...   -> v3_padat
        lainnya              -> v1_awal
        convnext / efficientnet / mobilenet / vgg  -> arsitektur
    File di kurva/  lebih diutamakan daripada kurva/bawaan/ (hasil run teman).
"""
import json
import re
from pathlib import Path

from .config import ARCHS, BASE_DIR, MOBILENET_V1_VARIANT, VERSIONS

CURVES_DIR = BASE_DIR / "kurva"

# --- Pola teks per epoch (3 gaya cetak yang berbeda di notebook) -------------
# v1 ConvNeXt / EfficientNet / VGG16 :  [Head] Ep 01/05 | Train Loss: 1.28 Acc: 0.763 | Val Loss: 2.50 Acc: 0.266
RE_V1 = re.compile(r"\[(Head|Fine)\] Ep (\d+)/\d+ \| Train Loss: ([\d.]+) Acc: ([\d.]+) \| "
                   r"Val Loss: ([\d.]+) Acc: ([\d.]+)")
# v1 MobileNet :  [Head Ep 01] Train Loss: 1.07 Acc: 0.834 | Val Loss: 0.88 Acc: 0.908 Macro-F1: 0.8915
RE_MN = re.compile(r"\[(Head|Fine) Ep (\d+)\] Train Loss: ([\d.]+) Acc: ([\d.]+) \| "
                   r"Val Loss: ([\d.]+) Acc: ([\d.]+) Macro-F1: ([\d.]+)")
# v2 / v3 :  (Tahap 1 - )Epoch [01/25] Train Loss: 0.41 Acc: 88.10% | Val Loss: 0.52 Acc: 85.00% Macro-F1: 84.20%
RE_V2 = re.compile(r"(?:Tahap (\d) - )?Epoch \[(\d+)/\d+\] Train Loss: ([\d.]+) Acc: ([\d.]+)% \| "
                   r"Val Loss: ([\d.]+) Acc: ([\d.]+)% Macro-F1: ([\d.]+)%")
RE_TEST_ACC = re.compile(r"(?:Test Accuracy|Top-1 Accuracy)\s*:\s*([\d.]+)%")
RE_TEST_F1 = re.compile(r"Macro(?:-F1 Score|-F1|\s*F1-Score)\s*:\s*([\d.]+)%")
RE_MN_HEADER = re.compile(r"MEMULAI TRAINING: (\S+)")


def identify(filename: str):
    """Nama file notebook -> (versi, arsitektur) atau None."""
    n = filename.lower().replace(" ", "_")
    if "ultra_robust" in n or "ultrarobust" in n:
        ver = "v3_padat"
    elif "canonical" in n:
        ver = "v2_clean"
    else:
        ver = "v1_awal"
    for key, arch in (("convnext", "convnext_tiny"), ("efficientnet", "efficientnetv2_s"),
                      ("effnet", "efficientnetv2_s"), ("mobilenet", "mobilenetv3_cbam"), ("vgg", "vgg16")):
        if key in n:
            return ver, arch
    return None


def _output_text(nb: dict) -> str:
    """Gabungkan semua teks output dari semua sel."""
    parts = []
    for cell in nb.get("cells", []):
        for out in cell.get("outputs", []):
            if "text" in out:
                t = out["text"]
                parts.append("".join(t) if isinstance(t, list) else t)
            elif "data" in out and "text/plain" in out["data"]:
                t = out["data"]["text/plain"]
                parts.append("".join(t) if isinstance(t, list) else t)
    return "\n".join(parts)


def parse_notebook(path: Path, version: str, arch: str):
    """Ambil kurva + metrik test dari satu notebook. None kalau tidak ada data epoch."""
    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    text = _output_text(nb)

    # Notebook v1 MobileNet melatih beberapa varian berurutan -> ambil varian yang dipakai API
    if version == "v1_awal" and arch == "mobilenetv3_cbam":
        segments = re.split(r"(?=🚀 MEMULAI TRAINING: )", text)
        wanted = f"mobilenetv3_{MOBILENET_V1_VARIANT}"
        chosen = [s for s in segments if (m := RE_MN_HEADER.search(s)) and m.group(1) == wanted]
        if chosen:
            text = chosen[-1]

    ep = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_f1": []}
    phases = []  # nama fase per epoch: untuk garis pemisah tahap di grafik
    for line in text.splitlines():
        if m := RE_V2.search(line):
            stage, _, tl, ta, vl, va, vf = m.groups()
            phase = f"Tahap {stage}" if stage else "Training"
            vals = (float(tl), float(ta), float(vl), float(va), float(vf))
        elif m := RE_MN.search(line):
            ph, _, tl, ta, vl, va, vf = m.groups()
            phase = "Head" if ph == "Head" else "Fine-tune"
            vals = (float(tl), float(ta) * 100, float(vl), float(va) * 100, float(vf) * 100)
        elif m := RE_V1.search(line):
            ph, _, tl, ta, vl, va = m.groups()
            phase = "Head" if ph == "Head" else "Fine-tune"
            vals = (float(tl), float(ta) * 100, float(vl), float(va) * 100, None)
        else:
            continue
        for k, v in zip(("train_loss", "train_acc", "val_loss", "val_acc", "val_f1"), vals):
            ep[k].append(None if v is None else round(v, 4))
        phases.append(phase)

    if not phases:
        return None

    test = {}
    for line in text.splitlines():
        if "Epoch [" in line or " Ep " in line or "Val" in line:
            continue
        if m := RE_TEST_ACC.search(line):
            test["accuracy"] = float(m.group(1))
        if m := RE_TEST_F1.search(line):
            test["macro_f1"] = float(m.group(1))

    best_i = max(range(len(phases)), key=lambda i: (ep["val_f1"][i] or 0, ep["val_acc"][i] or 0))
    if all(v is None for v in ep["val_f1"]):
        best_i = min(range(len(phases)), key=lambda i: ep["val_loss"][i])
    boundaries = [i for i in range(1, len(phases)) if phases[i] != phases[i - 1]]

    return {
        "id": f"{arch}__{version}",
        "arch": arch, "arch_label": ARCHS[arch]["label"],
        "version": version, "version_label": VERSIONS[version]["label"],
        "source": str(path.relative_to(CURVES_DIR)).replace("\\", "/"),
        "epochs": len(phases),
        "history": ep,
        "phases": phases,
        "phase_boundaries": boundaries,     # index epoch (0-based) tempat fase berganti
        "best_epoch": best_i + 1,
        "best": {k: ep[k][best_i] for k in ep},
        "test": test or None,
    }


def load_all_curves():
    """Kembalikan dict model_id -> kurva. File di kurva/ menang atas kurva/bawaan/."""
    found = {}
    if not CURVES_DIR.exists():
        return found
    files = sorted(CURVES_DIR.rglob("*.ipynb"),
                   key=lambda p: (0 if "bawaan" in p.parts else 1, p.stat().st_mtime))
    for p in files:                              # yang diproses belakangan menimpa yang awal
        ident = identify(p.name)
        if not ident:
            continue
        data = parse_notebook(p, *ident)
        if data:
            found[data["id"]] = data
    return found
