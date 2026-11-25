"""
config.py - "Daftar isi" semua model.

Kalau nanti kamu menambah / mengganti model, cukup ubah file ini.
API tidak perlu diubah sama sekali.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Lokasi folder
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent          # folder balimask-api/
WEIGHTS_DIR = Path(os.getenv("BALIMASK_WEIGHTS", BASE_DIR / "weights"))
STATIC_DIR = BASE_DIR / "static"

# ---------------------------------------------------------------------------
# Pengaturan lewat environment variable (opsional)
# ---------------------------------------------------------------------------
# BALIMASK_DEMO=1  -> model yang bobotnya belum ada akan mengeluarkan prediksi
#                     PALSU (acak tapi konsisten). Hanya untuk mencoba tampilan.
# Mode demo otomatis mati begitu ada minimal satu file bobot asli, supaya
# model yang belum dilatih tampil "Belum dilatih" (bukan prediksi palsu).
_ADA_BOBOT = any(WEIGHTS_DIR.rglob("*.pt*")) if WEIGHTS_DIR.exists() else False
DEMO_MODE = os.getenv("BALIMASK_DEMO", "0") == "1" and not _ADA_BOBOT
# BALIMASK_LAZY=1  -> model baru di-load saat pertama kali dipakai (hemat RAM saat start)
LAZY_LOAD = os.getenv("BALIMASK_LAZY", "0") == "1"
# Batas ukuran file upload (MB)
MAX_UPLOAD_MB = float(os.getenv("BALIMASK_MAX_MB", "10"))
# Varian MobileNetV3 di notebook v1 (ablation): none | se_head | cbam_head | cbam_late
MOBILENET_V1_VARIANT = os.getenv("BALIMASK_MNV1_VARIANT", "cbam_head")

# ---------------------------------------------------------------------------
# 13 kelas - URUTANNYA HARUS SAMA dengan saat training
# (di notebook: sorted(nama folder) -> urutan alfabet)
# ---------------------------------------------------------------------------
CLASSES = [
    "barong-bangkal", "barong-ket", "bujuh", "dalem", "jauk-keras",
    "jauk-manis", "keras", "macan", "penasar", "sidakarya",
    "telek", "tua", "wijil",
]
NUM_CLASSES = len(CLASSES)
IMG_SIZE = 224

# ---------------------------------------------------------------------------
# 3 versi training
# ---------------------------------------------------------------------------
VERSIONS = {
    "v1_awal": {
        "label": "Dataset Awal",
        "desc": "Baseline · split Roboflow ±4.400 citra (notebooks_v1_baseline)",
    },
    "v2_clean": {
        "label": "Dataset Clean",
        "desc": "Kanonik 801 citra unik, bebas leakage · 25 epoch",
    },
    "v3_padat": {
        "label": "Training Padat",
        "desc": "Ultra-Robust · 2 tahap (5+35 epoch) + Random Erasing",
    },
}

# ---------------------------------------------------------------------------
# 4 arsitektur
# ---------------------------------------------------------------------------
ARCHS = {
    "convnext_tiny":   {"label": "ConvNeXt-Tiny"},
    "efficientnetv2_s": {"label": "EfficientNetV2-S"},
    "mobilenetv3_cbam": {"label": "MobileNetV3 + CBAM"},
    "vgg16":           {"label": "VGG16"},
}

# ---------------------------------------------------------------------------
# Cara preprocessing saat evaluasi (HARUS sama dengan eval_transform di notebook)
#   "resize"      -> Resize((224,224))                      (v1: ConvNeXt, EffNet, VGG16)
#   "resize_crop" -> Resize(255) + CenterCrop(224)          (v1 MobileNet, semua v2 & v3)
# ---------------------------------------------------------------------------
def _preprocess_for(version: str, arch: str) -> str:
    if version == "v1_awal" and arch != "mobilenetv3_cbam":
        return "resize"
    return "resize_crop"


def build_registry():
    """Menghasilkan daftar 12 model (4 arsitektur x 3 versi)."""
    registry = []
    for arch_id, arch in ARCHS.items():
        for ver_id, ver in VERSIONS.items():
            registry.append({
                "id": f"{arch_id}__{ver_id}",           # contoh: vgg16__v2_clean
                "arch": arch_id,
                "arch_label": arch["label"],
                "version": ver_id,
                "version_label": ver["label"],
                "version_desc": ver["desc"],
                # Letakkan file best_model.pth hasil training di sini:
                "weight_path": WEIGHTS_DIR / ver_id / f"{arch_id}.pth",
                "preprocess": _preprocess_for(ver_id, arch_id),
            })
    return registry


MODEL_REGISTRY = build_registry()
