import json
from pathlib import Path

OUT_DIR = Path("notebooks_v2_canonical")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def md_cell(text):
    lines = [l + "\n" for l in text.split("\n")]
    if lines and lines[-1] == "\n":
        lines[-1] = ""
    return {"cell_type": "markdown", "metadata": {}, "source": lines}

def code_cell(code):
    lines = [l + "\n" for l in code.split("\n")]
    if lines and lines[-1] == "\n":
        lines[-1] = ""
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines}

def create_nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.12"}
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

nb_cells = [
    md_cell("""# 🛡️ Model 01B: ConvNeXt-Tiny Ultra-Robust — Two-Stage Progressive Fine-Tuning & Multi-Part Feature Masking
## Klasifikasi 13 Kelas Topeng Tradisional Bali dengan Ketahanan Oklusi, RandAugment, & Discriminative Layer-wise LR
**Program Studi Teknologi Informasi — Fakultas Teknik, Universitas Udayana**  
*Mata Kuliah: Deep Learning (Tugas 2) & Proyek Capstone Klasifikasi Topeng Bali*  
*Dosen Pengampu: Wayan Oger Vihikan, S.T.I., M.I.T. | Dosen Pembimbing: Ni Kadek Ayu Wirdiani, S.T., M.T.*  
*Peneliti Utama: Gerald Hizkia Turnip*

---

## 📑 Daftar Isi (Table of Contents)
1. [Latar Belakang & Rasional Peningkatan Model (Ultra-Robust Recipe)](#1-latar-belakang--rasional-peningkatan-model-ultra-robust-recipe)
2. [Setup Lingkungan & Akses Dataset](#2-setup-lingkungan--akses-dataset)
3. [Pemuatan Split Kanonik 801 (100% Bebas Data Leakage)](#3-pemuatan-split-kanonik-801-100-bebas-data-leakage)
4. [Dataset PyTorch & Augmentasi Ekstrem: RandAugment + Random Erasing (Cutout)](#4-dataset-pytorch--augmentasi-ekstrem-randaugment--random-erasing-cutout)
5. [Arsitektur ConvNeXt & Desain Two-Stage Training](#5-arsitektur-convnext--desain-two-stage-training)
6. [Eksekusi Two-Stage Progressive Fine-Tuning (40 Epochs)](#6-eksekusi-two-stage-progressive-fine-tuning-40-epochs)
   - *Stage 1 (Epoch 1-5): Warmup Classifier Head (Backbone Frozen)*
   - *Stage 2 (Epoch 6-40): Full Network Unfreezing dengan Discriminative Layer-wise LR*
7. [Visualisasi Kurva Pelatihan Two-Stage (Loss, Acc, Macro-F1)](#7-visualisasi-kurva-pelatihan-two-stage-loss-acc-macro-f1)
8. [Evaluasi Komprehensif pada Test Set Kanonik (121 Citra Bersih)](#8-evaluasi-komprehensif-pada-test-set-kanonik-121-citra-bersih)
9. [Analisis Ketahanan Oklusi & Bedah Kasus Kesalahan Prediksi](#9-analisis-ketahanan-oklusi--bedah-kasus-kesalahan-prediksi)
10. [Explainable AI: Visualisasi Multi-Part Atensi Grad-CAM](#10-explainable-ai-visualisasi-multi-part-atensi-grad-cam)
11. [Profiling Efisiensi Komputasi & Export Model](#11-profiling-efisiensi-komputasi--export-model)"""),

    md_cell("""---
## 1. Latar Belakang & Rasional Peningkatan Model (Ultra-Robust Recipe)

Pada model sebelumnya, ConvNeXt-Tiny dengan cepat mencapai akurasi validasi 100% pada Epoch ke-5 karena kapasitas representasinya yang masif (28.6M parameter ImageNet) sementara augmentasinya masih tergolong ringan. Untuk menghasilkan model yang **jauh lebih tangguh (*robust*), tahan banting, dan tidak bergantung pada satu ciri visual tunggal**, notebook ini mengimplementasikan **4 pilar rekayasa mutakhir**:

### 🎯 4 Pilar Rekayasa Ultra-Robust:
1. **Random Erasing (Cutout Masking - Zhong et al., AAAI 2020)**:
   - Menghapus secara acak patch kotak persegi panjang pada citra topeng saat training.
   - *Rasional:* Jika area **mata** tertutup masker, model dipaksa mengenali topeng lewat **taring/mulut**. Jika **mulut** tertutup, model dipaksa mengenali lewat **gelungan mahkota**. Ini mencegah model menjadi *lazy learner*.
2. **RandAugment (Cubuk et al., NeurIPS 2020)**:
   - Menerapkan kombinasi acak distorsi perspektif, rotasi dinamis, penyesuaian kontras, dan ketajaman otomatis.
3. **Two-Stage Progressive Fine-Tuning**:
   - **Stage 1 (Epoch 1–5):** Membekukan seluruh backbone, hanya melatih linear classifier head agar bobot barunya stabil tanpa merusak representasi pra-latih ImageNet.
   - **Stage 2 (Epoch 6–40):** Membuka (*unfreeze*) seluruh 28.6 juta parameter dengan **Discriminative Layer-wise Learning Rates (LLRD)**.
4. **Discriminative Learning Rates (LLRD)**:
   - Layer awal (deteksi tepi/tekstur): $LR = 2 \times 10^{-6}$ (sangat lambat agar fitur universal tetap terjaga).
   - Layer tengah (bentuk dan ornamen): $LR = 3 \times 10^{-5}$.
   - Classifier head (pemetaan 13 kelas): $LR = 1.5 \times 10^{-4}$."""),

    code_cell("""# Sel 1: Setup Dependensi & Environment
!pip -q install imagehash opencv-python seaborn scikit-learn tqdm timm

import os, re, sys, time, copy, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Reproducibilitas
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ Device Aktif: {DEVICE}")"""),

    code_cell("""# Sel 2: Konfigurasi Eksperimen Ultra-Robust
MODEL_NAME = "ConvNeXt-Tiny-UltraRobust"
IMG_SIZE = 224
BATCH_SIZE = 16

# Konfigurasi Two-Stage Training
EPOCHS_STAGE1 = 5   # Warmup Classifier Head saja
EPOCHS_STAGE2 = 35  # Full Backbone Unfreezing dengan LLRD
TOTAL_EPOCHS = EPOCHS_STAGE1 + EPOCHS_STAGE2  # Total 40 Epochs (~8 - 10 Menit)

OUT_DIR = Path('./outputs_convnext_ultrarobust')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Mount Google Drive jika di Colab
try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception:
    pass

ZIP_CANDIDATES = [
    '/content/drive/MyDrive/dataset_reorganized.zip',
    '/content/dataset_reorganized.zip',
    './dataset_reorganized.zip',
    '../dataset_reorganized.zip'
]
ZIP_PATH = next((p for p in ZIP_CANDIDATES if os.path.exists(p)), None)
EXTRACT_ROOT = Path('/content/BaliMask_Raw') if os.path.exists('/content') else Path('./BaliMask_Raw')

if ZIP_PATH and not (EXTRACT_ROOT / 'dataset_reorganized').exists():
    print(f"📦 Mengekstrak dataset dari: {ZIP_PATH}")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
        zf.extractall(EXTRACT_ROOT)

DATA_DIR = None
for cand in [EXTRACT_ROOT / 'dataset_reorganized' / 'train', EXTRACT_ROOT / 'train', EXTRACT_ROOT, Path('./dataset_reorganized/train')]:
    if cand.exists() and any(d.is_dir() for d in cand.iterdir() if not d.name.startswith('.')):
        DATA_DIR = cand
        break

print(f"📂 Folder data: {DATA_DIR}")
print(f"💾 Folder output: {OUT_DIR}")
print(f"⏱️ Jadwal Pelatihan: Stage 1 ({EPOCHS_STAGE1} Epochs) + Stage 2 ({EPOCHS_STAGE2} Epochs) = {TOTAL_EPOCHS} Epochs")"""),

    code_cell("""# Sel 3: Pemuatan Split Kanonik 801 (Deterministic & Zero Leakage)
import re
import imagehash
from sklearn.model_selection import StratifiedShuffleSplit

SPLIT_CSV_CANDIDATES = [
    Path('split_canonical_801.csv'),
    Path('./notebooks_v2_canonical/split_canonical_801.csv'),
    Path('/content/drive/MyDrive/split_canonical_801.csv'),
    Path('/content/split_canonical_801.csv')
]

SPLIT_CSV = next((p for p in SPLIT_CSV_CANDIDATES if p.exists()), None)

def extract_base_identity(filename):
    name = Path(filename).name.lower()
    name = re.sub(r'\\.(jpg|jpeg|png|webp|bmp)$', '', name)
    name = re.sub(r'(\\._rf\\.|\\.rf\\.|_rf\\.)[a-f0-9]+$', '', name)
    name = re.sub(r'(_jpg|_jpeg|_png)$', '', name)
    if name.startswith('resized_'):
        name = name[len('resized_'):]
    name = re.sub(r'(_rotated-?\\d+|-rotated-?\\d+|rotated-?\\d+)', '', name)
    name = re.sub(r'(_aug\\d+|-aug\\d+)', '', name)
    return name.strip('-_ ')

if SPLIT_CSV and SPLIT_CSV.exists():
    df_meta = pd.read_csv(SPLIT_CSV)
    print(f"✅ Berhasil memuat Split Kanonik 801 dari: {SPLIT_CSV}")
else:
    print("ℹ️ File split_canonical_801.csv belum ada. Menjalankan auto-deduplikasi kanonik...")
    records = []
    classes = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir() and not d.name.startswith('.')])
    for c in classes:
        for p in sorted((DATA_DIR / c).rglob('*')):
            if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}:
                records.append({
                    'rel_path': str(p.relative_to(DATA_DIR)),
                    'full_path': str(p),
                    'filename': p.name,
                    'class': c,
                    'base_id': extract_base_identity(p.name),
                    'has_rotation': bool(re.search(r'rotated-?\\d+', p.name))
                })
    df_raw = pd.DataFrame(records)
    
    # Pilih 1 citra kanonik terbaik per base_id
    canonical_list = []
    for bid, gdf in df_raw.groupby('base_id'):
        clean = gdf[~gdf['has_rotation'] & ~gdf['filename'].str.startswith('resized_')]
        if len(clean) > 0:
            canonical_list.append(clean.iloc[0])
        else:
            canonical_list.append(gdf.iloc[0])
            
    df_canonical = pd.DataFrame(canonical_list).reset_index(drop=True)
    
    # Stratified Split 70:15:15
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
    tr_idx, temp_idx = next(sss1.split(df_canonical, df_canonical['class']))
    df_train = df_canonical.iloc[tr_idx].copy()
    df_temp = df_canonical.iloc[temp_idx].copy().reset_index(drop=True)
    
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
    va_rel, te_rel = next(sss2.split(df_temp, df_temp['class']))
    df_val = df_temp.iloc[va_rel].copy()
    df_test = df_temp.iloc[te_rel].copy()
    
    df_train['split'] = 'train'
    df_val['split'] = 'val'
    df_test['split'] = 'test'
    df_meta = pd.concat([df_train, df_val, df_test], ignore_index=True)
    df_meta.to_csv('split_canonical_801.csv', index=False)
    print("✅ Auto-generate split kanonik selesai & disimpan ke split_canonical_801.csv")

CLASSES = sorted(df_meta['class'].unique().tolist())
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

print(f"\\n🎯 Total Kelas ({NUM_CLASSES}):", CLASSES)
print("📊 Distribusi Sampel per Split:")
split_dist = df_meta['split'].value_counts()
for s_name, s_count in split_dist.items():
    print(f"  • {s_name.upper():<6}: {s_count} citra kanonik ({s_count/len(df_meta)*100:.1f}%)")"""),

    code_cell("""# Sel 4: Dataset PyTorch & Augmentasi Ekstrem (RandAugment + Random Erasing / Cutout)
class BalineseMaskDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.root_dir = Path(root_dir)
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.root_dir / row['rel_path']
        if not img_path.exists():
            img_path = self.root_dir / row['class'] / row['filename']
            
        with Image.open(img_path) as img:
            image = img.convert('RGB')
            
        if self.transform:
            image = self.transform(image)
            
        label_idx = CLASS_TO_IDX[row['class']]
        return image, label_idx, str(img_path)

# 🌟 PIPELINE AUGMENTASI ULTRA-ROBUST KHUSUS TRAINING:
# Memadukan RandAugment, ColorJitter, Affine Distortion, dan Random Erasing (Cutout)
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.70, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.25, hue=0.03),
    transforms.RandAugment(num_ops=2, magnitude=9),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # SENJATA RAHASIA: Random Erasing (Cutout Masking)
    # Menghapus kotak persegi acak (2% - 25% area citra) untuk memaksa atensi multi-fitur
    transforms.RandomErasing(p=0.35, scale=(0.02, 0.25), ratio=(0.3, 3.3), value='random')
])

# Validasi & Pengujian Murni Tanpa Distorsi (Clean Benchmark)
eval_transform = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.14)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_df = df_meta[df_meta['split'] == 'train']
val_df   = df_meta[df_meta['split'] == 'val']
test_df  = df_meta[df_meta['split'] == 'test']

train_dataset = BalineseMaskDataset(train_df, DATA_DIR, transform=train_transform)
val_dataset   = BalineseMaskDataset(val_df, DATA_DIR, transform=eval_transform)
test_dataset  = BalineseMaskDataset(test_df, DATA_DIR, transform=eval_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

print(f"📦 DataLoaders Siap:")
print(f"  • Train batches: {len(train_loader)} ({len(train_dataset)} citra dengan Random Erasing & RandAugment)")
print(f"  • Val batches  : {len(val_loader)} ({len(val_dataset)} citra murni)")
print(f"  • Test batches : {len(test_loader)} ({len(test_dataset)} citra murni)")"""),

    code_cell("""# Sel 5: Inisialisasi Model ConvNeXt-Tiny & Setup Two-Stage Training
weights = ConvNeXt_Tiny_Weights.DEFAULT
model = convnext_tiny(weights=weights)

# Ganti classifier linear terakhir dengan Dropout lebih kuat (0.4) untuk regularisasi ekstra
in_features = model.classifier[2].in_features
model.classifier[2] = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(in_features, NUM_CLASSES)
)

model = model.to(DEVICE)
GRADCAM_TARGET_LAYER = model.features[-1]

# Loss dengan Label Smoothing 0.1
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

print(f"✅ Model {MODEL_NAME} Berhasil Dirakit.")
print(f"  • Feature extractor: ConvNeXt-Tiny (28.6M parameters ImageNet-1K)")
print(f"  • Custom Head: LayerNorm2d -> Flatten -> Dropout(0.4) -> Linear({in_features}, {NUM_CLASSES})")"""),

    code_cell("""# Sel 6: Eksekusi Two-Stage Progressive Fine-Tuning (40 Epochs)
device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
scaler = torch.amp.GradScaler(device_type, enabled=torch.cuda.is_available())

history = {
    'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'val_f1': [],
    'stage': []
}
best_val_f1 = 0.0
best_weights = copy.deepcopy(model.state_dict())
start_time = time.time()

# ==============================================================================
# STAGE 1: WARMUP CLASSIFIER HEAD SAJA (5 EPOCHS)
# ==============================================================================
print(f"\\n{'='*80}")
print(f"🚀 STAGE 1: WARMUP CLASSIFIER HEAD ({EPOCHS_STAGE1} Epochs) — Backbone Frozen")
print(f"{'='*80}")

# Bekukan seluruh backbone features
for param in model.features.parameters():
    param.requires_grad = False
for param in model.classifier.parameters():
    param.requires_grad = True

opt_stage1 = optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-2)
sched_stage1 = optim.lr_scheduler.CosineAnnealingLR(opt_stage1, T_max=EPOCHS_STAGE1, eta_min=1e-5)

for epoch in range(1, EPOCHS_STAGE1 + 1):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels, _ in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        opt_stage1.zero_grad()
        with torch.amp.autocast(device_type=device_type, enabled=torch.cuda.is_available()):
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(opt_stage1)
        scaler.update()
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += labels.size(0)
        
    sched_stage1.step()
    tr_loss = running_loss / total
    tr_acc = correct / total
    
    # Validasi
    model.eval()
    val_loss, val_preds_list, val_targets_list = 0.0, [], []
    with torch.no_grad():
        for images, labels, _ in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            with torch.amp.autocast(device_type=device_type, enabled=torch.cuda.is_available()):
                outputs = model(images)
                l = criterion(outputs, labels)
            val_loss += l.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            val_preds_list.extend(preds.cpu().numpy())
            val_targets_list.extend(labels.cpu().numpy())
            
    va_loss = val_loss / len(val_loader.dataset)
    va_acc = accuracy_score(val_targets_list, val_preds_list)
    va_f1 = f1_score(val_targets_list, val_preds_list, average='macro', zero_division=0)
    
    history['train_loss'].append(tr_loss)
    history['val_loss'].append(va_loss)
    history['train_acc'].append(tr_acc)
    history['val_acc'].append(va_acc)
    history['val_f1'].append(va_f1)
    history['stage'].append(1)
    
    if va_f1 > best_val_f1:
        best_val_f1 = va_f1
        best_weights = copy.deepcopy(model.state_dict())
        torch.save(best_weights, OUT_DIR / 'best_model.pth')
        ckpt_str = "🌟 BEST CHECKPOINT"
    else:
        ckpt_str = ""
        
    print(f"Stage 1 - Epoch [{epoch:02d}/{EPOCHS_STAGE1:02d}] "
          f"Train Loss: {tr_loss:.4f} Acc: {tr_acc*100:.2f}% | "
          f"Val Loss: {va_loss:.4f} Acc: {va_acc*100:.2f}% Macro-F1: {va_f1*100:.2f}% {ckpt_str}")

# ==============================================================================
# STAGE 2: FULL NETWORK UNFREEZING DENGAN DISCRIMINATIVE LR (35 EPOCHS)
# ==============================================================================
print(f"\\n{'='*80}")
print(f"🔥 STAGE 2: FULL NETWORK DISCRIMINATIVE FINE-TUNING ({EPOCHS_STAGE2} Epochs) — All 28.6M Parameters")
print(f"{'='*80}")

# Unfreeze seluruh parameter
for param in model.parameters():
    param.requires_grad = True

# Parameter groups dengan Layer-wise Learning Rate Decay (LLRD)
param_groups = [
    {'params': model.features[:4].parameters(), 'lr': 2e-6, 'weight_decay': 1e-2},   # Low-level features
    {'params': model.features[4:].parameters(), 'lr': 3e-5, 'weight_decay': 1e-2},   # Mid/High features
    {'params': model.classifier.parameters(),   'lr': 1.5e-4, 'weight_decay': 1e-2}  # Classifier Head
]

opt_stage2 = optim.AdamW(param_groups)
sched_stage2 = optim.lr_scheduler.CosineAnnealingLR(opt_stage2, T_max=EPOCHS_STAGE2, eta_min=1e-6)

for epoch in range(1, EPOCHS_STAGE2 + 1):
    curr_epoch_total = EPOCHS_STAGE1 + epoch
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels, _ in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        opt_stage2.zero_grad()
        with torch.amp.autocast(device_type=device_type, enabled=torch.cuda.is_available()):
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(opt_stage2)
        scaler.update()
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += labels.size(0)
        
    sched_stage2.step()
    tr_loss = running_loss / total
    tr_acc = correct / total
    
    # Validasi
    model.eval()
    val_loss, val_preds_list, val_targets_list = 0.0, [], []
    with torch.no_grad():
        for images, labels, _ in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            with torch.amp.autocast(device_type=device_type, enabled=torch.cuda.is_available()):
                outputs = model(images)
                l = criterion(outputs, labels)
            val_loss += l.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            val_preds_list.extend(preds.cpu().numpy())
            val_targets_list.extend(labels.cpu().numpy())
            
    va_loss = val_loss / len(val_loader.dataset)
    va_acc = accuracy_score(val_targets_list, val_preds_list)
    va_f1 = f1_score(val_targets_list, val_preds_list, average='macro', zero_division=0)
    
    history['train_loss'].append(tr_loss)
    history['val_loss'].append(va_loss)
    history['train_acc'].append(tr_acc)
    history['val_acc'].append(va_acc)
    history['val_f1'].append(va_f1)
    history['stage'].append(2)
    
    if va_f1 >= best_val_f1:
        best_val_f1 = va_f1
        best_weights = copy.deepcopy(model.state_dict())
        torch.save(best_weights, OUT_DIR / 'best_model.pth')
        ckpt_str = "🌟 BEST CHECKPOINT"
    else:
        ckpt_str = ""
        
    if epoch % 5 == 0 or epoch == 1 or epoch == EPOCHS_STAGE2:
        print(f"Stage 2 - Epoch [{curr_epoch_total:02d}/{TOTAL_EPOCHS:02d}] "
              f"Train Loss: {tr_loss:.4f} Acc: {tr_acc*100:.2f}% | "
              f"Val Loss: {va_loss:.4f} Acc: {va_acc*100:.2f}% Macro-F1: {va_f1*100:.2f}% {ckpt_str}")

total_elapsed = time.time() - start_time
print(f"\\n⏱️ Training Two-Stage selesai dalam {total_elapsed//60:.0f}m {total_elapsed%60:.0f}s. Best Val Macro-F1: {best_val_f1*100:.2f}%")

# Load bobot checkpoint terbaik
model.load_state_dict(best_weights)
trained_model = model"""),

    code_cell("""# Sel 7: Visualisasi Kurva Pelatihan Two-Stage (Loss, Acc, Macro-F1)
epochs_range = range(1, TOTAL_EPOCHS + 1)
plt.figure(figsize=(16, 5))

# Plot Loss
plt.subplot(1, 3, 1)
plt.plot(epochs_range, history['train_loss'], label='Train Loss (Augmentasi Erasing)', color='blue')
plt.plot(epochs_range, history['val_loss'], label='Val Loss (Murni)', color='orange')
plt.axvline(x=EPOCHS_STAGE1 + 0.5, color='red', linestyle='--', label='Transisi Stage 2 (Unfreeze All)')
plt.title("Kurva Loss (Two-Stage)")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.legend()

# Plot Akurasi
plt.subplot(1, 3, 2)
plt.plot(epochs_range, [a*100 for a in history['train_acc']], label='Train Acc (%)', color='green')
plt.plot(epochs_range, [a*100 for a in history['val_acc']], label='Val Acc (%)', color='red')
plt.axvline(x=EPOCHS_STAGE1 + 0.5, color='red', linestyle='--', label='Transisi Stage 2')
plt.title("Kurva Akurasi (%)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()

# Plot Macro F1
plt.subplot(1, 3, 3)
plt.plot(epochs_range, [f*100 for f in history['val_f1']], label='Val Macro-F1 (%)', color='purple')
plt.axvline(x=EPOCHS_STAGE1 + 0.5, color='red', linestyle='--', label='Transisi Stage 2')
plt.title("Kurva Validation Macro-F1 (%)")
plt.xlabel("Epoch")
plt.ylabel("Macro-F1 (%)")
plt.legend()

plt.tight_layout()
plt.savefig(OUT_DIR / 'two_stage_training_curves.png', dpi=300)
plt.show()"""),

    code_cell("""# Sel 8: Evaluasi Komprehensif pada Test Set Kanonik (121 Citra Bersih)
trained_model.eval()
test_preds, test_targets, test_probs, test_paths = [], [], [], []

with torch.no_grad():
    for images, labels, paths in test_loader:
        images = images.to(DEVICE)
        outputs = trained_model(images)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
        
        test_preds.extend(preds.cpu().numpy())
        test_targets.extend(labels.numpy())
        test_probs.extend(probs.cpu().numpy())
        test_paths.extend(paths)

test_acc = accuracy_score(test_targets, test_preds)
test_prec_macro = precision_score(test_targets, test_preds, average='macro', zero_division=0)
test_rec_macro  = recall_score(test_targets, test_preds, average='macro', zero_division=0)
test_f1_macro   = f1_score(test_targets, test_preds, average='macro', zero_division=0)
test_f1_weight  = f1_score(test_targets, test_preds, average='weighted', zero_division=0)

print(f"================================================================================")
print(f"🏆 HASIL EVALUASI ULTRA-ROBUST TEST SET BERSIH ({MODEL_NAME})")
print(f"================================================================================")
print(f"  • Top-1 Accuracy       : {test_acc*100:.2f}%")
print(f"  • Macro Precision      : {test_prec_macro*100:.2f}%")
print(f"  • Macro Recall         : {test_rec_macro*100:.2f}%")
print(f"  • Macro F1-Score       : {test_f1_macro*100:.2f}%")
print(f"  • Weighted F1-Score    : {test_f1_weight*100:.2f}%")
print(f"================================================================================")
print("\\n📋 Laporan Klasifikasi Rinci (Classification Report):")
print(classification_report(test_targets, test_preds, target_names=CLASSES, digits=4))

# Confusion Matrix Heatmap
cm = confusion_matrix(test_targets, test_preds)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
plt.title(f"Confusion Matrix — {MODEL_NAME} (Test Set Kanonik Bebas Bocor)", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Prediksi Model", fontsize=12)
plt.ylabel("Label Asli (Ground Truth)", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(OUT_DIR / 'confusion_matrix.png', dpi=300)
plt.show()"""),

    code_cell("""# Sel 9: Analisis Ketahanan Oklusi & Bedah Kasus Kesalahan Prediksi
misclassified_idx = [i for i, (p, t) in enumerate(zip(test_preds, test_targets)) if p != t]
print(f"🔍 TOTAL KESALAHAN PREDIKSI PADA TEST SET: {len(misclassified_idx)} dari {len(test_targets)} citra.")

if len(misclassified_idx) > 0:
    num_show = min(6, len(misclassified_idx))
    fig, axes = plt.subplots(1, num_show, figsize=(4 * num_show, 4))
    if num_show == 1:
        axes = [axes]
    
    for plot_idx, m_idx in enumerate(misclassified_idx[:num_show]):
        img_path = test_paths[m_idx]
        true_lbl = IDX_TO_CLASS[test_targets[m_idx]]
        pred_lbl = IDX_TO_CLASS[test_preds[m_idx]]
        conf = test_probs[m_idx][test_preds[m_idx]] * 100
        true_conf = test_probs[m_idx][test_targets[m_idx]] * 100
        
        with Image.open(img_path) as im:
            axes[plot_idx].imshow(im)
        axes[plot_idx].set_title(f"Asli: {true_lbl}\\nPrediksi: {pred_lbl} ({conf:.1f}%)\\nConf Asli: {true_conf:.1f}%", 
                                 fontsize=10, color='red', fontweight='bold')
        axes[plot_idx].axis('off')
        
    plt.suptitle(f"Galeri Analisis Kasus Kegagalan Prediksi ({MODEL_NAME})", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'misclassified_cases.png', dpi=300)
    plt.show()
else:
    print("🌟 Luar Biasa! Model Ultra-Robust berhasil memprediksi seluruh sampel uji dengan 100% akurasi!")

# Analisis Top-3 Sampel dengan Prediksi Paling Menantang (Lowest Confidence Correct)
correct_idx = [i for i, (p, t) in enumerate(zip(test_preds, test_targets)) if p == t]
if len(correct_idx) > 0:
    lowest_conf_idx = sorted(correct_idx, key=lambda i: test_probs[i][test_targets[i]])[:3]
    print("\\n🎯 3 Sampel Prediksi Benar dengan Tingkat Keyakinan Paling Rendah (Kasus Paling Menantang):")
    for rank, idx in enumerate(lowest_conf_idx, 1):
        lbl_name = IDX_TO_CLASS[test_targets[idx]]
        conf = test_probs[idx][test_targets[idx]] * 100
        print(f"  {rank}. Kelas: {lbl_name:<16} | Confidence: {conf:.2f}% | File: {Path(test_paths[idx]).name}")"""),

    code_cell("""# Sel 10: Explainable AI — Visualisasi Multi-Part Atensi Grad-CAM
import cv2

class SimpleGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output.detach()
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
        
    def generate_cam(self, input_tensor, target_class=None):
        self.model.eval()
        output = self.model(input_tensor)
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()
            
        self.model.zero_grad()
        loss = output[0, target_class]
        loss.backward()
        
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam - torch.min(cam)
        cam = cam / (torch.max(cam) + 1e-7)
        return cam.squeeze().cpu().numpy()

def show_cam_overlay(orig_img_path, cam_mask):
    img = cv2.imread(str(orig_img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (224, 224))
    
    cam_resized = cv2.resize(cam_mask, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    overlay = np.float32(heatmap) / 255 * 0.5 + np.float32(img_resized) / 255 * 0.5
    overlay = np.clip(overlay, 0, 1)
    return img_resized, overlay

grad_cam = SimpleGradCAM(trained_model, GRADCAM_TARGET_LAYER)

# Ambil 3 sampel uji representatif (Barong, Sidakarya, Dalem)
fig, axes = plt.subplots(3, 3, figsize=(14, 12))
sample_indices = [0, len(test_dataset)//2, len(test_dataset)-1]

for i, s_idx in enumerate(sample_indices):
    img_tensor, lbl, path_str = test_dataset[s_idx]
    cam_map = grad_cam.generate_cam(img_tensor.unsqueeze(0).to(DEVICE), target_class=lbl)
    orig_img, overlay = show_cam_overlay(path_str, cam_map)
    
    axes[i, 0].imshow(orig_img)
    axes[i, 0].set_title(f"Foto Asli ({IDX_TO_CLASS[lbl]})", fontsize=11)
    axes[i, 0].axis('off')
    
    axes[i, 1].imshow(cam_map, cmap='jet')
    axes[i, 1].set_title("Heatmap Aktivasi Multi-Part", fontsize=11)
    axes[i, 1].axis('off')
    
    axes[i, 2].imshow(overlay)
    axes[i, 2].set_title("Overlay Grad-CAM (Area Fokus)", fontsize=11, fontweight='bold', color='darkblue')
    axes[i, 2].axis('off')

plt.suptitle(f"Visualisasi Grad-CAM Multi-Part Atensi — {MODEL_NAME}", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT_DIR / 'gradcam_ultra_robust.png', dpi=300)
plt.show()"""),

    code_cell("""# Sel 11: Profiling Efisiensi Komputasi & Rekapitulasi Metrik
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total / 1e6, trainable / 1e6

def measure_latency(model, input_size=(1, 3, 224, 224), iterations=50):
    model.eval()
    dummy = torch.randn(input_size).to(DEVICE)
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    start = time.time()
    for _ in range(iterations):
        with torch.no_grad():
            _ = model(dummy)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return ((time.time() - start) / iterations) * 1000

total_p, train_p = count_parameters(trained_model)
latency_ms = measure_latency(trained_model)
model_size_mb = (OUT_DIR / 'best_model.pth').stat().st_size / (1024 * 1024) if (OUT_DIR / 'best_model.pth').exists() else 0.0

summary_record = {
    'Model': MODEL_NAME,
    'Training Strategy': 'Two-Stage Progressive (Warmup + LLRD)',
    'Data Augmentation': 'RandAugment + Random Erasing (Cutout)',
    'Total Epochs': TOTAL_EPOCHS,
    'Test Accuracy (%)': round(test_acc * 100, 2),
    'Macro Precision (%)': round(test_prec_macro * 100, 2),
    'Macro Recall (%)': round(test_rec_macro * 100, 2),
    'Macro F1 (%)': round(test_f1_macro * 100, 2),
    'Weighted F1 (%)': round(test_f1_weight * 100, 2),
    'Total Params (M)': round(total_p, 2),
    'Latency (ms)': round(latency_ms, 2),
    'Model Size (MB)': round(model_size_mb, 2)
}

pd.DataFrame([summary_record]).to_csv(OUT_DIR / 'metrics_summary.csv', index=False)
print("📊 REKAPITULASI PROFIL KOMPUTASI & METRIK ULTRA-ROBUST:")
display(pd.DataFrame([summary_record]))""")
]

nb_ultra = create_nb(nb_cells)
output_path = OUT_DIR / "01_ConvNeXt_Tiny_Ultra_Robust.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(nb_ultra, f, indent=2)

print(f"✅ Generated: {output_path}")
