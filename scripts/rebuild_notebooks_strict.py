"""
Generator skrip untuk membangun notebook ConvNeXt-Tiny Ultra-Robust.
Standar penulisan:
1. Tidak ada emotikon sama sekali.
2. Tidak ada en dash atau em dash (hanya tanda hubung standar -).
3. Struktur heading terpisah per sel markdown (#, ##, ###) agar daftar isi Colab runtut ke bawah.
4. Penjelasan konsep komprehensif pada setiap sel markdown sebelum sel kode.
"""

import json
from pathlib import Path

OUT_DIR = Path("notebooks_v2_canonical")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def md_cell(text):
    text = text.replace("—", "-").replace("–", "-")
    lines = [l + "\n" for l in text.split("\n")]
    if lines and lines[-1] == "\n":
        lines[-1] = ""
    return {"cell_type": "markdown", "metadata": {}, "source": lines}

def code_cell(code):
    code = code.replace("—", "-").replace("–", "-")
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

cells = [
    # Header Notebook
    md_cell("""<a id="top"></a>
# Klasifikasi 13 Kelas Topeng Tradisional Bali Menggunakan ConvNeXt-Tiny
### Pendekatan Fine-Tuning Dua Tahap (Two-Stage Progressive Training) dan Regularisasi Oklusi Multi-Fitur (Random Erasing)

| Informasi Proyek | Keterangan |
|---|---|
| Mata Kuliah | Deep Learning (Tugas 2) & Proyek Capstone |
| Dosen Pengampu | Wayan Oger Vihikan, S.T.I., M.I.T. |
| Dosen Pembimbing | Ni Kadek Ayu Wirdiani, S.T., M.T. |
| Institusi | Program Studi Teknologi Informasi, Fakultas Teknik, Universitas Udayana |
| Peneliti | Gerald Hizkia Turnip |"""),

    # Daftar Isi
    md_cell("""<a id="toc"></a>
## 0. Daftar Isi

| Bagian | Topik Pembahasan | Deskripsi Teknis |
|---|---|---|
| [1](#bab1) | Persiapan Lingkungan dan Konfigurasi | Instalasi dependensi, alokasi GPU, seed reproduksibilitas |
| [2](#bab2) | Integritas Dataset dan Pemuatan Data | Verifikasi split kanonik 801 citra bebas kebocoran data |
| [3](#bab3) | Rekayasa Data dan Augmentasi Multi-Fitur | Implementasi RandAugment dan Random Erasing Cutout |
| [4](#bab4) | Arsitektur Model ConvNeXt-Tiny | Konstruksi backbone, custom classification head, receptive field |
| [5](#bab5) | Pelatihan Dua Tahap (Two-Stage Fine-Tuning) | Warmup classifier head dan unfreezing dengan layer-wise LR |
| [6](#bab6) | Evaluasi Komprehensif pada Data Uji | Pengujian pada 121 citra unseen, confusion matrix, classification report |
| [7](#bab7) | Analisis Kesalahan Prediksi dan Kasus Menantang | Identifikasi sampel salah dan sampel dengan konfidensi terendah |
| [8](#bab8) | Explainable AI: Atensi Spasial Grad-CAM | Visualisasi heatmap fitur distingtif (mata, taring, mahkota) |
| [9](#bab9) | Profil Efisiensi Komputasi dan Ekspor Model | Pengukuran latensi inferensi, parameter, dan ukuran berkas |"""),

    # Bab 1
    md_cell("""<a id="bab1"></a>
## 1. Persiapan Lingkungan dan Konfigurasi
[kembali ke daftar isi](#toc)

Tahap ini memverifikasi ketersediaan akselerator perangkat keras (GPU), memasang pustaka pendukung, dan mengunci parameter acak (random seed) agar seluruh eksperimen bersifat deterministik dan dapat direproduksi secara identik."""),

    md_cell("""### 1.1 Pemasangan Pustaka dan Konfigurasi Perangkat Keras
Pustaka yang digunakan mencakup PyTorch untuk pemodelan deep learning, torchvision untuk transformasi citra dan model pra-latih, scikit-learn untuk metrik evaluasi, serta opencv-python untuk rendering Explainable AI."""),

    code_cell(r"""# 1.1 Pemasangan dependensi dan inisialisasi pustaka
!pip -q install imagehash opencv-python seaborn scikit-learn tqdm timm

import os
import re
import sys
import time
import copy
import zipfile
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
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# Penguncian seed untuk reproduksibilitas penuh
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Perangkat komputasi aktif: {DEVICE}")
if torch.cuda.is_available():
    print(f"Nama GPU: {torch.cuda.get_device_name(0)}")
    print(f"Alokasi Memori: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")"""),

    md_cell("""### 1.2 Konfigurasi Parameter Eksperimen
Parameter pelatihan dikelompokkan ke dalam struktur global:
- MODEL_NAME: Identitas model untuk pencatatan eksperimen.
- IMG_SIZE: Resolusi spasial citra masukan (224x224 piksel standar ImageNet).
- BATCH_SIZE: Ukuran batch mini (16 citra per iterasi gradien).
- EPOCHS_STAGE1: Jumlah epoch pemanasan classifier head (5 epoch).
- EPOCHS_STAGE2: Jumlah epoch fine-tuning seluruh parameter (35 epoch).
- TOTAL_EPOCHS: Total durasi pelatihan (40 epoch)."""),

    code_cell(r"""# 1.2 Inisialisasi parameter eksperimen global
MODEL_NAME = "ConvNeXt-Tiny-UltraRobust"
IMG_SIZE = 224
BATCH_SIZE = 16

EPOCHS_STAGE1 = 5
EPOCHS_STAGE2 = 35
TOTAL_EPOCHS = EPOCHS_STAGE1 + EPOCHS_STAGE2

OUT_DIR = Path('./outputs_convnext_ultrarobust')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Deteksi mount Google Drive
try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception:
    pass

# Deteksi arsip dataset
ZIP_CANDIDATES = [
    '/content/drive/MyDrive/dataset_reorganized.zip',
    '/content/dataset_reorganized.zip',
    './dataset_reorganized.zip',
    '../dataset_reorganized.zip'
]
ZIP_PATH = next((p for p in ZIP_CANDIDATES if os.path.exists(p)), None)
EXTRACT_ROOT = Path('/content/BaliMask_Raw') if os.path.exists('/content') else Path('./BaliMask_Raw')

if ZIP_PATH and not (EXTRACT_ROOT / 'dataset_reorganized').exists():
    print(f"Mengekstrak dataset dari: {ZIP_PATH}")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
        zf.extractall(EXTRACT_ROOT)

DATA_DIR = None
SEARCH_PATHS = [
    EXTRACT_ROOT / 'dataset_reorganized' / 'train',
    EXTRACT_ROOT / 'train',
    EXTRACT_ROOT,
    Path('./dataset_reorganized/train')
]
for cand in SEARCH_PATHS:
    if cand.exists() and any(d.is_dir() for d in cand.iterdir() if not d.name.startswith('.')):
        DATA_DIR = cand
        break

print(f"Direktori data aktif: {DATA_DIR}")
print(f"Direktori luaran: {OUT_DIR}")
print(f"Jadwal pelatihan: Tahap 1 ({EPOCHS_STAGE1} epoch) + Tahap 2 ({EPOCHS_STAGE2} epoch) = {TOTAL_EPOCHS} epoch")"""),

    # Bab 2
    md_cell("""<a id="bab2"></a>
## 2. Integritas Dataset dan Pemuatan Data
[kembali ke daftar isi](#toc)

Dataset publik topeng Bali memiliki kerentanan kebocoran data bawaan (target data leakage) karena ekspor default Roboflow menyebarkan citra hasil augmentasi offline dari satu objek fisik ke dalam folder train, valid, dan test.

Bagian ini memuat partisi kanonik 801 citra fisik unik yang telah dideduplikasi menggunakan regular expression pada nama berkas dan verifikasi visual perceptual hashing (pHash). Pembagian data dilakukan secara stratified:
- Training: 560 citra kanonik (70.0%)
- Validation: 120 citra kanonik (15.0%)
- Testing: 121 citra kanonik (15.1%)"""),

    md_cell("""### 2.1 Pemuatan Berkas Partisi Kanonik
Berkas partisi split_canonical_801.csv dibaca langsung jika sudah tersedia. Jika belum ada, fungsi ekstraksi identitas dasar akan memproses dataset mentah secara deterministik."""),

    code_cell(r"""# 2.1 Pemuatan atau pembentukan berkas partisi kanonik 801 citra
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
    print(f"Memuat partisi kanonik dari: {SPLIT_CSV}")
else:
    print("Membentuk partisi kanonik baru secara deterministik...")
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
    
    canonical_list = []
    for bid, gdf in df_raw.groupby('base_id'):
        clean = gdf[~gdf['has_rotation'] & ~gdf['filename'].str.startswith('resized_')]
        if len(clean) > 0:
            canonical_list.append(clean.iloc[0])
        else:
            canonical_list.append(gdf.iloc[0])
            
    df_canonical = pd.DataFrame(canonical_list).reset_index(drop=True)
    
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
    print("Partisi kanonik selesai dibentuk dan disimpan ke split_canonical_801.csv")

CLASSES = sorted(df_meta['class'].unique().tolist())
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

print(f"Total kelas terdaftar: {NUM_CLASSES}")
print("Daftar kelas:", CLASSES)"""),

    md_cell("""### 2.2 Distribusi Sampel per Kelas dan Subset Data
Tabel berikut menyajikan rekapitulasi jumlah sampel pada masing-masing subset (train, validation, testing) untuk memastikan proporsi kelas seimbang secara stratified."""),

    code_cell(r"""# 2.2 Analisis distribusi kelas pada subset data
dist_table = pd.crosstab(df_meta['class'], df_meta['split'])[['train', 'val', 'test']]
dist_table['total'] = dist_table.sum(axis=1)
display(dist_table)

print("Distribusi total sampel:")
for s_name, s_count in df_meta['split'].value_counts().items():
    print(f"- {s_name.upper():<6}: {s_count} citra ({s_count/len(df_meta)*100:.1f}%)")"""),

    # Bab 3
    md_cell("""<a id="bab3"></a>
## 3. Rekayasa Data dan Augmentasi Multi-Fitur
[kembali ke daftar isi](#toc)

Untuk mencegah model mengalami kejenuhan awal (overfitting pada ciri visual tunggal), subset pelatihan diperkaya dengan augmentasi dinamis (online data augmentation). Subset validasi dan pengujian dipertahankan murni tanpa distorsi."""),

    md_cell("""### 3.1 Rasional Ilmiah Random Erasing (Cutout Masking)
Random Erasing (Zhong et al., 2020) menghapus bidang persegi panjang acak berukuran antara 2% hingga 25% dari luas citra topeng pada saat pelatihan.
- Tujuan: Memaksa model mempelajari representasi multi-fitur. Jika area mata tertutup, model diwajibkan mengenali topeng melalui bentuk taring, mulut, atau gelungan mahkota.
- Parameter: Probabilitas aplikasi p = 0.35, skala luas = (0.02, 0.25), rasio aspek = (0.3, 3.3)."""),

    md_cell("""### 3.2 Konstruksi Dataset dan DataLoader PyTorch
Pipa augmentasi pelatihan menggabungkan:
1. RandomResizedCrop (skala 0.70 hingga 1.0) untuk ketahanan terhadap variasi jarak kamera.
2. RandomHorizontalFlip (p = 0.5) untuk invariansi arah pandang.
3. RandomRotation (hingga 20 derajat) untuk simulasi kemiringan kepala penari.
4. ColorJitter untuk variasi pencahayaan panggung dan kontras kayu.
5. RandAugment (2 operasi dengan magnitude 9) untuk kombinasi distorsi non-linear.
6. RandomErasing setelah normalisasi tensor."""),

    code_cell(r"""# 3.2 Implementasi kelas Dataset dan pembuatan DataLoader
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

# Pipa augmentasi khusus data latih
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.70, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.25, hue=0.03),
    transforms.RandAugment(num_ops=2, magnitude=9),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.35, scale=(0.02, 0.25), ratio=(0.3, 3.3), value='random')
])

# Pipa evaluasi tanpa distorsi untuk validasi dan pengujian
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

print(f"DataLoaders terkonfigurasi:")
print(f"- Train batches: {len(train_loader)} ({len(train_dataset)} citra)")
print(f"- Val batches  : {len(val_loader)} ({len(val_dataset)} citra)")
print(f"- Test batches : {len(test_loader)} ({len(test_dataset)} citra)")"""),

    # Bab 4
    md_cell("""<a id="bab4"></a>
## 4. Arsitektur Model dan Strategi Pelatihan Dua Tahap
[kembali ke daftar isi](#toc)

Arsitektur ConvNeXt (Liu et al., CVPR 2022) mengadopsi prinsip desain Vision Transformer (ViT) ke dalam struktur konvolusi murni:
1. Receptive field besar melalui kernel depthwise 7x7.
2. Inverted bottleneck ratio 1:4 untuk preservasi kapasitas fitur representasional.
3. Penggunaan LayerNorm dan aktivasi GELU menggantikan BatchNorm dan ReLU."""),

    md_cell("""### 4.1 Inisialisasi Backbone dan Custom Classification Head
Struktur kepala klasifikasi bawaan torchvision ConvNeXt adalah:
- classifier[0]: LayerNorm2d (memproses tensor 4 dimensi [B, 768, 1, 1]).
- classifier[1]: Flatten (mengubah tensor menjadi 2 dimensi [B, 768]).
- classifier[2]: Linear (pemetaan fitur ke jumlah kelas target).

Untuk menjaga kestabilan tensor dan meningkatkan regularisasi, classifier[2] diganti dengan sekuens Dropout(p=0.4) dan Linear(768, 13). LayerNorm2d dan Flatten bawaan dipertahankan."""),

    code_cell(r"""# 4.1 Konstruksi model ConvNeXt-Tiny dan modifikasi head
weights = ConvNeXt_Tiny_Weights.DEFAULT
model = convnext_tiny(weights=weights)

in_features = model.classifier[2].in_features
model.classifier[2] = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(in_features, NUM_CLASSES)
)

model = model.to(DEVICE)
GRADCAM_TARGET_LAYER = model.features[-1]

# Loss function dengan Label Smoothing 0.1
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

print(f"Model {MODEL_NAME} siap.")
print(f"- Fitur keluaran backbone: {in_features} kanal")
print(f"- Modifikasi head: LayerNorm2d -> Flatten -> Dropout(0.4) -> Linear({in_features}, {NUM_CLASSES})")"""),

    # Bab 5
    md_cell("""<a id="bab5"></a>
## 5. Pelatihan Dua Tahap (Two-Stage Progressive Training)
[kembali ke daftar isi](#toc)

Pelatihan dilakukan melalui dua fase bertahap untuk mencegah degradasi bobot representasi ImageNet:

- **Tahap 1 (Warmup Head, Epoch 1-5):**
  Seluruh layer backbone dibekukan (requires_grad = False). Hanya linear classifier head yang diperbarui dengan learning rate 1e-3 untuk mengadaptasikan bobot acak ke distribusi fitur topeng Bali.

- **Tahap 2 (Full Fine-Tuning dengan Discriminative LR, Epoch 6-40):**
  Seluruh 28.6 juta parameter dibuka (unfreeze). Pembelajaran menggunakan Layer-wise Learning Rate Decay (LLRD):
  - Layer awal (features[:4]): LR = 2e-6 (mempertahankan filter deteksi tepi universal).
  - Layer tengah/tinggi (features[4:]): LR = 3e-5 (adaptasi bentuk morfologi).
  - Classifier head: LR = 1.5e-4 (penyesuaian klasifikasi akhir)."""),

    md_cell("""### 5.1 Eksekusi Pelatihan Dua Tahap (Total 40 Epoch)
Proses pelatihan menggunakan automatic mixed precision (torch.amp) untuk efisiensi komputasi pada GPU."""),

    code_cell(r"""# 5.1 Loop pelatihan dua tahap
device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
scaler = torch.amp.GradScaler(device_type, enabled=torch.cuda.is_available())

history = {
    'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'val_f1': [],
    'stage': []
}
best_val_f1 = 0.0
best_weights = copy.deepcopy(model.state_dict())
start_time = time.time()

# ------------------------------------------------------------------------------
# Tahap 1: Pemanasan Classifier Head (5 Epoch)
# ------------------------------------------------------------------------------
print("Tahap 1: Pemanasan Classifier Head (5 Epoch, Backbone Dibekukan)")
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
        ckpt_str = "[Checkpoint Terbaik]"
    else:
        ckpt_str = ""
        
    print(f"Tahap 1 - Epoch [{epoch:02d}/{EPOCHS_STAGE1:02d}] "
          f"Train Loss: {tr_loss:.4f} Acc: {tr_acc*100:.2f}% | "
          f"Val Loss: {va_loss:.4f} Acc: {va_acc*100:.2f}% Macro-F1: {va_f1*100:.2f}% {ckpt_str}")

# ------------------------------------------------------------------------------
# Tahap 2: Fine-Tuning Seluruh Parameter dengan Discriminative LR (35 Epoch)
# ------------------------------------------------------------------------------
print("\nTahap 2: Fine-Tuning Seluruh Parameter dengan Discriminative LR (35 Epoch)")
for param in model.parameters():
    param.requires_grad = True

param_groups = [
    {'params': model.features[:4].parameters(), 'lr': 2e-6, 'weight_decay': 1e-2},
    {'params': model.features[4:].parameters(), 'lr': 3e-5, 'weight_decay': 1e-2},
    {'params': model.classifier.parameters(),   'lr': 1.5e-4, 'weight_decay': 1e-2}
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
        ckpt_str = "[Checkpoint Terbaik]"
    else:
        ckpt_str = ""
        
    if epoch % 5 == 0 or epoch == 1 or epoch == EPOCHS_STAGE2:
        print(f"Tahap 2 - Epoch [{curr_epoch_total:02d}/{TOTAL_EPOCHS:02d}] "
              f"Train Loss: {tr_loss:.4f} Acc: {tr_acc*100:.2f}% | "
              f"Val Loss: {va_loss:.4f} Acc: {va_acc*100:.2f}% Macro-F1: {va_f1*100:.2f}% {ckpt_str}")

total_elapsed = time.time() - start_time
print(f"\nPelatihan selesai dalam {total_elapsed//60:.0f}m {total_elapsed%60:.0f}s. Best Val Macro-F1: {best_val_f1*100:.2f}%")

model.load_state_dict(best_weights)
trained_model = model"""),

    md_cell("""### 5.2 Visualisasi Kurva Pembelajaran Two-Stage
Grafik di bawah ini menggambarkan dinamika konvergensi loss dan akurasi sepanjang 40 epoch. Garis putus-putus merah menandai titik transisi di Epoch 5 saat seluruh backbone dibuka untuk fine-tuning diskriminatif."""),

    code_cell(r"""# 5.2 Visualisasi grafik loss dan metrik akurasi
epochs_range = range(1, TOTAL_EPOCHS + 1)
plt.figure(figsize=(16, 5))

# Plot Loss
plt.subplot(1, 3, 1)
plt.plot(epochs_range, history['train_loss'], label='Train Loss (Random Erasing)', color='blue')
plt.plot(epochs_range, history['val_loss'], label='Val Loss (Murni)', color='orange')
plt.axvline(x=EPOCHS_STAGE1 + 0.5, color='red', linestyle='--', label='Transisi Tahap 2')
plt.title("Kurva Loss (Two-Stage)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# Plot Akurasi
plt.subplot(1, 3, 2)
plt.plot(epochs_range, [a*100 for a in history['train_acc']], label='Train Acc (%)', color='green')
plt.plot(epochs_range, [a*100 for a in history['val_acc']], label='Val Acc (%)', color='red')
plt.axvline(x=EPOCHS_STAGE1 + 0.5, color='red', linestyle='--', label='Transisi Tahap 2')
plt.title("Kurva Akurasi (%)")
plt.xlabel("Epoch")
plt.ylabel("Akurasi (%)")
plt.legend()

# Plot Macro F1
plt.subplot(1, 3, 3)
plt.plot(epochs_range, [f*100 for f in history['val_f1']], label='Val Macro-F1 (%)', color='purple')
plt.axvline(x=EPOCHS_STAGE1 + 0.5, color='red', linestyle='--', label='Transisi Tahap 2')
plt.title("Kurva Validation Macro-F1 (%)")
plt.xlabel("Epoch")
plt.ylabel("Macro-F1 (%)")
plt.legend()

plt.tight_layout()
plt.savefig(OUT_DIR / 'two_stage_training_curves.png', dpi=300)
plt.show()"""),

    # Bab 6
    md_cell("""<a id="bab6"></a>
## 6. Evaluasi Komprehensif pada Data Uji (Test Set 121 Citra)
[kembali ke daftar isi](#toc)

Evaluasi akhir dilakukan terhadap 121 citra kanonik pada himpunan uji yang belum pernah dilihat oleh model (unseen physical masks). Pengukuran mencakup Top-1 Accuracy, Macro-Precision, Macro-Recall, Macro-F1, dan Weighted-F1."""),

    md_cell("""### 6.1 Perhitungan Metrik Evaluasi dan Classification Report
Hasil evaluasi disajikan secara rinci per kelas menggunakan classification report scikit-learn."""),

    code_cell(r"""# 6.1 Evaluasi pada himpunan data uji
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

print("HASIL EVALUASI TEST SET (121 Citra Kanonik Bebas Bocor):")
print(f"- Top-1 Accuracy    : {test_acc*100:.2f}%")
print(f"- Macro Precision   : {test_prec_macro*100:.2f}%")
print(f"- Macro Recall      : {test_rec_macro*100:.2f}%")
print(f"- Macro F1-Score    : {test_f1_macro*100:.2f}%")
print(f"- Weighted F1-Score : {test_f1_weight*100:.2f}%")

print("\nLaporan Klasifikasi Rinci:")
print(classification_report(test_targets, test_preds, target_names=CLASSES, digits=4))"""),

    md_cell("""### 6.2 Visualisasi Confusion Matrix
Matriks konfusi di bawah ini memperlihatkan pemetaan prediksi model terhadap label sebenarnya pada ke-13 kelas topeng."""),

    code_cell(r"""# 6.2 Heatmap Confusion Matrix
cm = confusion_matrix(test_targets, test_preds)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
plt.title(f"Confusion Matrix: {MODEL_NAME} (Test Set Kanonik Bebas Bocor)", fontsize=13, pad=15)
plt.xlabel("Prediksi Model", fontsize=11)
plt.ylabel("Label Asli (Ground Truth)", fontsize=11)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(OUT_DIR / 'confusion_matrix.png', dpi=300)
plt.show()"""),

    # Bab 7
    md_cell("""<a id="bab7"></a>
## 7. Analisis Kesalahan Prediksi dan Kasus Menantang
[kembali ke daftar isi](#toc)

Untuk memenuhi kriteria evaluasi kritis, bagian ini membedah sampel yang mengalami kesalahan klasifikasi serta sampel dengan probabilitas keyakinan paling rendah."""),

    md_cell("""### 7.1 Identifikasi dan Visualisasi Sampel Salah Prediksi
Menampilkan galeri citra yang gagal diprediksi dengan benar, membandingkan ground truth dengan kelas prediksi dan konfidensinya."""),

    code_cell(r"""# 7.1 Galeri sampel salah klasifikasi
misclassified_idx = [i for i, (p, t) in enumerate(zip(test_preds, test_targets)) if p != t]
print(f"Total kesalahan prediksi pada test set: {len(misclassified_idx)} dari {len(test_targets)} citra.")

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
        axes[plot_idx].set_title(f"Asli: {true_lbl}\nPred: {pred_lbl} ({conf:.1f}%)\nAsli Conf: {true_conf:.1f}%", 
                                 fontsize=10, color='red')
        axes[plot_idx].axis('off')
        
    plt.suptitle("Galeri Sampel Gagal Klasifikasi", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'misclassified_cases.png', dpi=300)
    plt.show()
else:
    print("Tidak ditemukan kesalahan prediksi pada himpunan uji yang dievaluasi.")"""),

    md_cell("""### 7.2 Analisis Kasus Menantang (Lowest-Confidence Correct Predictions)
Sampel-sampel berikut berhasil diprediksi dengan benar oleh model, namun memiliki margin probabilitas terendah akibat kemiripan ornamen morfologis antarkelas."""),

    code_cell(r"""# 7.2 Identifikasi top-3 prediksi benar paling menantang
correct_idx = [i for i, (p, t) in enumerate(zip(test_preds, test_targets)) if p == t]
if len(correct_idx) > 0:
    lowest_conf_idx = sorted(correct_idx, key=lambda i: test_probs[i][test_targets[i]])[:3]
    print("Top-3 sampel prediksi benar dengan tingkat keyakinan paling rendah:")
    for rank, idx in enumerate(lowest_conf_idx, 1):
        lbl_name = IDX_TO_CLASS[test_targets[idx]]
        conf = test_probs[idx][test_targets[idx]] * 100
        print(f"{rank}. Kelas: {lbl_name:<16} | Keyakinan: {conf:.2f}% | Berkas: {Path(test_paths[idx]).name}")"""),

    # Bab 8
    md_cell("""<a id="bab8"></a>
## 8. Explainable AI: Atensi Spasial Grad-CAM
[kembali ke daftar isi](#toc)

Gradient-weighted Class Activation Mapping (Grad-CAM, Selvaraju et al., 2017) digunakan untuk memverifikasi secara visual wilayah spasial mana yang menjadi dasar keputusan klasifikasi model."""),

    md_cell("""### 8.1 Implementasi Grad-CAM pada Layer Konvolusi Terakhir
Grad-CAM menghitung gradien kelas target terhadap peta fitur pada layer features[-1] (stage konvolusi terakhir ConvNeXt), lalu menggabungkannya melalui Global Average Pooling."""),

    code_cell(r"""# 8.1 Modul Grad-CAM dan fungsi overlay
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

grad_cam = SimpleGradCAM(trained_model, GRADCAM_TARGET_LAYER)"""),

    md_cell("""### 8.2 Visualisasi Overlay Heatmap pada Citra Asli
Visualisasi di bawah ini menampilkan foto asli, heatmap aktivasi, dan overlay Grad-CAM untuk membuktikan bahwa model memfokuskan atensi pada bagian distingtif topeng (mata, taring, dan mahkota)."""),

    code_cell(r"""# 8.2 Visualisasi perbandingan Grad-CAM pada sampel uji
fig, axes = plt.subplots(3, 3, figsize=(14, 12))
sample_indices = [0, len(test_dataset)//2, len(test_dataset)-1]

for i, s_idx in enumerate(sample_indices):
    img_tensor, lbl, path_str = test_dataset[s_idx]
    cam_map = grad_cam.generate_cam(img_tensor.unsqueeze(0).to(DEVICE), target_class=lbl)
    orig_img, overlay = show_cam_overlay(path_str, cam_map)
    
    axes[i, 0].imshow(orig_img)
    axes[i, 0].set_title(f"Foto Asli: {IDX_TO_CLASS[lbl]}", fontsize=11)
    axes[i, 0].axis('off')
    
    axes[i, 1].imshow(cam_map, cmap='jet')
    axes[i, 1].set_title("Heatmap Aktivasi Spasial", fontsize=11)
    axes[i, 1].axis('off')
    
    axes[i, 2].imshow(overlay)
    axes[i, 2].set_title("Overlay Grad-CAM pada Topeng", fontsize=11)
    axes[i, 2].axis('off')

plt.suptitle(f"Interpretasi Fitur Visual Menggunakan Grad-CAM: {MODEL_NAME}", fontsize=13)
plt.tight_layout()
plt.savefig(OUT_DIR / 'gradcam_ultra_robust.png', dpi=300)
plt.show()"""),

    # Bab 9
    md_cell("""<a id="bab9"></a>
## 9. Profil Efisiensi Komputasi dan Ekspor Model
[kembali ke daftar isi](#toc)

Bagian akhir mengukur aspek efisiensi komputasi model, mencakup jumlah parameter, latensi inferensi rata-rata per sampel pada GPU, dan ukuran berkas bobot yang diekspor."""),

    md_cell("""### 9.1 Pengukuran Parameter dan Latensi Inferensi
Pengukuran latensi dilakukan melalui 50 iterasi setelah tahap pemanasan (warmup) 10 iterasi."""),

    code_cell(r"""# 9.1 Pengukuran metrik komputasi dan penyimpanan
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
    'Strategi Pelatihan': 'Two-Stage Progressive (Warmup + LLRD)',
    'Augmentasi': 'RandAugment + Random Erasing Cutout',
    'Total Epoch': TOTAL_EPOCHS,
    'Test Accuracy (%)': round(test_acc * 100, 2),
    'Macro Precision (%)': round(test_prec_macro * 100, 2),
    'Macro Recall (%)': round(test_rec_macro * 100, 2),
    'Macro F1 (%)': round(test_f1_macro * 100, 2),
    'Weighted F1 (%)': round(test_f1_weight * 100, 2),
    'Total Parameter (M)': round(total_p, 2),
    'Latensi Inferensi (ms)': round(latency_ms, 2),
    'Ukuran Berkas (MB)': round(model_size_mb, 2)
}

df_summary = pd.DataFrame([summary_record])
df_summary.to_csv(OUT_DIR / 'metrics_summary.csv', index=False)
display(df_summary)

print(f"Berkas model tersimpan di: {OUT_DIR / 'best_model.pth'}")
print(f"Ukuran berkas model: {model_size_mb:.2f} MB")""")
]

nb = create_nb(cells)
out_file = OUT_DIR / "01_ConvNeXt_Tiny_Ultra_Robust.ipynb"
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print(f"Selesai membangun: {out_file}")
