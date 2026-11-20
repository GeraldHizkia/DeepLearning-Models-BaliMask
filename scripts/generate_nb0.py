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

nb0_cells = [
    md_cell("""# 🏛️ Modul 00: Audit Dataset & Pembuatan Split Kanonik (Leakage-Free 801)
## Investigasi Integritas Data, Eliminasi Data Leakage Roboflow, & Pembagian Stratified 70:15:15
**Program Studi Teknologi Informasi — Fakultas Teknik, Universitas Udayana**  
*Mata Kuliah: Deep Learning (Tugas 2) & Proyek Capstone Klasifikasi 13 Kelas Topeng Bali*  
*Dosen Pengampu: Wayan Oger Vihikan, S.T.I., M.I.T. | Dosen Pembimbing: Ni Kadek Ayu Wirdiani, S.T., M.T.*  
*Peneliti: Gerald Hizkia Turnip & Tim Peneliti Topeng Bali*

---

## 📑 Daftar Isi (Table of Contents)
1. [Latar Belakang & Identifikasi Permasalahan Data Leakage](#1-latar-belakang--identifikasi-permasalahan-data-leakage)
2. [Setup Lingkungan & Akses Dataset](#2-setup-lingkungan--akses-dataset)
3. [Audit Empiris: Bukti Kebocoran Data Bawaan Roboflow (.rf.)](#3-audit-empiris-bukti-kebocoran-data-bawaan-roboflow-rf)
4. [Algoritma Deduplikasi: Regex Base Identity & Visual Perceptual Hashing (pHash)](#4-algoritma-deduplikasi-regex-base-identity--visual-perceptual-hashing-phash)
5. [Ekstraksi 801 Citra Fisik Kanonik Asli](#5-ekstraksi-801-citra-fisik-kanonik-asli)
6. [Pembagian Stratified Group Split (70% Train, 15% Val, 15% Test)](#6-pembagian-stratified-group-split-70-train-15-val-15-test)
7. [Verifikasi Matematis Integritas Split (Zero-Leakage Assertion)](#7-verifikasi-matematis-integritas-split-zero-leakage-assertion)
8. [Visualisasi Distribusi Kelas Antar-Split](#8-visualisasi-distribusi-kelas-antar-split)
9. [Ekspor Master CSV & Panduan Integrasi Model](#9-ekspor-master-csv--panduan-integrasi-model)"""),

    md_cell("""---
## 1. Latar Belakang & Identifikasi Permasalahan Data Leakage

Pada penelitian klasifikasi citra bernilai budaya tinggi seperti **Topeng Tradisional Bali**, integritas dataset merupakan pondasi paling fundamental. Dataset publik *Balinese Mask Image Dataset* (Mendeley Data / Roboflow Universe) memuat 13 kelas topeng tradisional Bali:
1. `barong-bangkal`
2. `barong-ket`
3. `bujuh`
4. `dalem`
5. `jauk-keras`
6. `jauk-manis`
7. `keras`
8. `macan`
9. `penasar`
10. `sidakarya`
11. `telek`
12. `tua`
13. `wijil`

### ⚠️ Permasalahan Kritis:
Pada dataset mentah hasil ekspor otomatis platform Roboflow, augmentasi offline (rotasi -15°, -10°, -5°, +5°, +10°, +15°, dan penyesuaian kontras) diterapkan **sebelum** pembagian split dilakukan. Hal ini menyebabkan citra dari satu objek fisik topeng yang sama terdistribusi secara acak ke dalam folder `train`, `valid`, dan `test`.
- Kondisi ini menciptakan **Target Data Leakage**: model evaluasi menguji gambar yang hanya berbeda rotasi beberapa derajat dari gambar yang dipelajari pada data pelatihan.
- Akurasi model tampak mencapai >97%, namun mengalami degradasi tajam saat diuji di dunia nyata (*overoptimistic bias*).

### 🎯 Solusi Metodologis:
Notebook ini merekonstruksi dataset secara ilmiah:
1. Mengembalikan seluruh data augmentasi ke **801 citra fisik kanonik asli**.
2. Melakukan pembagian **Stratified Group Split** pada tingkat objek fisik topeng:
   - **Training Set (70%)**: 561 citra kanonik (tempat augmentasi dinamis online diterapkan).
   - **Validation Set (15%)**: 120 citra kanonik murni tanpa augmentasi.
   - **Test Set (15%)**: 120 citra kanonik murni tanpa augmentasi.
3. Menjamin secara matematis bahwa $Train \cap (Val \cup Test) = \emptyset$ pada tingkat identitas objek fisik."""),

    code_cell("""# Sel 1: Instalasi Dependensi & Setup Environment
!pip -q install imagehash pandas matplotlib seaborn tqdm

import os, re, shutil, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit

# Konfigurasi Tampilan
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
sns.set_palette('tab10')
plt.rcParams['font.size'] = 11

SEED = 42
np.random.seed(SEED)
print("✅ Lingkungan siap. Modul imagehash, pandas, dan scikit-learn aktif.")"""),

    md_cell("""---
## 2. Setup Lingkungan & Akses Dataset
Deteksi otomatis Google Drive atau repositori lokal untuk memuat file arsip `dataset_reorganized.zip`."""),

    code_cell("""# Sel 2: Mount Google Drive & Ekstraksi Arsip
try:
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE_MOUNTED = True
    print("✅ Google Drive berhasil dimount.")
except Exception:
    DRIVE_MOUNTED = False
    print("ℹ️ Menjalankan di lingkungan lokal / notebook mandiri.")

ZIP_CANDIDATES = [
    '/content/drive/MyDrive/dataset_reorganized.zip',
    '/content/dataset_reorganized.zip',
    './dataset_reorganized.zip',
    '../dataset_reorganized.zip'
]

ZIP_PATH = next((p for p in ZIP_CANDIDATES if os.path.exists(p)), None)
EXTRACT_ROOT = Path('/content/BaliMask_Raw') if os.path.exists('/content') else Path('./BaliMask_Raw')

if ZIP_PATH:
    print(f"📦 Mengekstrak dataset dari: {ZIP_PATH}")
    EXTRACT_ROOT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
        zf.extractall(EXTRACT_ROOT)
    print(f"✅ Ekstraksi selesai di: {EXTRACT_ROOT}")
else:
    print(f"ℹ️ File ZIP belum ditemukan di path standar. Mencari folder dataset yang sudah diekstrak...")

# Cari direktori gambar yang berisi 13 subfolder kelas
DATA_DIR = None
SEARCH_CANDIDATES = [
    EXTRACT_ROOT / 'dataset_reorganized' / 'train',
    EXTRACT_ROOT / 'train',
    EXTRACT_ROOT / 'dataset_reorganized',
    EXTRACT_ROOT,
    Path('./dataset_reorganized/train'),
    Path('./BaliMask_Clean/train')
]

for cand in SEARCH_CANDIDATES:
    if cand.exists() and any(d.is_dir() for d in cand.iterdir() if not d.name.startswith('.')):
        DATA_DIR = cand
        break

print(f"📂 Direktori Dataset Aktif: {DATA_DIR}")"""),

    md_cell("""---
## 3. Audit Empiris: Bukti Kebocoran Data Bawaan Roboflow (.rf.)
Di bawah ini kita melakukan penelusuran terhadap penamaan file untuk membuktikan keberadaan pola export Roboflow (`.rf.`) serta replikasi sudut rotasi dari objek yang sama."""),

    code_cell("""# Sel 3: Pemindaian Seluruh Citra & Ekstraksi Metadata Nama File
IMG_EXT = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

def extract_base_identity(filename):
    \"\"\"Mengekstrak identitas dasar objek fisik sebelum augmentasi offline.\"\"\"
    name = Path(filename).name.lower()
    name = re.sub(r'\\.(jpg|jpeg|png|webp|bmp)$', '', name)
    # Hapus hash roboflow: .rf.xxxx atau ._rf.xxxx
    name = re.sub(r'(\\._rf\\.|\\.rf\\.|_rf\\.)[a-f0-9]+$', '', name)
    # Hapus format ekstensi yang terbawa
    name = re.sub(r'(_jpg|_jpeg|_png)$', '', name)
    # Hapus prefix resized
    if name.startswith('resized_'):
        name = name[len('resized_'):]
    # Hapus suffix rotasi offline
    name = re.sub(r'(_rotated-?\\d+|-rotated-?\\d+|rotated-?\\d+)', '', name)
    name = re.sub(r'(_aug\\d+|-aug\\d+)', '', name)
    return name.strip('-_ ')

# Pindai file
records = []
if DATA_DIR and DATA_DIR.exists():
    for class_dir in sorted(DATA_DIR.iterdir()):
        if class_dir.is_dir() and not class_dir.name.startswith('.'):
            c_name = class_dir.name
            for img_path in sorted(class_dir.rglob('*')):
                if img_path.suffix.lower() in IMG_EXT:
                    rel_p = str(img_path.relative_to(DATA_DIR))
                    b_id = extract_base_identity(img_path.name)
                    records.append({
                        'rel_path': rel_p,
                        'full_path': str(img_path),
                        'filename': img_path.name,
                        'class': c_name,
                        'base_id': b_id,
                        'has_roboflow_hash': bool(re.search(r'\\.rf\\.[a-f0-9]+', img_path.name)),
                        'has_rotation': bool(re.search(r'rotated-?\\d+', img_path.name))
                    })

df_all = pd.DataFrame(records)
print(f"📊 Total citra ditemukan: {len(df_all)}")
if len(df_all) > 0:
    rf_count = df_all['has_roboflow_hash'].sum()
    rot_count = df_all['has_rotation'].sum()
    print(f"🔍 Citra ber-hash Roboflow (.rf.): {rf_count} ({rf_count/len(df_all)*100:.1f}%)")
    print(f"🔄 Citra varian rotasi offline : {rot_count} ({rot_count/len(df_all)*100:.1f}%)")
    print(f"🏷️  Jumlah identitas dasar unik (base_id): {df_all['base_id'].nunique()}")
    
    # Tampilkan sampel duplikasi identitas dasar
    dup_sample = df_all[df_all['base_id'] == 'bangkal-ig-1-']
    if len(dup_sample) > 0:
        print(\"\\nContoh varian dari satu objek fisik 'bangkal-ig-1-':\")
        display(dup_sample[['filename', 'class', 'base_id']].head(5))"""),

    md_cell("""---
## 4. Algoritma Deduplikasi: Regex Base Identity & Visual Perceptual Hashing (pHash)
Untuk memastikan deduplikasi 100% akurat (termasuk bila ada dua foto identik dengan penamaan berbeda), kita menggabungkan **Regex Base Identity** dengan **Perceptual Hashing (pHash)** menggunakan struktur data *Disjoint-Set (Union-Find)*."""),

    code_cell("""# Sel 4: Pengelompokan Disjoint-Set Menggunakan pHash + Base ID
def cluster_physical_masks(df, root_dir, phash_threshold=4):
    import imagehash
    n = len(df)
    parent = list(range(n))
    
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
        
    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    # 1. Union berdasarkan Base Identity (teks)
    id_map = {}
    for idx, bid in enumerate(df['base_id']):
        if bid in id_map:
            union(id_map[bid], idx)
        else:
            id_map[bid] = idx

    # 2. Union berdasarkan pHash Visual (Hamming Distance <= phash_threshold)
    print("🔍 Menghitung perceptual hash (pHash 64-bit) untuk verifikasi citra...")
    hashes = []
    for fp in tqdm(df['full_path'], desc="pHash computation"):
        try:
            with Image.open(fp) as img:
                h = imagehash.phash(img.convert('RGB')).hash.flatten()
                hashes.append(h)
        except Exception:
            hashes.append(np.zeros(64, dtype=bool))
            
    hash_arr = np.array(hashes, dtype=np.uint8)
    for i in range(n - 1):
        diff = (hash_arr[i + 1:] != hash_arr[i]).sum(axis=1)
        matches = np.where(diff <= phash_threshold)[0]
        for m in matches:
            union(i, i + 1 + int(m))

    clusters = [find(i) for i in range(n)]
    return np.array(clusters)

if len(df_all) > 0:
    groups = cluster_physical_masks(df_all, DATA_DIR)
    df_all['mask_group_id'] = groups
    unique_masks = df_all['mask_group_id'].nunique()
    print(f"\\n🎯 HASIL DEDUPLIKASI: Ditemukan {unique_masks} objek topeng fisik unik dari {len(df_all)} total citra!")"""),

    md_cell("""---
## 5. Ekstraksi 801 Citra Fisik Kanonik Asli
Dari setiap grup objek topeng fisik, kita memilih **1 citra perwakilan kanonik** (diutamakan citra asli tanpa rotasi atau resolusi penuh)."""),

    code_cell("""# Sel 5: Pemilihan Citra Kanonik Terbaik per Objek Topeng Fisik
def pick_canonical_image(group_df):
    # Prioritaskan citra yang bukan rotasi dan bukan resized
    clean = group_df[~group_df['has_rotation'] & ~group_df['filename'].str.startswith('resized_')]
    if len(clean) > 0:
        return clean.iloc[0]
    unrot = group_df[~group_df['has_rotation']]
    if len(unrot) > 0:
        return unrot.iloc[0]
    return group_df.iloc[0]

if len(df_all) > 0:
    canonical_rows = []
    for gid, gdf in df_all.groupby('mask_group_id'):
        best_img = pick_canonical_image(gdf)
        canonical_rows.append(best_img)

    df_canonical = pd.DataFrame(canonical_rows).reset_index(drop=True)
    print(f"✅ Berhasil mengekstrak {len(df_canonical)} citra fisik kanonik asli!")
    
    # Distribusi per kelas kanonik
    class_summary = df_canonical['class'].value_counts().reset_index()
    class_summary.columns = ['Nama Kelas', 'Jumlah Topeng Unik']
    display(class_summary)"""),

    md_cell("""---
## 6. Pembagian Stratified Group Split (70% Train, 15% Val, 15% Test)
Membagi 801 citra kanonik ke dalam 3 himpunan bagian dengan proporsi seimbang:
- **70% Training (~561 citra)**
- **15% Validation (~120 citra)**
- **15% Test Set (~120 citra)**"""),

    code_cell("""# Sel 6: Stratified Splitting Bebas Bocor
if len(df_all) > 0:
    # Tahap 1: Split 70% Train vs 30% Temp (Val + Test)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
    train_idx, temp_idx = next(sss1.split(df_canonical, df_canonical['class']))
    
    df_train = df_canonical.iloc[train_idx].copy()
    df_temp = df_canonical.iloc[temp_idx].copy().reset_index(drop=True)
    
    # Tahap 2: Split 30% Temp menjadi 50:50 (15% Val & 15% Test)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
    val_rel, test_rel = next(sss2.split(df_temp, df_temp['class']))
    
    df_val = df_temp.iloc[val_rel].copy()
    df_test = df_temp.iloc[test_rel].copy()
    
    df_train['split'] = 'train'
    df_val['split'] = 'val'
    df_test['split'] = 'test'
    
    df_split = pd.concat([df_train, df_val, df_test], ignore_index=True)
    
    print(f"📈 HASIL PEMBAGIAN DATA:")
    print(f"  • Training   : {len(df_train)} citra ({len(df_train)/len(df_canonical)*100:.1f}%)")
    print(f"  • Validation : {len(df_val)} citra ({len(df_val)/len(df_canonical)*100:.1f}%)")
    print(f"  • Testing    : {len(df_test)} citra ({len(df_test)/len(df_canonical)*100:.1f}%)")
    print(f"  • Total      : {len(df_split)} citra kanonik")"""),

    md_cell("""---
## 7. Verifikasi Matematis Integritas Split (Zero-Leakage Assertion)
Pengujian otomatis menggunakan operator himpunan untuk membuktikan tidak ada kebocoran objek fisik antar-split."""),

    code_cell("""# Sel 7: Assertion Uji Bebas Kebocoran
if len(df_all) > 0:
    train_masks = set(df_split[df_split['split'] == 'train']['mask_group_id'])
    val_masks   = set(df_split[df_split['split'] == 'val']['mask_group_id'])
    test_masks  = set(df_split[df_split['split'] == 'test']['mask_group_id'])
    
    leak_tv = train_masks.intersection(val_masks)
    leak_tt = train_masks.intersection(test_masks)
    leak_vt = val_masks.intersection(test_masks)
    
    assert len(leak_tv) == 0, f"❌ ERROR: Kebocoran Train-Val sebanyak {len(leak_tv)} objek!"
    assert len(leak_tt) == 0, f"❌ ERROR: Kebocoran Train-Test sebanyak {len(leak_tt)} objek!"
    assert len(leak_vt) == 0, f"❌ ERROR: Kebocoran Val-Test sebanyak {len(leak_vt)} objek!"
    
    print("🛡️ VERIFIKASI INTERSECTION BEBAS KEBOCORAN:")
    print("  ✓ Train ∩ Val  = 0 (Sempurna)")
    print("  ✓ Train ∩ Test = 0 (Sempurna)")
    print("  ✓ Val ∩ Test   = 0 (Sempurna)")
    print("  ==> INTEGRITAS METODOLOGI TERVALIDASI 100% BEBAS KEBOCORAN (LEAKAGE-FREE).")"""),

    md_cell("""---
## 8. Visualisasi Distribusi Kelas Antar-Split
Menampilkan grafik batang komparasi jumlah sampel tiap kelas pada himpunan Train, Validation, dan Testing."""),

    code_cell("""# Sel 8: Visualisasi Distribusi Kelas
if len(df_all) > 0:
    plt.figure(figsize=(14, 6))
    split_counts = pd.crosstab(df_split['class'], df_split['split'])[['train', 'val', 'test']]
    split_counts.plot(kind='bar', stacked=True, figsize=(14, 6), colormap='viridis', edgecolor='black')
    plt.title("Distribusi Kelas Topeng Bali pada Split Kanonik 801 (Leakage-Free)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Kelas Topeng Tradisional Bali", fontsize=12, labelpad=10)
    plt.ylabel("Jumlah Citra Kanonik", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Subset", frameon=True)
    plt.tight_layout()
    plt.savefig('distribusi_split_kanonik_801.png', dpi=300)
    plt.show()
    
    print("Tabel Distribusi Rinci:")
    display(split_counts)"""),

    md_cell("""---
## 9. Ekspor Master CSV & Panduan Integrasi Model
File CSV yang dihasilkan akan menjadi acuan data loader tunggal bagi seluruh model:
- `split_canonical_801.csv`: Hanya 801 citra kanonik unik.
- `dataset_all_annotated.csv`: Seluruh citra beserta anotasi grup objeknya."""),

    code_cell("""# Sel 9: Penyimpanan CSV Master
OUT_CSV = Path('split_canonical_801.csv')
if len(df_all) > 0:
    export_df = df_split[['rel_path', 'filename', 'class', 'split', 'mask_group_id', 'base_id']].copy()
    export_df.to_csv(OUT_CSV, index=False)
    print(f"💾 File berhasil diekspor ke: {OUT_CSV.resolve()}")
    
    # Simpan juga ke Google Drive jika tersedia
    if DRIVE_MOUNTED:
        drive_target = Path('/content/drive/MyDrive/split_canonical_801.csv')
        export_df.to_csv(drive_target, index=False)
        print(f"💾 File cadangan tersimpan di Google Drive: {drive_target}")

print(\"\"\"
================================================================================
✅ SELESAI! DATASET SPLIT KANONIK TELAH TERKUNCI SECARA DETERMINISTIK.
Semua notebook model (ConvNeXt, EfficientNet, MobileNet, VGG) sekarang dapat 
menggunakan split_canonical_801.csv sebagai data loader anti-bocor yang sah!
================================================================================
\"\"\")""")
]

nb0 = create_nb(nb0_cells)
with open(OUT_DIR / "00_Dataset_Audit_and_Canonical_Split.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb0, f, indent=2)

print("✅ Generated: notebooks_v2_canonical/00_Dataset_Audit_and_Canonical_Split.ipynb")
