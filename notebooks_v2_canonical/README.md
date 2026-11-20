# 🏛️ Suite Notebooks V2: Balinese Mask Image Classification (Canonical 801 Pipeline)
## Benchmark Riset Komparasi 4 Arsitektur Deep Learning Bebas Data Leakage

Repository ini memuat suite notebook eksperimen mutakhir untuk klasifikasi **13 Kelas Topeng Tradisional Bali** dengan metodologi **Dual-Track Benchmarking**:
1. **Track A (Baseline Roboflow — ~4.400 Citra):** Menguji data bawaan platform untuk melihat performa baseline.
2. **Track B (Canonical 801 Clean Split — 801 Citra Fisik Unik):** Mengeliminasi kebocoran data (*data leakage*) melalui deduplikasi identitas dasar dan *Perceptual Hashing (pHash)*, lalu menerapkan *online data augmentation* hanya pada training set.

---

## 📑 Struktur Berkas Notebook (`notebooks_v2_canonical/`)

| Nomor | Berkas Notebook | Peran & Deskripsi Utama | Target Arsitektur / Fokus |
| :---: | :--- | :--- | :--- |
| **00** | [00_Dataset_Audit_and_Canonical_Split.ipynb](00_Dataset_Audit_and_Canonical_Split.ipynb) | **Data Engineering & Audit Kebocoran Data**: Mengaudit bukti kebocoran Roboflow (`.rf.`), mengekstrak 801 citra kanonik, membuat Stratified Split 70:15:15, dan mengekspor `split_canonical_801.csv`. | Data Engineering & Integrity |
| **01** | [01_ConvNeXt_Tiny_Canonical.ipynb](01_ConvNeXt_Tiny_Canonical.ipynb) | **Model Utama (Gerald)**: Pure Modern ConvNet dengan kernel 7x7, LayerNorm, AdamW, Cosine Annealing, Label Smoothing, Grad-CAM, dan Error Analysis. | **ConvNeXt-Tiny** |
| **02** | [02_EfficientNetV2_S_Canonical.ipynb](02_EfficientNetV2_S_Canonical.ipynb) | **Model Efisiensi Parameter**: Fused-MBConv yang dioptimasi untuk rasio akurasi-ke-parameter tinggi dan pelatihan cepat. | **EfficientNetV2-S** |
| **03** | [03_MobileNetV3_CBAM_Canonical.ipynb](03_MobileNetV3_CBAM_Canonical.ipynb) | **Model Edge AI & Dual Attention**: Mengintegrasikan CBAM (*Channel & Spatial Attention*) pada layer konvolusi terakhir MobileNetV3. | **MobileNetV3 + CBAM** |
| **04** | [04_VGG16_Canonical.ipynb](04_VGG16_Canonical.ipynb) | **Classical Baseline Reference**: Arsitektur konvolusi homogen klasik dengan Batch Normalization sebagai standar pembanding historis. | **VGG16-BN** |
| **05** | [05_Master_Comparison_and_Error_Analysis.ipynb](05_Master_Comparison_and_Error_Analysis.ipynb) | **Dashboard Evaluasi & Sintesis Ilmiah**: Rekapitulasi komparasi Track A vs Track B, Bubble Chart trade-off, bedah analisis error, dan komparasi Grad-CAM. | **Master Comparison** |

---

## 🚀 Panduan Eksekusi di Google Colab

Setiap notebook didesain dengan prinsip **Zero-Friction Execution**:
1. **Buka Notebook di Google Colab**:
   - Unggah notebook ke Google Colab atau buka langsung dari repositori GitHub.
2. **Siapkan Dataset**:
   - Letakkan file `dataset_reorganized.zip` di root Google Drive Anda (`MyDrive/dataset_reorganized.zip`) atau di direktori kerja Colab (`/content/dataset_reorganized.zip`).
   - Notebook akan secara otomatis mendeteksi, me-mount Google Drive, dan mengekstrak file arsip tanpa intervensi manual.
3. **Pilih GPU Runtime**:
   - Di menu Google Colab: `Runtime` -> `Change runtime type` -> pilih **T4 GPU**.
4. **Jalankan Semua Sel (`Run all`)**:
   - Seluruh sel kode telah dilengkapi penanganan pengecualian (*graceful fallbacks*), pencatatan metrik, serta rendering visualisasi grafik (kurva pelatihan, confusion matrix, galeri error, dan overlay Grad-CAM) langsung di bawah sel eksekusi.

---

## 🎯 Pemenuhan Rubrik Penilaian Tugas 2 Deep Learning (Pak Oger)

1. **Perumusan Masalah & Kajian Literatur (Poin 2 & 13)**:
   - Terintegrasi di Bagian 1 setiap notebook dengan rasional ilmiah pemilihan arsitektur dan kaitannya dengan karakteristik visual artefak topeng Bali.
2. **Dataset & Preprocessing (Poin 4)**:
   - Terdokumentasi lengkap di Notebook 00: audit empiris format Roboflow, deduplikasi Regex + pHash, pembagian stratified bebas bocor, dan augmentasi online dinamis pada train set.
3. **Pengembangan Model & Fine-Tuning (Poin 5)**:
   - Menggunakan transfer learning bobot terbaru ImageNet-1K, pembekuan layer awal (*early stage freezing*), fine-tuning custom classification head, serta teknik modern (Label Smoothing 0.1, AdamW, Cosine Annealing).
4. **Evaluasi & Analisis Komparatif (Poin 8 & 10)**:
   - Menguji Top-1 Accuracy, Macro-Precision, Macro-Recall, Macro-F1, Weighted-F1, Confusion Matrix beranotasi, serta tabel komparasi 4 arsitektur berdampingan.
5. **Analisis Error & Explainability (Poin 9)**:
   - Menampilkan sampel prediksi gagal beserta persentase confidence, dibedah menggunakan kajian morfologi topeng Bali (*domain-knowledge error analysis*), dan divisualisasikan dengan Heatmap Grad-CAM di atas foto asli.

---

## 🌟 Nilai Tambah untuk Publikasi Capstone (Bu Wirdiani)

Penelitian ini mentransformasikan proyek capstone dari sekadar "mencoba 4 model" menjadi **penelitian ilmiah berbobot publikasi jurnal bereputasi**:
- Menguak fenomena umum *overoptimistic evaluation* pada dataset visual berbasis web akibat augmentasi offline sebelum split.
- Membuktikan secara kuantitatif daya tahan generalisasi arsitektur modern (ConvNeXt-Tiny dan MobileNetV3+CBAM) menghadapi data dunia nyata yang belum pernah dilihat sebelumnya (*unseen physical artifacts*).
