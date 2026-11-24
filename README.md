# Klasifikasi 13 Kelas Topeng Tradisional Bali Berbasis Deep Learning

Repositori ini memuat implementasi dan pengujian model deep learning untuk klasifikasi fine-grained 13 kelas topeng tradisional Bali. Proyek ini dikembangkan dalam rangka Tugas 2 Mata Kuliah Deep Learning dan Proyek Capstone di Program Studi Teknologi Informasi, Fakultas Teknik, Universitas Udayana.

---

## Struktur Repositori

```text
├── docs/
│   ├── Proposal Klasifikasi Topeng Bali (2).pptx
│   ├── Tugas Mahasiswa 2.pdf
│   ├── Naskah_Asistensi_18Nov.md
│   ├── Naskah_Asistensi_18Nov_Lengkap.md
│   ├── 01_train_vgg16.md
│   ├── 02_train_efficientnetv2s.md
│   ├── 03_train_convnext_tiny.md
│   └── 04_train_mobilenetv3_attention.md
│
├── notebooks_v1_baseline/
│   ├── ConvNeXt.ipynb
│   ├── EfficientNetV2_S.ipynb
│   ├── Evaluasi_GradCAM_Analisis_Error.ipynb
│   ├── Evaluasi_GradCAM_ErrorAnalysis_AUTO.ipynb
│   ├── Master_Comparison_Evaluation.ipynb
│   ├── MobileNetV3_Attention.ipynb
│   ├── VGG16.ipynb
│   └── topeng.ipynb
│
├── notebooks_v2_canonical/
│   ├── 00_Dataset_Audit_and_Canonical_Split.ipynb
│   ├── 01_ConvNeXt_Tiny_Canonical.ipynb
│   ├── 01_ConvNeXt_Tiny_Ultra_Robust.ipynb
│   ├── 02_EfficientNetV2_S_Canonical.ipynb
│   ├── 03_MobileNetV3_CBAM_Canonical.ipynb
│   ├── 04_VGG16_Canonical.ipynb
│   ├── 05_Master_Comparison_and_Error_Analysis.ipynb
│   ├── split_canonical_801.csv
│   └── README.md
│
├── .gitignore
└── README.md
```

---

## Deskripsi Folder

### 1. docs/
Berisi dokumen panduan tugas kuliah, proposal awal proyek, naskah asistensi dosen, serta catatan teknis pelatihan awal.

### 2. notebooks_v1_baseline/
Berisi berkas eksperimen awal menggunakan partisi data bawaan Roboflow (~4.400 citra). Folder ini digunakan sebagai tolok ukur (baseline) untuk menganalisis performa awal sebelum mitigasi kebocoran data diterapkan.

### 3. notebooks_v2_canonical/
Berisi berkas eksperimen utama dengan partisi kanonik 801 citra fisik unik yang bebas dari kebocoran data (zero data leakage):
- 00_Dataset_Audit_and_Canonical_Split.ipynb: Audit data dan pembentukan partisi 70:15:15.
- 01_ConvNeXt_Tiny_Canonical.ipynb: Model ConvNeXt-Tiny versi standar.
- 01_ConvNeXt_Tiny_Ultra_Robust.ipynb: Model ConvNeXt-Tiny dengan two-stage fine-tuning dan Random Erasing.
- 02_EfficientNetV2_S_Canonical.ipynb: Model EfficientNetV2-S.
- 03_MobileNetV3_CBAM_Canonical.ipynb: Model MobileNetV3 dengan modul atensi CBAM.
- 04_VGG16_Canonical.ipynb: Model VGG16-BN sebagai baseline klasik.
- 05_Master_Comparison_and_Error_Analysis.ipynb: Dashboard komparasi master lintas model.

---

## Daftar Kelas Topeng

Model dilatih untuk mengenali 13 kelas topeng tradisional Bali:
1. barong-bangkal
2. barong-ket
3. bujuh
4. dalem
5. jauk-keras
6. jauk-manis
7. keras
8. macan
9. penasar
10. sidakarya
11. telek
12. tua
13. wijil

---

## Panduan Eksekusi

1. Buka berkas notebook yang dituju melalui Google Colab atau JupyterLab.
2. Pastikan runtime akselerator GPU telah aktif (disarankan GPU T4 atau lebih tinggi).
3. Jalankan sel secara berurutan mulai dari sel persiapan lingkungan hingga evaluasi akhir.
