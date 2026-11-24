# Dokumentasi Pipeline Kanonik 801 Citra Topeng Tradisional Bali

Repositori ini memuat implementasi model deep learning untuk klasifikasi 13 kelas topeng tradisional Bali menggunakan partisi data kanonik 801 citra fisik unik bebas dari kebocoran data (zero data leakage).

---

## Daftar Berkas Notebook

| Berkas | Fokus Utama | Target Arsitektur |
|---|---|---|
| 00_Dataset_Audit_and_Canonical_Split.ipynb | Audit bukti kebocoran Roboflow (.rf.) dan ekstraksi 801 citra kanonik | Data Engineering |
| 01_ConvNeXt_Tiny_Canonical.ipynb | Model ConvNeXt-Tiny versi standar | ConvNeXt-Tiny |
| 01_ConvNeXt_Tiny_Ultra_Robust.ipynb | Model ConvNeXt-Tiny dengan two-stage fine-tuning dan Random Erasing | ConvNeXt-Tiny Ultra-Robust |
| 02_EfficientNetV2_S_Canonical.ipynb | Model EfficientNetV2-S berbasis Fused-MBConv | EfficientNetV2-S |
| 03_MobileNetV3_CBAM_Canonical.ipynb | Model MobileNetV3-Large dengan integrasi atensi ganda CBAM | MobileNetV3 + CBAM |
| 04_VGG16_Canonical.ipynb | Model VGG16 dengan Batch Normalization sebagai baseline klasik | VGG16-BN |
| 05_Master_Comparison_and_Error_Analysis.ipynb | Dashboard komparasi master lintas arsitektur dan analisis error | Master Evaluation |

---

## Metodologi Partisi Kanonik

1. Deduplikasi Berbasis Identitas Fisik:
   Nama berkas yang memuat variasi rotasi offline dan hash ekspor Roboflow dikembalikan ke identitas dasar objek fisik (base_id).
2. Verifikasi Perceptual Hashing (pHash):
   Pemeriksaan kemiripan visual dilakukan menggunakan algoritma pHash 64-bit dengan threshold Hamming Distance 4 untuk memastikan tidak ada duplikat visual antar-subset.
3. Pembagian Stratified 70:15:15:
   - Training: 560 citra kanonik (tempat augmentasi dinamis online diterapkan).
   - Validation: 120 citra kanonik murni tanpa distorsi.
   - Testing: 121 citra kanonik murni tanpa distorsi.
