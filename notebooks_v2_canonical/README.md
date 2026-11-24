# Dokumentasi Pipeline Kanonik 801 Citra Topeng Tradisional Bali

Repositori ini memuat implementasi model deep learning untuk klasifikasi 13 kelas topeng tradisional Bali menggunakan partisi data kanonik 801 citra fisik unik bebas dari kebocoran data (zero data leakage).

---

## Daftar Berkas Notebook

| Berkas | Fokus Utama | Target Arsitektur |
|---|---|---|
| 00_Dataset_Audit_and_Canonical_Split.ipynb | Audit bukti kebocoran Roboflow (.rf.) dan ekstraksi 801 citra kanonik | Data Engineering |
| 01_ConvNeXt_Tiny_Canonical.ipynb | Model ConvNeXt-Tiny versi baseline kanonik | ConvNeXt-Tiny |
| 01_ConvNeXt_Tiny_Ultra_Robust.ipynb | Model ConvNeXt-Tiny dengan two-stage fine-tuning dan Random Erasing | ConvNeXt-Tiny Ultra-Robust |
| 02_EfficientNetV2_S_Canonical.ipynb | Model EfficientNetV2-S versi baseline kanonik | EfficientNetV2-S |
| 02_EfficientNetV2_S_Ultra_Robust.ipynb | Model EfficientNetV2-S dengan two-stage fine-tuning dan Random Erasing | EfficientNetV2-S Ultra-Robust |
| 03_MobileNetV3_CBAM_Canonical.ipynb | Model MobileNetV3-Large dengan modul atensi CBAM versi baseline kanonik | MobileNetV3 + CBAM |
| 03_MobileNetV3_CBAM_Ultra_Robust.ipynb | Model MobileNetV3-Large CBAM dengan two-stage fine-tuning dan Random Erasing | MobileNetV3 + CBAM Ultra-Robust |
| 04_VGG16_Canonical.ipynb | Model VGG16 with BatchNorm versi baseline kanonik | VGG16-BN |
| 04_VGG16_Ultra_Robust.ipynb | Model VGG16-BN dengan two-stage fine-tuning selektif dan Random Erasing | VGG16-BN Ultra-Robust |
| 05_Master_Comparison_and_Error_Analysis.ipynb | Dashboard komparasi master multi-track, error analysis budaya, dan Grad-CAM | Master Evaluation |

---

## Metodologi Partisi Kanonik

1. Deduplikasi Berbasis Identitas Fisik:
   Nama berkas yang memuat variasi rotasi offline dan hash ekspor Roboflow dikembalikan ke identitas dasar objek fisik (base_id).
2. Verifikasi Perceptual Hashing (pHash):
   Pemeriksaan kemiripan visual dilakukan menggunakan algoritma pHash 64-bit dengan threshold Hamming Distance 4 untuk memastikan tidak ada duplikat visual antar-subset.
3. Pembagian Stratified 70:15:15:
   - Training: 561 citra kanonik (tempat augmentasi dinamis online RandAugment dan Random Erasing diterapkan).
   - Validation: 119 citra kanonik murni tanpa distorsi.
   - Testing: 121 citra kanonik murni tanpa distorsi.
