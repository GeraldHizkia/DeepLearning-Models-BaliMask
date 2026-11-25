# Taruh file bobot (.pth) di sini

Ambil `best_model.pth` (atau `*_best.pt`) dari tiap notebook, lalu **ganti namanya** sesuai tabel:

```
weights/
├── v1_awal/                      ← notebooks_v1_baseline/
│   ├── convnext_tiny.pth         ← ConvNeXt.ipynb            (convnext_tiny_best.pt)
│   ├── efficientnetv2_s.pth      ← EfficientNetV2_S.ipynb    (effnetv2s_best.pt)
│   ├── mobilenetv3_cbam.pth      ← MobileNetV3_Attention.ipynb  (mobilenetv3_cbam_head/seed_*/best_model.pth)
│   └── vgg16.pth                 ← VGG16.ipynb               (vgg16_best.pt)
├── v2_clean/                     ← notebooks_v2_canonical/*_Canonical.ipynb
│   ├── convnext_tiny.pth         ← 01_ConvNeXt_Tiny_Canonical
│   ├── efficientnetv2_s.pth      ← 02_EfficientNetV2_S_Canonical
│   ├── mobilenetv3_cbam.pth      ← 03_MobileNetV3_CBAM_Canonical
│   └── vgg16.pth                 ← 04_VGG16_Canonical
└── v3_padat/                     ← notebooks_v2_canonical/*_Ultra_Robust.ipynb
    ├── convnext_tiny.pth         ← 01_ConvNeXt_Tiny_Ultra_Robust
    ├── efficientnetv2_s.pth      ← 02_EfficientNetV2_S_Ultra_Robust
    ├── mobilenetv3_cbam.pth      ← 03_MobileNetV3_CBAM_Ultra_Robust
    └── vgg16.pth                 ← 04_VGG16_Ultra_Robust
```

Ekstensi `.pt` atau `.pth` sama saja. Yang penting namanya cocok.
Kalau untuk MobileNet v1 kamu ingin varian lain (misal `se_head`), jalankan server dengan
`BALIMASK_MNV1_VARIANT=se_head`.

Model yang filenya belum ada tetap tampil di interface dengan status **"Bobot belum ada"**,
jadi kamu bisa mengisinya satu per satu.
