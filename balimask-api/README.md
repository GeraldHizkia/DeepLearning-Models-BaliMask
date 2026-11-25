# BaliMask API · TopengLab

API + interface uji untuk membandingkan **12 model** klasifikasi Topeng Bali
(4 arsitektur × 3 versi training) dengan satu kali upload gambar.

## Mulai cepat

```bash
cd balimask-api
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1) Coba tampilan dulu TANPA bobot (prediksi palsu):
BALIMASK_DEMO=1 uvicorn app.main:app --reload         # Windows PowerShell: $env:BALIMASK_DEMO="1"; uvicorn app.main:app --reload

# 2) Setelah file .pth dimasukkan ke weights/ (lihat weights/README.md):
uvicorn app.main:app --reload
```

Buka:
- http://127.0.0.1:8000: interface uji
- http://127.0.0.1:8000/docs: dokumentasi API (Swagger), bisa langsung dicoba

## Endpoint

| Method | Path | Fungsi |
|---|---|---|
| GET  | `/api/health` | Cek server hidup & jumlah model siap |
| GET  | `/api/models` | Daftar 12 model + status (`ready`/`missing`/`error`/`demo`) |
| POST | `/api/predict` | Upload gambar → prediksi semua model + ringkasan (voting & ensemble) |
| POST | `/api/predict/{model_id}` | Prediksi dengan satu model saja, contoh `vgg16__v2_clean` |

Contoh dengan curl:
```bash
curl -F "file=@topeng.jpg" "http://127.0.0.1:8000/api/predict?top_k=3"
```

## Struktur

```
balimask-api/
├── app/
│   ├── config.py         ← daftar kelas, versi, arsitektur, lokasi bobot
│   ├── architectures.py  ← kerangka model (disalin dari notebook)
│   ├── inference.py      ← load model + preprocessing + prediksi
│   └── main.py           ← endpoint FastAPI
├── static/index.html     ← interface uji (HTML+CSS+JS murni)
├── weights/              ← taruh file .pth di sini
└── PANDUAN_API.md        ← penjelasan konsep, baca ini untuk belajar
```

## Environment variable (opsional)

| Variabel | Default | Fungsi |
|---|---|---|
| `BALIMASK_DEMO` | `0` | `1` = model tanpa bobot mengeluarkan prediksi palsu (untuk tes UI) |
| `BALIMASK_LAZY` | `0` | `1` = model di-load saat pertama dipakai (start lebih cepat, hemat RAM) |
| `BALIMASK_WEIGHTS` | `./weights` | Lokasi folder bobot |
| `BALIMASK_MNV1_VARIANT` | `cbam_head` | Varian MobileNet v1: `none`/`se_head`/`cbam_head`/`cbam_late` |
| `BALIMASK_MAX_MB` | `10` | Batas ukuran upload |

Kebutuhan RAM: sekitar 2 GB untuk ke-12 model di CPU. VGG16 v1 saja sudah ±540 MB.
