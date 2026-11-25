# Panduan Belajar: Membuat API untuk Model Deep Learning

Panduan ini menjelaskan **cara berpikir** di balik kode di folder `app/`.
Bacanya berurutan. Tiap bagian merujuk ke file yang bisa langsung kamu buka.

---

## 1. Apa itu API? (analogi warung)

Bayangkan **warung makan**:

| Warung | API kita |
|---|---|
| Pembeli | Frontend (`index.html`), aplikasi HP, Postman, curl |
| Menu | Daftar **endpoint** (`/api/predict`, `/api/models`, …) |
| Pesanan ("nasi goreng 1, pedas") | **Request** (method + URL + data) |
| Dapur | Model PyTorch di server |
| Makanan yang diantar | **Response** (JSON) |

Pembeli tidak perlu tahu cara memasak. Frontend juga tidak perlu tahu PyTorch,
cukup tahu "kirim gambar ke `/api/predict`, nanti dapat JSON".
Inilah inti API: **memisahkan yang memakai dari yang mengerjakan**.

---

## 2. Anatomi satu request

```
POST /api/predict?top_k=3          ← METHOD + PATH + QUERY PARAMETER
Content-Type: multipart/form-data  ← HEADER (jenis data yang dikirim)

file = <isi gambar.jpg>            ← BODY
```

- **Method**: `GET` = minta data (tidak mengubah apa pun), `POST` = kirim data untuk diproses.
  Upload gambar pakai `POST` karena kita *mengirim* sesuatu.
- **Path**: alamat endpoint.
- **Query parameter** (`?top_k=3`): opsi kecil di URL.
- **Body**: datanya. Untuk file pakai `multipart/form-data`, bukan JSON,
  karena JSON tidak efisien untuk data biner.

Lalu server menjawab dengan **status code** + body JSON:

| Kode | Arti | Contoh di kode kita |
|---|---|---|
| 200 | Berhasil | prediksi sukses |
| 400 | Request salah | file kosong |
| 404 | Tidak ditemukan | `model_id` ngawur |
| 413 | Terlalu besar | file > 10 MB |
| 415 | Tipe tidak didukung | upload PDF, bukan gambar |
| 500 | Server error | bug di kode kita |

Coba lihat `read_image()` di `app/main.py`: setiap `raise HTTPException(...)` adalah
cara kita "menolak pesanan dengan sopan" dan menjelaskan alasannya.

---

## 3. Kenapa FastAPI?

```python
@app.post("/api/predict")
async def predict_all(file: UploadFile = File(...), top_k: int = Query(3, ge=1, le=13)):
    ...
```

Dari satu fungsi itu, FastAPI otomatis:
1. **Mendaftarkan endpoint** `POST /api/predict` (dari decorator `@app.post`)
2. **Membaca & memvalidasi input**: `top_k` harus int antara 1-13. Kalau user kirim `top_k=99`,
   FastAPI langsung membalas error 422 tanpa kamu tulis `if`.
3. **Mengubah return dict → JSON**
4. **Membuat dokumentasi interaktif** di `/docs`. Buka ini dan klik "Try it out".
   Ini cara tercepat untuk belajar dan mengetes API tanpa frontend.

---

## 4. Bagian tersulit: memuat model dengan benar

### 4.1 `.pth` itu isinya apa?

Di notebook kalian menyimpan:
```python
torch.save(model.state_dict(), 'best_model.pth')
```
`state_dict` = **kamus berisi angka bobot**, misalnya:
```
{"features.0.0.weight": tensor(...), "classifier.2.1.weight": tensor(...), ...}
```
Kerangkanya tidak ikut tersimpan. Analoginya, yang tersimpan hanya isi lemari, bukan lemarinya.
Jadi di API kita harus **membangun ulang lemari yang persis sama** (`app/architectures.py`),
baru isinya dimasukkan:

```python
model = build_model("vgg16", "v2_clean")      # bangun kerangka
model.load_state_dict(torch.load(path))       # tuang bobot
model.eval()                                   # mode evaluasi
```

### 4.2 Kenapa v1 dan v2/v3 punya fungsi berbeda?

Karena head classifier-nya memang beda di notebook:

| Model | v1 (baseline) | v2 & v3 (canonical / ultra-robust) |
|---|---|---|
| ConvNeXt | `Linear(768,256)→GELU→Dropout→Linear(256,13)` | `Dropout→Linear(768,13)` |
| VGG16 | **VGG16 biasa**, ganti `classifier[6]` | **VGG16-BN**, classifier baru 25088→1024→256→13 |
| MobileNetV3 | CBAM ditempel di `features[17]` (gaya ChannelGate/SpatialGate) | Kelas `MobileNetV3_CBAM` dengan `self.cbam` |
| EfficientNetV2-S | sama | sama |

Kalau kerangkanya beda sedikit saja, `load_state_dict(strict=True)` akan error
`Missing key(s)` / `Unexpected key(s)`. **Error ini justru teman kita**: dia memberi tahu
kerangka belum cocok. Dropout tidak punya bobot, jadi nilai p-nya boleh beda.

> [v] Kode ini sudah diuji: model dibangun dari kode sel notebook asli, `state_dict`-nya disimpan,
> lalu ke-12 berhasil di-load oleh API dengan `strict=True`.

### 4.3 `weights=None`

Di notebook kita pakai `weights=...DEFAULT` (ImageNet). Di API kita pakai `weights=None`
karena bobot ImageNet toh akan **ditimpa** bobot hasil training. Hasilnya start lebih cepat
dan tidak perlu internet.

### 4.4 `model.eval()` dan `torch.inference_mode()`

- `eval()` mematikan Dropout dan membuat BatchNorm memakai statistik hasil training.
  **Lupa ini = prediksi acak-acakan.**
- `inference_mode()` mematikan perhitungan gradien, jadi lebih cepat dan hemat memori.

---

## 5. Preprocessing harus SAMA PERSIS dengan saat evaluasi

Ini kesalahan paling sering di deployment: model bagus di notebook tapi jelek di API.

| Versi | `eval_transform` di notebook | Di API (`config.py → preprocess`) |
|---|---|---|
| v1 ConvNeXt/EffNet/VGG | `Resize((224,224))` | `"resize"` |
| v1 MobileNet, semua v2 & v3 | `Resize(255) → CenterCrop(224)` | `"resize_crop"` |
| Semua | `Normalize(mean ImageNet, std ImageNet)` | sama |

Plus dua hal kecil di `main.py`:
- `img.convert("RGB")`: PNG bisa punya 4 channel (RGBA), model butuh 3.
- `ImageOps.exif_transpose`: foto HP sering tersimpan miring, lalu "diputar" lewat metadata EXIF.

Urutan 13 kelas di `config.CLASSES` juga harus sama dengan saat training (alfabetis,
dari `sorted(nama folder)`). Kalau urutannya beda, label yang tampil akan tertukar.

---

## 6. Alur lengkap satu prediksi

```
[Browser]                        [FastAPI]                              [PyTorch]
pilih gambar
FormData.append("file", f)
fetch POST /api/predict  ───────► read_image(): validasi ukuran & tipe
                                  PIL.Image → RGB
                                  manager.predict_all()
                                     untuk tiap 12 model:
                                       transform(img) → tensor [1,3,224,224] ─► model(x) → logits [1,13]
                                                                               softmax → probabilitas
                                       ambil top-k, catat waktu (ms)
                                     _summarize(): voting & rata-rata prob
                     ◄─────────── return dict → JSON
render kartu 4×3
```

Optimasi kecil di `inference.py`: tensor hasil preprocessing **di-cache per jenis
transform**, jadi gambar yang sama tidak di-resize 12 kali.

---

## 7. Model di-load SEKALI, bukan per request

```python
@asynccontextmanager
async def lifespan(app):
    manager.load_all()   # jalan sekali saat server start
    yield
```

Memuat 12 model butuh beberapa detik. Kalau dimuat ulang tiap request, satu prediksi bisa
butuh 10 detik lebih. Karena itu model disimpan di memori (`manager.models`) dan dipakai berulang.
Kalau RAM laptop terbatas, pakai `BALIMASK_LAZY=1`: model baru dimuat saat pertama dibutuhkan.

---

## 8. Ringkasan komparasi: voting & ensemble

- **Voting (konsensus)**: tiap model memberi 1 suara untuk label top-1-nya; label terbanyak menang.
  `agreement = suara / jumlah model` → seberapa kompak model-model setuju.
- **Ensemble**: rata-rata vektor probabilitas 12 model, lalu ambil yang tertinggi.
  Biasanya lebih stabil daripada satu model.
- **Label asli (opsional di UI)**: kalau kamu tahu jawabannya, UI menandai [v]/[x] per model.
  Berguna untuk mengecek kasus sulit seperti *macan vs barong-bangkal vs barong-ket*.

---

## 9. CORS: kenapa ada `CORSMiddleware`?

Browser melarang halaman dari `http://localhost:5500` memanggil API di `http://localhost:8000`
(beda port = beda "origin") **kecuali** server mengizinkan. Di sini interface disajikan dari
server yang sama, jadi sebenarnya aman tanpa CORS. Middleware tetap dipasang supaya kalau nanti
frontend-nya dipisah (React, Flutter Web, dsb.), API langsung bisa dipakai.
Untuk produksi, ganti `allow_origins=["*"]` dengan domain spesifik.

---

## 10. Latihan untuk memperdalam

1. **Tambah endpoint** `GET /api/classes` yang mengembalikan daftar 13 kelas. (5 menit)
2. **Pydantic response model**: buat class `PredictResponse(BaseModel)` dan pasang
   `@app.post(..., response_model=PredictResponse)`, lalu lihat perubahan di `/docs`.
3. **Grad-CAM endpoint**: `POST /api/gradcam/{model_id}` yang mengembalikan heatmap PNG (base64).
   Target layer sudah ada di notebook (`model.features[-1]`).
4. **Batch**: terima beberapa file sekaligus (`files: list[UploadFile]`).
5. **Test otomatis** dengan `fastapi.testclient.TestClient` + `pytest`.
6. **Deploy**: bungkus dengan Dockerfile, atau coba Hugging Face Spaces.

---

## 11. Troubleshooting

| Gejala | Penyebab & solusi |
|---|---|
| Kartu "Gagal load: Missing key(s)…" | Kerangka tidak cocok. Cek file .pth ditaruh di folder versi yang benar (v1 vs v2/v3) |
| `Unexpected key(s) … ema` / `optimizer` | Checkpoint berisi dict lain; `_extract_state_dict()` sudah menangani key umum, tambahkan key-mu bila perlu |
| `size mismatch … [13, …]` | Jumlah kelas beda. Cek `CLASSES` |
| Semua prediksi aneh tapi tidak error | Preprocessing atau urutan kelas tidak sama dengan training |
| `weights_only` error saat load | File disimpan sebagai seluruh model (`torch.save(model)`). Simpan ulang sebagai `state_dict()` |
| Server lambat start | Pakai `BALIMASK_LAZY=1` |
| Interface "API tidak terhubung" | Server belum jalan, atau `API_BASE` di `index.html` salah |
