"""
main.py - Pintu masuk API (FastAPI).

Jalankan:   uvicorn app.main:app --reload
Lalu buka:  http://127.0.0.1:8000        (interface uji)
            http://127.0.0.1:8000/docs   (dokumentasi API otomatis / Swagger)
"""
import io
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError

from .config import CLASSES, DEMO_MODE, MAX_UPLOAD_MB, STATIC_DIR, VERSIONS, ARCHS
from .inference import DEVICE, manager
from .curves import load_all_curves


# 1) Lifespan: kode yang jalan SEKALI saat server start -> load semua model
@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Device: {DEVICE} | Demo mode: {DEMO_MODE}")
    manager.load_all()
    yield


app = FastAPI(
    title="BaliMask Classifier API",
    description="Klasifikasi 13 kelas Topeng Bali - komparasi 4 arsitektur × 3 versi training.",
    version="1.0.0",
    lifespan=lifespan,
)

# 2) CORS: izinkan frontend dari domain/port lain memanggil API ini
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---------------------------------------------------------------------------
# Helper: baca & validasi gambar dari upload
# ---------------------------------------------------------------------------
async def read_image(file: UploadFile) -> Image.Image:
    data = await file.read()
    if not data:
        raise HTTPException(400, "File kosong.")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File terlalu besar (maks {MAX_UPLOAD_MB:g} MB).")
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img)        # foto HP sering tersimpan miring
        return img.convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(415, "File bukan gambar yang valid (gunakan JPG/PNG/WEBP).")


# ---------------------------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------------------------
@app.get("/api/health", tags=["Info"])
def health():
    """Cek apakah server hidup + berapa model yang siap."""
    models = manager.describe()
    return {
        "status": "ok",
        "device": str(DEVICE),
        "demo_mode": DEMO_MODE,
        "models_ready": sum(m["status"] == "ready" for m in models),
        "models_total": len(models),
    }


@app.get("/api/models", tags=["Info"])
def list_models():
    """Daftar 12 model beserta statusnya (ready / missing / error / demo)."""
    return {
        "classes": CLASSES,
        "versions": VERSIONS,
        "archs": ARCHS,
        "models": manager.describe(),
    }


@app.get("/api/curves", tags=["Info"])
def training_curves():
    """Kurva training (loss, akurasi, F1 per epoch) + metrik test tiap model.

    Dibaca dari file .ipynb di folder kurva/ (lihat app/curves.py).
    Dibaca ulang setiap request, jadi cukup taruh file baru lalu refresh halaman.
    """
    curves = load_all_curves()
    return {
        "versions": VERSIONS,
        "archs": ARCHS,
        "available": sorted(curves),
        "curves": curves,
    }


@app.post("/api/predict", tags=["Prediksi"])
async def predict_all(
    file: UploadFile = File(..., description="Gambar topeng (JPG/PNG/WEBP)"),
    top_k: int = Query(3, ge=1, le=13, description="Jumlah prediksi teratas per model"),
):
    """Upload 1 gambar -> diprediksi oleh SEMUA model -> hasil komparasi."""
    img = await read_image(file)
    t0 = time.perf_counter()
    results, summary = manager.predict_all(img, top_k=top_k)
    return {
        "filename": file.filename,
        "image_size": list(img.size),
        "device": str(DEVICE),
        "demo_mode": DEMO_MODE,
        "total_ms": round((time.perf_counter() - t0) * 1000, 1),
        "summary": summary,
        "results": results,
    }


@app.post("/api/predict/{model_id}", tags=["Prediksi"])
async def predict_one(model_id: str, file: UploadFile = File(...), top_k: int = Query(3, ge=1, le=13)):
    """Prediksi dengan SATU model saja, contoh model_id: vgg16__v2_clean"""
    if model_id not in manager.entries:
        raise HTTPException(404, f"Model '{model_id}' tidak dikenal. Lihat GET /api/models.")
    img = await read_image(file)
    results, _ = manager.predict_all(img, top_k=top_k, only=[model_id])
    return results[0]


# ---------------------------------------------------------------------------
# Frontend (interface uji) disajikan dari folder static/
# ---------------------------------------------------------------------------
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index():
    # no-cache: browser selalu mengambil tampilan terbaru setelah file diubah
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})
