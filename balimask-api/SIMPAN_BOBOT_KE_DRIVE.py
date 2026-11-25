# ============================================================
#  TEMPEL SEL INI DI AKHIR SETIAP NOTEBOOK TRAINING, LALU RUN.
#  Gunanya: menyalin best_model.pth dari disk Colab (yang akan
#  terhapus) ke Google Drive, sekalian memberi nama yang benar
#  supaya langsung dikenali API.
# ============================================================
from pathlib import Path
import shutil

from google.colab import drive
drive.mount('/content/drive')

DEST = Path('/content/drive/MyDrive/TopengBali_Bobot')

# Untuk MobileNetV3 notebook v1 (punya 4 varian ablation).
# Ganti kalau mau varian lain: mobilenetv3_base / _se_head / _cbam_head / _cbam_late
VARIAN_MOBILENET_V1 = 'mobilenetv3_cbam_head'


def tujuan(nama_folder):
    """Tebak versi + arsitektur dari nama folder outputs_*."""
    n = nama_folder.lower()
    if 'ultrarobust' in n or 'ultra_robust' in n:
        versi = 'v3_padat'
    elif 'canonical' in n:
        versi = 'v2_clean'
    else:
        versi = 'v1_awal'

    if 'convnext' in n:
        arsitektur = 'convnext_tiny'
    elif 'effnet' in n or 'efficientnet' in n:
        arsitektur = 'efficientnetv2_s'
    elif 'mobilenet' in n:
        arsitektur = 'mobilenetv3_cbam'
    elif 'vgg' in n:
        arsitektur = 'vgg16'
    else:
        return None
    return versi, arsitektur


def cari_checkpoint(d):
    """Cari file bobot di dalam folder outputs_*."""
    for pola in ('best_model.pth', '*_best.pt', '*_best.pth'):
        hit = sorted(d.glob(pola))
        if hit:
            return hit[0]
    # Notebook MobileNetV3 v1 menyimpan per varian & per seed
    hit = sorted(d.glob(f'{VARIAN_MOBILENET_V1}/seed_*/best_model.pth'))
    if not hit:
        hit = sorted(d.glob('*/seed_*/best_model.pth'))
    return hit[0] if hit else None


# Kumpulkan semua folder outputs_* (tanpa duplikat)
folders = {}
for induk in (Path('.'), Path('/content'), Path('/content/drive/MyDrive')):
    try:
        for d in induk.glob('outputs_*'):
            if d.is_dir():
                folders[d.resolve()] = d
    except OSError:
        pass

if not folders:
    print('Tidak ada folder outputs_* ditemukan.')
    print('Jalankan sel training dulu, baru sel ini.')

n_ok = 0
for d in sorted(folders.values()):
    t = tujuan(d.name)
    if t is None:
        print(f'[lewati] {d.name}: tidak dikenali')
        continue
    versi, arsitektur = t
    ckpt = cari_checkpoint(d)
    if ckpt is None:
        print(f'[kosong] {d.name}: belum ada file bobot')
        continue

    dst = DEST / versi / f'{arsitektur}.pth'
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(ckpt, dst)
    n_ok += 1
    print(f'[OK] {ckpt.relative_to(d.parent)}'
          f'  ->  MyDrive/TopengBali_Bobot/{versi}/{arsitektur}.pth'
          f'  ({dst.stat().st_size / 1e6:.1f} MB)')

print(f'\n{n_ok} file bobot tersimpan di Google Drive.')
print('Sudah lengkap kalau total ada 12 file (4 arsitektur x 3 versi).')

# Rekap isi folder tujuan
print('\nIsi MyDrive/TopengBali_Bobot sekarang:')
total = 0
for versi in ('v1_awal', 'v2_clean', 'v3_padat'):
    ada = sorted((DEST / versi).glob('*.pth')) if (DEST / versi).exists() else []
    total += len(ada)
    nama = ', '.join(p.stem for p in ada) if ada else '(kosong)'
    print(f'  {versi:10s} [{len(ada)}/4]  {nama}')
print(f'  TOTAL: {total}/12')
