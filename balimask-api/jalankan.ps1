# ===========================================================
#   BaliMask API - setup otomatis + jalankan server
#   Jangan dijalankan langsung. Klik dua kali: JALANKAN.bat
# ===========================================================
$ErrorActionPreference = 'Continue'
$root = $PSScriptRoot
Set-Location $root

$venv   = Join-Path $root '.venv'
$py     = Join-Path $venv 'Scripts\python.exe'
$oldEnv = Join-Path (Split-Path -Parent $root) '.venv'
$port   = 8000

function Say($t, $c = 'Gray') { Write-Host $t -ForegroundColor $c }

Say ''
Say '===============================================' Cyan
Say '   BaliMask API  -  TopengLab' Cyan
Say '===============================================' Cyan
Say ''

# ----------------------------------------------------------
# 1. Cek apakah virtual environment sudah sehat
# ----------------------------------------------------------
$sehat = $false
if (Test-Path $py) {
    Say 'Memeriksa virtual environment...'
    & $py -c "import numpy, PIL.Image, pydantic_core, torch, torchvision, fastapi, uvicorn" 2>$null
    $sehat = ($LASTEXITCODE -eq 0)
}

if ($sehat) {
    Say 'Virtual environment sehat. Lanjut.' Green
}
else {
    Say 'Virtual environment belum ada atau rusak. Akan dibuat ulang.' Yellow
    Say '(Penyebab sebelumnya: library ter-install untuk Python 3.14,' DarkGray
    Say ' sedangkan venv memakai Python 3.12, jadi file .pyd tidak cocok.)' DarkGray
    Say ''

    # ---- Cari Python yang cocok ----
    $basePy  = $null
    $baseVer = $null
    foreach ($c in @(
        "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe",
        "C:\Python312\python.exe",
        "C:\Python311\python.exe"
    )) {
        if (Test-Path $c) { $basePy = $c; break }
    }
    if (-not $basePy -and (Get-Command py -ErrorAction SilentlyContinue)) {
        foreach ($v in @('3.12', '3.11', '3.13')) {
            & py "-$v" -c "import sys" 2>$null
            if ($LASTEXITCODE -eq 0) { $baseVer = $v; break }
        }
    }
    if (-not $basePy -and -not $baseVer) {
        Say 'GAGAL: Python 3.11 / 3.12 / 3.13 tidak ditemukan.' Red
        Say 'Install Python 3.12 dari python.org, lalu jalankan file ini lagi.' Red
        Read-Host 'Tekan Enter untuk menutup'
        exit 1
    }
    if ($basePy) { Say "Memakai Python: $basePy" } else { Say "Memakai Python: py -$baseVer" }

    # ---- Hapus venv lama yang rusak ----
    foreach ($d in @($venv, $oldEnv)) {
        if (Test-Path $d) {
            Say "Menghapus venv lama yang rusak: $d" DarkGray
            Remove-Item -Recurse -Force $d -ErrorAction SilentlyContinue
        }
    }
    Get-ChildItem -Path $root -Recurse -Directory -Filter '__pycache__' -ErrorAction SilentlyContinue |
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

    # ---- Buat venv baru ----
    Say ''
    Say 'Membuat virtual environment baru...'
    if ($basePy) { & $basePy -m venv $venv } else { & py "-$baseVer" -m venv $venv }
    if (-not (Test-Path $py)) {
        Say 'GAGAL membuat virtual environment.' Red
        Read-Host 'Tekan Enter untuk menutup'
        exit 1
    }
    & $py --version

    # ---- Install library ----
    Say ''
    Say '-----------------------------------------------' DarkGray
    Say ' Mengunduh PyTorch versi CPU (+/- 250 MB).' Yellow
    Say ' Ini bagian paling lama, bisa 5-15 menit. Tunggu saja.' Yellow
    Say '-----------------------------------------------' DarkGray
    & $py -m pip install --upgrade pip --disable-pip-version-check
    & $py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --disable-pip-version-check

    Say ''
    Say ' Mengunduh FastAPI dan library lainnya...' Yellow
    & $py -m pip install -r (Join-Path $root 'requirements.txt') --disable-pip-version-check

    # ---- Verifikasi ----
    Say ''
    Say 'Memeriksa hasil instalasi...'
    & $py -c "import numpy, PIL.Image, pydantic_core, torch, torchvision, fastapi, uvicorn; print('  numpy', numpy.__version__, '| torch', torch.__version__)"
    if ($LASTEXITCODE -ne 0) {
        Say ''
        Say 'Masih ada library yang gagal dimuat. Periksa pesan error di atas atau cek requirements.txt.' Red
        Read-Host 'Tekan Enter untuk menutup'
        exit 1
    }
    Say 'Semua library berhasil dimuat.' Green
}

# ----------------------------------------------------------
# 2. Cek file bobot. Kalau kosong -> mode demo
# ----------------------------------------------------------
Say ''
$bobot = @(Get-ChildItem -Path (Join-Path $root 'weights') -Recurse -File -ErrorAction SilentlyContinue |
           Where-Object { $_.Extension -in @('.pth', '.pt') })
if ($bobot.Count -eq 0) {
    $env:BALIMASK_DEMO = '1'
    Say 'Folder weights masih kosong, jadi server jalan dalam MODE DEMO.' Yellow
    Say 'Prediksinya palsu/acak, hanya untuk mencoba tampilan dan alur API.' Yellow
    Say 'Cara memasukkan bobot asli ada di weights\README.md' DarkGray
}
else {
    $env:BALIMASK_DEMO = '0'
    Say "Ditemukan $($bobot.Count) file bobot. Server memakai model asli." Green
    Say 'Model dimuat saat server start, mohon tunggu sampai muncul "Application startup complete".' DarkGray
}

# ----------------------------------------------------------
# 2b. Matikan server lama yang mungkin masih berjalan
#     (kalau tidak, browser tetap membuka server versi lama)
# ----------------------------------------------------------
$lama = @()
try {
    $lama += Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction Stop |
             Where-Object { $_.CommandLine -like '*uvicorn*app.main*' } |
             Select-Object -ExpandProperty ProcessId
} catch {}
try {
    $lama += Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction Stop |
             Select-Object -ExpandProperty OwningProcess
} catch {}
$lama = $lama | Where-Object { $_ -and $_ -ne $PID } | Sort-Object -Unique
if ($lama) {
    Say "Menghentikan server lama yang masih berjalan (PID: $($lama -join ', '))..." Yellow
    foreach ($id in $lama) { Stop-Process -Id $id -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Seconds 2
}

# ----------------------------------------------------------
# 3. Buka browser otomatis begitu server siap
# ----------------------------------------------------------
$waiter = "for(`$i=0;`$i -lt 600;`$i++){ try { `$c = New-Object Net.Sockets.TcpClient; `$c.Connect('127.0.0.1',$port); `$c.Close(); Start-Process 'http://127.0.0.1:$port'; break } catch { Start-Sleep -Milliseconds 500 } }"
Start-Process powershell -WindowStyle Hidden -ArgumentList '-NoProfile', '-Command', $waiter

# ----------------------------------------------------------
# 4. Jalankan server
# ----------------------------------------------------------
Say ''
Say '===============================================' Cyan
Say "  Interface uji : http://127.0.0.1:$port" Cyan
Say "  Dokumentasi   : http://127.0.0.1:$port/docs" Cyan
Say '  Tekan Ctrl+C untuk menghentikan server.' Cyan
Say '===============================================' Cyan
Say ''

& $py -m uvicorn app.main:app --host 127.0.0.1 --port $port

Say ''
Read-Host 'Server berhenti. Tekan Enter untuk menutup jendela ini'
