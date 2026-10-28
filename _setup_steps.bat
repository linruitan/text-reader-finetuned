@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONUNBUFFERED=1
set "CONDA_ROOT=%USERPROFILE%\anaconda3"
if not exist "%CONDA_ROOT%\Scripts\activate.bat" (
  echo [ERROR] Anaconda not found at %CONDA_ROOT%
  exit /b 1
)
call "%CONDA_ROOT%\Scripts\activate.bat"

echo [1/5] Creating conda env "fahua" (Python 3.10 + ffmpeg) ...
call conda env list | findstr /b /c:"fahua " >nul
if errorlevel 1 (
  call conda create -n fahua -y -c conda-forge python=3.10 ffmpeg || exit /b 1
)
call conda activate fahua || exit /b 1
python -m pip install --upgrade pip "setuptools<81" wheel

echo [2/5] Installing PyTorch 2.7.1 (CUDA 12.8, for RTX 50 series) ...
pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128 || exit /b 1

echo [3/5] Installing other packages ...
pip install -r requirements-windows.txt || exit /b 1
pip install pyworld==0.3.4 || python make_stub.py

echo [4/5] Checking CosyVoice code ...
if not exist CosyVoice\cosyvoice (
  echo [ERROR] CosyVoice folder missing
  exit /b 1
)

echo [5/5] Downloading models (about 9GB) ...
python download_models.py || exit /b 1

python -c "import torch; print('CUDA available:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"
where ffmpeg
echo SETUP_OK
