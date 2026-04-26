$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Definition
$PYTHON_EXE = "$SCRIPT_DIR\venv\Scripts\python.exe"

Write-Host "--- StreamClipMaker GPU Upgrade ---" -ForegroundColor Cyan
Write-Host "Updating environment at: $SCRIPT_DIR"

if (!(Test-Path $PYTHON_EXE)) {
    Write-Host "Error: Virtual environment not found at $PYTHON_EXE" -ForegroundColor Red
    exit 1
}

Write-Host "`n[1/3] Installing CUDA-enabled Torch (this may take a few minutes)..." -ForegroundColor Yellow
& $PYTHON_EXE -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --force-reinstall --no-cache-dir

Write-Host "`n[2/3] Switching to ONNX Runtime GPU (pinned 1.20.1)..." -ForegroundColor Yellow
& $PYTHON_EXE -m pip uninstall -y onnxruntime onnxruntime-gpu
& $PYTHON_EXE -m pip install onnxruntime-gpu==1.20.1

Write-Host "`n[3/3] Verifying installation..." -ForegroundColor Yellow
& $PYTHON_EXE -c "import torch; print('Torch CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

Write-Host "`n✅ Upgrade complete! Restart StreamClipMaker to use GPU mode." -ForegroundColor Green
