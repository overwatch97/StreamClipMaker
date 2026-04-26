@echo off
:: Ensure the script runs in the directory where it's located
cd /d "%~dp0"

echo.
echo ===========================================
echo   StreamClipMaker - Initial Setup (F: Drive)
echo ===========================================
echo.
echo 1. Creating Virtual Environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo [ERROR] Python not found or failed to create venv. 
    echo Please make sure Python is installed and in your PATH.
    pause
    exit /b
)

echo.
echo 2. Installing PyTorch with CUDA Support...
call venv\Scripts\activate
python -m pip install --upgrade pip
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

echo.
echo 3. Fixing ONNX Runtime GPU (preventing conflicts)...
pip uninstall -y onnxruntime onnxruntime-gpu
pip install onnxruntime-gpu==1.24.4

echo.
echo 4. Installing remaining Dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install requirements.
    pause
    exit /b
)

echo.
echo 5. Finalizing Setup...
echo DONE! You can now use Launch_GUI.bat from the F: drive.
echo.
pause
