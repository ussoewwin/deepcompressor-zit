@echo off
setlocal

cd /d D:\nu\deepcompressor\deepcompressor-zit
set PYTHONPATH=D:\nu\deepcompressor\deepcompressor-zit
set XFORMERS_DISABLED=1

echo === Z-Image Turbo Calibration Data Collection ===

python -m deepcompressor.app.diffusion.dataset.collect.calib ^
    examples/diffusion/configs/model/zit.yaml ^
    examples/diffusion/configs/collect/qdiff.yaml

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo === Calibration FAILED ===
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo === Calibration COMPLETE ===
pause

