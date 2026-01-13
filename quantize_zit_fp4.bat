@echo off
setlocal

cd /d D:\nu\deepcompressor\deepcompressor-zit
set PYTHONPATH=D:\nu\deepcompressor\deepcompressor-zit
set XFORMERS_DISABLED=1

echo === Z-Image Turbo R128 SVDQ-FP4 Quantization ===

python -m deepcompressor.app.diffusion.ptq ^
    examples/diffusion/configs/model/zit.yaml ^
    examples/diffusion/configs/svdquant/fp4.yaml ^
    --skip-eval --skip-gen ^
    --export-nunchaku-zit "D:\nu\svdq-fp4_r128-z-image-turbo-custom.safetensors"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo === Quantization FAILED ===
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo === Quantization COMPLETE ===
echo Output: D:\nu\svdq-fp4_r128-z-image-turbo-custom.safetensors
pause

