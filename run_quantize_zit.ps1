
$ErrorActionPreference = "Stop"

$Python = "D:\nu\venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    Write-Host "WARNING: venv python not found at $Python. Falling back to 'python'."
    $Python = "python"
}

Write-Host "=== Running Pre-flight Verification ==="
& $Python tools/verify_struct.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "CRITICAL FAILURE: Struct verification failed. Aborting."
    exit 1
}

$ConfigPath = "examples/diffusion/configs/svdquant/svdq-fp4-r128.yaml"
& $Python tools/verify_config.py $ConfigPath
if ($LASTEXITCODE -ne 0) {
    Write-Host "CRITICAL FAILURE: Config verification failed. Aborting."
    exit 1
}
Write-Host "=== Verification Passed ==="

Write-Host "=== Cleaning up previous cache files ==="
if (Test-Path "jobs") { Remove-Item -Recurse -Force "jobs" }
# if (Test-Path "datasets") { Remove-Item -Recurse -Force "datasets" } # Optional, can keep datasets

Write-Host "=== Starting ZIT Quantization (PTQ) ==="

$env:PYTHONPATH = "$PWD"
$env:XFORMERS_DISABLED = "1"

& $Python -m deepcompressor.app.diffusion.ptq `
    examples/diffusion/configs/model/zit.yaml `
    $ConfigPath

Write-Host "=== Quantization complete! ==="
Write-Host "Output saved to: quantized_models/"

# Run verification automatically
Write-Host "=== Running Post-Quantization Verification ==="
$GenModel = "quantized_models/z_image_turbo-r128-svdq-fp4.safetensors" 
# Note: Check where ptq.py saves it. Usually based on config name.
# Update this path if needed based on actual output.
# For now, let's assume default behavior.

# Locate the output file
$OutputFiles = Get-ChildItem -Path "quantized_models" -Filter "*.safetensors" | Sort-Object LastWriteTime -Descending
if ($OutputFiles) {
    $Latest = $OutputFiles[0].FullName
    Write-Host "Verifying latest output: $Latest"
    & $Python tools/verify_model.py --gen_path "$Latest"
}
else {
    Write-Host "WARNING: No output model found to verify."
}
