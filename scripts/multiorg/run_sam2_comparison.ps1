# MultiOrg SAM2 对照实验 - zero-shot vs finetuned_v2
# 每轮约 40-60 分钟，共 2 轮 1.5-2 小时
# 跑完看 results\ 下三个 JSON

Set-Location "C:\Users\decha\organoid-fl"

# 激活 venv
& .\.venv\Scripts\Activate.ps1

# 检查文件是否存在
$rdetrCkpt = "output\checkpoint_best_regular.pth"
$zeroShotCkpt = "sam2_checkpoints\sam2_hiera_small.pt"
$finetunedV2Ckpt = "runs\sam2_finetune_v2\sam2_finetuned.pt"

if (-not (Test-Path $rdetrCkpt)) {
    Write-Host "[ERROR] RF-DETR checkpoint not found: $rdetrCkpt"
    exit 1
}
if (-not (Test-Path $zeroShotCkpt)) {
    Write-Host "[ERROR] Zero-shot SAM2 checkpoint not found: $zeroShotCkpt"
    exit 1
}
if (-not (Test-Path $finetunedV2Ckpt)) {
    Write-Host "[WARN] Finetuned v2 checkpoint not found: $finetunedV2Ckpt"
    Write-Host "       Round 2 will be skipped."
}

# ============================================================
# Round 1: Zero-shot SAM2
# ============================================================
Write-Host "`n============================================================"
Write-Host "[1/2] Zero-shot SAM2"
Write-Host "============================================================"
Write-Host "Start: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

python scripts\multiorg\multiorg_sam2.py `
    --weights $rdetrCkpt `
    --model-variant small `
    --src "D:\datasets\mutliorg\MultiOrg_v2\test" `
    --dst "results\multiorg_sam2_zeroshot" `
    --windows 512 --overlap 0.3 --conf 0.25 --score-filter 0.3 `
    --sam2-checkpoint $zeroShotCkpt

Write-Host "End:   $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

# ============================================================
# Round 2: Finetuned v2 (4-point polygon GT)
# ============================================================
if (Test-Path $finetunedV2Ckpt) {
    Write-Host "`n============================================================"
    Write-Host "[2/2] Finetuned v2 (4-point polygon GT)"
    Write-Host "============================================================"
    Write-Host "Start: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

    python scripts\multiorg\multiorg_sam2.py `
        --weights $rdetrCkpt `
        --model-variant small `
        --src "D:\datasets\mutliorg\MultiOrg_v2\test" `
        --dst "results\multiorg_sam2_finetuned_v2" `
        --windows 512 --overlap 0.3 --conf 0.25 --score-filter 0.3 `
        --sam2-checkpoint $finetunedV2Ckpt

    Write-Host "End:   $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
} else {
    Write-Host "`n[SKIP] Round 2: $finetunedV2Ckpt not found"
}

# ============================================================
# Done
# ============================================================
Write-Host "`n============================================================"
Write-Host "ALL DONE"
Write-Host "============================================================"
Write-Host "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "Results:"
Write-Host "  results\multiorg_sam2_zeroshot\multiorg_sam2_results.json"
if (Test-Path $finetunedV2Ckpt) {
    Write-Host "  results\multiorg_sam2_finetuned_v2\multiorg_sam2_results.json"
}
Write-Host "  results\multiorg_sam2_pseudo_finetuned\multiorg_sam2_results.json (already done)"
