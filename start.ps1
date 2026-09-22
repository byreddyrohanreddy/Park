# VoiceCare AI - PowerShell Launcher
Set-Location -Path $PSScriptRoot
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Starting VoiceCare AI (PD-VoiceNet) System" -ForegroundColor Cyan
Write-Host "  Targeting Conda Environment: ml" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
node run.js
