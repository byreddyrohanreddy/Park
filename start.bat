@echo off
title VoiceCare AI - Parkinson's Detection System
cd /d "%~dp0"
echo ============================================================
echo   Starting VoiceCare AI (PD-VoiceNet) System
echo   Targeting Conda Environment: ml
echo ============================================================
echo.

node run.js
pause
