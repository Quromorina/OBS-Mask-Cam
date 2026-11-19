@echo off
rem Double-click to reset a USB camera that stopped responding (runs reset-camera.ps1).
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0reset-camera.ps1" %*
