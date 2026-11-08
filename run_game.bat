@echo off
cd /d "%~dp0"
python -m turkeyworld.play --size 256 --display-size 1024 --fps 60
if errorlevel 1 pause
