@echo off
setlocal EnableExtensions
set "PYEXE=python"
set "ROOT=%~dp0"
%PYEXE% src\rl_train.py --epochs 1000 --save-dir models
echo Press any key to close...
pause>nul
endlocal
