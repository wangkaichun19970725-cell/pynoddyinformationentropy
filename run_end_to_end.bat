@echo off
setlocal EnableExtensions
set "PYEXE=python"
set "ROOT=%~dp0"
set "OUT=%ROOT%outputs"

if not exist "%OUT%" mkdir "%OUT%"

echo [1/4] generate a small local dataset with pynoddy
%PYEXE% src\gen_dataset.py --num_models 50 --out data --seed 42

echo [2/4] build entropy from a small ensemble
%PYEXE% src\build_entropy_from_ensemble.py --case data\train\model_00001 --out %OUT%\entropy

echo [3/4] train RL (or skip if model exists)
%PYEXE% src\rl_train.py --epochs 200 --save-dir models

echo [4/4] plan drilling on a fresh case
%PYEXE% src\plan_drilling.py --g00 data\test\model_00002\case.g00 --g12 data\test\model_00002\case.g12 --n 5 --out %OUT%\plans

echo Done. Results in %OUT%
echo Press any key to close...
pause>nul
endlocal
