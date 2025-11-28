@echo off
chcp 65001 >nul
title Building Wheatgrass Analysis - CLEAN VERSION
echo 🌾 Building Clean Wheatgrass Analysis...
echo ======================================
echo.

echo Step 1: Cleaning previous builds...
rmdir /s /q build 2>nul
rmdir /s /q dist 2>nul

echo Step 2: Building without segment_anything...
pyinstaller --onefile --console --name=WheatgrassAnalysis ^
--hidden-import=cv2 ^
--hidden-import=matplotlib ^
--hidden-import=matplotlib.backends.backend_tkagg ^
--hidden-import=numpy ^
--hidden-import=scipy ^
--hidden-import=scipy.signal ^
--hidden-import=scipy.ndimage ^
--hidden-import=PIL ^
--hidden-import=tkinter ^
--hidden-import=requests ^
--hidden-import=tqdm ^
--exclude-module=segment_anything ^
--exclude-module=torch ^
--exclude-module=torchvision ^
--collect-data=matplotlib ^
--noupx ^
main.py

echo.
if exist "dist\WheatgrassAnalysis.exe" (
    echo ✅ BUILD SUCCESSFUL!
    echo 🚀 Executable: dist\WheatgrassAnalysis.exe
    echo.
    echo 🎯 Now the executable should work without PyTorch errors!
) else (
    echo ❌ Build failed!
)

pause