# Quick Start - Windows Build Fix

## Problem
The error `MSB1009: Project file does not exist` means CMake didn't generate the Visual Studio solution files.

## Solution

### Step 1: Check Prerequisites
```powershell
# Verify CUDA is installed
nvcc --version

# Verify CMake is installed
cmake --version

# Should be CMake 3.20 or higher
```

### Step 2: Clean and Reconfigure
```powershell
# Navigate to project directory
cd C:\path\to\hello-world

# Remove old build directory if exists
Remove-Item -Recurse -Force build -ErrorAction SilentlyContinue

# Create fresh build directory
mkdir build
cd build
```

### Step 3: Configure with CMake

**If you have vcpkg installed:**
```powershell
cmake .. -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_TOOLCHAIN_FILE=C:\vcpkg\scripts\buildsystems\vcpkg.cmake
```

**If you DON'T have vcpkg yet:**
```powershell
# First install vcpkg
cd C:\
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg integrate install

# Install dependencies
.\vcpkg install glfw3:x64-windows
.\vcpkg install glad:x64-windows
.\vcpkg install glm:x64-windows
.\vcpkg install imgui[glfw-binding,opengl3-binding]:x64-windows

# Then go back and configure
cd C:\path\to\hello-world\build
cmake .. -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_TOOLCHAIN_FILE=C:\vcpkg\scripts\buildsystems\vcpkg.cmake
```

### Step 4: Build
```powershell
# After successful configure, you should see:
# -- Generating done
# -- Build files written to: C:/path/to/hello-world/build

# Then build:
cmake --build . --config Release
```

## Common Issues

### Issue 1: "Could not find CUDA"
- Install CUDA Toolkit from: https://developer.nvidia.com/cuda-downloads
- Restart after installation
- Verify: `nvcc --version`

### Issue 2: "Could not find glfw3"
- Install vcpkg (see above)
- Install packages via vcpkg
- Use `-DCMAKE_TOOLCHAIN_FILE` in cmake command

### Issue 3: "Python version not found"
- Ignore this - we don't need Python for C++ build!
- Close VSCode if it's open (might interfere)

### Issue 4: CMake generates but build fails
- Check the actual error in build output
- Most common: Missing CUDA (install toolkit)
- Or missing vcpkg packages (install them)

## Verification

After successful build, you should have:
```
build\bin\Release\bec4d_sim.exe
```

Run it:
```powershell
cd build\bin\Release
.\bec4d_sim.exe
```

## Alternative: Build in Visual Studio

1. After CMake configure succeeds, you'll have `BEC4D.sln` in build directory
2. Open it in Visual Studio 2022
3. Right-click solution → Build Solution
4. Set bec4d_sim as startup project
5. Press F5 to run

## Still Having Issues?

Share the **full output** of the CMake configure command. The error messages will tell us what's missing.
