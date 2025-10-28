# Build Instructions for C++ Implementation

## Quick Start (Windows + Visual Studio 2022)

### Prerequisites
1. **Visual Studio 2022** with C++ Desktop Development
2. **CUDA Toolkit 12.x**: https://developer.nvidia.com/cuda-downloads
3. **vcpkg** package manager

### Step-by-Step

#### 1. Install vcpkg
```powershell
git clone https://github.com/microsoft/vcpkg.git C:\vcpkg
cd C:\vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg integrate install
```

#### 2. Install Dependencies
```powershell
cd C:\vcpkg
.\vcpkg install glfw3:x64-windows glad:x64-windows glm:x64-windows
.\vcpkg install imgui[glfw-binding,opengl3-binding]:x64-windows
.\vcpkg install hdf5[cpp]:x64-windows
```

This will take 10-20 minutes.

#### 3. Configure CMake
```powershell
cd /path/to/hello-world
mkdir build
cd build

cmake .. -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake -G "Visual Studio 17 2022" -A x64
```

#### 4. Build
```powershell
cmake --build . --config Release
```

Or open `build/BEC4D.sln` in Visual Studio 2022 and build there.

#### 5. Run
```powershell
.\bin\Release\bec4d_sim.exe
```

## Linux Build

### Prerequisites
```bash
sudo apt install build-essential cmake
sudo apt install libglfw3-dev libglm-dev
sudo apt install nvidia-cuda-toolkit
```

### Build
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
./bin/bec4d_sim
```

## Troubleshooting

### "CUDA not found"
- Verify CUDA install: `nvcc --version`
- Add to PATH: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin`

### "Cannot find glfw3"
- Ensure vcpkg integration: `vcpkg integrate install`
- Specify toolchain file in CMake command

### "Unsupported compute capability"
Edit `CMakeLists.txt` line 13:
```cmake
set(CMAKE_CUDA_ARCHITECTURES 61)  # Change to your GPU's compute capability
```

Find your GPU's capability: https://developer.nvidia.com/cuda-gpus

## Notes

### Compilation Time
- First build: 5-15 minutes (depends on CPU)
- Subsequent builds: <1 minute (incremental)

### GPU Requirements
- Minimum: NVIDIA GPU with Compute Capability 3.5+
- Recommended: GTX 1060 or better
- Optimal: GTX 1070+ or RTX 20xx/30xx/40xx

### Memory Requirements
- **Compile time**: 4GB RAM minimum
- **Runtime**: Depends on simulation size
  - N=32: ~100 MB
  - N=64: ~500 MB
  - N=128: ~2 GB

## Development

### Adding New Source Files
1. Create `.cpp` or `.cu` file in appropriate `src/` subdirectory
2. Add to `CMakeLists.txt` in corresponding source list
3. Reconfigure CMake: `cmake ..`

### Debugging
```powershell
cmake --build . --config Debug
.\bin\Debug\bec4d_sim.exe
```

### IDE Integration
- **Visual Studio**: Open `build/BEC4D.sln`
- **CLion**: Open root `CMakeLists.txt`
- **VS Code**: Use CMake Tools extension

## Performance Tips

### Optimize CUDA Kernels
- Profile with `nvprof` or Nsight Compute
- Adjust block size in `evolution_kernels.cu` (currently 256)

### Reduce Compilation Time
- Use precompiled headers (add to CMakeLists.txt)
- Build in Release mode for production
- Use `ccache` on Linux

## Support

For build issues, check:
1. CUDA installation: `nvidia-smi` and `nvcc --version`
2. CMake version: `cmake --version` (need 3.20+)
3. Compiler version: MSVC 19.30+ or GCC 9+
