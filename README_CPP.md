# 4D BEC Simulator - C++/CUDA Implementation

High-performance 4D Bose-Einstein Condensate simulator with real-time OpenGL visualization.

## Overview

This is a complete C++ rewrite of the Python `sim_v006.py` with the same features and architecture:

- **CUDA GPU acceleration** for 50-100+ timesteps/second (GTX 1070 target)
- **4D hypersphere simulation** with Gross-Pitaevskii equation
- **Real-time visualization** with OpenGL + ImGui
- **4D to 3D projection** (perspective, stereographic, orthogonal)
- **Interactive controls** for viewing angles, rotation, and parameters

## Features

### Simulation
- Fused CUDA kernel for high-performance evolution
- Struct-of-Arrays memory layout for coalesced access
- KD-tree neighbor lookup for irregular 4D mesh
- Quantized vortex detection
- Configurable physical parameters (R, δ, g, Ω)

### Visualization
- Real-time OpenGL rendering
- 4D rotation with 6 degrees of freedom (XY, XZ, XW, YZ, YW, ZW planes)
- Multiple projection methods
- Phase-colored particles with density-based brightness
- Vortex core markers
- 4D North/South pole indicators
- Interactive camera controls

### Performance
- **Target**: 50-100 steps/s on GTX 1070
- **Memory**: <2GB for N=128 grid
- **Initialization**: <30s for N=128

## Requirements

### Hardware
- **GPU**: NVIDIA GPU with CUDA support (Compute Capability 6.1+)
  - Tested on: GTX 1070 (8GB VRAM)
  - Also works on: RTX 20xx/30xx/40xx series
- **CPU**: Multi-core recommended (i7-4790K or better)
- **RAM**: 16GB minimum, 32GB recommended

### Software

#### Windows (VS2022)
- **Visual Studio 2022** Enterprise/Professional/Community
- **CUDA Toolkit 12.x** (https://developer.nvidia.com/cuda-downloads)
- **CMake 3.20+** (included with VS2022)
- **vcpkg** (for dependencies)

#### Dependencies (via vcpkg)
```bash
vcpkg install glfw3 glad glm imgui[glfw-binding,opengl3-binding] hdf5[cpp]
```

## Build Instructions

### Windows + Visual Studio 2022

#### 1. Install CUDA Toolkit
Download and install from: https://developer.nvidia.com/cuda-downloads

#### 2. Install vcpkg
```powershell
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg integrate install
```

#### 3. Install Dependencies
```powershell
.\vcpkg install glfw3:x64-windows
.\vcpkg install glad:x64-windows
.\vcpkg install glm:x64-windows
.\vcpkg install imgui[glfw-binding,opengl3-binding]:x64-windows
.\vcpkg install hdf5[cpp]:x64-windows
```

#### 4. Configure and Build
```powershell
cd /path/to/hello-world
mkdir build
cd build

cmake .. -DCMAKE_TOOLCHAIN_FILE=C:/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake -G "Visual Studio 17 2022" -A x64

cmake --build . --config Release
```

#### 5. Run
```powershell
.\bin\Release\bec4d_sim.exe
```

### Linux (with CUDA)

#### 1. Install CUDA
Follow instructions at: https://developer.nvidia.com/cuda-downloads

#### 2. Install Dependencies
```bash
# Ubuntu/Debian
sudo apt install cmake build-essential
sudo apt install libglfw3-dev libglm-dev libhdf5-dev

# Build glad (or use vcpkg)
# Build imgui (or use vcpkg)
```

#### 3. Build
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
./bin/bec4d_sim
```

## Usage

### Running the Simulation

#### Quick Test (Small Grid)
```cpp
bec4d::SimulationParams params;
params.R = 200.0f;        // Smaller for testing
params.delta = 10.0f;
params.N = 32;            // 32^4 grid
params.dt = 0.001f;
params.n_neighbors = 6;
params.random_seed = 42;
```

#### Full Simulation (Production)
```cpp
params.R = 1000.0f;       // Full size
params.delta = 25.0f;
params.N = 128;           // 128^4 grid (large!)
params.g = 0.05f;
params.omega = 0.03f;
```

### Controls

#### Mouse
- **Left Drag**: Rotate 3D camera
- **Right Drag**: Pan camera
- **Scroll**: Zoom in/out

#### UI Panel
- **Projection Method**: Perspective / Stereographic / Orthogonal
- **4D Rotation**: XW, YW, ZW sliders (viewing angles)
- **Display**: Point size, vortex size, show poles
- **Animation**: Play/pause, speed control
- **Simulation**: Start/stop, save snapshot

### File I/O

#### Export Simulation Data (HDF5)
```cpp
app.exportToHDF5("simulation_N128_seed42.h5");
```

#### Load Previous Simulation
```cpp
app.loadFromHDF5("simulation_N128_seed42.h5");
```

## Project Structure

```
hello-world/
├── CMakeLists.txt              # Main build configuration
├── README_CPP.md               # This file
│
├── include/                    # Public headers
│   ├── simulation_params.h
│   ├── hypersphere_bec.h
│   ├── projector_4d.h
│   ├── renderer.h
│   ├── application.h
│   ├── camera.h
│   ├── shader.h
│   └── ...
│
├── src/
│   ├── main.cpp                # Entry point
│   │
│   ├── core/                   # Utilities
│   │   ├── math_utils.cpp
│   │   └── timer.cpp
│   │
│   ├── simulation/             # Physics simulation
│   │   ├── simulation_params.cpp
│   │   ├── hypersphere_bec.cpp
│   │   ├── neighbor_tree.cpp
│   │   └── cuda/
│   │       ├── evolution_kernels.cu  # CUDA kernels
│   │       └── laplacian_kernels.cu
│   │
│   ├── visualization/          # OpenGL rendering
│   │   ├── renderer.cpp
│   │   ├── shader.cpp
│   │   ├── camera.cpp
│   │   └── projector_4d.cpp
│   │
│   └── ui/                     # ImGui UI
│       ├── application.cpp
│       └── controls.cpp
│
└── build/                      # Build output (generated)
    ├── bin/
    │   └── bec4d_sim.exe
    └── ...
```

## Architecture

### Simulation Pipeline

```
1. Initialization
   ├── Find shell points (vectorized scan of 4D grid)
   ├── Build KD-tree for neighbor lookup
   ├── Initialize wavefunction (uniform + noise)
   └── Allocate GPU memory

2. Time Evolution (GPU)
   ├── Fused CUDA kernel:
   │   ├── Compute Laplacian (via neighbors)
   │   ├── Compute rotation term (angular momentum)
   │   ├── Apply Gross-Pitaevskii equation
   │   └── Euler timestep
   └── Swap buffers (double buffering)

3. Snapshot Capture
   ├── Copy density & phase from GPU
   ├── Detect vortices (low density + phase winding)
   └── Compute statistics

4. Visualization
   ├── Apply 4D rotation matrix (user-controlled)
   ├── Project to 3D (perspective/stereographic/orthogonal)
   ├── Color by phase, brightness by density
   └── Render with OpenGL
```

### CUDA Kernel Optimization

The key performance improvement comes from the **fused kernel**:

```cuda
__global__ void fused_evolve_kernel(...) {
    // Single kernel does EVERYTHING:
    // 1. Laplacian (neighbor interpolation)
    // 2. Rotation term (angular momentum)
    // 3. Interaction term (g |ψ|²)
    // 4. Time evolution (Euler step)

    // vs Python v005: 20+ separate kernel launches
}
```

**Expected speedup**: 30-70x vs Python v005

## Comparison with Python Version

| Feature | Python (v005) | Python (v006) | C++ (this) |
|---------|---------------|---------------|------------|
| Language | Python + CuPy | Python + Numba | C++ + CUDA |
| Performance (GTX 1070) | 1.4 steps/s | 50-100 steps/s | 50-100 steps/s |
| Memory layout | AoS (bad) | SoA (good) | SoA (good) |
| Kernel fusion | No | Yes | Yes |
| Visualization | Web (Three.js) | Web (Three.js) | Native (OpenGL) |
| File format | JSON/MessagePack | HDF5 | HDF5 |
| UI responsiveness | Slow | Slow | **Fast** |

## Tips & Tricks

### Performance Tuning

1. **Grid size**: Start with N=32 for testing, increase to N=128 for production
2. **Neighbors**: 6 neighbors is usually sufficient, 12+ for high accuracy
3. **Time step**: dt=0.001 is stable, can increase to 0.005 if needed
4. **GPU memory**: Monitor with `nvidia-smi`, reduce N if running out

### Vortex Formation

To see quantized vortices form:
- Use **omega >= 0.03** (faster rotation)
- Run for **5000+ steps** (takes time to nucleate)
- Look for **black spheres** with low density

### 4D Visualization

The 4D rotation controls let you "rotate the sculpture" before projecting to 3D:
- **XW rotation**: Most dramatic (moves points in/out of 3D slice)
- **YW rotation**: Similar to XW but perpendicular
- **ZW rotation**: Rotates "up-down" in 4D

Try animating XW slowly while the simulation runs!

## Troubleshooting

### CUDA Errors

**Error**: `CUDA device not found`
- Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
- Check GPU: `nvidia-smi`
- Update drivers: https://www.nvidia.com/Download/index.aspx

**Error**: `Unsupported compute capability`
- Edit `CMakeLists.txt`: Change `CMAKE_CUDA_ARCHITECTURES` to match your GPU
- Find your GPU's compute capability: https://developer.nvidia.com/cuda-gpus

### Build Errors

**Error**: `Cannot find glfw3`
- Install via vcpkg: `vcpkg install glfw3`
- Add to CMake: `-DCMAKE_TOOLCHAIN_FILE=path/to/vcpkg.cmake`

**Error**: `LINK : fatal error LNK1104: cannot open file 'cudart.lib'`
- CUDA Toolkit not properly installed
- Add CUDA to PATH: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin`

### Runtime Errors

**Error**: Black screen / no visualization
- Check OpenGL version: Need OpenGL 3.3+
- Update graphics drivers
- Try different projection method

**Error**: Low FPS (<10)
- Reduce point count (use downsampling slider)
- Reduce grid size (N=64 instead of N=128)
- Close other GPU-intensive applications

**Error**: Simulation unstable (NaN values)
- Reduce time step: `params.dt = 0.0005`
- Reduce rotation rate: `params.omega = 0.01`
- Check physical parameters (g, R, delta)

## References

### Physics
- **Gross-Pitaevskii Equation**: https://en.wikipedia.org/wiki/Gross–Pitaevskii_equation
- **Quantum Vortices**: https://en.wikipedia.org/wiki/Quantum_vortex
- **BEC**: https://en.wikipedia.org/wiki/Bose–Einstein_condensate

### Math
- **4D Geometry**: https://en.wikipedia.org/wiki/Four-dimensional_space
- **Stereographic Projection**: https://en.wikipedia.org/wiki/Stereographic_projection
- **Hypersphere**: https://en.wikipedia.org/wiki/N-sphere

### Programming
- **CUDA C++ Programming Guide**: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- **OpenGL**: https://www.khronos.org/opengl/
- **GLM Math Library**: https://github.com/g-truc/glm

## License

Same as parent project.

## Acknowledgments

Based on the Python simulation architecture from `sim_v006.py`.

## Support

For issues and questions:
1. Check Troubleshooting section above
2. Verify CUDA installation: `nvcc --version`
3. Check GPU compatibility: `nvidia-smi`
4. Report issues with system specs and error messages
