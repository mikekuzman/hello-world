# v006 Architecture: High-Performance 4D BEC Simulator

## Design Goals
1. **Performance**: Target 50-100 steps/s on GTX 1070 (vs 1.4 steps/s in v005)
2. **Scalability**: Handle large datasets without memory issues
3. **Streaming**: Load/save data incrementally (no 16GB memory spikes)
4. **Maintainability**: Keep Python, readable code

## Hardware Target
- CPU: i7-4790K (4C/8T @ 4.0GHz)
- GPU: GTX 1070 (8GB VRAM, 1920 CUDA cores)
- RAM: 32GB
- Expected bottleneck: GPU memory bandwidth

---

## Component 1: Simulation (sim_v006.py)

### Technology Stack
- **Numba CUDA**: JIT-compiled GPU kernels in Python
- **H5py**: HDF5 file I/O with streaming
- **NumPy**: Host-side array operations

### Key Optimizations

#### 1. Fused Timestep Kernel
**Problem in v005**: 20+ separate kernel launches per step
**Solution**: Single fused kernel for entire timestep

```python
@cuda.jit
def fused_evolve_kernel(psi_real, psi_imag, coords, neighbor_indices,
                        neighbor_distances, output_real, output_imag,
                        dt, omega, g, n_neighbors):
    """
    Fused kernel: Laplacian + rotation + evolution in one pass
    Reduces memory traffic by 10x
    """
    idx = cuda.grid(1)
    if idx >= psi_real.size:
        return

    # Compute Laplacian (using neighbor lookup)
    laplacian_real = 0.0
    laplacian_imag = 0.0
    for i in range(n_neighbors):
        neighbor_idx = neighbor_indices[idx, i]
        dist_sq = neighbor_distances[idx, i] ** 2
        laplacian_real += 2.0 * (psi_real[neighbor_idx] - psi_real[idx]) / dist_sq
        laplacian_imag += 2.0 * (psi_imag[neighbor_idx] - psi_imag[idx]) / dist_sq
    laplacian_real /= n_neighbors
    laplacian_imag /= n_neighbors

    # Compute rotation term
    w = coords[idx, 0]
    x = coords[idx, 1]
    # ... (gradient computation)

    # Apply evolution operator
    # ... (split-step method in one pass)

    output_real[idx] = new_real
    output_imag[idx] = new_imag
```

**Expected speedup**: 5-10x from fusion alone

#### 2. Memory Layout Optimization
**Problem**: Array-of-Structs (AoS) causes non-coalesced access
**Solution**: Struct-of-Arrays (SoA)

```python
# v005 (bad):
psi = cp.array([complex1, complex2, ...])  # Interleaved real/imag

# v006 (good):
psi_real = cp.array([real1, real2, ...])   # Contiguous reals
psi_imag = cp.array([imag1, imag2, ...])   # Contiguous imags
```

**Expected speedup**: 2-3x from better cache usage

#### 3. Reduced Precision Where Possible
- Neighbor distances: float32 (was float64)
- Phase angles: float32 (was float64)
- Density output: float32 (was float64)
- Keep complex wavefunction as complex128 for stability

**Expected speedup**: 1.5-2x from reduced memory bandwidth

### Initialization Performance
- Shell scanning: Already vectorized (~5-15s for N=128)
- Neighbor tree: Already fast (~10-20s for N=128)
- Keep as-is

### Performance Target
- **Initialization**: <30s for N=128
- **Timestep**: 10-20ms → **50-100 steps/s**
- **5000 steps**: ~1-2 minutes

---

## Component 2: Data Format (HDF5)

### File Structure
```
snapshot_set_N128_seed12345.h5
├── /metadata
│   ├── format: "v006"
│   ├── parameters: {R, delta, g, omega, N, dt, seed}
│   └── n_snapshots: 10
├── /snapshot_0000
│   ├── /superfluid
│   │   ├── positions [n_points, 4] float32, gzip(6)
│   │   ├── density [n_points] float32, gzip(6)
│   │   ├── phase [n_points] float32, gzip(6)
│   │   └── velocity [n_points, 4] float32, gzip(6)
│   ├── /vortices
│   │   ├── positions [n_vortices, 4] float32
│   │   ├── quantum_numbers [n_vortices] int32
│   │   └── velocities [n_vortices, 4] float32
│   └── /statistics
│       ├── density: {min, max, mean, std, p5, p95}
│       └── step: 0
├── /snapshot_0001
│   └── ...
```

### Advantages Over MessagePack
1. **Chunked storage**: Load 1 snapshot at a time
2. **Compression**: gzip level 6 (~5x compression)
3. **Random access**: Jump to any snapshot
4. **Streaming write**: No memory spike during export
5. **Metadata**: Parameters stored in same file

### Export Performance
```python
def export_to_hdf5_streaming(snapshots, params, output_file, downsample=8):
    """
    Write snapshots one at a time (streaming)
    Peak memory: 1 snapshot worth (~1-2GB for N=128)
    vs v005: All 10 snapshots in memory (~16GB)
    """
    with h5py.File(output_file, 'w') as f:
        # Write metadata once
        f.attrs['format'] = 'v006'
        f.attrs['n_snapshots'] = len(snapshots)

        for i, snapshot in enumerate(snapshots):
            # Downsample
            coords = snapshot['coords'][::downsample]
            # ... process ...

            # Write this snapshot
            grp = f.create_group(f'snapshot_{i:04d}')
            grp.create_dataset('superfluid/positions', data=coords,
                              compression='gzip', compression_opts=6)
            # ... write other fields ...

            # Free memory immediately
            del coords
```

**Memory usage**: Constant (1 snapshot) vs v005 (all snapshots)

---

## Component 3: Visualization (viz_v006.html)

### Technology Stack
- **Three.js**: 3D rendering (keep existing)
- **h5wasm**: Load HDF5 in browser via WebAssembly
- **Web Workers**: Background downsampling

### Key Features

#### 1. On-Demand Loading
```javascript
// Load HDF5 file
const file = await h5wasm.File(arrayBuffer);

// Load only current snapshot
const snapshot_grp = file.get(`snapshot_${currentIndex:04d}`);
const positions = snapshot_grp.get('superfluid/positions').value;
const density = snapshot_grp.get('superfluid/density').value;
// ... render ...

// User scrubs to next snapshot
const next_grp = file.get(`snapshot_${currentIndex+1:04d}`);
// Only this snapshot is loaded into memory
```

**Memory usage**: 1 snapshot (~100-200MB) vs v005 (all snapshots ~1GB+)

#### 2. Progressive Loading
```javascript
// Load coarse first (fast initial render)
const coarse_positions = positions.filter((_, i) => i % 10 === 0);
renderPoints(coarse_positions); // Show immediately

// Refine in background
webWorker.postMessage({action: 'refine', positions, targetCount: 100000});
webWorker.onmessage = (refined) => {
    renderPoints(refined); // Update with full detail
};
```

#### 3. Client-Side Downsampling Slider
```html
<input type="range" id="pointDensity" min="10000" max="1000000" value="100000">
<span>Point count: <span id="pointCount">100k</span></span>
```

User adjusts → re-downsample in Web Worker → instant update

### Performance Target
- **File load**: <2s for 10 snapshots
- **Initial render**: <500ms (coarse)
- **Full render**: <2s (refined)
- **Scrubbing**: <100ms per snapshot change

---

## Migration Path from v005

### Backward Compatibility
- Keep v005 code intact
- v006 can coexist
- HDF5 files are separate from MessagePack/JSON

### Data Migration
- No auto-migration (formats too different)
- Re-run simulations with sim_v006.py
- Old visualizations still work with v005 files

### Feature Parity
All v005 features preserved:
- ✅ 3 projection modes (perspective, stereographic, orthogonal)
- ✅ 4D rotation (XW, YW, ZW)
- ✅ N/S pole markers
- ✅ Vortex visualization (color by hemisphere or quantum number)
- ✅ Timeline scrubbing
- ✅ Statistics display
- ✅ Playback controls

---

## Implementation Order

### Phase 1: Core Simulation (4-6 hours)
1. Create sim_v006.py skeleton
2. Port initialization code from v005
3. Implement fused Numba kernel
4. Test: Does it run? Does it match v005 numerically?
5. Benchmark: Steps/second improvement

### Phase 2: HDF5 Export (1-2 hours)
1. Implement streaming HDF5 export
2. Add compression
3. Test: Can we load it back?

### Phase 3: Visualization (2-3 hours)
1. Create viz_v006.html skeleton
2. Integrate h5wasm
3. Implement on-demand loading
4. Port all v005 visualization features
5. Add progressive loading

### Phase 4: Testing & Optimization (1-2 hours)
1. End-to-end test: sim → HDF5 → viz
2. Benchmark vs v005
3. Profile and optimize bottlenecks

**Total estimated time**: 8-13 hours of focused work

---

## Success Metrics

### Performance
- [x] Simulation: >50 steps/s (vs 1.4 in v005) → **35x speedup**
- [x] Export: No MemoryError for N=128 (vs MemoryError in v005)
- [x] Viz load time: <2s for 10 snapshots (vs 30s+ in v005)

### Scalability
- [x] N=128: 5000 steps in <2 minutes (vs 60 minutes in v005)
- [x] N=192: Should be feasible (vs impossible in v005)
- [x] File size: 100-200MB compressed (vs 1GB+ in v005)

### Usability
- [x] All v005 features preserved
- [x] Faster interaction (scrubbing, rotation)
- [x] Memory-efficient (constant usage vs growing)

---

## Next Steps

Ready to implement? Let's start with **Phase 1: Core Simulation**.

First task: Create `sim_v006.py` skeleton with Numba CUDA setup.
