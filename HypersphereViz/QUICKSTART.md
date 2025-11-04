# Quick Start Guide

## Getting Started in 5 Minutes

### 1. Prerequisites
- Windows 10/11
- Visual Studio 2022 or 2026 Insiders
- DirectX 12 compatible GPU

### 2. Build
```
1. Open HypersphereViz.sln in Visual Studio
2. Select Release configuration (for best performance)
3. Press Ctrl+Shift+B to build
```

### 3. Run
```
Press Ctrl+F5 to run without debugging
```

### 4. Try It Out

**Switch projections:**
- Press `1` for Perspective (default - most intuitive)
- Press `2` for Stereographic (conformal)
- Press `3` for Orthographic (simple drop)

**Adjust 4D rotations:**
- Press `Q` to speed up WX rotation (see points flow differently!)
- Press `W` to speed up WY rotation
- Press `E` to speed up WZ rotation
- Press `A/S/D` to slow down respective rotations

**Watch the magic:**
- Points change color based on their W coordinate (4th dimension)
- Red/Blue special markers show north/south poles
- The morphing effect shows 4D rotation projected to 3D

### 5. Experiment

**Want more points?**
Edit `D3D12Renderer.h` line ~30:
```cpp
m_pointCount(100000)  // Change to 500000 for half a million points!
```

**Want faster rotation?**
Edit `D3D12Renderer.h` lines ~33-35:
```cpp
m_rotationSpeedWX(1.0f)  // Increase for faster WX rotation
m_rotationSpeedWY(0.5f)
m_rotationSpeedWZ(1.2f)
```

## What You're Seeing

You're watching a **4D object** (hypersphere) rotating in **4D space**, projected down to 3D so you can see it!

- **Color gradient**: Shows position in the 4th dimension (W coordinate)
- **Morphing effect**: Shows how 4D rotation looks when projected to 3D
- **Special points**: Red/blue markers at W = ±1 (north/south poles)

The three projection methods show different ways to "flatten" 4D space to 3D:
- **Perspective**: Like a 4D camera (most natural looking)
- **Stereographic**: Projects from north pole (preserves angles)
- **Orthographic**: Simply removes W coordinate (simple but loses info)

## Tips

- Start with perspective projection (key `1`)
- Use Q/W/E to experiment with different rotation combinations
- Try all three projections to see how they differ
- Watch the FPS in the console to see performance
- For maximum FPS, use Release build and close other apps

## Need Help?

See README.md for full documentation, controls, and customization options.

---

**Have fun exploring the 4th dimension!**
