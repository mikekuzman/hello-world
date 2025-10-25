# 4D Visualization Projects

Two complementary projects for visualizing 4D hypersphere geometry and physics.

## Projects

### 1. 🌊 [Quantum Superfluid Visualizer](SUPERFLUID_README.md) (NEW!)

Interactive visualization of 4D Bose-Einstein Condensate simulations on a rotating hypersphere.

**Features:**
- Load simulation snapshots from `sim_v005.py`
- Visualize superfluid density and phase fields
- Detect and render quantized vortices
- 4D viewing angles (rotate frozen snapshots in 4D)
- Time evolution playback
- Physics-accurate representation

**[→ Full Documentation](SUPERFLUID_README.md)**

### 2. 📐 Static Geometry Demo

Generates random points on a 4D spherical shell and projects them to 3D for visualization. Demonstrates mathematical projection methods with interactive 4D rotation.

---

## Geometry Demo Overview

This project generates random points on a 4D spherical shell and projects them to 3D for visualization. It demonstrates how higher-dimensional geometry can be visualized by projecting down to dimensions we can perceive.

### What is a 4D Sphere?

A 4D sphere (or 3-sphere) is defined by the equation: **x² + y² + z² + w² = r²**

It's a 3-dimensional surface embedded in 4-dimensional space, just as a regular sphere is a 2D surface in 3D space.

## Projection Methods

### 1. Perspective Projection (Recommended)
The most "physically correct" method - analogous to how a camera works. Points are projected from a viewpoint in 4D space onto a 3D hyperplane.

**Formula:** `(x, y, z, w) → (x, y, z) × d / (d - w)`

Where `d` is the distance of the viewpoint from the origin.

### 2. Stereographic Projection
Projects from a pole of the 4D sphere onto 3D space. This projection is **conformal** (preserves angles).

**Formula:** `(x, y, z, w) → (x, y, z) × r / (r - w)`

### 3. Orthogonal Projection
Simply drops the w-coordinate. Simplest but loses the most information.

**Formula:** `(x, y, z, w) → (x, y, z)`

## Usage

### 1. Generate Data

```bash
python3 sphere_4d_projection.py
```

This creates three JSON files:
- `sphere_4d_perspective.json`
- `sphere_4d_stereographic.json`
- `sphere_4d_orthogonal.json`

### 2. Visualize

Open `visualize.html` in a web browser. The visualization:
- Shows 2000 points from the 4D shell projected to 3D
- **4D North & South Poles** - Special markers showing the poles of the 4D sphere (where w = ±radius)
  - North pole (N) in red - positive w direction
  - South pole (S) in blue - negative w direction
  - Poles rotate with the 4D sphere, showing how fixed points in 4D move when projected
- **Real-time 4D rotation** - The sphere rotates in 4D space before projection (ON by default)
- Separate 3D rotation for the visualization camera view (OFF by default)
- Colors points by their w-coordinate (red = positive, blue = negative)
- Allows switching between projection methods
- Interactive controls for 4D rotation speeds in different planes
- Mouse rotation and zoom controls

### 3. Use in Your Own Code

```python
from sphere_4d_projection import generate_visualization_data

# Generate data
data = generate_visualization_data(
    n_points=5000,
    radius=1.0,
    shell_thickness=0.02,
    projection_method='perspective',
    distance=2.5
)

# data contains:
# - points_4d: Original 4D coordinates
# - points_3d: Projected 3D coordinates
# - colors: Normalized w-values for coloring
# - metadata: Generation parameters
```

## Mathematical Details

### Uniform Random Point Generation

Points are generated using the Marsaglia (1972) method:
1. Sample 4 values from a normal distribution
2. Normalize to get a point on the unit 4-sphere
3. Scale by radius ± random variation for shell thickness

This ensures **uniform distribution** on the hypersphere surface.

### Why These Projections?

**Perspective projection** is the most intuitive because it mimics how we perceive lower dimensions. When you look at a 3D object, your 2D retina receives a perspective projection of it. Similarly, a hypothetical 4D being would see our 3D world as a 3D "slice" or projection.

**Stereographic projection** is mathematically elegant and preserves angles, making it useful for certain geometric analyses.

### 4D Rotation

In 4D space, rotation is more complex than in 3D. While 3D has 3 rotation axes (x, y, z), **4D has 6 planes of rotation**:

**3D-like rotations** (don't involve w):
- **XY plane** - Like rotating around the z-axis in 3D
- **XZ plane** - Like rotating around the y-axis in 3D
- **YZ plane** - Like rotating around the x-axis in 3D

**4D-specific rotations** (involve the 4th dimension):
- **XW plane** - Rotates x and w coordinates
- **YW plane** - Rotates y and w coordinates
- **ZW plane** - Rotates z and w coordinates

The visualization uses XW, YW, and ZW rotations by default, which creates the characteristic "morphing" effect as points move through the 4th dimension. You can adjust the rotation speeds independently for each plane to create different patterns.

When the hypersphere rotates in 4D space, points with different w-coordinates move in and out of the 3D "slice" we can see, creating mesmerizing flowing patterns that would be impossible with only 3D rotation.

## Requirements

- Python 3.x
- NumPy
- Modern web browser (for visualization)

```bash
pip install numpy
```

## Customization

Edit parameters in `sphere_4d_projection.py`:

```python
n_points = 2000          # Number of points
radius = 1.0             # 4D sphere radius
shell_thickness = 0.02   # Shell thickness (2% of radius)
distance = 2.5           # Viewpoint distance (perspective only)
```

## Repository Files

### Quantum Superfluid Project
- `sim_v005.py` - 4D BEC simulation script (requires GPU with CUDA)
- `superfluid_viz.html` - Interactive quantum superfluid visualizer (loads v005 JSON)
- `SUPERFLUID_README.md` - Complete documentation for superfluid visualizer

### Geometry Demo Project
- `sphere_4d_projection.py` - Main Python module with all projection logic
- `visualize.html` - Interactive Three.js visualization with 4D rotation
- `sphere_4d_*.json` - Generated data files (created after running Python script)

## Learn More

- [Hypersphere (Wikipedia)](https://en.wikipedia.org/wiki/N-sphere)
- [Stereographic Projection](https://en.wikipedia.org/wiki/Stereographic_projection)
- [Four-dimensional space](https://en.wikipedia.org/wiki/Four-dimensional_space)