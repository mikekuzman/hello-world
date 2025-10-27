# 4D Quantum Superfluid Visualizer

Interactive visualization of 4D Bose-Einstein Condensate (BEC) simulations on a hyperspherical shell.

## Overview

This visualizer loads simulation data from `sim_v005.py` and displays the quantum superfluid dynamics, including:
- **Superfluid density** and **phase** fields
- **Vortices** (topological defects)
- **4D North & South poles** (reference points)
- **Time evolution** through snapshot playback

## Key Concept: 4D Viewing Angles

**IMPORTANT**: The 4D rotation controls in this visualizer are **NOT** the physics simulation rotation!

- **Simulation rotation** (Ω = 0.03 in w-x plane): Fixed during simulation, creates vortices via centrifugal effects
- **Viewing angles** (XW/YW/ZW sliders): Interactive controls to rotate the *frozen snapshot* in 4D space before projection

Think of it like rotating a sculpture to view it from different angles, but in 4D!

## Usage

### 1. Generate Simulation Data

Run the simulation to create a snapshot file:

```bash
python sim_v006.py
```

This creates: `snapshot_set_v006_N<grid>_seed<seed>_<init>.msgpack`

**V006 New Features:**
- Energy diagnostics and conservation tracking
- Checkpoint/resume functionality
- Progress estimation with ETA
- Global visualization downsampling parameter
- Enhanced file naming with initial condition type

### 2. Load and Visualize

1. Open `superfluid_viz.html` in a web browser
2. Click "Choose File" and select your `snapshot_set_*.json`
3. The visualization loads automatically

### 3. Controls

**Snapshot Playback:**
- Slider: Jump to specific snapshot
- Play/Pause: Animate through time evolution

**4D Viewing Angle:**
- XW, YW, ZW sliders: Rotate the 4D hypersphere to view from different 4D perspectives
- Reset View: Return to default viewing angle (0, 0, 0)

**Display:**
- Point Size: Adjust superfluid particle size
- Vortex Size: Adjust vortex core size
- Show 4D Poles: Toggle N/S pole markers

**Projection:**
- Perspective (default): Most physically intuitive
- Stereographic: Conformal projection
- Orthogonal: Simple 4D → 3D drop

## Visualization Guide

### Superfluid Points
- **Color** = Phase angle (hue cycle: red → yellow → green → cyan → blue → magenta → red)
- **Brightness** = Local density (brighter = higher density)
- Points form the condensate "background"

### Vortices
- **Black spheres** with **white wireframe borders**
- Mark topological defects where superfluid density → 0
- Vortices have quantized circulation (quantum numbers)
- Form when the hypersphere rotates fast enough (centrifugal instability)

### 4D Poles
- **N (Red)**: North pole at w = +R
- **S (Blue)**: South pole at w = -R
- Reference points fixed in 4D space
- Move in 3D projection as you change viewing angles

### Statistics Panel
- **Sound Speed** (c_s): Speed of density waves in the superfluid
- **Healing Length** (ξ): Characteristic length scale of the condensate
- **Vortex Count**: Number of quantized vortex lines detected

## Physics Background

### What is a 4D Superfluid?

This simulation models a **Bose-Einstein Condensate** (BEC) on a 4D hypersphere:
- **Quantum superfluid**: All particles occupy the same quantum state
- **Zero viscosity**: Flows without friction
- **Quantized vortices**: Rotation creates topological defects with integer circulation

### The Gross-Pitaevskii Equation

The simulation solves:

```
iℏ ∂ψ/∂t = [-ℏ²/(2m) ∇⁴² + g|ψ|² - Ω·L] ψ
```

Where:
- ψ = Complex order parameter (superfluid wavefunction)
- ∇⁴² = 4D Laplacian operator
- g = Interaction strength (repulsive)
- Ω·L = Rotation term (angular momentum operator)

### Vortex Formation

When the hypersphere rotates faster than a critical speed:
1. Centrifugal effects overcome surface tension
2. Vortices nucleate with quantized circulation
3. Vortex cores have zero density (topological defects)
4. Each vortex carries quantum number n = ±1, ±2, ...

## File Format

The visualizer expects JSON/MessagePack files in `snapshot_set_v006` format (also compatible with v005):

```json
{
  "format": "snapshot_set_v006",
  "version": "006",
  "parameters": {
    "R", "delta", "g", "omega", "N", "dt",
    "initial_condition_type",  // NEW in v006
    "viz_downsample",          // NEW in v006
    "track_energy": true       // NEW in v006
  },
  "snapshots": [
    {
      "metadata": { "step": 0, "n_points": 50000, "n_vortices": 12 },
      "superfluid": {
        "positions": [[w,x,y,z], ...],  // 4D coordinates
        "density": [ρ₁, ρ₂, ...],        // |ψ|²
        "phase": [φ₁, φ₂, ...],          // arg(ψ)
        "velocity": [[vw,vx,vy,vz], ...]
      },
      "statistics": {
        "density": { "min", "max", "mean", "std", "p5", "p95" },
        ...
      },
      "energy_data": {                   // NEW in v006
        "total_energy", "kinetic_energy",
        "potential_energy", "rotational_energy",
        "energy_per_particle", "time"
      },
      "vortices": {
        "data": [
          {
            "position": [w,x,y,z],
            "quantum_number": 1,
            "velocity": [vw,vx,vy,vz]
          },
          ...
        ]
      },
      "phonons": { "sound_speed", "healing_length", ... },
      "rotons": { "count", "positions", ... }
    }
  ]
}
```

## Tips for Exploration

1. **Start with default view**: Load data, press Play to watch vortex nucleation
2. **Rotate in 4D**: Adjust XW/YW/ZW to see vortices from different 4D angles
3. **Watch poles**: N and S poles trace paths as you rotate the viewing angle
4. **Late-time snapshots**: Vortex lattices form at equilibrium (fascinating patterns!)
5. **Try projections**: Stereographic projection shows different structure than perspective

## V006 Features ✓

- [x] **Global subsampling parameter**: Control export resolution with viz_downsample
- [x] **Energy diagnostics**: Track total, kinetic, potential, and rotational energy
- [x] **Checkpoint/resume**: Save and resume simulations mid-run
- [x] **Progress estimation**: Real-time ETA and performance metrics
- [x] **Enhanced file naming**: Include initial condition type in filenames

## Planned Features (Future)

- [ ] Velocity field streamlines
- [ ] Vortex line tracking (connect cores across time)
- [ ] Phase winding visualization
- [ ] Roton excitation markers
- [ ] Export rendered frames

## Technical Details

- **Rendering**: Three.js WebGL
- **4D Rotation**: Full matrix composition (6 degrees of freedom)
- **Projection**: Perspective/stereographic/orthogonal
- **Performance**: ~50k points at 60fps (depends on GPU)

## Troubleshooting

**"Unsupported format" error**: Make sure you're using v006 (or v005) JSON/MessagePack files from `sim_v006.py`

**Black screen**: Check browser console for errors, ensure WebGL is enabled

**Slow performance**: Try smaller simulations (reduce N parameter in sim_v006.py) or adjust viz_downsample parameter

**Vortices not visible**: They may not have nucleated yet - try later snapshots, or increase Ω in simulation

## References

- Gross-Pitaevskii Equation: https://en.wikipedia.org/wiki/Gross%E2%80%93Pitaevskii_equation
- Quantum Vortices: https://en.wikipedia.org/wiki/Quantum_vortex
- Bose-Einstein Condensate: https://en.wikipedia.org/wiki/Bose%E2%80%93Einstein_condensate
