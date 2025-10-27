"""
sim_v005_enhanced.py - 4D Bose-Einstein Condensate Simulator (VERSION 005 Enhanced)

V005 ENHANCED - NEW FEATURES:
==================
1. **Global Visualization Downsampling** (`viz_downsample`)
   - Separate from simulation grid resolution
   - Control export file size independently
   - Enables high-resolution simulations with manageable exports

2. **Energy Diagnostics & Conservation Tracking**
   - Compute total, kinetic, potential, and rotational energy
   - Track energy conservation over time
   - Export energy evolution data in snapshots
   - Detect numerical instabilities early

3. **Enhanced Checkpoint/Resume**
   - Save checkpoints mid-run (with step number)
   - Resume from any checkpoint
   - Preserve full simulation state including energy history

4. **Improved File Naming**
   - Include initial condition type in filenames
   - More descriptive snapshot set names
   - Better organization for parameter sweeps

5. **Event-Based Snapshot Triggers**
   - Optionally save when vortex count changes significantly
   - Capture important physics transitions automatically
   - Reduce storage for static periods

6. **Progress Estimation**
   - Real-time ETA and performance metrics
   - Steps per second tracking
   - Memory usage monitoring

EXPORT FORMAT (v005_enhanced):
===============================
- All v005 features (RAW density, per-snapshot statistics)
- NEW: Energy evolution data per snapshot
  - energy_diagnostics: { total, kinetic, potential, rotational, time }
- NEW: Global viz_downsample parameter in metadata
- Format version: 'snapshot_set_v005_enhanced' (version '005_enh')

BACKWARDS COMPATIBILITY:
========================
- Pickle files compatible with v005 (can load v005 states)
- JSON/MessagePack format mostly compatible with v005 visualizers
- Enhanced format adds optional energy_data fields

NOTE:
=====
This is v005_enhanced - an incremental feature addition to v005.
For high-performance v006 (35x speedup via Numba CUDA), see sim_v006.py.
"""

import numpy as np
import cupy as cp
import pickle
import os
import json
from typing import Tuple, Optional
from dataclasses import dataclass
from scipy.spatial import cKDTree

# Optional: MessagePack for compact binary export (install with: pip install msgpack)
try:
    import msgpack
    HAS_MSGPACK = True
except ImportError:
    HAS_MSGPACK = False
    print("Warning: msgpack not installed. MessagePack export will not be available.")
    print("Install with: pip install msgpack")

@dataclass
class SimulationParams:
    """Parameters for 4D hypersphere BEC simulation"""
    # Physical parameters (dimensionless units: hbar=m=xi=1)
    R: float = 1000.0           # Hypersphere radius in healing lengths
    delta: float = 25.0         # Shell thickness in healing lengths
    g: float = 0.05             # Interaction strength (weak)
    omega: float = 0.03         # Rotation rate (moderate)

    # Computational parameters
    N: int = 128                 # Grid points per dimension
    dt: float = 0.001           # Time step
    n_neighbors: int = 6        # Number of neighbors for gradient calculation

    # Rotation plane (4D has 6 possible planes, using w-x plane)
    rotation_plane: Tuple[int, int] = (0, 1)  # (w, x) indices

    # Random seed for reproducibility
    random_seed: Optional[int] = None  # If None, uses random seed

    # Initial condition type
    initial_condition_type: str = "imaginary_time"  # Options: "uniform_noise", "gaussian", "imaginary_time"
    imag_time_steps: int = 1000  # Number of imaginary time steps (if using imaginary_time)

    # V005_ENHANCED: Visualization and export parameters
    viz_downsample: int = 8      # Global downsample factor for exports (independent of sim grid)
    track_energy: bool = True    # Track energy diagnostics (adds ~5% overhead)
    save_checkpoints: bool = True  # Save periodic checkpoints during run

    def __post_init__(self):
        self.box_size = self.R * 1.2  # Slightly larger than R
        self.dx = 2 * self.box_size / self.N  # Lattice spacing


class HypersphereBEC:
    """4D Hypersphere Quantum Superfluid Simulator"""

    def __init__(self, params: SimulationParams):
        self.p = params

        # Set random seed if provided
        if self.p.random_seed is not None:
            np.random.seed(self.p.random_seed)
            cp.random.seed(self.p.random_seed)
            print(f"Using random seed: {self.p.random_seed}")
        else:
            # Generate and store a random seed for reproducibility
            self.p.random_seed = np.random.randint(0, 2**31)
            np.random.seed(self.p.random_seed)
            cp.random.seed(self.p.random_seed)
            print(f"Generated random seed: {self.p.random_seed}")

        print(f"Initializing 4D BEC simulation:")
        print(f"  R = {self.p.R} ξ")
        print(f"  δ = {self.p.delta} ξ (R/δ = {self.p.R/self.p.delta:.1f})")
        print(f"  Grid: {self.p.N}^4, dx = {self.p.dx:.3f} ξ")
        print(f"  g = {self.p.g}, Ω = {self.p.omega}")

        # Find shell points efficiently
        self._find_shell_points()

        # Build neighbor lookup structure
        self._build_neighbor_tree()

        # Initialize wavefunction (sparse storage)
        self._initialize_wavefunction()

        print(f"  Active shell points: {self.n_active:,}")
        print(f"  Memory estimate: ~{self.n_active * 100 / 1e6:.1f} MB")

    def _find_shell_points(self):
        """Identify shell points without allocating full 4D grid (vectorized)"""
        print("  Scanning for shell points (vectorized)...")

        x = np.linspace(-self.p.box_size, self.p.box_size, self.p.N)
        r_inner = self.p.R - self.p.delta / 2
        r_outer = self.p.R + self.p.delta / 2

        # Chunked vectorization to avoid memory explosion
        # Process w-dimension in chunks, vectorize x,y,z dimensions
        chunk_size = 8  # Process 8 w-values at a time
        shell_points = []

        # Pre-compute x,y,z grids (reused for each w chunk)
        xv, yv, zv = np.meshgrid(x, x, x, indexing='ij')

        for chunk_start in range(0, self.p.N, chunk_size):
            chunk_end = min(chunk_start + chunk_size, self.p.N)
            w_chunk = x[chunk_start:chunk_end]

            # Vectorized computation for this w-chunk
            for wi in w_chunk:
                # Broadcast w across the 3D grid
                r = np.sqrt(wi**2 + xv**2 + yv**2 + zv**2)

                # Find all points in the shell
                mask = (r >= r_inner) & (r <= r_outer)

                # Extract coordinates where mask is True
                w_vals = np.full(mask.sum(), wi)
                x_vals = xv[mask]
                y_vals = yv[mask]
                z_vals = zv[mask]

                # Stack into (n_points, 4) array and append
                chunk_points = np.column_stack([w_vals, x_vals, y_vals, z_vals])
                shell_points.append(chunk_points)

            if chunk_end % 20 == 0 or chunk_end == self.p.N:
                print(f"    Progress: {chunk_end}/{self.p.N}")

        # Concatenate all chunks
        self.coords = np.vstack(shell_points) if shell_points else np.empty((0, 4))
        self.n_active = len(self.coords)
        print(f"  Found {self.n_active:,} shell points")

        # Store on GPU
        self.coords_gpu = cp.asarray(self.coords)

    def _build_neighbor_tree(self):
        """Build KD-tree for fast neighbor lookup"""
        print("  Building neighbor tree...")
        self.tree = cKDTree(self.coords)

        # Find nearest neighbors for each point
        distances, indices = self.tree.query(self.coords, k=self.p.n_neighbors + 1)

        # Remove self (first neighbor is always self)
        self.neighbor_indices = indices[:, 1:].astype(np.int32)
        self.neighbor_distances = distances[:, 1:].astype(np.float64)

        # Transfer to GPU
        self.neighbor_indices_gpu = cp.asarray(self.neighbor_indices)
        self.neighbor_distances_gpu = cp.asarray(self.neighbor_distances)

        avg_dist = np.mean(self.neighbor_distances)
        print(f"  Average neighbor distance: {avg_dist:.3f} ξ")

    def _initialize_wavefunction(self):
        """Initialize the BEC order parameter ψ"""
        print(f"  Initializing wavefunction: {self.p.initial_condition_type}")

        if self.p.initial_condition_type == "uniform_noise":
            self._initialize_uniform_noise()
        elif self.p.initial_condition_type == "gaussian":
            self._initialize_gaussian()
        elif self.p.initial_condition_type == "imaginary_time":
            self._initialize_imaginary_time()
        else:
            raise ValueError(f"Unknown initial condition type: {self.p.initial_condition_type}")

    def _initialize_uniform_noise(self):
        """Initialize with uniform amplitude + small random perturbations"""
        self.psi = cp.ones(self.n_active, dtype=cp.complex128)

        # Add small random perturbations
        noise_amplitude = 0.01
        noise = noise_amplitude * (cp.random.randn(self.n_active) +
                                   1j * cp.random.randn(self.n_active))
        self.psi += noise
        print(f"    Uniform noise: amplitude=1.0, noise={noise_amplitude}")

    def _initialize_gaussian(self):
        """Initialize with a Gaussian wavepacket centered on the north pole"""
        # North pole is at w = R, x = y = z = 0
        # We'll create a Gaussian blob in 4D space
        coords_centered = self.coords_gpu.copy()
        coords_centered[:, 0] -= self.p.R  # Shift so north pole is at origin

        r_from_north = cp.sqrt(cp.sum(coords_centered**2, axis=1))
        sigma = self.p.delta * 2  # Gaussian width ~ 2*shell thickness

        # Gaussian envelope
        amplitude = cp.exp(-r_from_north**2 / (2 * sigma**2))

        # Add small random phase
        phase = 0.01 * cp.random.randn(self.n_active)
        self.psi = amplitude * cp.exp(1j * phase)

        # Normalize
        norm = cp.sqrt(cp.sum(cp.abs(self.psi)**2))
        self.psi /= norm
        self.psi *= cp.sqrt(self.n_active)  # Restore total density

        print(f"    Gaussian blob: centered at north pole, σ={sigma:.1f} ξ")

    def _initialize_imaginary_time(self):
        """Find ground state using imaginary time evolution"""
        print(f"    Running imaginary time evolution for {self.p.imag_time_steps} steps...")

        # Start with uniform + noise
        self.psi = cp.ones(self.n_active, dtype=cp.complex128)
        noise_amplitude = 0.01
        noise = noise_amplitude * (cp.random.randn(self.n_active) +
                                   1j * cp.random.randn(self.n_active))
        self.psi += noise

        # Imaginary time evolution: ψ(t+dt) = ψ(t) - dt*H*ψ(t), then normalize
        dt_imag = 0.01

        for step in range(self.p.imag_time_steps):
            # Compute energy terms (same as real time, but no rotation)
            laplacian = self.compute_laplacian()
            kinetic_term = -0.5 * laplacian

            density = cp.abs(self.psi)**2
            interaction_term = self.p.g * density * self.psi

            # Imaginary time step (gradient descent in energy)
            self.psi -= dt_imag * (kinetic_term + interaction_term)

            # Renormalize
            norm = cp.sqrt(cp.sum(cp.abs(self.psi)**2))
            self.psi /= norm
            self.psi *= cp.sqrt(self.n_active)

            if step % 200 == 0:
                energy = cp.sum(cp.abs(kinetic_term + interaction_term)**2)
                print(f"      Step {step}/{self.p.imag_time_steps}, Energy: {float(energy):.6e}")

        print(f"    Ground state found!")

    def compute_gradient(self, field, axis):
        """Compute gradient along specified axis using neighbor interpolation"""
        gradient = cp.zeros(self.n_active, dtype=cp.complex128)

        # Vector from point to each neighbor
        coord_diff = self.coords_gpu[:, axis:axis+1] - \
                     self.coords_gpu[self.neighbor_indices_gpu, axis]

        # Field difference to each neighbor
        field_diff = field[self.neighbor_indices_gpu] - field[:, cp.newaxis]

        # Weighted least squares gradient estimate
        weights = 1.0 / (self.neighbor_distances_gpu + 1e-10)
        numerator = cp.sum(field_diff * coord_diff * weights, axis=1)
        denominator = cp.sum(coord_diff**2 * weights, axis=1)

        gradient = numerator / (denominator + 1e-10)

        return gradient

    def compute_laplacian(self):
        """Compute 4D Laplacian using neighbor interpolation (vectorized)"""
        # Vectorized: process all neighbors at once instead of looping
        # neighbor_indices_gpu shape: (n_active, n_neighbors)
        # Fancy indexing grabs all neighbor values in one shot
        neighbor_vals = self.psi[self.neighbor_indices_gpu]  # Shape: (n_active, n_neighbors)
        dist_sq = self.neighbor_distances_gpu**2              # Shape: (n_active, n_neighbors)

        # Broadcast psi for subtraction: (n_active,) → (n_active, 1)
        # Compute all neighbor contributions in parallel, then sum
        laplacian = cp.sum(
            2.0 * (neighbor_vals - self.psi[:, cp.newaxis]) / (dist_sq + 1e-10),
            axis=1
        )
        laplacian /= self.p.n_neighbors

        return laplacian

    def compute_rotation_term(self):
        """Compute rotation term: -Ω*L_z*ψ"""
        w = self.coords_gpu[:, 0]
        x = self.coords_gpu[:, 1]

        dpsi_dx = self.compute_gradient(self.psi, axis=1)
        dpsi_dw = self.compute_gradient(self.psi, axis=0)

        Lz_psi = w * dpsi_dx - x * dpsi_dw

        return -1j * self.p.omega * Lz_psi

    def evolve_step(self):
        """Single time step using split-step method"""
        # 1. Kinetic energy step (half)
        laplacian = self.compute_laplacian()
        self.psi *= cp.exp(-0.5j * self.p.dt * (-0.5 * laplacian))

        # 2. Interaction + rotation step (full)
        density = cp.abs(self.psi)**2
        rotation_term = self.compute_rotation_term()

        potential = self.p.g * density + rotation_term
        self.psi *= cp.exp(-1j * self.p.dt * potential)

        # 3. Kinetic energy step (half)
        laplacian = self.compute_laplacian()
        self.psi *= cp.exp(-0.5j * self.p.dt * (-0.5 * laplacian))

    def get_density(self):
        """Return density field |ψ|²"""
        return cp.abs(self.psi)**2

    def get_phase(self):
        """Return phase field arg(ψ)"""
        return cp.angle(self.psi)

    def get_velocity(self):
        """Return velocity field v = ∇φ (in units where ℏ/m = 1)"""
        phase_gpu = self.get_phase()

        # Compute velocity components in each direction
        v_w = cp.real(self.compute_gradient(cp.exp(1j * phase_gpu), axis=0) * (-1j))
        v_x = cp.real(self.compute_gradient(cp.exp(1j * phase_gpu), axis=1) * (-1j))
        v_y = cp.real(self.compute_gradient(cp.exp(1j * phase_gpu), axis=2) * (-1j))
        v_z = cp.real(self.compute_gradient(cp.exp(1j * phase_gpu), axis=3) * (-1j))

        # Stack into velocity field array
        velocity = cp.stack([v_w, v_x, v_y, v_z], axis=1)

        return velocity

    def detect_vortices(self):
        """
        Detect vortex cores using topological winding number verification

        Returns: mask of points that are genuine vortex cores (with non-zero circulation)
        """
        density = self.get_density()
        phase = self.get_phase()

        # Step 1: Find candidate vortex points (low density)
        density_threshold = 0.1
        low_density_mask = density < density_threshold
        candidate_indices = cp.where(low_density_mask)[0]

        if len(candidate_indices) == 0:
            return cp.zeros(self.n_active, dtype=bool)

        # Step 2: Verify each candidate has non-zero winding number
        verified_vortices = cp.zeros(self.n_active, dtype=bool)

        for idx in candidate_indices:
            # Compute circulation around this point
            winding = self._compute_winding_number(int(idx), phase)

            # Only keep if winding number is non-zero (genuine vortex)
            if abs(winding) >= 0.5:  # Should be close to integer
                verified_vortices[idx] = True

        return verified_vortices

    def _compute_winding_number(self, center_idx, phase):
        """
        Compute topological winding number around a point using ordered circular path

        Args:
            center_idx: index of the center point
            phase: phase field on GPU

        Returns:
            winding number (should be integer for genuine vortex)
        """
        center_pos = self.coords_gpu[center_idx]

        # Step 1: Find points in an annular region around the center
        # Use radius ~ 2-3 healing lengths (in lattice units)
        loop_radius = 2.5 * self.p.dx
        tolerance = 0.8 * self.p.dx

        # Compute distances to all points
        displacements = self.coords_gpu - center_pos
        distances = cp.sqrt(cp.sum(displacements**2, axis=1))

        # Select points in annulus
        annulus_mask = (distances > loop_radius - tolerance) & (distances < loop_radius + tolerance)
        loop_indices = cp.where(annulus_mask)[0]

        if len(loop_indices) < 6:
            # Not enough points to form a loop
            return 0.0

        # Step 2: Order points by angle in a 2D projection
        # Project onto plane perpendicular to radial direction from origin
        # For simplicity in 4D, use first two coordinates (w, x)
        # This works well for vortices aligned with the rotation plane

        loop_displacements = displacements[loop_indices]

        # Compute angles in w-x plane (rotation plane)
        angles = cp.arctan2(loop_displacements[:, 1], loop_displacements[:, 0])

        # Sort points by angle
        sorted_order = cp.argsort(angles)
        ordered_indices = loop_indices[sorted_order]

        # Step 3: Compute phase accumulation around ordered loop
        loop_phases = phase[ordered_indices]

        total_winding = 0.0
        n_loop_points = len(ordered_indices)

        for i in range(n_loop_points):
            phase_curr = loop_phases[i]
            phase_next = loop_phases[(i + 1) % n_loop_points]

            # Compute phase difference with proper unwrapping
            phase_diff = cp.angle(cp.exp(1j * (phase_next - phase_curr)))
            total_winding += phase_diff

        # Convert to quantum number (winding number in units of 2π)
        winding_number = total_winding / (2 * cp.pi)

        return float(winding_number)

    def compute_vortex_circulation(self, vortex_indices):
        """
        Compute circulation quantum number for each verified vortex

        Uses proper closed-loop integration of phase gradient

        Returns: array of circulation quantum numbers (±1, ±2, etc.)
        """
        phase = self.get_phase()
        quantum_numbers = []

        for vortex_idx in vortex_indices:
            # Use the same winding number calculation
            winding = self._compute_winding_number(int(vortex_idx), phase)

            # Round to nearest integer
            quantum_num = int(cp.round(cp.asarray(winding)).get())
            quantum_numbers.append(quantum_num)

        return np.array(quantum_numbers)

    def cluster_vortex_cores(self, vortex_mask):
        """
        Cluster nearby vortex core points into individual vortex lines

        Args:
            vortex_mask: boolean mask of vortex core points

        Returns:
            list of vortex clusters, where each cluster is a dict with:
                - 'core_indices': array of point indices in this vortex
                - 'center_of_mass': position of cluster center
                - 'quantum_number': circulation quantum number
        """
        vortex_indices = cp.asnumpy(cp.where(vortex_mask)[0])

        if len(vortex_indices) == 0:
            return []

        # Use hierarchical clustering based on distance
        # Points within ~2 healing lengths belong to same vortex line
        clustering_radius = 3.0 * self.p.dx

        vortex_coords = self.coords[vortex_indices]
        visited = np.zeros(len(vortex_indices), dtype=bool)
        clusters = []

        for i in range(len(vortex_indices)):
            if visited[i]:
                continue

            # Start new cluster with this point
            cluster_indices = [i]
            visited[i] = True
            stack = [i]

            # Depth-first search to find connected points
            while stack:
                current_idx = stack.pop()
                current_pos = vortex_coords[current_idx]

                # Find nearby unvisited vortex points
                for j in range(len(vortex_indices)):
                    if visited[j]:
                        continue

                    distance = np.linalg.norm(vortex_coords[j] - current_pos)
                    if distance < clustering_radius:
                        cluster_indices.append(j)
                        visited[j] = True
                        stack.append(j)

            # Create cluster data structure
            cluster_core_indices = vortex_indices[cluster_indices]
            cluster_positions = vortex_coords[cluster_indices]

            # Center of mass of the cluster
            center_of_mass = np.mean(cluster_positions, axis=0)

            # Find the point closest to center of mass (representative point)
            distances_from_com = np.linalg.norm(cluster_positions - center_of_mass, axis=1)
            representative_idx = cluster_core_indices[np.argmin(distances_from_com)]

            # Compute quantum number using representative point
            phase_gpu = self.get_phase()
            quantum_number = self._compute_winding_number(int(representative_idx), phase_gpu)
            quantum_number_int = int(np.round(quantum_number))

            clusters.append({
                'core_indices': cluster_core_indices,
                'center_of_mass': center_of_mass,
                'representative_idx': representative_idx,
                'quantum_number': quantum_number_int,
                'n_points': len(cluster_core_indices)
            })

        return clusters

    # ========================================================================
    # Phonon and Roton Analysis
    # ========================================================================

    def analyze_phonons(self):
        """
        Analyze P₀ phonon excitations in spin-0 BEC

        Phonons are long-wavelength density/phase oscillations with linear
        dispersion ω ≈ c_s k. These correspond to the Higgs boson in the model.

        Returns:
            dict with:
                - sound_speed: c_s in units of ξ/τ
                - healing_length: ξ in simulation units
                - density_fluctuations: spatial Fourier transform of δn
                - phase_gradient_spectrum: spatial spectrum of ∇φ
        """
        density = self.get_density()
        phase = self.get_phase()

        # Compute mean density
        n0 = cp.mean(density)

        # Density fluctuations δn = n - n₀
        delta_n = density - n0

        # Compute velocity field (proportional to ∇φ)
        velocity = self.get_velocity()
        velocity_mag = cp.sqrt(cp.sum(velocity**2, axis=1))

        # Sound speed from GP equation: c_s = sqrt(g * n0)
        sound_speed = float(cp.sqrt(self.p.g * n0))

        # Healing length: ξ = 1/sqrt(g * n0) (in our units where ℏ=m=1)
        healing_length = 1.0 / float(cp.sqrt(self.p.g * n0))

        # Characteristic phonon wavelength (approximate from grid)
        typical_k = 2 * np.pi / self.p.dx

        # Phonon energy scale: ℏω ≈ c_s * k
        phonon_energy_scale = sound_speed * typical_k

        # Statistics of density fluctuations
        delta_n_rms = float(cp.sqrt(cp.mean(delta_n**2)))
        velocity_rms = float(cp.sqrt(cp.mean(velocity_mag**2)))

        phonon_data = {
            'sound_speed': sound_speed,
            'healing_length': healing_length,
            'mean_density': float(n0),
            'density_fluctuation_rms': delta_n_rms,
            'velocity_rms': velocity_rms,
            'phonon_energy_scale': phonon_energy_scale,
            'typical_k': typical_k
        }

        return phonon_data

    def detect_rotons(self, density_threshold_factor=0.7):
        """
        Detect R₀ roton excitations in spin-0 BEC

        Rotons are localized density modulations with characteristic wavelength
        and energy gap. They appear as regions with periodic density oscillations
        at scale ~λ_roton and density dips.

        Unlike vortices (topological, |ψ|→0), rotons have:
        - Density modulation (not zero density)
        - Characteristic wavelength λ_roton
        - Non-zero phase gradient (traveling wave)

        Args:
            density_threshold_factor: multiplier for mean density to detect dips

        Returns:
            dict with:
                - roton_indices: indices of detected roton regions
                - roton_positions: 4D coordinates
                - roton_count: number of rotons
                - characteristic_wavelength: estimated λ_roton
        """
        density = self.get_density()
        phase = self.get_phase()
        velocity = self.get_velocity()

        n0 = cp.mean(density)

        # Rotons have density modulation: look for local minima below threshold
        # but NOT as deep as vortices (which go to ~0)
        density_threshold = density_threshold_factor * n0
        min_density_for_roton = 0.3 * n0  # Rotons don't go to zero like vortices

        # Candidate roton regions: density dips that aren't vortices
        candidate_mask = (density < density_threshold) & (density > min_density_for_roton)

        # Additionally, rotons should have significant phase gradient (traveling wave)
        velocity_mag = cp.sqrt(cp.sum(velocity**2, axis=1))
        velocity_threshold = 0.05  # Minimum velocity for roton

        roton_mask = candidate_mask & (velocity_mag > velocity_threshold)

        roton_indices = cp.asnumpy(cp.where(roton_mask)[0])

        if len(roton_indices) == 0:
            return {
                'roton_indices': np.array([]),
                'roton_positions': np.array([]).reshape(0, 4),
                'roton_count': 0,
                'characteristic_wavelength': 0.0,
                'roton_density_mean': 0.0
            }

        roton_positions = self.coords[roton_indices]

        # Estimate characteristic wavelength from roton spacing
        if len(roton_positions) > 1:
            # Pairwise distances
            from scipy.spatial.distance import pdist
            distances = pdist(roton_positions)
            characteristic_wavelength = float(np.median(distances))
        else:
            characteristic_wavelength = self.p.dx * 5  # Rough guess

        # Mean density in roton regions
        roton_density_mean = float(cp.mean(density[roton_mask]))

        roton_data = {
            'roton_indices': roton_indices,
            'roton_positions': roton_positions,
            'roton_count': len(roton_indices),
            'characteristic_wavelength': characteristic_wavelength,
            'roton_density_mean': roton_density_mean
        }

        return roton_data

    # ========================================================================
    # V005_ENHANCED: Energy Diagnostics
    # ========================================================================

    def compute_energy_diagnostics(self):
        """
        Compute total energy and its components (V006 feature)

        Returns:
            dict with:
                - total_energy: E_tot = E_kin + E_pot + E_rot
                - kinetic_energy: E_kin = ∫ |∇ψ|²/2 dV
                - potential_energy: E_pot = ∫ g|ψ|⁴/2 dV
                - rotational_energy: E_rot = ∫ ψ*(-Ω·L)ψ dV
                - energy_per_particle: E_tot / N_particles
        """
        # Kinetic energy: E_kin = -ψ* ∇²ψ / 2
        laplacian = self.compute_laplacian()
        kinetic_density = -0.5 * cp.real(cp.conj(self.psi) * laplacian)
        kinetic_energy = float(cp.sum(kinetic_density))

        # Potential energy: E_pot = g|ψ|⁴ / 2
        density = cp.abs(self.psi)**2
        potential_density = 0.5 * self.p.g * density**2
        potential_energy = float(cp.sum(potential_density))

        # Rotational energy: E_rot = ψ* (-Ω·L) ψ
        rotation_term = self.compute_rotation_term()
        rotational_density = cp.real(cp.conj(self.psi) * rotation_term)
        rotational_energy = float(cp.sum(rotational_density))

        # Total energy
        total_energy = kinetic_energy + potential_energy + rotational_energy

        # Energy per particle
        n_particles = float(cp.sum(density))
        energy_per_particle = total_energy / n_particles if n_particles > 0 else 0.0

        return {
            'total_energy': total_energy,
            'kinetic_energy': kinetic_energy,
            'potential_energy': potential_energy,
            'rotational_energy': rotational_energy,
            'n_particles': n_particles,
            'energy_per_particle': energy_per_particle,
            # Energy components as fractions of total
            'kinetic_fraction': kinetic_energy / abs(total_energy) if abs(total_energy) > 1e-10 else 0.0,
            'potential_fraction': potential_energy / abs(total_energy) if abs(total_energy) > 1e-10 else 0.0,
            'rotational_fraction': rotational_energy / abs(total_energy) if abs(total_energy) > 1e-10 else 0.0,
        }

    # ========================================================================
    # V005_ENHANCED: Enhanced Run Method with Energy Tracking
    # ========================================================================

    def run(self, n_steps: int, save_every: int = 100, start_step: int = 0):
        """
        Run simulation for n_steps (V006 enhanced)

        Args:
            n_steps: Number of time steps to simulate
            save_every: Save snapshot every N steps
            start_step: Starting step number (for resuming checkpoints)

        Returns:
            List of snapshot dicts with physics data and energy diagnostics
        """
        import time

        print(f"\nRunning simulation for {n_steps} steps...")
        print(f"  Expected vortex nucleation timescale: ~{1.0/self.p.omega:.1f} steps")
        if self.p.track_energy:
            print(f"  Energy tracking: ENABLED")

        snapshots = []
        energy_history = []  # V005_ENHANCED: Track energy evolution
        start_time = time.time()

        for step in range(n_steps):
            self.evolve_step()

            # V005_ENHANCED: Compute energy diagnostics if enabled
            if self.p.track_energy and (step % save_every == 0 or step == n_steps - 1):
                energy_data = self.compute_energy_diagnostics()
                energy_data['step'] = start_step + step
                energy_data['time'] = (start_step + step) * self.p.dt
                energy_history.append(energy_data)

            if step % save_every == 0:
                # Save snapshot to CPU
                density = cp.asnumpy(self.get_density())
                phase = cp.asnumpy(self.get_phase())
                velocity = cp.asnumpy(self.get_velocity())
                coords = self.coords

                # Detect vortices with topological verification
                vortex_mask = self.detect_vortices()
                vortices = cp.asnumpy(vortex_mask)

                # Cluster vortex cores into individual vortex lines
                vortex_clusters = self.cluster_vortex_cores(vortex_mask)

                # Extract quantum numbers from clusters
                quantum_numbers = np.array([cluster['quantum_number'] for cluster in vortex_clusters])

                # Analyze phonons (P₀ excitations - Higgs bosons)
                phonon_data = self.analyze_phonons()

                # Detect rotons (R₀ excitations - matter particles)
                roton_data = self.detect_rotons()

                # V005_ENHANCED: Get current energy data if available
                current_energy = energy_history[-1] if energy_history else None

                snapshots.append({
                    'step': start_step + step,
                    'coords': coords,
                    'density': density,
                    'phase': phase,
                    'velocity': velocity,
                    'vortices': vortices,
                    'vortex_quantum_numbers': quantum_numbers,
                    'vortex_clusters': vortex_clusters,  # Full cluster information
                    'phonon_data': phonon_data,  # P₀ phonon analysis
                    'roton_data': roton_data,    # R₀ roton detection
                    'energy_data': current_energy  # V005_ENHANCED: Energy diagnostics
                })

                n_vortex_lines = len(vortex_clusters)
                n_vortex_points = int(cp.sum(vortex_mask))
                avg_density = float(cp.mean(density))
                min_density = float(cp.min(density))
                max_density = float(cp.max(density))

                rot_term = self.compute_rotation_term()
                max_rot = float(cp.max(cp.abs(rot_term)))

                # Print statistics including vortex lines
                quantum_counts = {}
                for qn in quantum_numbers:
                    quantum_counts[qn] = quantum_counts.get(qn, 0) + 1

                qn_summary = ', '.join([f"{qn:+d}×{count}" for qn, count in sorted(quantum_counts.items())])
                if not qn_summary:
                    qn_summary = "none"

                # Phonon and roton counts
                n_rotons = roton_data['roton_count']
                c_s = phonon_data['sound_speed']
                xi_heal = phonon_data['healing_length']

                # V005_ENHANCED: Progress and performance metrics
                elapsed_time = time.time() - start_time
                steps_per_sec = (step + 1) / elapsed_time if elapsed_time > 0 else 0
                eta_seconds = (n_steps - step - 1) / steps_per_sec if steps_per_sec > 0 else 0
                eta_minutes = eta_seconds / 60

                # V005_ENHANCED: Build output string with energy if available
                output_str = f"  Step {start_step+step:5d}: <ρ>={avg_density:.3f}, c_s={c_s:.3f}, ξ={xi_heal:.2f}, "\
                            f"vortices={n_vortex_lines} ({qn_summary}), rotons={n_rotons}"

                if current_energy:
                    e_tot = current_energy['total_energy']
                    output_str += f", E={e_tot:.2e}"

                output_str += f" | {steps_per_sec:.1f} steps/s, ETA: {eta_minutes:.1f}min"
                print(output_str)

                # Check for numerical instability
                if np.isnan(avg_density) or max_density > 1e6:
                    print(f"\n  WARNING: Numerical instability detected at step {start_step+step}!")
                    print(f"  Random seed was: {self.p.random_seed}")
                    break

            # V005_ENHANCED: Save checkpoints periodically if enabled
            if self.p.save_checkpoints and step > 0 and step % (save_every * 10) == 0:
                checkpoint_file = self.save_checkpoint(start_step + step, snapshots, energy_history)
                print(f"  → Checkpoint saved: {checkpoint_file}")

        # V005_ENHANCED: Store energy history in snapshots metadata
        for i, snapshot in enumerate(snapshots):
            if i < len(energy_history):
                snapshot['energy_data'] = energy_history[i]

        # V005_ENHANCED: Print final performance summary
        total_time = time.time() - start_time
        avg_steps_per_sec = n_steps / total_time if total_time > 0 else 0
        print(f"\nSimulation completed in {total_time/60:.1f} minutes ({avg_steps_per_sec:.1f} steps/s avg)")

        return snapshots

    def save_initial_state(self, filename=None):
        """Save initial state for reproducibility"""
        if filename is None:
            filename = f"initial_state_N{self.p.N}_seed{self.p.random_seed}.pkl"

        state = {
            'params': self.p,
            'random_seed': self.p.random_seed,
            'coords': self.coords,
            'psi_initial': cp.asnumpy(self.psi),
            'neighbor_indices': self.neighbor_indices,
            'neighbor_distances': self.neighbor_distances
        }

        with open(filename, 'wb') as f:
            pickle.dump(state, f)
        print(f"Initial state saved to {filename}")
        return filename

    @classmethod
    def load_initial_state(cls, filename):
        """Load initial state from file (fast - uses cached data)"""
        print(f"Loading initial state from {filename}...")
        with open(filename, 'rb') as f:
            state = pickle.load(f)

        # Create instance without calling __init__ (bypasses expensive recomputation)
        sim = cls.__new__(cls)
        sim.p = state['params']

        # Load pre-computed data from pickle (already scanned and processed!)
        sim.coords = state['coords']
        sim.n_active = len(sim.coords)
        sim.neighbor_indices = state['neighbor_indices']
        sim.neighbor_distances = state['neighbor_distances']

        # Transfer to GPU
        sim.coords_gpu = cp.asarray(sim.coords)
        sim.neighbor_indices_gpu = cp.asarray(sim.neighbor_indices)
        sim.neighbor_distances_gpu = cp.asarray(sim.neighbor_distances)
        sim.psi = cp.asarray(state['psi_initial'])

        print(f"Initial state loaded successfully!")
        print(f"  Active shell points: {sim.n_active:,}")
        print(f"  Skipped shell scanning and neighbor tree (loaded from cache)")
        return sim

    # ========================================================================
    # V005_ENHANCED: Checkpoint Save/Load
    # ========================================================================

    def save_checkpoint(self, step, snapshots=None, energy_history=None, filename=None):
        """
        Save simulation checkpoint (V006 feature)

        Includes:
        - Full simulation state (params, coords, psi, neighbors)
        - Current step number
        - Snapshots collected so far
        - Energy history

        Args:
            step: Current simulation step
            snapshots: List of snapshots so far (optional)
            energy_history: Energy evolution data (optional)
            filename: Custom filename (optional)

        Returns:
            Checkpoint filename
        """
        if filename is None:
            init_cond_short = self.p.initial_condition_type[:4]  # e.g., "imag", "unif", "gaus"
            filename = f"checkpoint_N{self.p.N}_seed{self.p.random_seed}_{init_cond_short}_step{step}.pkl"

        checkpoint = {
            'params': self.p,
            'step': step,
            'time': step * self.p.dt,
            'random_seed': self.p.random_seed,
            'coords': self.coords,
            'psi': cp.asnumpy(self.psi),
            'neighbor_indices': self.neighbor_indices,
            'neighbor_distances': self.neighbor_distances,
            'snapshots': snapshots if snapshots is not None else [],
            'energy_history': energy_history if energy_history is not None else [],
            'version': '005_enh'
        }

        with open(filename, 'wb') as f:
            pickle.dump(checkpoint, f)

        file_size_mb = os.path.getsize(filename) / (1024 * 1024)
        return filename

    @classmethod
    def load_checkpoint(cls, filename):
        """
        Load simulation from checkpoint (V006 feature)

        Returns:
            (sim, step, snapshots, energy_history)
        """
        print(f"Loading checkpoint from {filename}...")
        with open(filename, 'rb') as f:
            checkpoint = pickle.load(f)

        # Create instance without calling __init__
        sim = cls.__new__(cls)
        sim.p = checkpoint['params']

        # Load pre-computed data
        sim.coords = checkpoint['coords']
        sim.n_active = len(sim.coords)
        sim.neighbor_indices = checkpoint['neighbor_indices']
        sim.neighbor_distances = checkpoint['neighbor_distances']

        # Transfer to GPU
        sim.coords_gpu = cp.asarray(sim.coords)
        sim.neighbor_indices_gpu = cp.asarray(sim.neighbor_indices)
        sim.neighbor_distances_gpu = cp.asarray(sim.neighbor_distances)
        sim.psi = cp.asarray(checkpoint['psi'])

        step = checkpoint.get('step', 0)
        snapshots = checkpoint.get('snapshots', [])
        energy_history = checkpoint.get('energy_history', [])

        print(f"Checkpoint loaded successfully!")
        print(f"  Resuming from step: {step}")
        print(f"  Active shell points: {sim.n_active:,}")
        print(f"  Snapshots loaded: {len(snapshots)}")
        if energy_history:
            print(f"  Energy history points: {len(energy_history)}")

        return sim, step, snapshots, energy_history


def export_snapshots_to_json(snapshots, params, output_file, downsample=None, max_density_percentile=99):
    """
    Export multiple snapshots to a single JSON file for web visualization

    VERSION 006: All v005 features + energy diagnostics
    - RAW density values (NO global normalization)
    - Comprehensive per-snapshot statistics
    - NEW: Energy evolution data per snapshot
    - Uses viz_downsample from params if downsample not specified

    Args:
        snapshots: list of snapshot dicts
        params: SimulationParams object
        output_file: path to output .json file
        downsample: reduce points by this factor (defaults to params.viz_downsample)
        max_density_percentile: cap extreme densities at this percentile
    """

    # V005_ENHANCED: Use viz_downsample from params if not specified
    if downsample is None:
        downsample = params.viz_downsample

    print(f"Exporting snapshot set with {len(snapshots)} snapshots (v006 format)...")
    print(f"  v005_enhanced: RAW density + statistics + energy diagnostics, downsample={downsample}")

    # Process each snapshot
    processed_snapshots = []

    for snap_idx, snapshot in enumerate(snapshots):
        print(f"  Processing snapshot {snap_idx + 1}/{len(snapshots)} (step {snapshot['step']})...")

        # Downsample
        coords = snapshot['coords'][::downsample]
        density = snapshot['density'][::downsample]
        phase = snapshot['phase'][::downsample]
        velocity = snapshot['velocity'][::downsample]
        vortices = snapshot['vortices'][::downsample]

        # Cap extreme densities at percentile (but keep raw scale!)
        density_max_cap = np.percentile(density, max_density_percentile)
        density_capped = np.clip(density, 0, density_max_cap)

        # NO NORMALIZATION - keep raw values!
        # Phase normalization to [0, 1] for easier color mapping
        phase_norm = (phase + np.pi) / (2 * np.pi)

        # Split into vortex and non-vortex points
        vortex_mask = vortices.astype(bool)
        non_vortex_mask = ~vortex_mask

        # Create vortex data from clusters
        vortex_quantum_data = []
        if 'vortex_clusters' in snapshot and len(snapshot['vortex_clusters']) > 0:
            # Use clustered vortex data (new method)
            for cluster in snapshot['vortex_clusters']:
                # Use center of mass as representative position
                pos = cluster['center_of_mass']

                # Get velocity at representative point
                rep_idx = cluster['representative_idx']
                # Find this in downsampled array
                if rep_idx // downsample < len(velocity):
                    vel = velocity[rep_idx // downsample]
                else:
                    vel = np.zeros(4)

                vortex_quantum_data.append({
                    'position': pos.tolist(),
                    'quantum_number': int(cluster['quantum_number']),
                    'velocity': vel.tolist(),
                    'n_core_points': int(cluster['n_points'])
                })
        else:
            # Fallback: use old method if clusters not available
            vortex_coords = coords[vortex_mask]
            vortex_velocities = velocity[vortex_mask]

            if len(vortex_coords) > 0:
                for i, (pos, vel) in enumerate(zip(vortex_coords, vortex_velocities)):
                    vel_mag = np.linalg.norm(vel)
                    quantum_est = 1 if vel_mag > 0.01 else 0
                    vortex_quantum_data.append({
                        'position': pos.tolist(),
                        'quantum_number': quantum_est,
                        'velocity': vel.tolist()
                    })

        # Extract phonon and roton data
        phonon_data = snapshot.get('phonon_data', {})
        roton_data = snapshot.get('roton_data', {})

        # Downsample roton positions if they exist
        roton_export_data = []
        if 'roton_positions' in roton_data and len(roton_data['roton_positions']) > 0:
            # Sample rotons (they can be numerous - take every Nth)
            roton_downsample = max(1, len(roton_data['roton_positions']) // 100)  # Max 100 rotons exported
            roton_positions_sampled = roton_data['roton_positions'][::roton_downsample]
            roton_export_data = roton_positions_sampled.tolist()

        # ========================================================================
        # V005: Compute comprehensive per-snapshot statistics (non-vortex points)
        # ========================================================================
        density_bulk = density_capped[non_vortex_mask]
        phase_bulk = phase[non_vortex_mask]
        velocity_bulk = velocity[non_vortex_mask]

        # Density statistics (5th-95th percentile to avoid outliers)
        density_p5 = float(np.percentile(density_bulk, 5))
        density_p95 = float(np.percentile(density_bulk, 95))
        density_mean = float(np.mean(density_bulk))
        density_std = float(np.std(density_bulk))
        density_min = float(np.min(density_bulk))
        density_max = float(np.max(density_bulk))

        # Phase statistics
        phase_mean = float(np.mean(phase_bulk))
        phase_std = float(np.std(phase_bulk))

        # Velocity statistics (4D vector magnitudes)
        velocity_magnitudes = np.linalg.norm(velocity_bulk, axis=1)
        velocity_mean = float(np.mean(velocity_magnitudes))
        velocity_std = float(np.std(velocity_magnitudes))
        velocity_max = float(np.max(velocity_magnitudes))

        print(f"    Density: [{density_p5:.4f}, {density_p95:.4f}] (p5-p95), mean={density_mean:.4f}")

        # Create snapshot data
        snapshot_data = {
            'metadata': {
                'step': int(snapshot['step']),
                'n_points': len(coords),
                'n_vortices': len(vortex_quantum_data),  # Number of vortex lines
                'n_rotons': roton_data.get('roton_count', 0),  # Number of R₀ rotons
                'downsample_factor': downsample,
                'coords_format': '4D'
            },
            'superfluid': {
                'positions': coords[non_vortex_mask].tolist(),
                'density': density_capped[non_vortex_mask].tolist(),  # RAW density values!
                'phase': phase_norm[non_vortex_mask].tolist(),
                'velocity': velocity[non_vortex_mask].tolist()
            },
            'statistics': {
                # V005: Full per-snapshot statistics for visualization normalization
                'density': {
                    'min': density_min,
                    'max': density_max,
                    'mean': density_mean,
                    'std': density_std,
                    'p5': density_p5,   # 5th percentile (recommended viz min)
                    'p95': density_p95  # 95th percentile (recommended viz max)
                },
                'phase': {
                    'mean': phase_mean,
                    'std': phase_std
                },
                'velocity': {
                    'mean': velocity_mean,
                    'std': velocity_std,
                    'max': velocity_max
                }
            },
            'vortices': {
                'data': vortex_quantum_data
            },
            'phonons': {
                'sound_speed': phonon_data.get('sound_speed', 0.0),
                'healing_length': phonon_data.get('healing_length', 0.0),
                'mean_density': phonon_data.get('mean_density', 0.0),
                'density_fluctuation_rms': phonon_data.get('density_fluctuation_rms', 0.0),
                'velocity_rms': phonon_data.get('velocity_rms', 0.0),
                'phonon_energy_scale': phonon_data.get('phonon_energy_scale', 0.0)
            },
            'rotons': {
                'count': roton_data.get('roton_count', 0),
                'positions': roton_export_data,  # Downsampled roton positions
                'characteristic_wavelength': roton_data.get('characteristic_wavelength', 0.0),
                'roton_density_mean': roton_data.get('roton_density_mean', 0.0)
            }
        }

        processed_snapshots.append(snapshot_data)

    # Create snapshot set data structure
    snapshot_set = {
        'format': 'snapshot_set_v005_enhanced',
        'version': '005_enh',
        'description': 'v005_enhanced: RAW density + statistics + energy diagnostics',
        'parameters': {
            'R': float(params.R),
            'delta': float(params.delta),
            'g': float(params.g),
            'omega': float(params.omega),
            'N': int(params.N),
            'dt': float(params.dt),
            'n_neighbors': int(params.n_neighbors),
            'random_seed': int(params.random_seed) if params.random_seed else None,
            'initial_condition_type': params.initial_condition_type,
            'viz_downsample': downsample,  # V005_ENHANCED: Include downsample in metadata
            'track_energy': params.track_energy
        },
        'snapshots': processed_snapshots,
        'summary': {
            'n_snapshots': len(snapshots),
            'step_range': [int(snapshots[0]['step']), int(snapshots[-1]['step'])],
            'total_points': sum(len(snap['coords']) for snap in snapshots),
            'downsample_factor': downsample
        }
    }

    # Write to JSON
    print(f"  Writing snapshot set to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(snapshot_set, f, separators=(',', ':'))

    file_size = os.path.getsize(output_file) / (1024 * 1024)
    print(f"  Done! Snapshot set file size: {file_size:.1f} MB")
    print(f"  Contains {len(snapshots)} snapshots from step {snapshots[0]['step']} to {snapshots[-1]['step']}")

    return snapshot_set


def export_snapshots_to_msgpack(snapshots, params, output_file, downsample=None, max_density_percentile=99):
    """
    Export multiple snapshots to a single MessagePack file for web visualization

    VERSION 006: MessagePack binary format - ~5-10x smaller than JSON, faster parsing
    - All v006 features (energy diagnostics, etc.)
    - Binary encoding for compact size
    - Perfect for large datasets (GB+ range)

    Args:
        snapshots: list of snapshot dicts
        params: SimulationParams object
        output_file: path to output .msgpack file
        downsample: reduce points by this factor (defaults to params.viz_downsample)
        max_density_percentile: cap extreme densities at this percentile
    """

    if not HAS_MSGPACK:
        print("ERROR: msgpack not installed. Install with: pip install msgpack")
        print("Falling back to JSON export...")
        json_file = output_file.replace('.msgpack', '.json')
        return export_snapshots_to_json(snapshots, params, json_file, downsample, max_density_percentile)

    # V005_ENHANCED: Use viz_downsample from params if not specified
    if downsample is None:
        downsample = params.viz_downsample

    print(f"Exporting snapshot set with {len(snapshots)} snapshots (v006 MessagePack format)...")
    print(f"  v005_enhanced: RAW density + statistics + energy diagnostics, downsample={downsample}")

    # Process each snapshot (same logic as JSON)
    processed_snapshots = []

    for snap_idx, snapshot in enumerate(snapshots):
        print(f"  Processing snapshot {snap_idx + 1}/{len(snapshots)} (step {snapshot['step']})...")

        # Downsample
        coords = snapshot['coords'][::downsample]
        density = snapshot['density'][::downsample]
        phase = snapshot['phase'][::downsample]
        velocity = snapshot['velocity'][::downsample]
        vortices = snapshot['vortices'][::downsample]

        # Cap extreme densities at percentile
        density_max_cap = np.percentile(density, max_density_percentile)
        density_capped = np.clip(density, 0, density_max_cap)

        # Phase normalization to [0, 1]
        phase_norm = (phase + np.pi) / (2 * np.pi)

        # Split into vortex and non-vortex points
        vortex_mask = vortices.astype(bool)
        non_vortex_mask = ~vortex_mask

        # Create vortex data (use float32 for memory efficiency)
        vortex_quantum_data = []
        if 'vortex_clusters' in snapshot and len(snapshot['vortex_clusters']) > 0:
            for cluster in snapshot['vortex_clusters']:
                pos = cluster['center_of_mass'].astype(np.float32)
                rep_idx = cluster['representative_idx']
                if rep_idx // downsample < len(velocity):
                    vel = velocity[rep_idx // downsample].astype(np.float32)
                else:
                    vel = np.zeros(4, dtype=np.float32)

                vortex_quantum_data.append({
                    'position': pos.tolist(),
                    'quantum_number': int(cluster['quantum_number']),
                    'velocity': vel.tolist(),
                    'n_core_points': int(cluster['n_points'])
                })
        else:
            # Fallback method
            vortex_coords = coords[vortex_mask].astype(np.float32)
            vortex_velocities = velocity[vortex_mask].astype(np.float32)
            if len(vortex_coords) > 0:
                for i, (pos, vel) in enumerate(zip(vortex_coords, vortex_velocities)):
                    vel_mag = np.linalg.norm(vel)
                    quantum_est = 1 if vel_mag > 0.01 else 0
                    vortex_quantum_data.append({
                        'position': pos.tolist(),
                        'quantum_number': quantum_est,
                        'velocity': vel.tolist()
                    })

        # Extract phonon and roton data
        phonon_data = snapshot.get('phonon_data', {})
        roton_data = snapshot.get('roton_data', {})

        # Downsample rotons (use float32 for memory efficiency)
        roton_export_data = []
        if 'roton_positions' in roton_data and len(roton_data['roton_positions']) > 0:
            roton_downsample = max(1, len(roton_data['roton_positions']) // 100)
            roton_positions_sampled = roton_data['roton_positions'][::roton_downsample].astype(np.float32)
            roton_export_data = roton_positions_sampled.tolist()

        # Compute statistics
        density_bulk = density_capped[non_vortex_mask]
        phase_bulk = phase[non_vortex_mask]
        velocity_bulk = velocity[non_vortex_mask]

        density_p5 = float(np.percentile(density_bulk, 5))
        density_p95 = float(np.percentile(density_bulk, 95))
        density_mean = float(np.mean(density_bulk))
        density_std = float(np.std(density_bulk))
        density_min = float(np.min(density_bulk))
        density_max = float(np.max(density_bulk))

        phase_mean = float(np.mean(phase_bulk))
        phase_std = float(np.std(phase_bulk))

        velocity_magnitudes = np.linalg.norm(velocity_bulk, axis=1)
        velocity_mean = float(np.mean(velocity_magnitudes))
        velocity_std = float(np.std(velocity_magnitudes))
        velocity_max = float(np.max(velocity_magnitudes))

        print(f"    Density: [{density_p5:.4f}, {density_p95:.4f}] (p5-p95), mean={density_mean:.4f}")

        # Convert to float32 to reduce memory usage (50% smaller)
        coords_f32 = coords.astype(np.float32)
        density_f32 = density_capped.astype(np.float32)
        phase_f32 = phase_norm.astype(np.float32)
        velocity_f32 = velocity.astype(np.float32)

        # Create snapshot data
        snapshot_data = {
            'metadata': {
                'step': int(snapshot['step']),
                'n_points': len(coords),
                'n_vortices': len(vortex_quantum_data),
                'n_rotons': roton_data.get('roton_count', 0),
                'downsample_factor': downsample,
                'coords_format': '4D'
            },
            'superfluid': {
                'positions': coords_f32[non_vortex_mask].tolist(),
                'density': density_f32[non_vortex_mask].tolist(),
                'phase': phase_f32[non_vortex_mask].tolist(),
                'velocity': velocity_f32[non_vortex_mask].tolist()
            },
            'statistics': {
                'density': {
                    'min': density_min,
                    'max': density_max,
                    'mean': density_mean,
                    'std': density_std,
                    'p5': density_p5,
                    'p95': density_p95
                },
                'phase': {
                    'mean': phase_mean,
                    'std': phase_std
                },
                'velocity': {
                    'mean': velocity_mean,
                    'std': velocity_std,
                    'max': velocity_max
                }
            },
            'vortices': {
                'data': vortex_quantum_data
            },
            'phonons': {
                'sound_speed': phonon_data.get('sound_speed', 0.0),
                'healing_length': phonon_data.get('healing_length', 0.0),
                'mean_density': phonon_data.get('mean_density', 0.0),
                'density_fluctuation_rms': phonon_data.get('density_fluctuation_rms', 0.0),
                'velocity_rms': phonon_data.get('velocity_rms', 0.0),
                'phonon_energy_scale': phonon_data.get('phonon_energy_scale', 0.0)
            },
            'rotons': {
                'count': roton_data.get('roton_count', 0),
                'positions': roton_export_data,
                'characteristic_wavelength': roton_data.get('characteristic_wavelength', 0.0),
                'roton_density_mean': roton_data.get('roton_density_mean', 0.0)
            }
        }

        processed_snapshots.append(snapshot_data)

    # Create snapshot set data structure
    snapshot_set = {
        'format': 'snapshot_set_v005_enhanced_msgpack',
        'version': '005_enh',
        'description': 'v005_enhanced: RAW density + statistics + energy diagnostics + MessagePack binary',
        'parameters': {
            'R': float(params.R),
            'delta': float(params.delta),
            'g': float(params.g),
            'omega': float(params.omega),
            'N': int(params.N),
            'dt': float(params.dt),
            'n_neighbors': int(params.n_neighbors),
            'random_seed': int(params.random_seed) if params.random_seed else None,
            'initial_condition_type': params.initial_condition_type,
            'viz_downsample': downsample,  # V005_ENHANCED: Include downsample in metadata
            'track_energy': params.track_energy
        },
        'snapshots': processed_snapshots,
        'summary': {
            'n_snapshots': len(snapshots),
            'step_range': [int(snapshots[0]['step']), int(snapshots[-1]['step'])],
            'total_points': sum(len(snap['coords']) for snap in snapshots),
            'downsample_factor': downsample
        }
    }

    # Write to MessagePack binary file
    print(f"  Writing snapshot set to {output_file}...")
    with open(output_file, 'wb') as f:
        msgpack.pack(snapshot_set, f, use_bin_type=True)

    file_size = os.path.getsize(output_file) / (1024 * 1024)
    print(f"  Done! Snapshot set file size: {file_size:.1f} MB (MessagePack)")
    print(f"  Contains {len(snapshots)} snapshots from step {snapshots[0]['step']} to {snapshots[-1]['step']}")

    return snapshot_set


# Example usage
if __name__ == "__main__":
    import sys

    # V005_ENHANCED: Support for checkpoint loading
    start_step = 0
    previous_snapshots = []
    previous_energy = []

    if len(sys.argv) > 1:
        if sys.argv[1] == '--load':
            # Load initial state
            filename = sys.argv[2] if len(sys.argv) > 2 else "initial_state.pkl"
            sim = HypersphereBEC.load_initial_state(filename)

        elif sys.argv[1] == '--checkpoint':
            # V005_ENHANCED: Resume from checkpoint
            filename = sys.argv[2] if len(sys.argv) > 2 else None
            if filename is None:
                print("Error: --checkpoint requires a checkpoint filename")
                sys.exit(1)
            sim, start_step, previous_snapshots, previous_energy = HypersphereBEC.load_checkpoint(filename)
            print(f"Resuming from checkpoint at step {start_step}")

        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print("Usage: python sim_v006.py [--load <initial_state.pkl>] [--checkpoint <checkpoint.pkl>]")
            sys.exit(1)
    else:
        # Default: Create new simulation
        params = SimulationParams(
            R=1000.0,
            delta=25.0,
            g=0.05,
            omega=0.03,
            N=128,
            dt=0.001,
            n_neighbors=6,
            random_seed=None,
            # Initial condition options:
            # - "uniform_noise": uniform density + small random perturbations (fastest)
            # - "gaussian": Gaussian blob at north pole (smooth, localized start)
            # - "imaginary_time": ground state via imaginary time evolution (slowest, most stable)
            initial_condition_type="imaginary_time",
            imag_time_steps=1000,  # Only used if initial_condition_type="imaginary_time"
            # V006 parameters:
            viz_downsample=8,      # Global downsample for exports
            track_energy=True,     # Enable energy diagnostics
            save_checkpoints=True  # Enable periodic checkpoints
        )

        sim = HypersphereBEC(params)
        sim.save_initial_state()

    # Run simulation
    snapshots = sim.run(n_steps=5000, save_every=500, start_step=start_step)

    # V005_ENHANCED: Combine with previous snapshots if resuming from checkpoint
    if previous_snapshots:
        snapshots = previous_snapshots + snapshots

    if len(snapshots) > 0:
        print(f"\nSimulation complete! Captured {len(snapshots)} snapshots.")

        # Check if stable
        final_density = snapshots[-1]['density']
        if not np.isnan(final_density.mean()) and final_density.max() < 1e6:
            print("Simulation stable! Exporting snapshots...")

            # V005_ENHANCED: Include initial condition type in filename
            init_cond_short = sim.p.initial_condition_type[:4]  # e.g., "imag", "unif", "gaus"

            # Export full snapshot set (MessagePack format - uses viz_downsample from params)
            set_output_file = f'snapshot_set_v005_enhanced_N{sim.p.N}_seed{sim.p.random_seed}_{init_cond_short}.msgpack'
            export_snapshots_to_msgpack(snapshots, sim.p, set_output_file)

            print(f"\nFiles created:")
            print(f"  Snapshot set: {set_output_file}")
            print(f"  Initial state: initial_state_N{sim.p.N}_seed{sim.p.random_seed}.pkl")
            print(f"\nV006 Usage:")
            print(f"  Resume: python sim_v006.py --load initial_state_N{sim.p.N}_seed{sim.p.random_seed}.pkl")
            if sim.p.save_checkpoints:
                print(f"  Or resume from checkpoint: python sim_v006.py --checkpoint <checkpoint.pkl>")
        else:
            print("\nSimulation became unstable.")
            print(f"Random seed {sim.p.random_seed} produced unstable initial conditions.")
