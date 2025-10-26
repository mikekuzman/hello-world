"""
sim_v005.py - 4D Bose-Einstein Condensate Simulator (VERSION 005)

V005 EXPORT FORMAT CHANGES:
==========================
- RAW density values exported (NO global normalization across snapshots)
- Comprehensive per-snapshot statistics added to JSON:
    statistics: {
        density: { min, max, mean, std, p5, p95 },  # p5/p95 = recommended viz range
        phase: { mean, std },
        velocity: { mean, std, max }
    }
- Format version: 'snapshot_set_v005' (version '005')
- Phase normalized to [0, 1] for easier color mapping
- Visualization (viz_v005.html) uses p5-p95 percentiles for local histogram stretch

BACKWARDS COMPATIBILITY:
- Pickle loading (.pkl files) unchanged - fully compatible with older sims
- JSON export format incompatible with old visualizers (use viz_v005.html)

WHY V005:
- Old format: Global normalization destroyed temporal contrast
- V005 fix: Each snapshot preserves raw density scale + statistics
- Result: Visualization can show actual density evolution over time
"""

import numpy as np
import cupy as cp
import pickle
import os
import json
import time
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
        t_start = time.time()

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

        t_elapsed = time.time() - t_start
        print(f"  Found {self.n_active:,} shell points in {t_elapsed:.2f}s")

        # Store on GPU
        self.coords_gpu = cp.asarray(self.coords)

    def _build_neighbor_tree(self):
        """Build KD-tree for fast neighbor lookup"""
        print("  Building neighbor tree...")
        t_start = time.time()

        self.tree = cKDTree(self.coords)

        # Find nearest neighbors for each point
        distances, indices = self.tree.query(self.coords, k=self.p.n_neighbors + 1)

        # Remove self (first neighbor is always self)
        self.neighbor_indices = indices[:, 1:].astype(np.int32)
        self.neighbor_distances = distances[:, 1:].astype(np.float64)

        # Transfer to GPU
        self.neighbor_indices_gpu = cp.asarray(self.neighbor_indices)
        self.neighbor_distances_gpu = cp.asarray(self.neighbor_distances)

        t_elapsed = time.time() - t_start
        avg_dist = np.mean(self.neighbor_distances)
        print(f"  Average neighbor distance: {avg_dist:.3f} ξ ({t_elapsed:.2f}s)")

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

    def run(self, n_steps: int, save_every: int = 100):
        """Run simulation for n_steps"""
        print(f"\nRunning simulation for {n_steps} steps...")
        print(f"  Expected vortex nucleation timescale: ~{1.0/self.p.omega:.1f} steps")

        snapshots = []
        t_sim_start = time.time()
        t_last_report = t_sim_start
        steps_since_report = 0

        for step in range(n_steps):
            t_step_start = time.time()
            self.evolve_step()
            cp.cuda.Stream.null.synchronize()  # Wait for GPU to finish
            t_step_elapsed = time.time() - t_step_start

            steps_since_report += 1

            if step % save_every == 0:
                t_save_start = time.time()
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

                snapshots.append({
                    'step': step,
                    'coords': coords,
                    'density': density,
                    'phase': phase,
                    'velocity': velocity,
                    'vortices': vortices,
                    'vortex_quantum_numbers': quantum_numbers,
                    'vortex_clusters': vortex_clusters,  # Full cluster information
                    'phonon_data': phonon_data,  # P₀ phonon analysis
                    'roton_data': roton_data     # R₀ roton detection
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

                t_save_elapsed = time.time() - t_save_start
                t_since_last = time.time() - t_last_report
                steps_per_sec = steps_since_report / t_since_last if t_since_last > 0 else 0

                print(f"  Step {step:5d}: <ρ>={avg_density:.3f}, c_s={c_s:.3f}, ξ={xi_heal:.2f}, "
                      f"vortices={n_vortex_lines} ({qn_summary}), rotons={n_rotons}")
                print(f"    Timing: {steps_per_sec:.1f} steps/s (snapshot save: {t_save_elapsed:.2f}s, step: {t_step_elapsed*1000:.1f}ms)")

                t_last_report = time.time()
                steps_since_report = 0

                # Check for numerical instability
                if np.isnan(avg_density) or max_density > 1e6:
                    print(f"\n  WARNING: Numerical instability detected at step {step}!")
                    print(f"  Random seed was: {self.p.random_seed}")
                    break

        t_sim_elapsed = time.time() - t_sim_start
        avg_steps_per_sec = n_steps / t_sim_elapsed if t_sim_elapsed > 0 else 0
        print(f"\n  Simulation complete: {n_steps} steps in {t_sim_elapsed:.1f}s ({avg_steps_per_sec:.1f} steps/s)")

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


def export_snapshots_to_json(snapshots, params, output_file, downsample=10, max_density_percentile=99):
    """
    Export multiple snapshots to a single JSON file for web visualization

    VERSION 005: Preserves RAW density values + comprehensive per-snapshot statistics
    - NO global normalization (visualization does local histogram stretch)
    - Full statistical summary per snapshot for proper color mapping
    - Density/phase/velocity min/max/mean/std per snapshot

    Args:
        snapshots: list of snapshot dicts
        params: SimulationParams object
        output_file: path to output .json file
        downsample: reduce points by this factor
        max_density_percentile: cap extreme densities at this percentile
    """

    print(f"Exporting snapshot set with {len(snapshots)} snapshots (v005 format)...")
    print("  v005: RAW density values + full per-snapshot statistics for local contrast")

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
        'format': 'snapshot_set_v005',
        'version': '005',
        'description': 'v005: RAW density values + comprehensive per-snapshot statistics',
        'parameters': {
            'R': float(params.R),
            'delta': float(params.delta),
            'g': float(params.g),
            'omega': float(params.omega),
            'N': int(params.N),
            'dt': float(params.dt),
            'n_neighbors': int(params.n_neighbors),
            'random_seed': int(params.random_seed) if params.random_seed else None
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


def export_snapshots_to_msgpack(snapshots, params, output_file, downsample=10, max_density_percentile=99):
    """
    Export multiple snapshots to a single MessagePack file for web visualization

    VERSION 005: MessagePack binary format - ~5-10x smaller than JSON, faster parsing
    - Same data structure as JSON export
    - Binary encoding for compact size
    - Perfect for large datasets (GB+ range)

    Args:
        snapshots: list of snapshot dicts
        params: SimulationParams object
        output_file: path to output .msgpack file
        downsample: reduce points by this factor
        max_density_percentile: cap extreme densities at this percentile
    """

    if not HAS_MSGPACK:
        print("ERROR: msgpack not installed. Install with: pip install msgpack")
        print("Falling back to JSON export...")
        json_file = output_file.replace('.msgpack', '.json')
        return export_snapshots_to_json(snapshots, params, json_file, downsample, max_density_percentile)

    # Estimate memory requirements
    n_snapshots = len(snapshots)
    avg_points = np.mean([len(snap['coords']) for snap in snapshots])
    points_after_downsample = avg_points / downsample
    bytes_per_point = 10 * 4  # (4 coords + 1 density + 1 phase + 4 velocity) × 4 bytes (float32)
    estimated_memory_gb = (n_snapshots * points_after_downsample * bytes_per_point) / 1e9

    print(f"Exporting snapshot set with {len(snapshots)} snapshots (v005 MessagePack format)...")
    print(f"  Estimated memory required: ~{estimated_memory_gb:.1f} GB")

    if estimated_memory_gb > 8:
        print(f"  WARNING: Estimated memory ({estimated_memory_gb:.1f} GB) may cause MemoryError!")
        print(f"  Current downsample={downsample}")
        suggested_downsample = int(downsample * np.sqrt(estimated_memory_gb / 4))
        print(f"  Suggested downsample={suggested_downsample} to reduce memory to ~4 GB")
        print(f"  Continuing anyway... (may fail)")

    print("  v005: RAW density values + MessagePack binary encoding")

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
        'format': 'snapshot_set_v005_msgpack',
        'version': '005',
        'description': 'v005: RAW density values + MessagePack binary format',
        'parameters': {
            'R': float(params.R),
            'delta': float(params.delta),
            'g': float(params.g),
            'omega': float(params.omega),
            'N': int(params.N),
            'dt': float(params.dt),
            'n_neighbors': int(params.n_neighbors),
            'random_seed': int(params.random_seed) if params.random_seed else None
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
    try:
        with open(output_file, 'wb') as f:
            msgpack.pack(snapshot_set, f, use_bin_type=True)

        file_size = os.path.getsize(output_file) / (1024 * 1024)
        print(f"  Done! Snapshot set file size: {file_size:.1f} MB (MessagePack)")
        print(f"  Contains {len(snapshots)} snapshots from step {snapshots[0]['step']} to {snapshots[-1]['step']}")

        return snapshot_set

    except MemoryError:
        print(f"\n  MemoryError: Dataset too large for MessagePack export!")
        print(f"  Estimated memory: ~{estimated_memory_gb:.1f} GB")
        print(f"  Falling back to JSON export with higher downsample...")

        # Suggest and use higher downsample
        new_downsample = int(downsample * np.sqrt(estimated_memory_gb / 2))
        print(f"  Auto-adjusting: downsample {downsample} → {new_downsample}")

        json_file = output_file.replace('.msgpack', '.json')
        return export_snapshots_to_json(snapshots, params, json_file, new_downsample, max_density_percentile)


# Example usage
if __name__ == "__main__":
    import sys

    # Check if loading previous state
    if len(sys.argv) > 1 and sys.argv[1] == '--load':
        if len(sys.argv) > 2:
            filename = sys.argv[2]
        else:
            filename = "initial_state.pkl"
        sim = HypersphereBEC.load_initial_state(filename)
    else:
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
            # - "uniform_noise": uniform density + small random perturbations (default, fastest)
            # - "gaussian": Gaussian blob at north pole (smooth, localized start)
            # - "imaginary_time": ground state via imaginary time evolution (slowest, most stable)
            initial_condition_type="imaginary_time",
            imag_time_steps=1000  # Only used if initial_condition_type="imaginary_time"
        )

        sim = HypersphereBEC(params)
        sim.save_initial_state()

    # Run simulation
    snapshots = sim.run(n_steps=5000, save_every=500)

    if len(snapshots) > 0:
        print(f"\nSimulation complete! Captured {len(snapshots)} snapshots.")

        # Check if stable
        final_density = snapshots[-1]['density']
        if not np.isnan(final_density.mean()) and final_density.max() < 1e6:
            print("Simulation stable! Exporting snapshots...")

            # Export full snapshot set (MessagePack format)
            set_output_file = f'snapshot_set_N{sim.p.N}_seed{sim.p.random_seed}.msgpack'
            export_snapshots_to_msgpack(snapshots, sim.p, set_output_file, downsample=8)

            print(f"\nFiles created:")
            print(f"  Snapshot set: {set_output_file}")
            print(f"\nTo reuse these initial conditions: python sim_v005.py --load initial_state_N{sim.p.N}_seed{sim.p.random_seed}.pkl")
        else:
            print("\nSimulation became unstable.")
            print(f"Random seed {sim.p.random_seed} produced unstable initial conditions.")
