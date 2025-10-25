#!/usr/bin/env python3
"""
4D to 3D Projection of a 4D Spherical Shell

This module generates random points on a 4D hypersphere (3-sphere) and projects
them to 3D space using various projection methods for visualization.

Mathematical Background:
- A 4D sphere (3-sphere) is defined as: x² + y² + z² + w² = r²
- A spherical shell has thickness δr, so r - δr/2 ≤ radius ≤ r + δr/2
"""

import numpy as np
import json
from typing import Tuple, List, Dict


class HypersphereProjector:
    """Generate and project points from 4D hypersphere to 3D space."""

    def __init__(self, radius: float = 1.0, shell_thickness: float = 0.01):
        """
        Initialize the projector.

        Args:
            radius: Radius of the 4D hypersphere
            shell_thickness: Thickness of the spherical shell in 4D space
        """
        self.radius = radius
        self.shell_thickness = shell_thickness

    @staticmethod
    def rotation_matrix_4d(angle_xy: float = 0, angle_xz: float = 0, angle_xw: float = 0,
                          angle_yz: float = 0, angle_yw: float = 0, angle_zw: float = 0) -> np.ndarray:
        """
        Create a 4D rotation matrix for rotation in multiple planes.

        In 4D, there are 6 independent planes of rotation (vs 3 axes in 3D):
        - XY plane (like standard 3D z-axis rotation)
        - XZ plane (like standard 3D y-axis rotation)
        - XW plane (rotation involving 4th dimension)
        - YZ plane (like standard 3D x-axis rotation)
        - YW plane (rotation involving 4th dimension)
        - ZW plane (rotation involving 4th dimension)

        Args:
            angle_xy: Rotation angle in XY plane (radians)
            angle_xz: Rotation angle in XZ plane (radians)
            angle_xw: Rotation angle in XW plane (radians)
            angle_yz: Rotation angle in YZ plane (radians)
            angle_yw: Rotation angle in YW plane (radians)
            angle_zw: Rotation angle in ZW plane (radians)

        Returns:
            4x4 rotation matrix
        """
        # XY plane rotation (affects x and y, z and w unchanged)
        if angle_xy != 0:
            c, s = np.cos(angle_xy), np.sin(angle_xy)
            R_xy = np.array([
                [c, -s, 0, 0],
                [s,  c, 0, 0],
                [0,  0, 1, 0],
                [0,  0, 0, 1]
            ])
        else:
            R_xy = np.eye(4)

        # XZ plane rotation
        if angle_xz != 0:
            c, s = np.cos(angle_xz), np.sin(angle_xz)
            R_xz = np.array([
                [c,  0, -s, 0],
                [0,  1,  0, 0],
                [s,  0,  c, 0],
                [0,  0,  0, 1]
            ])
        else:
            R_xz = np.eye(4)

        # XW plane rotation (involves 4th dimension!)
        if angle_xw != 0:
            c, s = np.cos(angle_xw), np.sin(angle_xw)
            R_xw = np.array([
                [c,  0,  0, -s],
                [0,  1,  0,  0],
                [0,  0,  1,  0],
                [s,  0,  0,  c]
            ])
        else:
            R_xw = np.eye(4)

        # YZ plane rotation
        if angle_yz != 0:
            c, s = np.cos(angle_yz), np.sin(angle_yz)
            R_yz = np.array([
                [1,  0,  0, 0],
                [0,  c, -s, 0],
                [0,  s,  c, 0],
                [0,  0,  0, 1]
            ])
        else:
            R_yz = np.eye(4)

        # YW plane rotation (involves 4th dimension!)
        if angle_yw != 0:
            c, s = np.cos(angle_yw), np.sin(angle_yw)
            R_yw = np.array([
                [1,  0,  0,  0],
                [0,  c,  0, -s],
                [0,  0,  1,  0],
                [0,  s,  0,  c]
            ])
        else:
            R_yw = np.eye(4)

        # ZW plane rotation (involves 4th dimension!)
        if angle_zw != 0:
            c, s = np.cos(angle_zw), np.sin(angle_zw)
            R_zw = np.array([
                [1,  0,  0,  0],
                [0,  1,  0,  0],
                [0,  0,  c, -s],
                [0,  0,  s,  c]
            ])
        else:
            R_zw = np.eye(4)

        # Compose all rotations
        R = R_xy @ R_xz @ R_xw @ R_yz @ R_yw @ R_zw
        return R

    def rotate_points_4d(self, points_4d: np.ndarray,
                        angle_xy: float = 0, angle_xz: float = 0, angle_xw: float = 0,
                        angle_yz: float = 0, angle_yw: float = 0, angle_zw: float = 0) -> np.ndarray:
        """
        Rotate 4D points using the specified rotation angles.

        Args:
            points_4d: Array of shape (n, 4) with 4D coordinates
            angle_xy, angle_xz, angle_xw, angle_yz, angle_yw, angle_zw: Rotation angles

        Returns:
            Rotated 4D points
        """
        R = self.rotation_matrix_4d(angle_xy, angle_xz, angle_xw, angle_yz, angle_yw, angle_zw)
        return (R @ points_4d.T).T

    def generate_random_points_on_4d_sphere(self, n_points: int) -> np.ndarray:
        """
        Generate uniformly distributed random points on a 4D hypersphere shell.

        Uses the Marsaglia (1972) method: generate points from 4D Gaussian
        distribution and normalize. For a shell, we add small random radial
        variation.

        Args:
            n_points: Number of points to generate

        Returns:
            Array of shape (n_points, 4) containing [x, y, z, w] coordinates
        """
        # Generate random points from 4D Gaussian distribution
        points = np.random.randn(n_points, 4)

        # Normalize to unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms

        # Add random radial variation for shell thickness
        # Random radius between (r - thickness/2) and (r + thickness/2)
        radii = self.radius + (np.random.rand(n_points, 1) - 0.5) * self.shell_thickness
        points = points * radii

        return points

    def project_perspective(self, points_4d: np.ndarray,
                          distance: float = 2.0) -> np.ndarray:
        """
        Perspective projection from 4D to 3D.

        Similar to how a 3D→2D camera works. We place a "camera" at distance
        along the w-axis and project onto the w=0 hyperplane.

        Formula: (x, y, z, w) → (x, y, z) * distance / (distance - w)

        Args:
            points_4d: Array of shape (n, 4) with 4D coordinates
            distance: Distance of viewpoint from origin along w-axis

        Returns:
            Array of shape (n, 3) with projected 3D coordinates
        """
        # Perspective division factor
        # We project from viewpoint at (0, 0, 0, distance) onto w=0 hyperplane
        scale = distance / (distance - points_4d[:, 3:4])

        # Apply perspective projection to x, y, z coordinates
        points_3d = points_4d[:, :3] * scale

        return points_3d

    def project_stereographic(self, points_4d: np.ndarray,
                             from_pole: str = 'north') -> np.ndarray:
        """
        Stereographic projection from 4D sphere to 3D space.

        Projects from the north pole (w = radius) or south pole (w = -radius)
        onto the w=0 hyperplane. This projection is conformal (preserves angles).

        Formula from north pole: (x, y, z, w) → (x, y, z) * r / (r - w)

        Args:
            points_4d: Array of shape (n, 4) with 4D coordinates
            from_pole: 'north' or 'south' pole to project from

        Returns:
            Array of shape (n, 3) with projected 3D coordinates
        """
        if from_pole == 'north':
            # Project from north pole (0, 0, 0, r)
            scale = self.radius / (self.radius - points_4d[:, 3:4])
        else:
            # Project from south pole (0, 0, 0, -r)
            scale = self.radius / (self.radius + points_4d[:, 3:4])

        points_3d = points_4d[:, :3] * scale

        return points_3d

    def project_orthogonal(self, points_4d: np.ndarray) -> np.ndarray:
        """
        Orthogonal projection - simply drop the w coordinate.

        Args:
            points_4d: Array of shape (n, 4) with 4D coordinates

        Returns:
            Array of shape (n, 3) with projected 3D coordinates
        """
        return points_4d[:, :3]


def generate_visualization_data(n_points: int = 1000,
                               radius: float = 1.0,
                               shell_thickness: float = 0.01,
                               projection_method: str = 'perspective',
                               distance: float = 2.0) -> Dict:
    """
    Generate complete visualization data for a 4D spherical shell.

    Args:
        n_points: Number of points to generate
        radius: Radius of the 4D sphere
        shell_thickness: Thickness of the shell
        projection_method: 'perspective', 'stereographic', or 'orthogonal'
        distance: Viewpoint distance for perspective projection

    Returns:
        Dictionary containing points in both 4D and 3D, ready for visualization
    """
    projector = HypersphereProjector(radius, shell_thickness)

    # Generate 4D points
    points_4d = projector.generate_random_points_on_4d_sphere(n_points)

    # Project to 3D
    if projection_method == 'perspective':
        points_3d = projector.project_perspective(points_4d, distance)
    elif projection_method == 'stereographic':
        points_3d = projector.project_stereographic(points_4d)
    elif projection_method == 'orthogonal':
        points_3d = projector.project_orthogonal(points_4d)
    else:
        raise ValueError(f"Unknown projection method: {projection_method}")

    # Color points by w-coordinate for better visualization
    w_values = points_4d[:, 3]
    w_normalized = (w_values - w_values.min()) / (w_values.max() - w_values.min())

    return {
        'points_4d': points_4d.tolist(),
        'points_3d': points_3d.tolist(),
        'colors': w_normalized.tolist(),  # Can map to color gradient
        'metadata': {
            'n_points': n_points,
            'radius': radius,
            'shell_thickness': shell_thickness,
            'projection_method': projection_method,
            'distance': distance if projection_method == 'perspective' else None
        }
    }


def save_for_threejs(data: Dict, filename: str = 'sphere_4d_data.json'):
    """
    Save visualization data in a format ready for Three.js.

    Args:
        data: Dictionary from generate_visualization_data()
        filename: Output JSON filename
    """
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(data['points_3d'])} points to {filename}")


def demo():
    """Demonstration of the 4D sphere projection."""
    print("=" * 60)
    print("4D Spherical Shell to 3D Projection Demo")
    print("=" * 60)

    # Parameters
    n_points = 2000
    radius = 1.0
    shell_thickness = 0.02  # 2% of radius

    print(f"\nGenerating {n_points} points on 4D sphere...")
    print(f"  Radius: {radius}")
    print(f"  Shell thickness: {shell_thickness}")

    # Generate with different projection methods
    methods = ['perspective', 'stereographic', 'orthogonal']

    for method in methods:
        print(f"\n{method.upper()} Projection:")

        data = generate_visualization_data(
            n_points=n_points,
            radius=radius,
            shell_thickness=shell_thickness,
            projection_method=method,
            distance=2.5  # Only used for perspective
        )

        # Save to file
        filename = f'sphere_4d_{method}.json'
        save_for_threejs(data, filename)

        # Show some statistics
        points_3d = np.array(data['points_3d'])
        print(f"  3D coordinate ranges:")
        print(f"    X: [{points_3d[:, 0].min():.3f}, {points_3d[:, 0].max():.3f}]")
        print(f"    Y: [{points_3d[:, 1].min():.3f}, {points_3d[:, 1].max():.3f}]")
        print(f"    Z: [{points_3d[:, 2].min():.3f}, {points_3d[:, 2].max():.3f}]")

    print("\n" + "=" * 60)
    print("Visualization files created!")
    print("You can load these JSON files in Three.js or any 3D viewer.")
    print("=" * 60)


if __name__ == '__main__':
    demo()
