#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace bec4d {

/**
 * Compute Laplacian using neighbor interpolation
 *
 * This kernel computes the 4D Laplacian operator ∇² ψ using
 * finite differences on an irregular mesh via K-nearest neighbors.
 *
 * The Laplacian at point i is approximated as:
 *   ∇²ψ(i) ≈ (2/K) Σ_j [(ψ(j) - ψ(i)) / r_ij²]
 *
 * where j ranges over the K nearest neighbors and r_ij is the distance.
 */
__global__ void compute_laplacian_kernel(
    const double* psi_real,
    const double* psi_imag,
    const int32_t* neighbor_indices,
    const float* neighbor_distances,
    double* laplacian_real_out,
    double* laplacian_imag_out,
    int n_neighbors,
    int n_points
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n_points) {
        return;
    }

    // Load center point value
    const double psi_r = psi_real[idx];
    const double psi_i = psi_imag[idx];

    // Accumulate contributions from neighbors
    double lap_r = 0.0;
    double lap_i = 0.0;

    for (int i = 0; i < n_neighbors; ++i) {
        const int neighbor_idx = neighbor_indices[idx * n_neighbors + i];
        const float dist = neighbor_distances[idx * n_neighbors + i];

        // Avoid division by zero
        const double dist_sq = static_cast<double>(dist * dist + 1e-10f);

        // Neighbor values
        const double neighbor_psi_r = psi_real[neighbor_idx];
        const double neighbor_psi_i = psi_imag[neighbor_idx];

        // Finite difference: 2 * (ψ_neighbor - ψ_center) / r²
        lap_r += 2.0 * (neighbor_psi_r - psi_r) / dist_sq;
        lap_i += 2.0 * (neighbor_psi_i - psi_i) / dist_sq;
    }

    // Average over neighbors
    laplacian_real_out[idx] = lap_r / static_cast<double>(n_neighbors);
    laplacian_imag_out[idx] = lap_i / static_cast<double>(n_neighbors);
}

/**
 * Compute gradient in 4D using neighbor interpolation
 *
 * Computes ∇ψ = (∂ψ/∂w, ∂ψ/∂x, ∂ψ/∂y, ∂ψ/∂z)
 */
__global__ void compute_gradient_kernel(
    const double* psi_real,
    const double* psi_imag,
    const float* coords,              // [n_points * 4]
    const int32_t* neighbor_indices,
    const float* neighbor_distances,
    double* grad_w_real_out,
    double* grad_w_imag_out,
    double* grad_x_real_out,
    double* grad_x_imag_out,
    double* grad_y_real_out,
    double* grad_y_imag_out,
    double* grad_z_real_out,
    double* grad_z_imag_out,
    int n_neighbors,
    int n_points
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n_points) {
        return;
    }

    // Center point coordinates
    const float w = coords[idx * 4 + 0];
    const float x = coords[idx * 4 + 1];
    const float y = coords[idx * 4 + 2];
    const float z = coords[idx * 4 + 3];

    // Center point value
    const double psi_r = psi_real[idx];
    const double psi_i = psi_imag[idx];

    // Accumulate gradient components
    double grad_w_r = 0.0, grad_w_i = 0.0;
    double grad_x_r = 0.0, grad_x_i = 0.0;
    double grad_y_r = 0.0, grad_y_i = 0.0;
    double grad_z_r = 0.0, grad_z_i = 0.0;

    for (int i = 0; i < n_neighbors; ++i) {
        const int neighbor_idx = neighbor_indices[idx * n_neighbors + i];
        const float dist = neighbor_distances[idx * n_neighbors + i] + 1e-10f;

        // Neighbor coordinates
        const float nw = coords[neighbor_idx * 4 + 0];
        const float nx = coords[neighbor_idx * 4 + 1];
        const float ny = coords[neighbor_idx * 4 + 2];
        const float nz = coords[neighbor_idx * 4 + 3];

        // Neighbor values
        const double npsi_r = psi_real[neighbor_idx];
        const double npsi_i = psi_imag[neighbor_idx];

        // Direction vectors (normalized by distance)
        const float dw = (nw - w) / dist;
        const float dx = (nx - x) / dist;
        const float dy = (ny - y) / dist;
        const float dz = (nz - z) / dist;

        // Value differences
        const double dpsi_r = npsi_r - psi_r;
        const double dpsi_i = npsi_i - psi_i;

        // Accumulate directional derivatives
        grad_w_r += dw * dpsi_r;
        grad_w_i += dw * dpsi_i;
        grad_x_r += dx * dpsi_r;
        grad_x_i += dx * dpsi_i;
        grad_y_r += dy * dpsi_r;
        grad_y_i += dy * dpsi_i;
        grad_z_r += dz * dpsi_r;
        grad_z_i += dz * dpsi_i;
    }

    // Average and store
    const double inv_neighbors = 1.0 / static_cast<double>(n_neighbors);

    grad_w_real_out[idx] = grad_w_r * inv_neighbors;
    grad_w_imag_out[idx] = grad_w_i * inv_neighbors;
    grad_x_real_out[idx] = grad_x_r * inv_neighbors;
    grad_x_imag_out[idx] = grad_x_i * inv_neighbors;
    grad_y_real_out[idx] = grad_y_r * inv_neighbors;
    grad_y_imag_out[idx] = grad_y_i * inv_neighbors;
    grad_z_real_out[idx] = grad_z_r * inv_neighbors;
    grad_z_imag_out[idx] = grad_z_i * inv_neighbors;
}

/**
 * Compute angular momentum operator L_wx = w * ∂/∂x - x * ∂/∂w
 *
 * Used for rotation term in Gross-Pitaevskii equation
 */
__global__ void compute_angular_momentum_kernel(
    const double* grad_w_real,
    const double* grad_w_imag,
    const double* grad_x_real,
    const double* grad_x_imag,
    const float* coords,
    double* Lz_real_out,
    double* Lz_imag_out,
    int n_points
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n_points) {
        return;
    }

    // Coordinates
    const float w = coords[idx * 4 + 0];
    const float x = coords[idx * 4 + 1];

    // Gradients
    const double gw_r = grad_w_real[idx];
    const double gw_i = grad_w_imag[idx];
    const double gx_r = grad_x_real[idx];
    const double gx_i = grad_x_imag[idx];

    // Angular momentum: L_wx = w * ∂ψ/∂x - x * ∂ψ/∂w
    Lz_real_out[idx] = w * gx_r - x * gw_r;
    Lz_imag_out[idx] = w * gx_i - x * gw_i;
}

/**
 * Host wrapper functions for launching kernels
 */

extern "C" {

void launch_compute_laplacian(
    const double* d_psi_real,
    const double* d_psi_imag,
    const int32_t* d_neighbor_indices,
    const float* d_neighbor_distances,
    double* d_laplacian_real,
    double* d_laplacian_imag,
    int n_neighbors,
    int n_points
) {
    const int threads_per_block = 256;
    const int blocks_per_grid = (n_points + threads_per_block - 1) / threads_per_block;

    compute_laplacian_kernel<<<blocks_per_grid, threads_per_block>>>(
        d_psi_real, d_psi_imag,
        d_neighbor_indices, d_neighbor_distances,
        d_laplacian_real, d_laplacian_imag,
        n_neighbors, n_points
    );

    cudaDeviceSynchronize();
}

void launch_compute_gradient(
    const double* d_psi_real,
    const double* d_psi_imag,
    const float* d_coords,
    const int32_t* d_neighbor_indices,
    const float* d_neighbor_distances,
    double* d_grad_w_real, double* d_grad_w_imag,
    double* d_grad_x_real, double* d_grad_x_imag,
    double* d_grad_y_real, double* d_grad_y_imag,
    double* d_grad_z_real, double* d_grad_z_imag,
    int n_neighbors,
    int n_points
) {
    const int threads_per_block = 256;
    const int blocks_per_grid = (n_points + threads_per_block - 1) / threads_per_block;

    compute_gradient_kernel<<<blocks_per_grid, threads_per_block>>>(
        d_psi_real, d_psi_imag, d_coords,
        d_neighbor_indices, d_neighbor_distances,
        d_grad_w_real, d_grad_w_imag,
        d_grad_x_real, d_grad_x_imag,
        d_grad_y_real, d_grad_y_imag,
        d_grad_z_real, d_grad_z_imag,
        n_neighbors, n_points
    );

    cudaDeviceSynchronize();
}

void launch_compute_angular_momentum(
    const double* d_grad_w_real,
    const double* d_grad_w_imag,
    const double* d_grad_x_real,
    const double* d_grad_x_imag,
    const float* d_coords,
    double* d_Lz_real,
    double* d_Lz_imag,
    int n_points
) {
    const int threads_per_block = 256;
    const int blocks_per_grid = (n_points + threads_per_block - 1) / threads_per_block;

    compute_angular_momentum_kernel<<<blocks_per_grid, threads_per_block>>>(
        d_grad_w_real, d_grad_w_imag,
        d_grad_x_real, d_grad_x_imag,
        d_coords,
        d_Lz_real, d_Lz_imag,
        n_points
    );

    cudaDeviceSynchronize();
}

} // extern "C"

} // namespace bec4d
