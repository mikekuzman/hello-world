#include "hypersphere_bec.h"
#include "../gpu_data.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>

namespace bec4d {

// Device function for Laplacian computation
__device__ void compute_laplacian_device(
    int idx,
    const double* psi_real,
    const double* psi_imag,
    const int32_t* neighbor_indices,
    const float* neighbor_distances,
    int n_neighbors,
    double& laplacian_real_out,
    double& laplacian_imag_out
) {
    const double psi_r = psi_real[idx];
    const double psi_i = psi_imag[idx];

    double laplacian_r = 0.0;
    double laplacian_i = 0.0;

    for (int i = 0; i < n_neighbors; ++i) {
        const int neighbor_idx = neighbor_indices[idx * n_neighbors + i];
        const double neighbor_psi_r = psi_real[neighbor_idx];
        const double neighbor_psi_i = psi_imag[neighbor_idx];
        const float dist = neighbor_distances[idx * n_neighbors + i];
        const double dist_sq = static_cast<double>(dist * dist + 1e-10f);

        laplacian_r += 2.0 * (neighbor_psi_r - psi_r) / dist_sq;
        laplacian_i += 2.0 * (neighbor_psi_i - psi_i) / dist_sq;
    }

    laplacian_real_out = laplacian_r / static_cast<double>(n_neighbors);
    laplacian_imag_out = laplacian_i / static_cast<double>(n_neighbors);
}

// Fused evolution kernel: Laplacian + rotation + evolution in ONE pass
__global__ void fused_evolve_kernel(
    const double* psi_real,
    const double* psi_imag,
    const float* coords,
    const int32_t* neighbor_indices,
    const float* neighbor_distances,
    double* output_real,
    double* output_imag,
    double dt,
    double omega,
    double g,
    int n_neighbors,
    int n_points
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n_points) {
        return;
    }

    // Load wavefunction at this point
    const double psi_r = psi_real[idx];
    const double psi_i = psi_imag[idx];

    // 1. Compute Laplacian using neighbor interpolation
    double laplacian_r, laplacian_i;
    compute_laplacian_device(idx, psi_real, psi_imag, neighbor_indices,
                           neighbor_distances, n_neighbors, laplacian_r, laplacian_i);

    // 2. Compute density
    const double density = psi_r * psi_r + psi_i * psi_i;

    // 3. Kinetic term: -0.5 * Laplacian
    const double kin_r = -0.5 * laplacian_r;
    const double kin_i = -0.5 * laplacian_i;

    // 4. Potential term: g * |psi|^2 * psi
    const double pot_r = g * density * psi_r;
    const double pot_i = g * density * psi_i;

    // 5. Rotation term (simplified - assumes rotation in w-x plane)
    // L_wx = w * p_x - x * p_w
    // For now, use simple approximation with nearest neighbors
    const float w = coords[idx * 4 + 0];
    const float x = coords[idx * 4 + 1];

    // Compute gradient (finite difference using neighbors)
    double grad_w_r = 0.0, grad_w_i = 0.0;
    double grad_x_r = 0.0, grad_x_i = 0.0;

    for (int i = 0; i < n_neighbors; ++i) {
        const int neighbor_idx = neighbor_indices[idx * n_neighbors + i];
        const float nw = coords[neighbor_idx * 4 + 0];
        const float nx = coords[neighbor_idx * 4 + 1];
        const double npsi_r = psi_real[neighbor_idx];
        const double npsi_i = psi_imag[neighbor_idx];
        const float dist = neighbor_distances[idx * n_neighbors + i] + 1e-10f;

        const float dw = (nw - w) / dist;
        const float dx = (nx - x) / dist;

        grad_w_r += dw * (npsi_r - psi_r);
        grad_w_i += dw * (npsi_i - psi_i);
        grad_x_r += dx * (npsi_r - psi_r);
        grad_x_i += dx * (npsi_i - psi_i);
    }

    grad_w_r /= n_neighbors;
    grad_w_i /= n_neighbors;
    grad_x_r /= n_neighbors;
    grad_x_i /= n_neighbors;

    // Angular momentum: L_wx = w * grad_x - x * grad_w
    const double Lz_r = w * grad_x_r - x * grad_w_r;
    const double Lz_i = w * grad_x_i - x * grad_w_i;

    // Rotation term: -i * omega * L_wx * psi
    // = -i * omega * (Lz_r + i*Lz_i)
    // = -i * omega * Lz_r + omega * Lz_i
    const double rot_r = omega * Lz_i;
    const double rot_i = -omega * Lz_r;

    // 6. First-order Euler step: psi_new = psi + dt * (-i) * (kin + pot + rot)
    // -i * (kin_r + i*kin_i) = kin_i - i*kin_r
    output_real[idx] = psi_r + dt * (kin_i + pot_i + rot_i);
    output_imag[idx] = psi_i - dt * (kin_r + pot_r + rot_r);
}

// Host function to launch kernel
void HypersphereBEC::evolveStep() {
    if (!gpu_data_->allocated) {
        return; // GPU not initialized
    }

    const int threads_per_block = 256;
    const int blocks_per_grid = (n_active_ + threads_per_block - 1) / threads_per_block;

    fused_evolve_kernel<<<blocks_per_grid, threads_per_block>>>(
        gpu_data_->d_psi_real,
        gpu_data_->d_psi_imag,
        gpu_data_->d_coords,
        gpu_data_->d_neighbor_indices,
        gpu_data_->d_neighbor_distances,
        gpu_data_->d_output_real,
        gpu_data_->d_output_imag,
        static_cast<double>(params_.dt),
        static_cast<double>(params_.omega),
        static_cast<double>(params_.g),
        params_.n_neighbors,
        n_active_
    );

    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel error: %s\n", cudaGetErrorString(err));
    }

    // Swap buffers (double buffering)
    std::swap(gpu_data_->d_psi_real, gpu_data_->d_output_real);
    std::swap(gpu_data_->d_psi_imag, gpu_data_->d_output_imag);
}

// GPU memory management
void HypersphereBEC::allocateGPUMemory() {
    const size_t n = n_active_;

    cudaMalloc(&gpu_data_->d_psi_real, n * sizeof(double));
    cudaMalloc(&gpu_data_->d_psi_imag, n * sizeof(double));
    cudaMalloc(&gpu_data_->d_output_real, n * sizeof(double));
    cudaMalloc(&gpu_data_->d_output_imag, n * sizeof(double));
    cudaMalloc(&gpu_data_->d_coords, n * 4 * sizeof(float));
    cudaMalloc(&gpu_data_->d_neighbor_indices, n * params_.n_neighbors * sizeof(int32_t));
    cudaMalloc(&gpu_data_->d_neighbor_distances, n * params_.n_neighbors * sizeof(float));

    gpu_data_->allocated = true;

    // Check for allocation errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA allocation error: %s\n", cudaGetErrorString(err));
        gpu_data_->allocated = false;
    }
}

void HypersphereBEC::copyToGPU() {
    if (!gpu_data_->allocated) return;

    const size_t n = n_active_;

    cudaMemcpy(gpu_data_->d_psi_real, psi_real_.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_data_->d_psi_imag, psi_imag_.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_data_->d_coords, coords_.data(), n * 4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_data_->d_neighbor_indices, neighbor_indices_.data(),
               n * params_.n_neighbors * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_data_->d_neighbor_distances, neighbor_distances_.data(),
               n * params_.n_neighbors * sizeof(float), cudaMemcpyHostToDevice);
}

void HypersphereBEC::copyFromGPU() {
    if (!gpu_data_->allocated) return;

    const size_t n = n_active_;

    cudaMemcpy(psi_real_.data(), gpu_data_->d_psi_real, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(psi_imag_.data(), gpu_data_->d_psi_imag, n * sizeof(double), cudaMemcpyDeviceToHost);
}

void HypersphereBEC::freeGPUMemory() {
    if (!gpu_data_->allocated) return;

    cudaFree(gpu_data_->d_psi_real);
    cudaFree(gpu_data_->d_psi_imag);
    cudaFree(gpu_data_->d_output_real);
    cudaFree(gpu_data_->d_output_imag);
    cudaFree(gpu_data_->d_coords);
    cudaFree(gpu_data_->d_neighbor_indices);
    cudaFree(gpu_data_->d_neighbor_distances);

    gpu_data_->allocated = false;
}

} // namespace bec4d
