#pragma once

#include <cstdint>

namespace bec4d {

/**
 * GPU data structure for CUDA device pointers
 * Shared between hypersphere_bec.cpp and evolution_kernels.cu
 */
struct GPUData {
    // Device pointers
    double* d_psi_real = nullptr;
    double* d_psi_imag = nullptr;
    double* d_output_real = nullptr;
    double* d_output_imag = nullptr;
    float* d_coords = nullptr;
    int32_t* d_neighbor_indices = nullptr;
    float* d_neighbor_distances = nullptr;

    bool allocated = false;
};

} // namespace bec4d
