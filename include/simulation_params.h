#pragma once

#include <cstdint>
#include <tuple>

namespace bec4d {

/**
 * Parameters for 4D hypersphere BEC simulation
 * Matches Python SimulationParams from sim_v006.py
 */
struct SimulationParams {
    // Physical parameters (dimensionless units: hbar=m=xi=1)
    float R = 1000.0f;              // Hypersphere radius in healing lengths
    float delta = 25.0f;            // Shell thickness in healing lengths
    float g = 0.05f;                // Interaction strength (weak)
    float omega = 0.03f;            // Rotation rate (moderate)

    // Computational parameters
    int N = 128;                    // Grid points per dimension
    float dt = 0.001f;              // Time step
    int n_neighbors = 6;            // Number of neighbors for gradient calculation

    // Rotation plane (4D has 6 possible planes, using w-x plane)
    std::tuple<int, int> rotation_plane = {0, 1};  // (w, x) indices

    // Random seed for reproducibility
    uint32_t random_seed = 0;

    // Initial condition type
    enum class InitialConditionType {
        UniformNoise,
        ImaginaryTime,
        Vortex
    };
    InitialConditionType initial_condition_type = InitialConditionType::UniformNoise;
    int imag_time_steps = 1000;

    // Derived parameters (computed in constructor)
    float box_size = 0.0f;
    float dx = 0.0f;

    // Constructor
    SimulationParams();

    // Update derived parameters
    void updateDerived();

    // Print parameters to console
    void print() const;
};

} // namespace bec4d
