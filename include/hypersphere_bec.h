#pragma once

#include "simulation_params.h"
#include <vector>
#include <memory>
#include <cstdint>
#include <functional>

namespace bec4d {

// Forward declarations
class NeighborTree;
struct SnapshotData;
struct GPUData;  // Forward declare at namespace level

/**
 * 4D Hypersphere Quantum Superfluid Simulator
 * C++ implementation of Python HypersphereBEC_v006
 *
 * High-performance CUDA-accelerated simulation of Bose-Einstein Condensate
 * on a 4D hyperspherical shell.
 */
class HypersphereBEC {
public:
    // Progress callback: (phase_name, progress_0_to_1)
    using ProgressCallback = std::function<void(const char*, float)>;

    explicit HypersphereBEC(const SimulationParams& params, ProgressCallback progress_cb = nullptr);
    ~HypersphereBEC();

    // Disable copy, allow move
    HypersphereBEC(const HypersphereBEC&) = delete;
    HypersphereBEC& operator=(const HypersphereBEC&) = delete;
    HypersphereBEC(HypersphereBEC&&) noexcept;
    HypersphereBEC& operator=(HypersphereBEC&&) noexcept;

    // Simulation control
    void evolveStep();                          // Single timestep
    void run(int n_steps, int save_every = 100); // Run simulation

    // Data access
    std::vector<float> getDensity() const;      // Get |ψ|²
    std::vector<float> getPhase() const;        // Get arg(ψ)
    std::vector<float> getCoords() const;       // Get 4D coordinates
    int getActivePointCount() const { return n_active_; }

    // Snapshot management
    struct Snapshot {
        int step;
        std::vector<float> coords;      // [n_active * 4] (w,x,y,z)
        std::vector<float> density;     // [n_active]
        std::vector<float> phase;       // [n_active]
        std::vector<float> velocity;    // [n_active * 4]

        // Statistics
        struct {
            float min, max, mean, std;
        } density_stats;

        int n_vortices;
    };

    const std::vector<Snapshot>& getSnapshots() const { return snapshots_; }

    // Performance metrics
    struct PerformanceMetrics {
        double init_time_s;
        double avg_step_time_ms;
        double steps_per_second;
        size_t memory_usage_mb;
    };

    const PerformanceMetrics& getMetrics() const { return metrics_; }

private:
    // Initialization
    void findShellPoints();
    void buildNeighborTree();
    void initializeWavefunction();

    // GPU management
    void allocateGPUMemory();
    void copyToGPU();
    void copyFromGPU();
    void freeGPUMemory();

    // Data members
    SimulationParams params_;
    int n_active_;

    // Host memory (CPU)
    std::vector<float> coords_;             // [n_active * 4] flat array
    std::vector<double> psi_real_;          // [n_active]
    std::vector<double> psi_imag_;          // [n_active]
    std::vector<int32_t> neighbor_indices_; // [n_active * n_neighbors]
    std::vector<float> neighbor_distances_; // [n_active * n_neighbors]

    // Device memory (GPU) - opaque pointer
    std::unique_ptr<GPUData> gpu_data_;

    // Neighbor tree
    std::unique_ptr<NeighborTree> neighbor_tree_;

    // Snapshots
    std::vector<Snapshot> snapshots_;

    // Performance tracking
    PerformanceMetrics metrics_;
    double start_time_;

    // Progress callback
    ProgressCallback progress_callback_;
};

} // namespace bec4d
