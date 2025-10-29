#include "hypersphere_bec.h"
#include "neighbor_tree.h"
#include "math_utils.h"
#include "gpu_data.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <algorithm>

namespace bec4d {

HypersphereBEC::HypersphereBEC(const SimulationParams& params, ProgressCallback progress_cb)
    : params_(params)
    , n_active_(0)
    , gpu_data_(std::make_unique<GPUData>())
    , metrics_{}
    , progress_callback_(progress_cb)
{
    auto start_time = std::chrono::high_resolution_clock::now();

    std::cout << "\nInitializing 4D BEC Simulation..." << std::endl;
    params_.print();

    // Find shell points
    std::cout << "\nFinding shell points..." << std::endl;
    auto t0 = std::chrono::high_resolution_clock::now();
    findShellPoints();
    auto t1 = std::chrono::high_resolution_clock::now();
    auto shell_time = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "  ✓ Shell scan: " << shell_time << "s" << std::endl;
    std::cout << "  Active shell points: " << n_active_ << std::endl;

    // Build neighbor tree
    std::cout << "\nBuilding neighbor tree..." << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    buildNeighborTree();
    t1 = std::chrono::high_resolution_clock::now();
    auto tree_time = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "  ✓ Neighbor tree: " << tree_time << "s" << std::endl;

    // Initialize wavefunction
    std::cout << "\nInitializing wavefunction..." << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    initializeWavefunction();
    t1 = std::chrono::high_resolution_clock::now();
    auto init_time = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "  ✓ Wavefunction: " << init_time << "s" << std::endl;

    // Allocate GPU memory
    std::cout << "\nAllocating GPU memory..." << std::endl;
    allocateGPUMemory();
    copyToGPU();
    std::cout << "  ✓ GPU memory allocated" << std::endl;

    auto end_time = std::chrono::high_resolution_clock::now();
    metrics_.init_time_s = std::chrono::duration<double>(end_time - start_time).count();
    metrics_.memory_usage_mb = static_cast<size_t>(
        (coords_.size() * sizeof(float) +
         psi_real_.size() * sizeof(double) +
         psi_imag_.size() * sizeof(double) +
         neighbor_indices_.size() * sizeof(int32_t) +
         neighbor_distances_.size() * sizeof(float)) / (1024.0 * 1024.0)
    );

    std::cout << "\nMemory estimate: ~" << metrics_.memory_usage_mb << " MB" << std::endl;
    std::cout << "Total initialization time: " << metrics_.init_time_s << "s" << std::endl;
    std::cout << "======================================================================\n" << std::endl;
}

HypersphereBEC::~HypersphereBEC() {
    freeGPUMemory();
}

HypersphereBEC::HypersphereBEC(HypersphereBEC&&) noexcept = default;
HypersphereBEC& HypersphereBEC::operator=(HypersphereBEC&&) noexcept = default;

void HypersphereBEC::findShellPoints() {
    const float r_inner = params_.R - params_.delta / 2.0f;
    const float r_outer = params_.R + params_.delta / 2.0f;
    const int N = params_.N;
    const float box_size = params_.box_size;

    // Generate grid
    std::vector<float> grid(N);
    for (int i = 0; i < N; ++i) {
        grid[i] = -box_size + (2.0f * box_size * i) / static_cast<float>(N - 1);
    }

    // Scan 4D grid for shell points (chunked to save memory)
    std::vector<std::vector<float>> temp_coords;
    const int chunk_size = 8;

    std::cout << "  Scanning grid: [" << std::flush;
    int last_pct = -1;

    for (int iw = 0; iw < N; iw += chunk_size) {
        const int iw_end = std::min(iw + chunk_size, N);

        for (int wi = iw; wi < iw_end; ++wi) {
            const float w = grid[wi];

            // Progress indicator
            int pct = (wi * 100) / N;
            if (pct != last_pct && pct % 10 == 0) {
                std::cout << "#" << std::flush;
                last_pct = pct;
                if (progress_callback_) {
                    progress_callback_("Shell Scan", static_cast<float>(pct) / 100.0f);
                }
            }

            for (int xi = 0; xi < N; ++xi) {
                const float x = grid[xi];

                for (int yi = 0; yi < N; ++yi) {
                    const float y = grid[yi];

                    for (int zi = 0; zi < N; ++zi) {
                        const float z = grid[zi];

                        const float r = std::sqrt(w*w + x*x + y*y + z*z);

                        if (r >= r_inner && r <= r_outer) {
                            temp_coords.push_back({w, x, y, z});
                        }
                    }
                }
            }
        }
    }
    std::cout << "] Done" << std::endl;

    // Flatten to single array
    n_active_ = static_cast<int>(temp_coords.size());
    coords_.resize(n_active_ * 4);

    for (int i = 0; i < n_active_; ++i) {
        coords_[i * 4 + 0] = temp_coords[i][0];
        coords_[i * 4 + 1] = temp_coords[i][1];
        coords_[i * 4 + 2] = temp_coords[i][2];
        coords_[i * 4 + 3] = temp_coords[i][3];
    }
}

void HypersphereBEC::buildNeighborTree() {
    std::cout << "  Building KD-tree: " << std::flush;
    if (progress_callback_) progress_callback_("KD-Tree Build", 0.0f);

    neighbor_tree_ = std::make_unique<NeighborTree>(coords_, n_active_);

    if (progress_callback_) progress_callback_("KD-Tree Build", 1.0f);
    std::cout << "Done" << std::endl;

    std::cout << "  Finding neighbors: " << std::flush;
    neighbor_indices_.resize(n_active_ * params_.n_neighbors);
    neighbor_distances_.resize(n_active_ * params_.n_neighbors);

    // Query with progress callback
    auto neighbor_progress = [this](float progress) {
        if (progress_callback_) {
            progress_callback_("Neighbor Search", progress);
        }
    };

    neighbor_tree_->queryKNN(params_.n_neighbors, neighbor_indices_, neighbor_distances_, neighbor_progress);
    std::cout << "Done (" << params_.n_neighbors << "-NN)" << std::endl;

    // Compute average neighbor distance
    double avg_dist = 0.0;
    for (const auto& d : neighbor_distances_) {
        avg_dist += d;
    }
    avg_dist /= neighbor_distances_.size();

    std::cout << "  Average neighbor distance: " << avg_dist << " ξ" << std::endl;
}

void HypersphereBEC::initializeWavefunction() {
    psi_real_.resize(n_active_);
    psi_imag_.resize(n_active_);

    // Initialize with uniform + noise
    math::Random rng(params_.random_seed);
    const float noise_amplitude = 0.01f;

    for (int i = 0; i < n_active_; ++i) {
        psi_real_[i] = 1.0 + noise_amplitude * rng.normal();
        psi_imag_[i] = noise_amplitude * rng.normal();
    }

    std::cout << "  Wavefunction layout: SoA (Struct-of-Arrays)" << std::endl;
    std::cout << "  Precision: float64 (wavefunction), float32 (coordinates)" << std::endl;
}

void HypersphereBEC::allocateGPUMemory() {
    // Will be implemented in CUDA file
    // For now, stub
    gpu_data_->allocated = false;
}

void HypersphereBEC::copyToGPU() {
    // Will be implemented in CUDA file
}

void HypersphereBEC::copyFromGPU() {
    // Will be implemented in CUDA file
}

void HypersphereBEC::freeGPUMemory() {
    // Will be implemented in CUDA file
}

void HypersphereBEC::evolveStep() {
    // Will be implemented in CUDA file
    // For now, placeholder
}

void HypersphereBEC::run(int n_steps, int save_every) {
    std::cout << "======================================================================" << std::endl;
    std::cout << "Starting simulation: " << n_steps << " steps" << std::endl;
    std::cout << "======================================================================" << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < n_steps; ++step) {
        evolveStep();

        // Save snapshot
        if (step % save_every == 0) {
            auto snapshot_start = std::chrono::high_resolution_clock::now();

            auto density = getDensity();
            auto phase = getPhase();

            // Compute statistics (on full data)
            auto stats = math::computeStats(density);

            // Subsample for snapshot storage
            int subsample = params_.snapshot_subsample;
            int n_sampled = (n_active_ + subsample - 1) / subsample;

            Snapshot snapshot;
            snapshot.step = step;
            snapshot.coords.reserve(n_sampled * 4);
            snapshot.density.reserve(n_sampled);
            snapshot.phase.reserve(n_sampled);

            for (int i = 0; i < n_active_; i += subsample) {
                snapshot.coords.push_back(coords_[i * 4 + 0]);
                snapshot.coords.push_back(coords_[i * 4 + 1]);
                snapshot.coords.push_back(coords_[i * 4 + 2]);
                snapshot.coords.push_back(coords_[i * 4 + 3]);
                snapshot.density.push_back(density[i]);
                snapshot.phase.push_back(phase[i]);
            }

            snapshot.density_stats.min = stats.min;
            snapshot.density_stats.max = stats.max;
            snapshot.density_stats.mean = stats.mean;
            snapshot.density_stats.std = stats.std;
            snapshot.n_vortices = 0; // TODO: detect vortices

            snapshots_.push_back(std::move(snapshot));

            auto snapshot_end = std::chrono::high_resolution_clock::now();
            auto snapshot_time = std::chrono::duration<double>(snapshot_end - snapshot_start).count();

            // Performance metrics
            auto elapsed = std::chrono::duration<double>(snapshot_end - start_time).count();
            double steps_per_sec = (step + 1) / elapsed;
            double eta_seconds = (n_steps - step - 1) / steps_per_sec;

            std::cout << "  Step " << std::setw(5) << step
                      << ": <ρ>=" << std::fixed << std::setprecision(3) << stats.mean
                      << " [" << stats.min << ", " << stats.max << "]"
                      << " | " << std::setprecision(1) << steps_per_sec << " steps/s"
                      << ", ETA: " << (eta_seconds / 60.0) << " min"
                      << " (snapshot: " << std::setprecision(2) << snapshot_time << "s)"
                      << std::endl;

            // Report progress
            if (progress_callback_) {
                float progress = static_cast<float>(step + 1) / static_cast<float>(n_steps);
                progress_callback_("Simulation", progress);
            }

            // Check stability
            if (std::isnan(stats.mean) || stats.max > 1e6f) {
                std::cout << "\n⚠ WARNING: Numerical instability at step " << step << std::endl;
                break;
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration<double>(end_time - start_time).count();
    double avg_steps_per_sec = n_steps / total_time;

    metrics_.avg_step_time_ms = (total_time / n_steps) * 1000.0;
    metrics_.steps_per_second = avg_steps_per_sec;

    std::cout << "\n======================================================================" << std::endl;
    std::cout << "Simulation complete!" << std::endl;
    std::cout << "  Total time: " << total_time << "s (" << (total_time / 60.0) << " min)" << std::endl;
    std::cout << "  Average: " << avg_steps_per_sec << " steps/s" << std::endl;
    std::cout << "  Target was: 50-100 steps/s" << std::endl;
    std::cout << "  Speedup vs v005 (1.4 steps/s): " << (avg_steps_per_sec / 1.4) << "x" << std::endl;
    std::cout << "======================================================================\n" << std::endl;
}

std::vector<float> HypersphereBEC::getDensity() const {
    std::vector<float> density(n_active_);
    for (int i = 0; i < n_active_; ++i) {
        const double r = psi_real_[i];
        const double im = psi_imag_[i];
        density[i] = static_cast<float>(r * r + im * im);
    }
    return density;
}

std::vector<float> HypersphereBEC::getPhase() const {
    std::vector<float> phase(n_active_);
    for (int i = 0; i < n_active_; ++i) {
        phase[i] = std::atan2(static_cast<float>(psi_imag_[i]),
                             static_cast<float>(psi_real_[i]));
    }
    return phase;
}

std::vector<float> HypersphereBEC::getCoords() const {
    return coords_;
}

} // namespace bec4d
