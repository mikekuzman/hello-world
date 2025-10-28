#include "math_utils.h"
#include <algorithm>
#include <numeric>
#include <cmath>

namespace bec4d {
namespace math {

template<typename T>
Stats<T> computeStats(const std::vector<T>& data) {
    if (data.empty()) {
        return {T(0), T(0), T(0), T(0), T(0), T(0)};
    }

    // Min/Max
    auto [min_it, max_it] = std::minmax_element(data.begin(), data.end());
    T min_val = *min_it;
    T max_val = *max_it;

    // Mean
    T sum = std::accumulate(data.begin(), data.end(), T(0));
    T mean_val = sum / static_cast<T>(data.size());

    // Standard deviation
    T sq_sum = std::accumulate(data.begin(), data.end(), T(0),
        [mean_val](T acc, T x) { return acc + (x - mean_val) * (x - mean_val); });
    T std_val = std::sqrt(sq_sum / static_cast<T>(data.size()));

    // Percentiles (5th and 95th)
    std::vector<T> sorted_data = data;
    std::sort(sorted_data.begin(), sorted_data.end());

    size_t idx_p5 = static_cast<size_t>(0.05 * sorted_data.size());
    size_t idx_p95 = static_cast<size_t>(0.95 * sorted_data.size());

    T p5_val = sorted_data[idx_p5];
    T p95_val = sorted_data[idx_p95];

    return {min_val, max_val, mean_val, std_val, p5_val, p95_val};
}

// Explicit instantiations
template Stats<float> computeStats(const std::vector<float>&);
template Stats<double> computeStats(const std::vector<double>&);

// Simple XorShift64* random number generator
Random::Random(uint32_t seed) : state_(seed) {
    if (state_ == 0) {
        state_ = 0x123456789ABCDEF0ULL;
    }
}

float Random::uniform() {
    // XorShift64*
    state_ ^= state_ >> 12;
    state_ ^= state_ << 25;
    state_ ^= state_ >> 27;
    uint64_t result = state_ * 0x2545F4914F6CDD1DULL;
    return static_cast<float>(result >> 32) / static_cast<float>(0xFFFFFFFFULL);
}

float Random::uniform(float min, float max) {
    return min + (max - min) * uniform();
}

float Random::normal(float mean, float stddev) {
    // Box-Muller transform
    static float spare = 0.0f;
    static bool has_spare = false;

    if (has_spare) {
        has_spare = false;
        return mean + stddev * spare;
    }

    has_spare = true;

    float u1, u2;
    do {
        u1 = uniform();
        u2 = uniform();
    } while (u1 <= 1e-8f);

    const float radius = std::sqrt(-2.0f * std::log(u1));
    const float theta = TWO_PI * u2;

    spare = radius * std::sin(theta);
    return mean + stddev * radius * std::cos(theta);
}

std::vector<Vortex> detectVortices(
    const std::vector<float>& coords,
    const std::vector<float>& phase,
    const std::vector<float>& density,
    int n_points,
    float density_threshold
) {
    // Simplified vortex detection
    // In production, use topological charge calculation around loops
    std::vector<Vortex> vortices;

    for (int i = 0; i < n_points; ++i) {
        if (density[i] < density_threshold) {
            // Potential vortex core
            Vortex v;
            v.position[0] = coords[i * 4 + 0];
            v.position[1] = coords[i * 4 + 1];
            v.position[2] = coords[i * 4 + 2];
            v.position[3] = coords[i * 4 + 3];
            v.quantum_number = 1; // TODO: compute from phase winding
            v.velocity[0] = 0.0f;
            v.velocity[1] = 0.0f;
            v.velocity[2] = 0.0f;
            v.velocity[3] = 0.0f;

            vortices.push_back(v);
        }
    }

    return vortices;
}

} // namespace math
} // namespace bec4d
