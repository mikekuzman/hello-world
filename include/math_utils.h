#pragma once

#include <cmath>
#include <vector>

namespace bec4d {
namespace math {

// Constants
constexpr float PI = 3.14159265358979323846f;
constexpr float TWO_PI = 2.0f * PI;
constexpr float HALF_PI = 0.5f * PI;

// Statistics
template<typename T>
struct Stats {
    T min, max, mean, std;
    T p5, p95;  // 5th and 95th percentiles
};

template<typename T>
Stats<T> computeStats(const std::vector<T>& data);

// Phase unwrapping
std::vector<float> unwrapPhase(const std::vector<float>& phase);

// Vortex detection
struct Vortex {
    float position[4];       // w, x, y, z
    int quantum_number;      // ±1, ±2, ...
    float velocity[4];       // dw, dx, dy, dz
};

std::vector<Vortex> detectVortices(
    const std::vector<float>& coords,     // [n * 4]
    const std::vector<float>& phase,      // [n]
    const std::vector<float>& density,    // [n]
    int n_points,
    float density_threshold = 0.1f
);

// Random number generation (thread-safe)
class Random {
public:
    explicit Random(uint32_t seed);

    float uniform();                          // [0, 1)
    float uniform(float min, float max);      // [min, max)
    float normal(float mean = 0.0f, float stddev = 1.0f);

private:
    uint64_t state_;
};

} // namespace math
} // namespace bec4d
