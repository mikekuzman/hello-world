#pragma once

#include <vector>
#include <cstdint>

namespace bec4d {

/**
 * KD-tree for 4D neighbor lookups
 * Used to compute Laplacian via finite differences on irregular mesh
 */
class NeighborTree {
public:
    NeighborTree(const std::vector<float>& coords, int n_points);
    ~NeighborTree();

    // Query k nearest neighbors for each point
    void queryKNN(
        int k,
        std::vector<int32_t>& indices_out,    // [n_points * k]
        std::vector<float>& distances_out     // [n_points * k]
    );

    // Query k nearest neighbors for a single point
    void queryPoint(
        const float* point_4d,
        int k,
        std::vector<int32_t>& indices_out,
        std::vector<float>& distances_out
    );

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace bec4d
