#include "neighbor_tree.h"
#include <algorithm>
#include <queue>
#include <cmath>
#include <limits>

namespace bec4d {

// Simple KD-tree implementation for 4D points
struct NeighborTree::Impl {
    struct Node {
        int point_idx;
        int split_dim;
        Node* left = nullptr;
        Node* right = nullptr;
    };

    Node* root = nullptr;
    const std::vector<float>* coords = nullptr;
    int n_points = 0;

    ~Impl() {
        deleteTree(root);
    }

    void deleteTree(Node* node) {
        if (!node) return;
        deleteTree(node->left);
        deleteTree(node->right);
        delete node;
    }

    float distance4D(int idx1, int idx2) const {
        const float* p1 = &(*coords)[idx1 * 4];
        const float* p2 = &(*coords)[idx2 * 4];

        float dw = p1[0] - p2[0];
        float dx = p1[1] - p2[1];
        float dy = p1[2] - p2[2];
        float dz = p1[3] - p2[3];

        return std::sqrt(dw*dw + dx*dx + dy*dy + dz*dz);
    }

    float distanceToPoint(int idx, const float* point) const {
        const float* p = &(*coords)[idx * 4];

        float dw = p[0] - point[0];
        float dx = p[1] - point[1];
        float dy = p[2] - point[2];
        float dz = p[3] - point[3];

        return std::sqrt(dw*dw + dx*dx + dy*dy + dz*dz);
    }

    // Build KD-tree recursively
    Node* buildTree(std::vector<int>& indices, int depth) {
        if (indices.empty()) {
            return nullptr;
        }

        if (indices.size() == 1) {
            Node* node = new Node();
            node->point_idx = indices[0];
            return node;
        }

        // Split dimension cycles through 0,1,2,3 (w,x,y,z)
        int split_dim = depth % 4;

        // Sort indices by split dimension
        std::sort(indices.begin(), indices.end(), [&](int a, int b) {
            return (*coords)[a * 4 + split_dim] < (*coords)[b * 4 + split_dim];
        });

        // Median
        size_t median = indices.size() / 2;

        Node* node = new Node();
        node->point_idx = indices[median];
        node->split_dim = split_dim;

        // Recursively build left and right subtrees
        std::vector<int> left_indices(indices.begin(), indices.begin() + median);
        std::vector<int> right_indices(indices.begin() + median + 1, indices.end());

        node->left = buildTree(left_indices, depth + 1);
        node->right = buildTree(right_indices, depth + 1);

        return node;
    }

    // Priority queue element for KNN search
    struct KNNEntry {
        int idx;
        float dist;

        bool operator<(const KNNEntry& other) const {
            return dist < other.dist;  // Max heap
        }
    };

    // Find K nearest neighbors using KD-tree
    void searchKNN(Node* node, const float* query_point, int k,
                   std::priority_queue<KNNEntry>& heap) const {
        if (!node) {
            return;
        }

        // Distance to this node's point
        float dist = distanceToPoint(node->point_idx, query_point);

        // Add to heap
        if (heap.size() < static_cast<size_t>(k)) {
            heap.push({node->point_idx, dist});
        } else if (dist < heap.top().dist) {
            heap.pop();
            heap.push({node->point_idx, dist});
        }

        // Determine which side to search first
        const float* node_point = &(*coords)[node->point_idx * 4];
        float diff = query_point[node->split_dim] - node_point[node->split_dim];

        Node* near_node = (diff < 0) ? node->left : node->right;
        Node* far_node = (diff < 0) ? node->right : node->left;

        // Search near side
        searchKNN(near_node, query_point, k, heap);

        // Check if we need to search far side
        if (heap.size() < static_cast<size_t>(k) || std::abs(diff) < heap.top().dist) {
            searchKNN(far_node, query_point, k, heap);
        }
    }
};

NeighborTree::NeighborTree(const std::vector<float>& coords, int n_points)
    : impl_(std::make_unique<Impl>())
{
    impl_->coords = &coords;
    impl_->n_points = n_points;

    // Build KD-tree
    std::vector<int> indices(n_points);
    for (int i = 0; i < n_points; ++i) {
        indices[i] = i;
    }

    impl_->root = impl_->buildTree(indices, 0);
}

NeighborTree::~NeighborTree() = default;

void NeighborTree::queryKNN(
    int k,
    std::vector<int32_t>& indices_out,
    std::vector<float>& distances_out,
    ProgressCallback progress_cb
) {
    indices_out.resize(impl_->n_points * k);
    distances_out.resize(impl_->n_points * k);

    // Report progress every 1% of points
    int progress_interval = std::max(1, impl_->n_points / 100);

    // Query each point
    for (int i = 0; i < impl_->n_points; ++i) {
        const float* query_point = &(*impl_->coords)[i * 4];

        std::priority_queue<Impl::KNNEntry> heap;
        impl_->searchKNN(impl_->root, query_point, k + 1, heap);  // +1 to exclude self

        // Extract results (excluding self)
        std::vector<Impl::KNNEntry> results;
        while (!heap.empty()) {
            results.push_back(heap.top());
            heap.pop();
        }

        // Reverse (heap is max-heap, we want ascending order)
        std::reverse(results.begin(), results.end());

        // Report progress periodically
        if (progress_cb && (i % progress_interval == 0 || i == impl_->n_points - 1)) {
            float progress = static_cast<float>(i + 1) / static_cast<float>(impl_->n_points);
            progress_cb(progress);
        }

        // Fill output arrays (skip first entry if it's the point itself)
        int out_idx = 0;
        for (const auto& entry : results) {
            if (entry.idx != i && out_idx < k) {
                indices_out[i * k + out_idx] = entry.idx;
                distances_out[i * k + out_idx] = entry.dist;
                ++out_idx;
            }
        }
    }
}

void NeighborTree::queryPoint(
    const float* point_4d,
    int k,
    std::vector<int32_t>& indices_out,
    std::vector<float>& distances_out
) {
    std::priority_queue<Impl::KNNEntry> heap;
    impl_->searchKNN(impl_->root, point_4d, k, heap);

    // Extract results
    std::vector<Impl::KNNEntry> results;
    while (!heap.empty()) {
        results.push_back(heap.top());
        heap.pop();
    }

    std::reverse(results.begin(), results.end());

    indices_out.resize(results.size());
    distances_out.resize(results.size());

    for (size_t i = 0; i < results.size(); ++i) {
        indices_out[i] = results[i].idx;
        distances_out[i] = results[i].dist;
    }
}

} // namespace bec4d
