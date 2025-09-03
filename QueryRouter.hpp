#ifndef QUERY_ROUTER_HPP
#define QUERY_ROUTER_HPP

#include <memory>
#include "hnswlib/hnswlib/hnswlib.h"
#include "utils/utils.hpp"

struct HnswObj {
    std::unique_ptr<hnswlib::SpaceInterface<float>> space;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> index;
};

HnswObj build_hnsw(
    const std::vector<std::vector<float>>& arr,
    int M = 16,
    int efConstruction = 200,
    int efSearch = 64
);

std::vector<std::vector<int>> run_knn(
    const std::vector<std::vector<float>>& points,
    const HnswObj& graph,
    int k = 5
);

std::vector<std::vector<int>> get_closest_centroids(
    const std::vector<std::vector<float>>& centroids, 
    const std::vector<std::vector<int>>& nearest_nbrs,
     const std::vector<std::vector<float>>& points
);

#endif // QUERY_ROUTER_HPP
