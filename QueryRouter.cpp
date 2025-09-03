#include "QueryRouter.hpp"

HnswObj build_hnsw(const std::vector<std::vector<float>>& arr,int M,
    int efConstruction,int efSearch) {
    const size_t n = arr.size();
    const size_t d = arr[0].size();

    HnswObj out;
    out.space = std::make_unique<hnswlib::L2Space>((int)d);
    out.index = std::make_unique<hnswlib::HierarchicalNSW<float>>(out.space.get(), n, M, efConstruction);
    // in the tutorials it expects a floar
    for (size_t i = 0; i < n; ++i) {
        const auto& v = arr[i];
        out.index->addPoint(v.data(), (size_t)(i)); //label is the index
    }
    out.index->setEf(efSearch);
    return out;
}


std::vector<std::vector<int>> run_knn(const std::vector<std::vector<float>>& points, const HnswObj& graph, int k) {
    const size_t N = points.size();
    const size_t dim = points[0].size();
    

    if (k <= 0) return std::vector<std::vector<int>>(N); 
    
    std::vector<std::vector<int>> knn_indices(N);

    for (size_t i = 0; i < N; ++i) {
        auto pq = graph.index->searchKnn(points[i].data(), (size_t)(k));
        std::vector<int> nbr;
        nbr.reserve(k);
        while (!pq.empty() && (int)(nbr.size()) < k) {
            auto [dist, label] = pq.top();
            pq.pop();
            if ((size_t)(label) == i) continue;
            nbr.push_back((int)(label));
        }

        knn_indices[i] = std::move(nbr);
    }
    return knn_indices;
}


std::vector<std::vector<int>> get_closest_centroids(
    const std::vector<std::vector<float>>& centroids, 
    const std::vector<std::vector<int>>& nearest_nbrs,
    const std::vector<std::vector<float>>& points
) {
   
    const size_t d = centroids[0].size();
    std::vector<std::vector<int>> out;
    out.resize(nearest_nbrs.size());

    for (size_t query_idx = 0; query_idx < nearest_nbrs.size(); query_idx++) {
        const auto& nbrs = nearest_nbrs[query_idx];
        auto& row = out[query_idx];
        row.resize(nbrs.size());

        for (size_t j = 0; j < nbrs.size(); j++) {
            const int point_idx = nbrs[j];  // neighbor point id for this query
            const auto& px = points[point_idx];
           
            int best_c = 0;
            double best_dist = DBL_MAX;
            for (int c = 0; c < static_cast<int>(centroids.size()); ++c) {
                double dist = (double)sqL2(px, centroids[c]);
                if (dist < best_dist) { best_dist = dist; best_c = c; }
            }
            row[j] = best_c; // store the closest centroid idx for this neighbor
        }
    }
    return out;
}