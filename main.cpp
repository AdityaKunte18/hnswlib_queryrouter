#include "utils/utils.hpp"
#include "QueryRouter.hpp"

int main(int argc, char** argv) {
    const std::string fvecs =  "/Users/adityakunte/Desktop/createlab/data/sift_1M/sift_base.fvecs";
    const std::string bvecs =  "/Users/adityakunte/Desktop/createlab/data/sift_1B/bigann_base.bvecs.gz";

    //vectors we will use to build the hnsw
    //we dont actually read 10 million, it depends on the TARGET_COUNT value inside this function
    //which is 100k
    std::vector<std::vector<float>> b_rand = ReadFirst10MBvecsGz(bvecs);
   
    
    std::cout << "bvecs Ncols: " << b_rand.size()  << "\n";
    const auto& first = b_rand[0];
    std::cout << "first vector size : " << first.size() << std::endl;
    for (size_t i = 0; i < first.size(); i++) {
        std::cout << (first[i]) << " ";
    }
    std::cout << "\n" << std::endl;

    // build hnsw
    HnswObj res = build_hnsw(b_rand);
    //run kmeans and get centroids
    std::vector<std::vector<float>> centroids = RunKmeans(b_rand, 5, 1000);
 
    std::cout << "first centroid size : " << centroids[0].size() << std::endl;
    for (size_t i = 0; i < first.size(); i++) {
        std::cout << (centroids[0][i]) << " ";
    }
    std::cout << "\n" << std::endl;

    //for each query point, run knn and store N candidates
    std::vector<std::vector<float>> queries = ReadFvecsRandom(fvecs, 20);
    std::vector<std::vector<int>> knn_indices = run_knn(queries,res);
    
    
    std::cout << "knn: fetched results for " << knn_indices.size() <<  " vectors ";
    std::cout << " first vec: " << std::endl;
    //knn[i] is a vector containing upto the k-nearest neighbors for queries[i]
    for (size_t i = 0; i < knn_indices[0].size(); i++) {
        std::cout << knn_indices[0][i] << " ";
    }
    std::cout << "\n" << std::endl;

    //after getting upto k candidates for each point, 
    //fetch the closest centroid for that point
    std::vector<std::vector<int>> closest_centroids =  get_closest_centroids(
        centroids,
        knn_indices,
        b_rand
    );

    std::cout << " closest centroid idxs for first query point " << std::endl;
    for (size_t i = 0; i < closest_centroids[0].size(); i++) {
        std::cout << closest_centroids[0][i] << " ";
    }
    std::cout << "\n" << std::endl;
    std::cout << "finished" << std::endl;
    return 0;
}
