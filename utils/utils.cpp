#include "utils.hpp"

//Each fvecs vector takes 4+d*4 bytes


// vector struct for fvecs: [int dimesion][<f_1><f_2>...<f_128>]|[int dimesion][<f_1><f_2>...<f_128>]
//128 cause sift dimensions are 128
std::vector<std::vector<float>> ReadFvecsRandom(const std::string& filename, int N) {
    FILE* filePointer = fopen(filename.c_str(), "rb");
    if (filePointer == NULL) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return {};
    }
    std::vector<std::vector<float>> res;

    int dimension = 0;
    size_t bytesRead = fread(&dimension, sizeof(int), 1, filePointer);
    

    std::cout << "Dimension : " << dimension << std::endl;
    
    fseek(filePointer, 0, SEEK_END);
    long fileSize = ftell(filePointer);
    fseek(filePointer, sizeof(int), SEEK_SET);
    long bytesPerVector = sizeof(int) + dimension * sizeof(float);
    long totalVectors = (fileSize) / bytesPerVector;
    std::cout << "Total vectors in file: " << totalVectors << std::endl;


    if (N > totalVectors) {N = (int)(totalVectors);}

    std::vector<long> indices(totalVectors);
    for (long i = 0; i < totalVectors; ++i) {indices[i] = i;}
    std::mt19937 rng(std::random_device{}());
    std::shuffle(indices.begin(), indices.end(), rng); //made a list of indices and shuffled them randomly
    indices.resize(N);
    std::sort(indices.begin(), indices.end());

    res.reserve(N);
    std::vector<float> buffer((size_t)(dimension));

     for (long idx : indices) {
        const long offset = idx * bytesPerVector;
        if (fseek(filePointer, offset, SEEK_SET) != 0) {
            std::cerr << "Error: fseek to vector " << idx << " failed.\n";
            res.clear();
            break;
        }

        int d_check = 0;
        if (fread(&d_check, sizeof(int), 1, filePointer) != 1) {
            std::cerr << "Error: Failed reading dimension for vector " << idx << ".\n";
            res.clear();
            break;
        }
        if (d_check != dimension) {
            std::cerr << "Error: Dimension mismatch at vector " << idx
                      << " (got " << d_check << ", expected " << dimension << ").\n";
            res.clear();
            break;
        }

        if (fread(buffer.data(), sizeof(float), (size_t)(dimension), filePointer)
            != (size_t)(dimension)) {
            std::cerr << "Error: Failed reading payload for vector " << idx << ".\n";
            res.clear();
            break;
        }

        res.emplace_back(buffer.begin(), buffer.end());
    }

    fclose(filePointer);
    return res;
}


std::vector<std::vector<float>> ReadFirst10MBvecsGz(const std::string& gz_filename) {
    // 10M vectors for HNSW takes a long time, need to use ParallelFor from the tutorials
    constexpr size_t TARGET_COUNT = 100000;  // 100k

    gzFile f = gzopen(gz_filename.c_str(), "rb");
    if (!f) {
        throw std::runtime_error("Failed to open gz file: " + gz_filename);
    }

    int32_t d = 0;
    if (gzread(f, &d, sizeof(int32_t)) != sizeof(int32_t) || d <= 0) {
        throw std::runtime_error("Bad bvecs header in gz stream");
    }

    std::vector<std::vector<float>> vectors;
    vectors.reserve(TARGET_COUNT);

    std::vector<uint8_t> payload((size_t)d);

    // Read the first payload (we already consumed its header)
    if (gzread(f, payload.data(), d) != d) {
        throw std::runtime_error("Truncated payload at first vector");
    }

    std::vector<float> firstVec(d);
    for (int i = 0; i < d; i++) {
        firstVec[i] = (float)(payload[i]);
    }
    vectors.push_back(std::move(firstVec));

    std::cout << "Dimension: " << d << std::endl;

    size_t count = 1;  // we already read one vector

    while (count < TARGET_COUNT) {
        int32_t d_check = 0;
        int hb = gzread(f, &d_check, sizeof(int32_t));
        if (hb == 0) {
            std::cerr << "Reached EOF after reading " << count << " vectors." << std::endl;
            break;  // EOF reached before 10M vectors
        }
        if (hb != sizeof(int32_t)) {
            throw std::runtime_error("Truncated header mid-stream");
        }
        if (d_check != d) {
            throw std::runtime_error("Dim mismatch in stream");
        }

        if (gzread(f, payload.data(), d) != d) {
            throw std::runtime_error("Truncated payload mid-stream");
        }

        std::vector<float> vec(d);
        for (int i = 0; i < d; i++) {
            vec[i] = (float)(payload[i]);
        }
        vectors.push_back(std::move(vec));
        ++count;

        if (count % 1'000'000ULL == 0) {
            std::cerr << "Read " << count << " vectors so far...\n";
        }
    }

    gzclose(f);
    return vectors;  // size = min(10M, total_vectors)
}


//ref: https://stackoverflow.com/questions/28287138/c-randomly-sample-k-numbers-from-range-0n-1-n-k-without-replacement

//todo 
//replace the index shuffling from ReadFvecsRandom with this?
static std::unordered_set<int> pickSet(int N, int k, std::mt19937& gen) {
    std::unordered_set<int> elems;
    elems.reserve(k);
    for (int r = N - k; r < N; ++r) {
        int v = std::uniform_int_distribution<>(0, r)(gen);
        if (!elems.insert(v).second) {
            elems.insert(r);
        }
    }
    return elems;
}

long long sqL2(const std::vector<float>& a, const std::vector<float>& b) {
    long long s = 0;
    const size_t d = a.size();
    for (size_t j = 0; j < d; ++j) {
        long long diff = (long long)(a[j]) - (long long)(b[j]);
        s += diff * diff;
    }
    return s;
}


std::vector<std::vector<float>> RunKmeans(std::vector<std::vector<float>>& arr, int k, int max_iters) {
    const int N = (int)(arr.size());
    const size_t d = arr[0].size();
    std::random_device rd;
    std::mt19937 gen(rd());
    auto chosen = pickSet(N, k, gen);

    std::vector<std::vector<float>> centroids;
    centroids.reserve(k);
    for (int idx : chosen) {
        centroids.push_back(arr[idx]);
    }

    // Main iteration loop
    for (int iter = 0; iter < max_iters; iter++) {
        if (iter == max_iters-1) {
            std::cout << "kmean: max iters hit " << std::endl;
        }
        // sums will store k x d
        std::vector<std::vector<double>> sums(k, std::vector<double>(d, 0.0));
        std::vector<int> counts(k, 0);

        // Assign each point to the nearest centroid
        for (const auto& x : arr) {
            int best_c = 0;
            double best_dist = DBL_MAX;
            for (int c = 0; c < k; ++c) {
                double dist = sqL2(x, centroids[c]);
                if (dist < best_dist) {
                    best_dist = dist;
                    best_c = c;
                }
            }

            counts[best_c] += 1;
            for (size_t j = 0; j < d; ++j) {
                sums[best_c][j] += x[j];
            }
        }

        // Update centroids
        bool converged = true;
        for (int c = 0; c < k; ++c) {
            if (counts[c] == 0) continue; 
            for (size_t j = 0; j < d; ++j) {
                float new_val = (float)(sums[c][j] / counts[c]);
                if (std::fabs(new_val - centroids[c][j]) > 1e-7) {
                    converged = false;
                }
                centroids[c][j] = new_val;
            }
        }

        if (converged) {
            std::cout << "converged adn exiting at iteration " << iter << std::endl;
            break;
        }
    }

    return centroids;
}
