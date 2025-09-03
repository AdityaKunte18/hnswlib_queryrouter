#ifndef UTILS_HPP
#define UTILS_HPP

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <cstdio>
#include <fstream> 
#include <unordered_map>
#include <float.h>
#include <unordered_set>
#include <algorithm>
#include <random>
#include <zlib.h>

/* ReadFvecsRandom
 *  
 *  samples N random columns without replacement 
 *  filename: string which points to the filepath 
 *  N: total number of column vectors to read (one data point is one column vector, and the rows are features)
 *  returns:
 *    a 2-d array of floats
 */
std::vector<std::vector<float>> ReadFvecsRandom(const std::string& filename, int N);


// used gpt to read in a .gz file as a temporary workaround
std::vector<std::vector<float>> ReadFirst10MBvecsGz(const std::string& gz_filename);

//runs k-means with hyperparameter K and max iterations
std::vector<std::vector<float>> RunKmeans(std::vector<std::vector<float>>& arr, int k=5, int max_iters=1000);

//distance between two fl point vectors
long long sqL2(const std::vector<float>& a, const std::vector<float>& b);
#endif
