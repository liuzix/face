#include <iostream>
#include <vector>

#include "feature.hpp"
#include "adaboost.hpp"

__device__ int Feature::compute(Sample& sample) {
    unsigned int s1 = sample.at(x2, y2);
    unsigned int s2 = sample.at((x1 + x2) / 2, y1);
    unsigned int s3 = sample.at(x2, y1);
    unsigned int s4 = sample.at((x1 + x2) / 2, y2);
    
    int sum = s1 + s2 - s3 - s4;

    s1 = sample.at((x1 + x2) / 2, y2);
    s2 = sample.at(x1, y1);
    s3 = sample.at((x1 + x2) / 2, y1);
    s4 = sample.at(x1, y2);


    return s1 + s2 - s3 - s4 - sum;
}

device_vector<Feature> Feature::generate(int dimX, int dimY) {
    std::vector<Feature> ret;
    ret.reserve(dimX * dimY * dimX);
    for (int x1 = 0; x1 < dimX; x1+=2) {
        for (int y1 = 0; y1 < dimY; y1+=2) {
            for (int x2 = x1 + 2; x2 < dimX; x2+=2) {
                for (int y2 = y1 + 2; y2 < dimY; y2+=2) {
                    Feature f(x1, y1, x2, y2);
                    ret.push_back(f);
                }
            }
        }
    }

    printf("Generated %ld features\n", ret.size());
    std::cout << "Copying features to device..." << std::endl;
    device_vector<Feature> dev = ret;
    cudaDeviceSynchronize();
    return dev;
}