#ifndef ADABOOST_H
#define ADABOOST_H

#include <vector>

#include "cudaManaged.hpp"
#include "jpeg.hpp"
#include "feature.hpp"

class Sample {
    unsigned int* data = nullptr;
    int dimX, dimY;
public:
    int y;

    explicit Sample (JPEGImage& jpeg, int _y) {
        //CHECK(cudaMalloc(&this->data, sizeof(unsigned int) * jpeg.dimX * jpeg.dimY));
        //CHECK(cudaMemcpy(this->data, jpeg.integral, sizeof(unsigned int) * jpeg.dimX * jpeg.dimY, cudaMemcpyDeviceToDevice));
        this->dimX = jpeg.dimX;
        this->dimY = jpeg.dimY;
        this->data = jpeg.integral;
        y = _y;
    }

    __host__ __device__ inline unsigned int& at(int x, int y) {
        return data[x + this->dimX * y];
    }
};


struct DecisionStump : public Managed {
    float threshold = 23333;
    char polarity = 1;
    int weight = 1;
    Feature* feature;

    /* returns weighted decision on a sample */
    __device__ inline int compute(Sample& sample) const {
        int f = feature->compute(sample);
        int v = f > threshold ? 1 : -1;
        return (int)polarity * v;
    }
};

struct Classifier {
    std::vector<DecisionStump> weakLearners;
    int threshold;

    // TODO: 
    // int classify (Sample& sample)
};


Classifier AdaBoostTrain(std::vector<Sample>& samples, device_vector<Feature>& features);

#endif