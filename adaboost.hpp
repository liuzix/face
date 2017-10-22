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
    float weight = 1.0;
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
    int classify (Sample& sample);
};


Classifier AdaBoostTrain(std::vector<Sample>& samples, device_vector<Feature>& features);

#endif