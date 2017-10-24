#ifndef ADABOOST_H
#define ADABOOST_H

#include <vector>
#include <cassert>

#include "cudaManaged.hpp"
#include "jpeg.hpp"
#include "feature.hpp"

class Sample {
    unsigned int* data = nullptr;
    int dimX, dimY;
public:
    int y;
    int other_x, other_y;
    explicit Sample (JPEGImage& jpeg, int _y) {
        assert(jpeg.integrated);
        this->dimX = jpeg.dimX;
        this->dimY = jpeg.dimY;
        this->other_x = jpeg.other_x;
        this->other_y = jpeg.other_y;
        this->data = jpeg.integral;
        jpeg.integral = nullptr;
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

    /* returns unweighted decision on a sample */
    __device__ inline int compute(Sample& sample) const {
        int f = feature->compute(sample);
        int v = f > threshold ? 1 : -1;
        return (int)polarity * v;
    }
};

struct Classifier {
    std::vector<DecisionStump> weakLearners;
    float threshold;
    float falsePositiveRate;

    std::vector<float> classify (device_vector<Sample>& samples);
    float getErrorRate(std::vector<Sample>& samples, std::vector<float> results);
    std::vector<Sample> getFaces (std::vector<Sample>& input);
};


Classifier AdaBoostTrain(std::vector<Sample>& samples, device_vector<Feature>& features);

#endif