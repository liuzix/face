#ifndef FEATURE_H
#define FEATURE_H

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "jpeg.hpp"

using namespace thrust;

typedef unsigned char uchar;

class Sample ;

class Feature {
public:
    uchar x1, y1, x2, y2;

    __host__ __device__ int compute(Sample& sample);

    Feature(uchar _x1, uchar _y1, uchar _x2, uchar _y2) {
        x1 = _x1;
        x2 = _x2;
        y1 = _y1;
        y2 = _y2;
    }

    __host__ __device__ Feature(const Feature& other) {
        x1 = other.x1;
        x2 = other.x2;
        y1 = other.y1;
        y2 = other.y2;
    }

    // to make nvcc happy
    __host__ __device__ ~Feature() {}

    static device_vector<Feature> generate(int dimX, int dimY);
};

#endif