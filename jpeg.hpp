#ifndef JPEG_H
#define JPEG_H

#include <cstdlib>
#include <cassert>


#include "cudaManaged.hpp"


/* This class provides the tool for accessing grey scale image */
class JPEGImage : public Managed {
private:
    unsigned char* originalData = nullptr;
    unsigned char* grayScaleData = nullptr;
    unsigned int* rows = nullptr;
    unsigned int* integral = nullptr;
public:
    friend class Sample;
    int dimX = 0;
    int dimY = 0;

    JPEGImage () {};

    ~JPEGImage ();

    explicit JPEGImage (const char* fileName) {
        load (fileName);
    }

    void load (const char* fileName);

    __host__ __device__ void toGray();
    
    __host__ __device__ inline unsigned int& at(int x, int y) {
        assert(integral);
        assert(0 <= x && x < this->dimX && 0 <= y && y < this->dimY);
        return integral[x + this->dimX * y];
    }

    __host__ __device__ void integrate();

};


__global__ void batchToGray(JPEGImage* input);

#endif