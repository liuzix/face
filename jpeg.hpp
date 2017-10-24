#ifndef JPEG_H
#define JPEG_H

#include <cstdlib>
#include <cassert>
#include <vector>
#define cimg_use_jpeg 
#include <CImg.h>

#include "cudaManaged.hpp"
using namespace cimg_library;

/* This class provides the tool for accessing grey scale image */
class JPEGImage : public Managed {
private:
    unsigned char* originalData = nullptr;
    unsigned char* grayScaleData = nullptr;
    unsigned int* rows = nullptr;
    unsigned int* integral = nullptr;
    bool integrated = false;
public:
    friend class Sample;
    int dimX = 0;
    int dimY = 0;

    JPEGImage () {};

    ~JPEGImage ();

    explicit JPEGImage (const char* fileName) {
        load (fileName);
    }

    explicit JPEGImage (CImg<unsigned char>& image, int x1, int y1);

    JPEGImage (JPEGImage && other) {
        originalData = other.originalData;
        grayScaleData = other.grayScaleData;
        rows = other.rows;
        integral = other.integral;
        integrated = other.integrated;
        dimX = other.dimX;
        dimY = other.dimY;

        other.originalData = nullptr;
        other.grayScaleData = nullptr;
        other.rows = nullptr;
        other.integral = nullptr;
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

std::vector<JPEGImage> getWindows (const char* fileName) ;

#endif