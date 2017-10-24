#include "jpeg.hpp"
#define cimg_use_jpeg

#include <iostream>
#include <vector>
#include <CImg.h>
#include <cuda_runtime.h>



#include "cudaManaged.hpp"

using namespace cimg_library;

void JPEGImage::load (const char* fileName) {
    CImg<unsigned char> image(fileName);
    this->dimX = image.width();
    this->dimY = image.height();

    size_t s = sizeof(unsigned char) * image.size();
    CHECK(cudaMallocManaged(&this->originalData, s));
    memcpy(this->originalData, image.data(), s);
    CHECK(cudaMallocManaged(&this->grayScaleData, s));

    size_t si =  image.size() * sizeof(unsigned int) / 3;
    CHECK(cudaMallocManaged(&this->rows, si));
    CHECK(cudaMallocManaged(&this->integral, si));
}

__host__ __device__ void JPEGImage::toGray() {
    size_t stride = this->dimX * this->dimY * 3;

    for (int x = 0; x < this->dimX; x++) {
        for (int y = 0; y < this->dimY; y++) {
            u_char R = this->originalData[x + this->dimX * y];
            u_char G = this->originalData[x + this->dimX * y + stride];
            u_char B = this->originalData[x + this->dimX * y + 2 * stride];
            u_char ave = (R + G + B) / 3;
            this->grayScaleData[x + this->dimX * y] = ave;
        }
    }
}

__host__ __device__ void JPEGImage::integrate() {
    for (int x = 0; x < this->dimX; x++) {
        for (int y = 0; y < this->dimY; y++) {
            if (x == 0) {
                rows[x + dimX * y] = grayScaleData[x + dimX * y];
            } else {
                rows[x + dimX * y] = rows[x - 1 + dimX * y] + grayScaleData[x + dimX * y];
            }

            if (y == 0) {
                integral[x + dimX * y] = rows[x + dimX * y];
            } else {
                integral[x + dimX * y] = integral[x + dimX * (y-1)] + rows[x + dimX * y];
            }
        }
    }
    this->integrated = true;
}

JPEGImage::~JPEGImage () {
    cudaDeviceSynchronize();
    if (this->originalData)
        cudaFree(this->originalData);
    if (this->grayScaleData)
        cudaFree(this->grayScaleData);
    if (this->rows)
        cudaFree(this->rows);
    if (this->integral)
        cudaFree(this->integral);
    CHECK(cudaPeekAtLastError());
}

JPEGImage::JPEGImage (CImg<unsigned char>& image, int x1, int y1) {
    int x2 = x1 + 63;
    int y2 = y1 + 63;

    this->dimX = 64;
    this->dimY = 64;

    this->other_x = x1;
    this->other_y = y1;

    auto window = image.get_crop(x1, y1, x2, y2 );
    assert(window.width() == 64 && window.height() == 64);
    
    size_t s = sizeof(unsigned char) * window.size();
    CHECK(cudaMallocManaged(&this->originalData, s));
    memcpy(this->originalData, window.data(), s);
    CHECK(cudaMallocManaged(&this->grayScaleData, s));

    size_t si = window.size() * sizeof(unsigned int) / 3;
    CHECK(cudaMallocManaged(&this->rows, si));
    CHECK(cudaMallocManaged(&this->integral, si));
}

std::vector<JPEGImage> getWindows (const char* fileName) {
    CImg<unsigned char> image(fileName);
    std::vector<JPEGImage> ret;

    for (int x = 0; x + 64 < image.width(); x+=6) {
        for (int y = 0; y + 64 < image.height(); y+=6) {
            ret.emplace_back(image, x, y);
        }
    }

    return ret;
}