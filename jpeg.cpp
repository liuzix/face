#include "jpeg.hpp"
#define cimg_use_jpeg

#include <iostream>
#include <CImg.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

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
}

JPEGImage::~JPEGImage () {
    cudaDeviceSynchronize();
    cudaFree(this->originalData);
    cudaFree(this->grayScaleData);
    cudaFree(this->rows);
    cudaFree(this->integral);
}

/* The cuda kernel for transforming all images to greyscale */
__global__ void batchToGray(JPEGImage* input) {
    int index = threadIdx.x + blockIdx.x;
    input[index].toGray();
    input[index].integrate();
}