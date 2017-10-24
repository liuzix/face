#include <iostream>
#include <string>
#include <cuda.h>
#include <math.h>
#include <thrust/device_vector.h>
#include <device_launch_parameters.h>

#include "jpeg.hpp"
#include "feature.hpp"
#include "adaboost.hpp"

#define NUM_IMAGES 750
#define THREADS_PER_BLOCK 500

using namespace std;

JPEGImage* faces;
JPEGImage* nonfaces;

/* The cuda kernel for transforming all images to greyscale */
__global__ void batchToGray(JPEGImage* input, int nImages) {
    int index = threadIdx.x + THREADS_PER_BLOCK * blockIdx.x;
    if (index > nImages ) return;
    input[index].toGray();
    input[index].integrate();
}

void loadImages () {
    faces = new JPEGImage[NUM_IMAGES];
    for (int i = 0; i < NUM_IMAGES; i++) {
        faces[i].load((string("faces/face") + to_string(i) + ".jpg").c_str());
    }

    batchToGray <<< ceilf((float)NUM_IMAGES / THREADS_PER_BLOCK), THREADS_PER_BLOCK >>> (faces, NUM_IMAGES);

    nonfaces = new JPEGImage[NUM_IMAGES];
    for (int i = 0; i < NUM_IMAGES; i++) {
        nonfaces[i].load((string("background/") + to_string(i) + ".jpg").c_str());
    }

    batchToGray <<< ceilf((float)NUM_IMAGES / THREADS_PER_BLOCK), THREADS_PER_BLOCK >>> (nonfaces, NUM_IMAGES);

    cudaDeviceSynchronize();
    CHECK(cudaPeekAtLastError());
}

std::vector<Sample> getFinalSamples() {
    auto jpegs = getWindows ("class.jpg");
    JPEGImage* p_jpegs = new JPEGImage[jpegs.size()];

    // I know this is not good practice....
    memcpy(p_jpegs, jpegs.data(), sizeof (JPEGImage) * jpegs.size());

    batchToGray <<<
        ceilf((float)jpegs.size() / THREADS_PER_BLOCK), THREADS_PER_BLOCK 
        >>> (p_jpegs, jpegs.size());
    
    cudaDeviceSynchronize();
    CHECK(cudaPeekAtLastError());

    std::vector<Sample> ret;
    for (int i = 0; i < jpegs.size(); i++) {
        ret.emplace_back(p_jpegs[i], 0);
    }

    return ret;
}





int main () {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, (size_t)((double)1.5 * 1024 * 1024 * 1024)); // limit = 1.5B
    cout << "Number of devices: " << deviceCount << endl;
    cout << "Loading and integrating images..." << endl;
    loadImages();


    cout << "Getting final samples.." << endl;
    std::vector<Sample> finalSamples = getFinalSamples();
    cout << "Number of windows = " << finalSamples.size() << endl;


    cout << "Generating features..." << endl;
    auto features = Feature::generate(64, 64);
    std::vector<Sample> samples;
    samples.reserve(NUM_IMAGES * 2);
    for (int i = 0; i < NUM_IMAGES; i++) {
        samples.emplace_back(faces[i], 1);
        //cout << i << endl;
    }
    for (int i = 0; i < NUM_IMAGES; i++) {
        samples.emplace_back(nonfaces[i], -1);
        //cout << i << endl;
    }
    cudaDeviceSynchronize();

    vector<Classifier> layers;

    for (int i = 0; i < 100; i++) {
        cout << "Starting adaboost... Round " << i << endl;
        Classifier new_layer = AdaBoostTrain(samples, features);
        samples = new_layer.getFaces(samples);
        cout << "remaining faces: " << samples.size() << endl;
        if (samples.size() - NUM_IMAGES < 20) {
            break;
        }
    }



    for (Classifier& cl: layers) {
        finalSamples = cl.getFaces(finalSamples);
    }

    for (Sample& s: finalSamples) {
        cout << s.other_x << ", " << s.other_y << endl;
    }
    
    
    return 0;
}