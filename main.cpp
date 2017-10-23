#include <iostream>
#include <string>
#include <cuda.h>


#include "jpeg.hpp"
#include "feature.hpp"
#include "adaboost.hpp"

#define NUM_IMAGES 500
#define THREADS_PER_BLOCK 500

using namespace std;

JPEGImage* faces;
JPEGImage* nonfaces;

void loadImages () {
    faces = new JPEGImage[NUM_IMAGES];
    for (int i = 0; i < NUM_IMAGES; i++) {
        faces[i].load((string("faces/face") + to_string(i) + ".jpg").c_str());
    }

    batchToGray <<< NUM_IMAGES / THREADS_PER_BLOCK, THREADS_PER_BLOCK >>> (faces);

    nonfaces = new JPEGImage[NUM_IMAGES];
    for (int i = 0; i < NUM_IMAGES; i++) {
        nonfaces[i].load((string("background/") + to_string(i) + ".jpg").c_str());
    }

    batchToGray <<< NUM_IMAGES / THREADS_PER_BLOCK, THREADS_PER_BLOCK >>> (nonfaces);

    cudaDeviceSynchronize();
    

}

int main () {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    cout << "Number of devices: " << deviceCount << endl;
    cout << "Loading and integrating images..." << endl;
    loadImages();

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
    }

    
    return 0;
}