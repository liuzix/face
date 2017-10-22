#include "adaboost.hpp"

#include <emmintrin.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

#define NUM_IMAGES 1000
#define THREADS_PER_BLOCK 512

#define SPLIT_NUM 10

struct weightValuePair {
    char v;
    float weight;
} ;

__global__ void evalOneFeature(size_t offset, DecisionStump* result, float* loss, 
                               Feature* features, 
                               Sample* samples, 
                               const float* weights,
                               int numFeatures,
                               float* keyBufs,
                               weightValuePair* wvBufs) {

    size_t index = offset + blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;

    if (blockIdx.x * THREADS_PER_BLOCK + threadIdx.x > numFeatures / SPLIT_NUM) return;
    if (index >= numFeatures) return;



    float* keyBuf = keyBufs + (blockIdx.x * THREADS_PER_BLOCK + threadIdx.x) * NUM_IMAGES;
    weightValuePair* wvBuf = wvBufs + (blockIdx.x * THREADS_PER_BLOCK + threadIdx.x) * NUM_IMAGES;

    for (int i = 0; i < NUM_IMAGES; i++) {
        keyBuf[i] = weights[i] * features[index].compute(samples[i]);
        wvBuf[i] = {.v = (char)samples[i].y, .weight = weights[i]};
    }

    __syncthreads();
    thrust::sort_by_key(thrust::seq, keyBuf, keyBuf + NUM_IMAGES, wvBuf);

    
    float T_plus = 0, T_minus = 0;
    for (int i = 0; i < NUM_IMAGES; i++) {
        if (wvBuf[i].v > 0) T_plus += wvBuf[i].weight;
        else T_minus += wvBuf[i].weight;
    }

    float S_plus = 0, S_minus = 0;
    float min_epsilon = 5000;
    int min_polarity;
    int min_index = 0;
    for (int i = 0; i < NUM_IMAGES; i++) {
        float epsilon = S_plus + T_minus - S_minus;
        if (epsilon < min_epsilon) {
            min_epsilon = epsilon;
            min_polarity = 1;
            min_index = i;
        }

        epsilon = S_minus + (T_plus - S_plus);
        if (epsilon < min_epsilon) {
            min_epsilon = epsilon;
            min_polarity = -1;
            min_index = i;
        }

        if (wvBuf[i].v > 0) S_plus += wvBuf[i].weight;
        else S_minus += wvBuf[i].weight;
    }


    
    result[index].feature = &features[index];
    result[index].threshold = keyBuf[min_index];
    assert(min_epsilon != 0);
    result[index].polarity = min_polarity;
    loss[index] = min_epsilon;
    //printf("epsilon = %f \n", min_epsilon);
}



Classifier AdaBoostTrain(std::vector<Sample>& samples, device_vector<Feature>& features) {
    std::cout << "Sample size = " << samples.size() << std::endl;
    device_vector<Sample> d_samples = samples;
    device_vector<Feature> d_features = features;
    device_vector<float> d_weights(samples.size(), 1);
    cudaDeviceSynchronize();
    Classifier ret;

    cudaDeviceSetLimit(cudaLimitMallocHeapSize, (size_t)((double)1.5 * 1024 * 1024 * 1024)); // limit = 1.5B

    for (int t = 0; t < 100; t++) {

        std::cout << "Adaboost: t = " << t << std::endl;
        DecisionStump* results = new DecisionStump[features.size()];
        float* loss;
        CHECK(cudaMallocManaged(&loss, sizeof(float) * features.size()));
        float* keyBufs;
        weightValuePair* wvBufs;
        CHECK(cudaMalloc(&keyBufs, sizeof(float) * (features.size() / SPLIT_NUM) * NUM_IMAGES));
        CHECK(cudaMalloc(&wvBufs, sizeof(weightValuePair) * (features.size() / SPLIT_NUM) * NUM_IMAGES));

        float* weights_p = thrust::raw_pointer_cast(d_weights.data());
        for (int i = 0; i < SPLIT_NUM + 1; i++) {


            printf("i = %d \n", i);


            size_t offset = (features.size() / SPLIT_NUM) * i;
            evalOneFeature <<< 
                ceil(((double)features.size() / SPLIT_NUM) / THREADS_PER_BLOCK), THREADS_PER_BLOCK 
                >>> (offset, results, loss, thrust::raw_pointer_cast(d_features.data()), 
                                    thrust::raw_pointer_cast(d_samples.data()),
                                    thrust::raw_pointer_cast(d_weights.data()),
                                    features.size(),
                                    keyBufs,
                                    wvBufs);
            
            cudaDeviceSynchronize();
            CHECK(cudaPeekAtLastError());

        }
        cudaFree(keyBufs);
        cudaFree(wvBufs);


        thrust::sort_by_key(thrust::device, loss, loss + features.size(), results);
        cudaDeviceSynchronize();
        CHECK(cudaPeekAtLastError());
        DecisionStump h_t = results[0];
        float sumWeight = thrust::reduce(thrust::device, weights_p, weights_p + samples.size());
        float normLoss = loss[0] / sumWeight;
        std::cout << "normalized loss = " << normLoss << std::endl;
        /*h_t.weight = 0.5 * log((1 - normLoss)/ normLoss);
        ret.weakLearners.push_back(h_t);

        float Z_t = 2 * sqrtf(normLoss * (1 - normLoss));

        cudaDeviceSynchronize();
        thrust::transform(thrust::device, d_weights.begin(), d_weights.end(), d_samples.begin(), d_weights.begin(),
        [=] __device__ (float D_t, Sample x_i) {
            return D_t * expf(-h_t.weight * x_i.y * h_t.compute(x_i));
        });
        cudaDeviceSynchronize();
        for (int i = 0; i < NUM_IMAGES; i++)
            std::cout << d_weights[i] << std::endl;
*/
        delete[] results;
    }

    return ret;
}