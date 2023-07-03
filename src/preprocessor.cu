#include <opencv2/opencv.hpp>
#include <iostream>
#include <SuperString.hh>
#include <string>
#include <thread>

#include "utility.h"
#include "threadpool.hpp"
#include "preprocessorKernels.h"

using namespace std;
using namespace cv;
using namespace astp;

typedef void** kernelptr;

void threadFunc(const int i) {
    // Allocate memory on host
    size_t size = sizeof(int);
    int* hostIn = new int(i);
    int* hostOut = new int();

    // Allocate memory on device
    int* kernelIn = nullptr;
    auto err = cudaMalloc((kernelptr)&kernelIn, size);

    int* kernelOut = nullptr;
    err = cudaMalloc((kernelptr)&kernelOut, size);

    // Copy to device memory
    err = cudaMemcpy(kernelIn, hostIn, size, cudaMemcpyHostToDevice);
    err = cudaMemcpy(kernelOut, hostOut, size, cudaMemcpyHostToDevice);

    // Call kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = 1;
    cout << "CUDA kernel launch with " << blocksPerGrid << " blocks of " << threadsPerBlock << " threads" << endl;

    multiplyByTen<<<blocksPerGrid, threadsPerBlock>>>(kernelIn, kernelOut);
    err = cudaGetLastError();

    // Results
    if (err != cudaSuccess) {
        cerr << "Failed to launch kernel (" << cudaGetErrorString(err) << ")!" << endl;
        exit(EXIT_FAILURE);
    } 
    else {
        // Copy from device memory
        err = cudaMemcpy(hostOut, kernelOut, size, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            cerr << "Failed to copy result from device to host (" << cudaGetErrorString(err) << ")!" << endl;
            exit(EXIT_FAILURE);
        }

        // Print result
        cout << "ThreadPool " << *hostOut << endl;
    }

    // Free GPU memory
    err = cudaFree(kernelIn);
    err = cudaFree(kernelOut);

    // Free host memory
    free(hostIn);
    free(hostOut);
}

int main() {
    ThreadPool tp; 
    for (int i = 0; i < 100; i++) {
        tp.push([i]() {
            threadFunc(i);
        });
    }

    tp.wait();
    return 0;
}























/*
    ThreadPool pool(500);
    std::vector<std::future<thread::id>> results;

    auto now = system_clock::now();
    auto now_ms = time_point_cast<milliseconds>(now);

    auto value = now_ms.time_since_epoch();

    int processes = 10;
    int finish_flags = 0;
    while (finish_flags < processes) {
        for (int i = 0; i < processes; ++i) {
            auto future = pool.enqueue([i] {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                cout << i << endl;
                return this_thread::get_id();
            });
            finish_flags++;
            results.emplace_back(std::move(future));
        }
    }

    auto new_now = system_clock::now();
    auto new_now_ms = time_point_cast<milliseconds>(new_now);

    auto new_value = new_now_ms.time_since_epoch();
    long time_taken = (new_value - value).count();

    cout << "Time taken: " << time_taken << endl;

    return 0;
    */