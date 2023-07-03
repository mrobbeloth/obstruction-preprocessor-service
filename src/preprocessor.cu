#include <opencv2/opencv.hpp>
#include <SuperString.hh>
#include <string>
#include <thread>
#include "utility.h"
using namespace std;
using namespace cv;

#include "threadpool.hpp"
#include <iostream>

using namespace astp;

int main() {
    ThreadPool tp; 
    for (int i = 0; i < 100; i++) {
        tp.push([i]() {
            std::cout << "ThreadPool " << i << std::endl; 
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