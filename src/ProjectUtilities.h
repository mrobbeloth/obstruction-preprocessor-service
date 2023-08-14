#include </usr/local/cuda/targets/x86_64-linux/include/cuda_runtime.h>
#pragma once
#include <iostream>
#include <stdlib.h>
#include <thread>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <time.h>
#include <iterator>
#include <list>
#include <set>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include <opencv2/dnn.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/cvstd.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/cvdef.h>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/gapi/own/scalar.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/photo/photo.hpp>

#include <rapidfuzz/fuzz.hpp>
#include <parallel_hashmap/phmap.h>
#include <plplot/plplot.h>
#include <OpenXLSX/OpenXLSX.hpp>

#include "BS_thread_pool.hpp"

using namespace std;
using namespace cv;
using namespace filesystem;
using namespace chrono;
using namespace BS;
using namespace cuda;

namespace ProjectUtilities {
    std::size_t number_of_files_in_directory(std::filesystem::path path);
    std::vector<std::string> findFiles(const std::string& directory, 
                                    const std::string& extension);
    Mat sharpen(Mat input);
    GpuMat sharpenGPU(GpuMat input);
    Mat unsharp_masking(Mat input);
    Mat setInitialLabelsGrayscale(int width, int height, int k);
    string printMatType(const Mat input);
    bool imageSave(string path, string fn, Mat imageData);
    vector<Point> findInMat(Mat input, int k, string direction);
    Mat* setInitialLabelsGrayscaler(int width, int height, int k);
    enum class Partitioning_Algorithm { OPENCV, NGB, NONE };
    long DetermineNodeSize(Mat border);
    bool isInside(Mat nonZeroBorderPoints, int n, Point p);
    Mat returnEveryNthElement(Mat p, int n);
    bool doIntersect(Point p1, Point q1, Point p2, Point q2);
    int orientation(Point p, Point q, Point r);
    bool onSegment (Point p, Point q, Point r);
    //Mat autoCropGrayScaleImage(Mat segment, bool apply_threshold);
    vector<double> convertMat1xn(vector<Mat> MatAL, bool duplicateBegEnd);

    template <typename K, typename V>
    string unorderedMapToString(const unordered_map<K, V>& map);

    template <typename T>
    string vecToString(const vector<T>& vec);

    string matToString(const Mat& mat);

    template <typename T = milliseconds>
    string currentTime();

    long convert_ns(long tic, long toc, string time_unit);
    long convert_ns(long time, string time_unit);
    long convert_mcs(long time, string time_unit);
}