#pragma once
#include <map>
#include <string>
#include <opencv2/core/mat.hpp>

using namespace std;
using namespace cv;

/*!
 * Container holding the result of opencv_kmeans_postProcess.
 * Mirrors the Java kMeansNGBContainer class: a partitioned image
 * (clustered_data) plus per-cluster pixel-count statistics (stats).
 */
class kMeansNGBContainer {
private:
    Mat clustered_data;
    map<string, Mat> stats;

public:
    kMeansNGBContainer() = default;
    kMeansNGBContainer(const Mat& clustered_data, const map<string, Mat>& stats);

    Mat getClustered_data() const;
    map<string, Mat> getStats() const;
};
