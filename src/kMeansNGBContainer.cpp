#include "kMeansNGBContainer.h"

kMeansNGBContainer::kMeansNGBContainer(const Mat& clustered_data,
                                       const map<string, Mat>& stats) {
    this->clustered_data = clustered_data.clone();
    for (const auto& kv : stats) {
        this->stats[kv.first] = kv.second;
    }
}

Mat kMeansNGBContainer::getClustered_data() const {
    return clustered_data;
}

map<string, Mat> kMeansNGBContainer::getStats() const {
    return stats;
}
