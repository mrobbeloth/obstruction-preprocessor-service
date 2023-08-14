#pragma once
#include <algorithm>
#include "includes.h"

class kMeansNGBContainer {
	private: 
        Mat* clustered_data;
        //unordered_map<string, Mat*> stats;
	
	public: 
        kMeansNGBContainer(Mat clustered_dat/*, unordered_map<string, Mat*> stats*/){
            *clustered_data = clustered_dat.clone();
            //stats = unordered_map<string, Mat*>(stats.size());

            /*unordered_set<string> keys;
            transform(stats.begin(), stats.end(), inserter(keys, keys.begin()), [](const auto& pair) {
                return pair.first;
            });*/

           /* for(string key : keys) {
                this->stats[key] = new Mat(*stats[key]);
            }*/
        }
        kMeansNGBContainer() {
            Mat* clustered_dat = new Mat();
            clustered_data = clustered_dat;
        }

        ~kMeansNGBContainer() {
            /*for (auto pair : stats) {
                delete pair.second;
            }
            stats.clear();*/
            delete clustered_data;
        }

        Mat getClustered_data() {
            return *clustered_data;
        }

        /*unordered_map<string, Mat*> getStats() {
            return stats;
        }	*/

        kMeansNGBContainer& operator=(const kMeansNGBContainer& other)
        {
            // Guard self assignment
            if (this->clustered_data == other.clustered_data) {
                return *this;
            }
            else {
                *(this->clustered_data) = *(other.clustered_data);
            } 
            return *this;
        }
};