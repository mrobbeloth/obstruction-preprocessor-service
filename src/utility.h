#include <iostream>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <chrono>
using namespace std;
using namespace std::chrono;
using namespace cv;


Mat sharpen(Mat input);
Mat unsharp_masking(Mat input);
Mat setInitialLabelsGrayscale(int width, int height, int k);

string printMatType(const Mat input);

//size_t number_of_files_in_directory(path path);


/*vector<string> findFiles(const string& directory, 
                                   const string& extension);*/


/*static path modifyFileName(path fn, string& appendStr, 
                                    string& extension);*/

//bool writeImagesToDisk(Mat imageData, string path, string fn);

//bool writeImagesToDisk(Mat imageData, string path, string fn, string appendStr, string format);