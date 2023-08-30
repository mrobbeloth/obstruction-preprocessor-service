#include <iostream>
#include <filesystem>
#include<algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/core/cvstd.hpp>
#include <opencv2/imgproc.hpp>
using namespace std;
using namespace cv;
using namespace filesystem;
using namespace chrono;
using namespace cuda;

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