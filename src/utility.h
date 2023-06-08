#include <iostream>
#include <filesystem>
#include<algorithm>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace filesystem;
using namespace cv;
std::size_t number_of_files_in_directory(std::filesystem::path path);
std::vector<std::string> findFiles(const std::string& directory, 
                                   const std::string& extension);
Mat sharpen(Mat input);
Mat unsharp_masking(Mat input);