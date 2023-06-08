#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include "utility.h"
using namespace std;
using namespace cv;
using namespace filesystem;
int main() {
    cout << "Starting PreProcessor Service" << endl;
    int fileRemoved = remove_all("../output");
    string path = "../data/coil-100/";
    bool result = create_directory("../output");
    vector<string> files =findFiles(path, ".png");
    for (const string& entry : files) {
        cout << "working with file:" << entry << endl;
        Mat img_grayscale = imread(path+entry, IMREAD_GRAYSCALE);
        cout << "Image Size:" << img_grayscale.size << endl;
        string outputFileName = "../output/"+entry;
        result = imwrite(outputFileName, img_grayscale);
        if (!result) {
            cout << "Failed to write " << outputFileName << endl;
        }
    }
   
    return 0;
}