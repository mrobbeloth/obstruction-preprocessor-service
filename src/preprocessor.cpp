#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <string>
#include "utility.h"

using namespace std;
using namespace cv;
using namespace filesystem;

int main(int argc, char* argv[]) {
    bool debugFlag = false;
    cout << "Starting PreProcessor Service" << endl;
    // check if debug mode is turned on or not
    cout << "argc: " << argc << endl;
    if (argc > 1) {     
       string param1 = string(argv[1]);
       cout << param1 << endl;
       if (param1 == "--debug") {
            cout << "Debug flag set" << endl;
            debugFlag = true;
       }
    }
    int fileRemoved = remove_all("../output");
    string path = "../data/coil-100/";
    bool result = create_directory("../output");
    vector<string> files =findFiles(path, ".png");
    for (const string& entry : files) {     
        // convert to grayscale
        cout << "working with file:" << entry << endl;
        Mat img_grayscale = imread(path+entry, IMREAD_GRAYSCALE);
        cout << "Image Size:" << img_grayscale.size << endl;
        string outputFileName = "../output/"+entry;
        if (debugFlag) {
            result = imwrite(outputFileName, img_grayscale);
            if (!result) {
                cout << "Failed to write " << outputFileName << endl;
            }
        }
        // Test if 8-bit unsigned
        int type = img_grayscale.type();
        if (type != CV_8U) {
            cout << "Converting " << entry << " to 8-bit unsigned grayscale" << endl;
            img_grayscale.convertTo(img_grayscale, CV_8U);

            if (debugFlag) {
                string outputFileName = "../output/8U_"+entry;
                result = imwrite(outputFileName, img_grayscale);
                if (!result) {
                    cout << "Failed to write " << outputFileName << endl;
                }
            }
        }

    }
   
    return 0;
}