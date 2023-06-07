#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
int main() {
    cout << "Starting PreProcessor Service" << endl;
    Mat img_grayscale = imread("data/coil-100/obj1__0.png", 0);
    imwrite("data/processed/grayscale.jpg", img_grayscale);
    return 0;
}