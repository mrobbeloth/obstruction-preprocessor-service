#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>
using namespace std;
using namespace cv;
using namespace filesystem;
int main() {
    cout << "Starting PreProcessor Service" << endl;
    int fileRemoved = remove_all("../output");
    cout << "Removed " << fileRemoved << endl;
    string path = "../data/coil-100";
    bool result = create_directory("../output");
    for (const auto & entry : filesystem::directory_iterator(path)) {
        Mat img_grayscale = imread(entry.path(), IMREAD_GRAYSCALE);
        cout << "Image Size:" << img_grayscale.size << endl;
        //string output_path = 
        //result = imwrite("../output/"+entry.path().string().substr, img_grayscale);
        //cout << "Result=" << result << endl;
    }
   
    return 0;
}