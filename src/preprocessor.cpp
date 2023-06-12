#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>
#include <filesystem>
#include <string>
#include "utility.h"

using namespace std;
using namespace cv;
using namespace filesystem;

void opencv_kmeans_postProcess(Mat data, Mat labels, Mat centers) {
    double *minVal;
    double *maxVal;

    // original java code includes check for three channels and
    // reshaping of data if needed, not sure it was ever used
    // originally thought I might use color RBG images probably 


    /* Setup data structure holding partitioned image data */
    //Mat clustered_data(data.rows, data.cols, data.type(), Scalar(0));
    centers.convertTo(centers, CV_8U); 
    cout << centers << endl;
    printMatType(centers);
    printMatType(labels);
    cout << centers.size().width << endl;
    cout << centers.size().height << endl;
    Mat res(labels.rows, labels.cols, CV_8U);
    cout << "Res rows "  << labels.rows << endl;
    cout << "Res cols "  << labels.cols << endl;
    cout << "Res type ";
    printMatType(res); 
    cout << "Label " << labels.size().width << " by " << labels.size().height << endl;
    for(int i = 0; i < res.rows;  i++) {
        cout << "i=" << i << endl;
            int label = labels.at<int>(i,0);
            cout << label << endl;
            res.at<uint8_t>(i,0) = centers.at<uint8_t>(label,0);
            cout << "UGH" << res.at<uint8_t>(i,0) << endl;
    }
    
    string fn = "blah.jpg";
    string fn2 = "blah2.jpg";
    Mat res2 = res.clone().reshape(1,data.rows);
    cout << (int)res2.at<uint8_t>(0,0) << endl;
    cout << (int)res2.at<uint8_t>(0,1) << endl;
    cout << (int)res2.at<uint8_t>(0,2) << endl;
    cout << (int)res2.at<uint8_t>(0,3) << endl;
    cout << res2.size().height << endl;
    cout << res2.size().width << endl;
    imwrite(fn2, data);
    imwrite(fn,res2);
    exit(0);
    // stats to keep here?

    //Size imageSize = data.size();
    //minMaxLoc(labels, minVal, maxVal, NULL);
    return;
} 

int main(int argc, char* argv[]) {
    bool debugFlag = false;
    int k = 4;
    int kMeansIterations = 16;
    int flags = KMEANS_PP_CENTERS; // 0x2

    cout << "Starting PreProcessor Service" << endl;
    // check if debug mode is turned on or not
    cout << "argc: " << argc << endl;
    if (argc > 1) {     
        if (argc > 2) {
            if (argc > 3) {
                string param3 = string(argv[3]);
                cout << "Max kMeans Iterations = " << param3 << endl;
                kMeansIterations = stoi(param3);
            }
            string param2 = string(argv[2]);
            cout << "K=" << param2 << endl;
            k = stoi(param2);
        }
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

        /* create a state of the image before preprocessing is done to use later in 
           following sharpen operation */
        Mat img_duplicate = img_grayscale.clone();

        /* Gaussian Blur image to reduce noise from original image. Will need
           to follow-up with edge detection 
           
           In the Java Code, I had also applied a non-local means denoising, 
           but believe that may have been overly aggressive, let's keep this 
           simple until we learn more, fastNlMeansDenoising could be used
           instead of this at some point in the future. 
           
           Alternatively we could use median blur or bilateral filter to
           try to reduce noise while keeping line edges */
        Mat gaussianApplied(img_grayscale.rows, img_grayscale.cols, 
                            img_grayscale.type());
        GaussianBlur(img_grayscale, gaussianApplied, Size(5,5),0, 0, BORDER_DEFAULT);
        if (debugFlag) {
            string outputFileName = "../output/Gaussian_"+entry;
            result = imwrite(outputFileName, gaussianApplied);
            if (!result) {
                cout << "Failed to write " << outputFileName << endl;
            }
        }

        /* follow up with sharpening */
        Mat sharpenApplied(img_grayscale.rows, img_grayscale.cols, 
                            img_grayscale.type());
        sharpenApplied = sharpen(gaussianApplied);
        if (debugFlag) {
            string outputFileName = "../output/Sharpen_"+entry;
            result = imwrite(outputFileName, gaussianApplied);
            if (!result) {
                cout << "Failed to write " << outputFileName << endl;
            }
        }

        // Merge original image with preprocessed image for clearest shape
        Mat mergedMat(img_grayscale.rows, img_grayscale.cols, 
                            img_grayscale.type());
        addWeighted(img_duplicate, 1.5, sharpenApplied,-0.5, 0, mergedMat);
        if (debugFlag) {
            string outputFileName = "../output/Merged_"+entry;
            result = imwrite(outputFileName, gaussianApplied);
            if (!result) {
                cout << "Failed to write " << outputFileName << endl;
            }
        }

        // Need to convert data to 32F for kmeans partitioning
        Mat matForKMeans(mergedMat.rows, mergedMat.cols, CV_32F);
        mergedMat.convertTo(matForKMeans, CV_32F);

        // Flatten image data for kmeans
        Mat colVec = matForKMeans.clone().reshape(
            1, matForKMeans.rows*matForKMeans.cols);
        Mat colVecFloat;
        colVec.convertTo(colVecFloat, CV_32F);

        Mat labels = setInitialLabelsGrayscale(matForKMeans.rows, matForKMeans.cols, 
                                                k);
        TermCriteria criteria(TermCriteria::Type::EPS+TermCriteria::Type::MAX_ITER,
                              kMeansIterations, 1.0);
        
        /*criteria is no more than x change in pixel centers or y iterations
          attempts is number of times to use algorithm using different init labeling
          
          will return labels with best compactness measure

            labels -- i/o integer array that store the cplsuter indices for every 
                        sample

            centers -- output matrix of the cluster centers, one row per each cluster
                        center

        Note: this does not change the image data sent to the array, the clustering
              of image data iteself has to be done in a post-processing array
              or what was called opencv_kmeans_PostProcess in original Java code
        */
        Mat centers;
        cout << colVecFloat.size().height << " by " << colVecFloat.size().width << endl;
        double compactness =
            kmeans(colVecFloat,k,labels, criteria, criteria.maxCount, flags, centers);
        cout << "Compactness=" << compactness << endl;
        cout << centers << endl;

        opencv_kmeans_postProcess(mergedMat, labels, centers);
    }
    return 0;
}