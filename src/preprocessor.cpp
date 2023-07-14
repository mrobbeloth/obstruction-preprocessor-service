#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/core/cvstd.hpp>
#include <opencv2/cudaarithm.hpp>
#include <filesystem>
#include <string>
#include <chrono>
#include "utility.h"

using namespace std;
using namespace cv;
using namespace filesystem;
using namespace chrono;
using namespace cuda;

Mat ScanSegments(Mat I, bool debug) {

    // verify basic charcteristics of image
    int rows = I.rows;
    int cols = I.cols;
    int channels = I.channels();

    if (debug) {
        cout << "rows=" << rows << " cols=" << cols << " channels=" 
             << channels << endl;
    }

    // prepare a vector of binary segments from grayscale image
    vector<Mat> segments;

    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            if (I.at<uint8_t>(i,j) == 0) {
                I.at<uint8_t>(i,j) = 1;
            } 
        }
    }

    // convert the input image to double precision
    Mat Temp = I.clone();
    I.convertTo(Temp, I.type());

    // find the first non-zero location
    vector<Point> points = findInMat(I, 1, "first");

    int n = 1;
    int indx = -1;
    int indy = -1;

    if(points.size() > 0) {
        indx = points[0].x;
        indy = points[0].y;
    }

    // keep goint while we still have regions to process
    if (debug) {
        cout << "ScanSegments(): starting to process regions" << endl;
    }

    while (points.size() > 0) {
        int i = indx;
        int j = indy;

        /*pass the image segment to the region growing code along with the
          coordiantes of the seed and max intensity distance of 1x10e-5
        
        This tends to eat the k-means segmented image starting at the start
        pixel. When the original segmented image is consumed, then we are
        done scanning for segments */

        if(debug) {
            cout << "ScanSegments(): calling region growing code" << endl;
        }

        // vector<Mat> JAndTemp =
        //TODO regiongrowing code here

        /* TODO pad the array and opy the extracted image segment with its 
           grown region into it */
    }

    //TODO placeholder
    return Mat();
}

Mat opencv_kmeans_postProcess(Mat data, Mat labels, Mat centers) {
    // original java code includes check for three channels and
    // reshaping of data if needed, not sure it was ever used
    // originally thought I might use color RBG images probably 

    /* Cluster centers identify in image data are in signed 32-bit floating
       point mode and need to be converted to 8-bit unsigned mode */
    centers.convertTo(centers, CV_8U); 
    Mat res(labels.rows, labels.cols, CV_8U);

    /* Map each label to a cluster center */
    for(int i = 0; i < res.rows;  i++) {
            int label = labels.at<int>(i,0);
            res.at<uint8_t>(i,0) = centers.at<uint8_t>(label,0);
    }
    
    /* Turn partitioned data back into a format suitable as an 
       image*/
    Mat res2 = res.clone().reshape(1,data.rows);

    /* return partitioned image */
    return res2;
} 

int main(int argc, char* argv[]) {
    bool debugFlag = true;
    int k = 4;
    int kMeansIterations = 16;
    int flags = KMEANS_PP_CENTERS; // 0x2

    int GPUCnt = getCudaEnabledDeviceCount();
    cout << "Number of CUDA enabled devices: " << GPUCnt << endl;

    // start code timing 
    auto start = chrono::high_resolution_clock::now();

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

        GpuMat gpu_img_grayscale;
        if (GPUCnt > 0) {
            gpu_img_grayscale.upload(img_grayscale);
            cout << "Image Size:" << gpu_img_grayscale.size() << endl;
        }
        else {
            cout << "Image Size:" << img_grayscale.size() << endl;
        }
        
        if (debugFlag) {
            if (GPUCnt > 0) {
                gpu_img_grayscale.download(img_grayscale);
            }
            result = imageSave("../output/", entry, img_grayscale);
            if (!result) {
                cerr << "Failed to write " << img_grayscale << endl;
            }
        }            

        // Test for GPU
        if (GPUCnt > 0) {
            int type = gpu_img_grayscale.type();
            // Test if 8-bit unsigned grayscale
            if (type != CV_8U) {
                cout << "Converting " << entry << " to 8-bit unsigned grayscale" << endl;
                gpu_img_grayscale.convertTo(gpu_img_grayscale, CV_8U);
                gpu_img_grayscale.download(img_grayscale);
                if (debugFlag) {
                    result = imageSave("../output/", "8U_"+entry, img_grayscale);
                    if (!result) {
                        cerr << "Failed to write " << img_grayscale << endl;
                    }
                }            
            }
            else {
                cout << "Image is already 8-bit unsigned grayscale" << endl;
            }
        }
        else {
            // CPU only
            int type = img_grayscale.type();
            // Test if 8-bit unsigned grayscale
            if (type != CV_8U) {
                cout << "Converting " << entry << " to 8-bit unsigned grayscale" << endl;
                img_grayscale.convertTo(img_grayscale, CV_8U);
                if (debugFlag) {
                    result = imageSave("../output/", "8U_"+entry, img_grayscale);
                    if (!result) {
                        cerr << "Failed to write " << img_grayscale << endl;
                    }
                }         
            }
            else {
                cout << "Image is already 8-bit unsigned grayscale" << endl;
            }
        }

        /* create a state of the image before preprocessing is done to use later in 
           following sharpen operation */
        Mat img_duplicate = img_grayscale.clone();
        GpuMat gpu_img_duplicate; 
        GpuMat gpu_img_src, gpu_img_dst; 
        
        if (GPUCnt > 0) {
          gpu_img_src = gpu_img_grayscale.clone();
          gpu_img_duplicate = gpu_img_grayscale.clone();
        }

        /* Gaussian Blur image to reduce noise from original image. Will need
           to follow-up with edge detection 
           
           In the Java Code, I had also applied a non-local means denoising, 
           but believe that may have been overly aggressive, let's keep this 
           simple until we learn more, fastNlMeansDenoising could be used
           instead of this at some point in the future. 
           
           Alternatively we could use median blur or bilateral filter to
           try to reduce noise while keeping line edges */
        cout << "Applying Gaussian Blur" << endl;
        Mat gaussianApplied(img_grayscale.rows, img_grayscale.cols, 
                            img_grayscale.type());

        if (GPUCnt > 0) {
            Ptr<Filter> gaussianFilter = createGaussianFilter(
                gpu_img_src.type(), gpu_img_src.type(), Size(5,5), 0.0, 0.0, BORDER_DEFAULT, -1);
            gaussianFilter->apply(gpu_img_src, gpu_img_dst);
            gpu_img_dst.download(gaussianApplied);  
            if (debugFlag) {
                string fn = "Gaussian_"+entry;              
                result = imageSave("../output/", fn, gaussianApplied);
                if (!result) {
                    cerr << "Failed to write " << fn << endl;
                }
            }
        }
        else {
            GaussianBlur(img_grayscale, gaussianApplied, Size(5,5),0, 0, BORDER_DEFAULT);

            if (debugFlag) {
                string fn = "Gaussian_"+entry;   
                result = imageSave("../output/", fn, gaussianApplied);
                if (!result) {
                    cerr << "Failed to write " << fn << endl;
                }
            }

        }
        cout << "Done applying Gaussian Blur" << endl;


        /* follow up with sharpening */
        Mat sharpenApplied(img_grayscale.rows, img_grayscale.cols, 
                            img_grayscale.type());
        GpuMat gpu_img_sharpened;
        if (GPUCnt > 0) {
            gpu_img_sharpened = sharpenGPU(gpu_img_dst);
            gpu_img_sharpened.download(sharpenApplied);
            if (debugFlag) {
                cout << "Applied GPU Sharpen Bilateral filter" << endl;
                string fn = "Sharpen_"+entry;
                result = imageSave("../output/", fn, sharpenApplied);
                if (!result) {
                    cerr << "Failed to write " << fn << endl;
                }
            }
        }
        else {
            sharpenApplied = sharpen(gaussianApplied);
            if (debugFlag) {
                cout << "Applied CPU Sharpen Bilateral filter" << endl;
                string fn = "Sharpen_"+entry;
                result = imageSave("../output/", fn, sharpenApplied);
                if (!result) {
                    cerr << "Failed to write " << fn << endl;
                }
            }
        }

        // Merge original image with preprocessed image for clearest shape
        Mat mergedMat(img_grayscale.rows, img_grayscale.cols, 
                    img_grayscale.type());
        GpuMat gpuMergedMat;
        if (GPUCnt > 0) {  
            cv::cuda::addWeighted(gpu_img_duplicate, 1.5, gpu_img_sharpened,-0.5, 0, gpuMergedMat);
            gpuMergedMat.download(mergedMat);
            if (debugFlag) {
                cout << "Used GPU to merge original image with preprocessed image" << endl;
                string fn = "Merged_"+entry;
                result = imageSave("../output/", fn, mergedMat);
                if (!result) {
                    cerr << "Failed to write " << fn << endl;
                }
            }
        }
        else {
            Mat mergedMat(img_grayscale.rows, img_grayscale.cols, 
                                img_grayscale.type());
            cv::addWeighted(img_duplicate, 1.5, sharpenApplied,-0.5, 0, mergedMat);

            if (debugFlag) {
                cout << "Used CPU to merge original image with preprocessed image" << endl;
                string fn = "Merged_"+entry;
                result = imageSave("../output/", fn, mergedMat);
                if (!result) {
                    cerr << "Failed to write " << fn << endl;
                }
            }
        }

        // Need to convert data to 32F for kmeans partitioning  
        // This code section is not amenable to GPU processing  
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
        double compactness =
            kmeans(colVecFloat,k,labels, criteria, criteria.maxCount, flags, centers);
        cout << "Compactness=" << compactness << endl;
    
        Mat partitionedImage = opencv_kmeans_postProcess(mergedMat, labels, centers);
        if (debugFlag) {
            string fn = "Partitioned_"+entry;
            result = imageSave("../output/", "Partitioned_"+entry, partitionedImage);
            if (!result) {
                cerr << "Failed to write " << fn << endl;
            }
        }

    }

    /* Scan partitioned or clustered data and produce one binary image for each segment */


    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "Preprocessing Execution time: " << duration.count() << " ms" << endl;
    return 0;
}