#include "compositeMat.hpp"
#include "includes.h"
#include "ProjectUtilities.h"
#include "TestFunctions.hpp"
using namespace ProjectUtilities;

//#include "ProjectUtilities.h"
// Utility.h must be included here to compile with the opencv files linked to preprocessor
// Include this when we finish the file
//  #include "LGAlgorithm.hpp"

/**
 * [build] /usr/bin/ld: CMakeFiles/preprocessor.dir/src/preprocessor.cpp.o: in function `preprocessTest(cv::Mat*, int, cv::Mat*, ProjectUtilities::Partitioning_Algorithm, int, bool, cv::Mat*, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, int, cv::TermCriteria, cv::Mat*)':
 [build] /home/riffle/git/obstruction-preprocessor-service/src/preprocessor.cpp:486: undefined reference to `cv::fastNlMeansDenoising(cv::_InputArray const&, cv::_OutputArray const&, float, int, int)'
*/

void preprocess (string, string, int, bool, bool, int, int, int);
void preprocessTest (Mat* clustered_data, int k, Mat* img_grayscale, Partitioning_Algorithm pa, int flags, bool debug_flag, Mat* labels, string filename, int attempts, TermCriteria criteria, Mat* centers);

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

Mat opencv_kmeans_postProcess1(Mat data, Mat labels, Mat centers) {
    // original java code includes check for three channels and
    // reshaping of data if needed, not sure it was ever used
    // originally thought I might use color RBG images probably 

    /* Cluster centers identify in image data are in signed 32-bit floating
      point mode and need to be converted to 8-bit unsigned mtring entry, string path, int GPUCnt, bool debugFlag, bool result, int kMetring entry, string path, int GPUCnt, bool debugFlag, bool result, int kMeansIterations, int flags, int k) {    ansIterations, int flags, int k) {    ode */
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
    bool debugFlag = false;
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

    // Start Multithreading Tasks for full utilization
    // Does not count extra files from the end of the database that aren't images
    int num_files = -2;
    for (auto& p : directory_iterator(path)) {
      num_files++;
    }
    thread_pool threads(num_files);
    cout << "Number of files to process: " << num_files << endl;
    for (int i = 0; i < num_files; i++) {
        string entry = files[i];
        threads.push_task(preprocess, entry, path, GPUCnt, debugFlag, result, kMeansIterations, flags, k);
    }
    
    /* Scan partitioned or clustered data and produce one binary image for each segment */
    threads.wait_for_tasks();
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "---------------------------------------" << endl;
    cout << "Preprocessing Execution time: " << duration.count() << " ms" << endl;
    cout << "Number of threads ran: " << threads.get_thread_count() << endl;
    cout << "---------------------------------------" << endl;

    cout << "Time to start test functions :D  :'(" << endl;
    auto startTest = chrono::high_resolution_clock::now();
    bool debug_flag = true;
    Mat* centers = new Mat();
    Partitioning_Algorithm pa = Partitioning_Algorithm::OPENCV;
    int attempts = 4;
    TermCriteria criteria(TermCriteria::EPS | TermCriteria::MAX_ITER, 20, 1.0);	
    
    // Deliverables
    Mat* labels = nullptr;
    // Start Multithreading Tasks for full utilization
    // Does not count extra files from the end of the database that aren't images
    num_files = -2;
    for (auto& p : directory_iterator(path)) {
        num_files++;
    }
    cout << "Number of files to process: " << num_files << endl;

    string filename = "TestImg";
    Mat* clustered_data = new Mat();
    
    for (int i = 0; i < num_files; i++) {
        string entry = files[i];
        Mat grayImg = imread(path+entry, IMREAD_GRAYSCALE);
        Mat* img_grayscale = &grayImg;
        // threads.push_task(preprocessTest, clustered_data, k, img_grayscale, pa, flags, debug_flag, labels, filename, attempts, criteria, centers);
        preprocessTest(clustered_data, k, img_grayscale, pa, flags, debug_flag, labels, filename, attempts, criteria, centers);
    }

    threads.wait_for_tasks();
    delete centers;
    delete labels;
    delete clustered_data;
    auto endTest = chrono::high_resolution_clock::now();
    auto testDuration = chrono::duration_cast<chrono::milliseconds>(endTest - startTest);
    cout << "---------------------------------------" << endl;
    cout << "Preprocessing Test Execution time: " << testDuration.count() << " ms" << endl;
    cout << "Number of threads ran: " << threads.get_thread_count() << endl;
    cout << "---------------------------------------" << endl;
    return 0;
}

void preprocess (string entry, string path, int GPUCnt, bool debugFlag, bool result, int kMeansIterations, int flags, int k) {    
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

    // create a state of the image before preprocessing is done to use later in 
        //following sharpen operation 
    Mat img_duplicate = img_grayscale.clone();
    GpuMat gpu_img_duplicate; 
    GpuMat gpu_img_src, gpu_img_dst; 
    
    if (GPUCnt > 0) {
        gpu_img_src = gpu_img_grayscale.clone();
        gpu_img_duplicate = gpu_img_grayscale.clone();
    }

    // Gaussian Blur image to reduce noise from original image. Will need
      //  to follow-up with edge detection 
        
      //  In the Java Code, I had also applied a non-local means denoising, 
      //  but believe that may have been overly aggressive, let's keep this 
      //  simple until we learn more, fastNlMeansDenoising could be used
      //  instead of this at some point in the future. 
        
      //  Alternatively we could use median blur or bilateral filter to
      //  try to reduce noise while keeping line edges 
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


    // follow up with sharpening 
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
    
    // criteria is no more than x change in pixel centers or y iterations
    //    attempts is number of times to use algorithm using different init labeling
    //    
    //    will return labels with best compactness measure
    //
    //    labels -- i/o integer array that store the cplsuter indices for every 
    //                sample
    //
    //    centers -- output matrix of the cluster centers, one row per each cluster
    //                center
    //
    //Note: this does not change the image data sent to the array, the clustering
    //        of image data iteself has to be done in a post-processing array
    //        or what was called opencv_kmeans_PostProcess in original Java code

    Mat centers;
    double compactness =
        kmeans(colVecFloat,k,labels, criteria, criteria.maxCount, flags, centers);
    cout << "Compactness=" << compactness << endl;

    Mat partitionedImage = opencv_kmeans_postProcess1(mergedMat, labels, centers);
    if (debugFlag) {
        string fn = "Partitioned_"+entry;
        result = imageSave("../output/", "Partitioned_"+entry, partitionedImage);
        if (!result) {
            cerr << "Failed to write " << fn << endl;
        }
    }

} // End multithreading for full utilization

void preprocessTest (Mat* clustered_data, int k, Mat* img_grayscale, Partitioning_Algorithm pa, int flags, bool debug_flag, Mat* labels, string filename, int attempts, TermCriteria criteria, Mat* centers) {
    // sanity check the number of clusters
    if (k < 2)
    {
        cerr << "The number of clusters must be greater than or equal to two." << endl;
        exit(1);
    }

    // sanity check that there is some data to work with
    if (img_grayscale->total() == 0)
    {
        cerr << "There must be some input data to work with for analysis." << endl;
        exit(2);
    }

    Mat* converted_data_8U = new Mat(img_grayscale->rows, img_grayscale->cols, CV_8U);
    img_grayscale->convertTo(*converted_data_8U, CV_8U);

    /* verify we have the actual full model image to work with
    * at the beginning of the process */
    if (debug_flag)
    {
        imwrite(String("../output/verify_full_image_in_ds_.jpg"),
                        *converted_data_8U);
    }

    if ((flags & KMEANS_USE_INITIAL_LABELS) == 0x1)
    {
        labels =
            setInitialLabelsGrayscaler(
                converted_data_8U->rows,
                converted_data_8U->cols, k);
                //converted_data_8U.height() instead of .cols originally
        cout << "Programming initial labels" << endl;
        cout << "Labels are:" << endl;
        cout << labels << endl;
    }
    else
    {
        labels = new Mat();
    }

    // start by smoothing the image -- let's get the obvious artifacts removed
    // start by smoothing the image -- let's get the obvious artifacts removed
    Mat* container = new Mat();
    long tic = stol(currentTime<nanoseconds>());
    // long tic = duration_cast<nanoseconds>(high_resolution_clock::now()).count();

    /* Aggressively sharpen and then remove noise */
    *converted_data_8U = sharpen(*converted_data_8U);
    if (debug_flag)
    {
        imwrite("../output/" + filename.substr(filename.find_last_of('/') + 1, filename.find_last_of('.')) + "_sharpen.jpg",
                        *converted_data_8U);
    }
    /* the h parameter here is quite high, 85, to remove lots of detail that
    * would otherwise generate extra segments from the clusters --
    * we loose fine details, but processing times are lower */
    cv::fastNlMeansDenoising(
        *converted_data_8U, *converted_data_8U, 85, 7, 21);
    imwrite("../output/" + filename.substr(filename.find_last_of('/') + 1, filename.find_last_of('.')) + "_denoise.jpg",
                    *converted_data_8U);
    /*CV_EXPORTS_W void fastNlMeansDenoising( InputArray src, OutputArray dst, float h = 3,     
                                            int templateWindowSize = 7, 
                                            int searchWindowSize = 21);*/

    // after smoothing, let's partition the image
    /* produce the segmented image using NGB or OpenCV Kmeans algorithm */
    if (pa == Partitioning_Algorithm::OPENCV)
    {
        Mat* colVec = new Mat(converted_data_8U->reshape(
            1, converted_data_8U->rows * converted_data_8U->cols));
        Mat* colVecFloat = new Mat(colVec->rows, colVec->cols, colVec->type());
        colVec->convertTo(*colVecFloat, CV_32F);

        /* labels -- i/o integer array that stores the cluster indices
        * for every sample
        *
        * centers --  Output matrix of the cluster centers, one row per
        * each cluster center.
        *
        * Note this does not change the image data sent to the array, the
        * clustering of image data itself has to be done in a
        * post processing step */
        cout << "flags=" << flags << endl;
        double compatness = kmeans(*colVecFloat, k, *labels, criteria, attempts,
                                        flags, *centers);
        cout << "Compatness=" << to_string(compatness) << endl;
        Mat* labelsFromImg = new Mat(labels->reshape(1, converted_data_8U->rows));

        /* Map each pixel in image to the proper partition given its labeling assignment
        * for a given cluster. so x1,y1 may have label 1, which is associated with
        * center 100,100, etc. */
        unordered_map<string, Mat*> stats;
        stats.insert(make_pair("testMat", labels));
        Mat res = opencv_kmeans_postProcess1(*converted_data_8U, *labelsFromImg, *centers);



        *container = res.clone();
        //*container = res; // Problem line --segfault when setting equal for some reason



        
    }
    else
    {
        cerr << "Paritioning algorithm not valid, returning" << endl;
    }

    // done with the converted data, so release this native memory
    cout << ("LGRunME(): Done with converted data, releasing native memory") << endl;
    if ((converted_data_8U != NULL) && (!converted_data_8U->empty()))
    {
        converted_data_8U->release();
    }
    else
    {
        cout << ("LGRunME(): Converted data is not present or is empty, not releasing") << endl;
    }

    *clustered_data = *container;
    long toc = stol(currentTime<nanoseconds>());
    cout << "Partitioning time: " << convert_ns(tic, toc, "MILLISECONDS") << " ms" << endl;

    // look at intermediate output from kmeans
    if (debug_flag && pa == Partitioning_Algorithm::OPENCV)
    {
        imwrite("../output/opencv_" + currentTime<milliseconds>() + ".jpg",
                            *clustered_data);
    }
    else if (debug_flag && pa == Partitioning_Algorithm::NGB)
    {
        imwrite("../output/kmeansNGB_" + currentTime<milliseconds>() + ".jpg",
                            *clustered_data);
    }

    // scan the image and produce one binary image for each segment
    if (debug_flag)
        cout << ("Calling ScanSegments") << endl;


    vector<Mat*> cmMats;
    cmMats.push_back(clustered_data);
    CompositeMat* cm = new CompositeMat(cmMats, ScanSegments(*clustered_data, false));
    // CompositeMat* cm = new CompositeMat(ScanSegments(clustered_data, false));


    if (debug_flag)
        cout << ("Finished ScanSegments") << endl;
    cm->setFilename(filename);
    vector<Mat*> cm_al_ms = cm->getListofMats();
    int segCnt = 0;
    for (auto m : cm_al_ms)
    {
        Mat* n = new Mat(m->rows, m->cols, m->type());
        if (m->type() != CV_8U)
        {
            n = new Mat(m->rows, m->cols, m->type());
            m->convertTo(*n, CV_8U);
        }
        else
        {
            n = m;
        }

        /* Just retain the edge pixels in white for each section for each
            * segment there will be more segments as the user asks for more
            * clusters -- no thresholds to get as many edges as possible,
            * lots of extraenous details removed in preprocessing ops */
        Canny(*n, *n, 0, 0);

        /* Dilate edges to make them stand out better*/
        Mat* element = new Mat(getStructuringElement(MORPH_RECT,
                                                    Size(2, 2),
                                                    Point(1, 1)));
        dilate(*n, *n, *element);
        if (debug_flag)
        {
            imwrite("../output/" + filename.substr(filename.find_last_of('/') + 1, filename.find_last_of('.')) +
                                    "_segments_after_threshold" + to_string(++segCnt) + "_" + currentTime() + ".jpg",
                                *n);	
        }

        /* WARNING: Do not autocrop otherwise L-G Graph Algorithm
            * calculations will be utterly wrong */
    }

    // Show time to scan each segment
    Mat* scanTimesPerSegment = new Mat(cm->getMat());
    int rowsSTPS = scanTimesPerSegment->rows;
    int colsSTPS = scanTimesPerSegment->cols;
    string sb = "Scan Times/Segment:";
    long totalTime = 0;
    if (rowsSTPS == 1)
    {
        sb.append("[");

        for (int i = 0; i < colsSTPS; i++)
        {
            double segScanTime = scanTimesPerSegment->at<int>(0, i, 0);
            long convertedSegScanTime = (long)(double)segScanTime;
            long time = convert_ns(convertedSegScanTime, "MILLISECONDS");

            sb.append(time + " ms ,");
            totalTime += (long)(double)segScanTime;
        }
        sb.append("]");
        sb.pop_back();
        sb.append("\n");
        sb.append("Average Scan Time/Segment: " +
                    convert_ns(totalTime/colsSTPS, "MILLISECONDS") +
                    string("ms\n"));
        

        sb.append("Total scan time: " + convert_ns(totalTime, "MILLISECONDS") + string(" ms") + "\n");
        cout << sb;
    }
}