#include "utility.h"
//!
/*!
 * Count the number of files in the directory
 * \param path -- where do I look w/ relative or absolute path
 * \return a count of the number of files in that directory
*/
std::size_t number_of_files_in_directory(std::filesystem::path path)
{
    using std::filesystem::directory_iterator;
    using fp = bool (*)( const std::filesystem::path&);
    return count_if(directory_iterator(path), directory_iterator{}, (fp)std::filesystem::is_regular_file);
}

//!
/*!
 * Find files in a directory given an extension pattern
 * \param directory -- where do I look w/ relative or absolute path
 * \param extension -- what type of file am I looking for
 * \return a vector of files from that directory that have the desired extension 
*/
std::vector<std::string> findFiles(const std::string& directory, 
                                   const std::string& extension)
{
    std::vector<std::string> pngFiles;
    
    for (const auto& entry : filesystem::directory_iterator(directory))
    {
        if (entry.is_regular_file() && entry.path().extension() == extension)
        {
            pngFiles.push_back(entry.path().filename().string());
        }
    }
    
    return pngFiles;
}

//!
/*!
 * Sharpen the image
 * \param input -- input array
 * \return sharpened array
*/
Mat sharpen(Mat input) {
    float kernel[9] = {0,-1,0,-1,5,-1,0,-1,0};
    Mat kernelArray = Mat(3,3,CV_32F,kernel);
    Point anchor(-1,-1);
    Mat output(input.rows, input.cols, input.type());
    filter2D(input, output, CV_8U, kernelArray, anchor);
    return output;
}

//!
/*!
 * Sharpen the image
 * \param input -- input array
 * \return sharpened array
*/
GpuMat sharpenGPU(GpuMat input) {
    Point anchor(-1,-1);
    GpuMat output(input.rows, input.cols, input.type());
    
    /* Note that bilteralFilter is in the cudaimgproc library, not cudafilters*/
    cv::cuda::bilateralFilter(input, output, 9, 50, 50);
    return output;
}

//!
 /*! Unsharp masking
     create blurred/"unsharp", negative image to create a mask of the original image. 
     Combine w/ original positive image 
     Creating an image that is less blurry than the original

     May increase noise too much; edge may end up overly strong
 * \param input -- input array
 * \return less blurry image than original
*/
Mat unsharp_masking(Mat input) {
    // there must be a better way to apply a scalar to the elements of this array
    float kernel[25] = {1,4,6,4,1,4,16,24,16,4,6,24,36,24,6,4,16,24,16,4,1,4,6,4,1};
    for(int i = 0; i < 25; i++) {
        kernel[i] = kernel[i]*(-1/256);
    }
    Mat kernelArray = Mat(5,5,CV_32F,kernel);
    Point anchor(-1,-1);
    Mat output(input.rows, input.cols, input.type());
    filter2D(input, output, CV_8U, kernelArray, anchor);
    return output;
}

//!
 /*! 
 \brief This method creates an evenly distributed set of initial center points
to use with the OpenCV partitioning algorithm -- needed to ensure that partitioning
between candidate and model images are similar -- e.g., the use of a fixed set of
centroid locations will lower confidence matches to unacceptable levels.

kmeans will squish 2d image into vector of values
 * \param
 * \return less blurry image than original
*/
Mat setInitialLabelsGrayscale(int width, int height, int k) {
    int totalCells = width * height;
    int index = 0;
    int count = 0;
    int jump = totalCells / k;

    Mat labels(k, 1, CV_32S);

    while(count < k) {
        index += jump;
        labels.at<int>(count) = index - 1;
        count++;
    }

    return labels;
}

//!
 /*! 
 \brief This method identifies the type of OpenCV Matrix

kmeans will squish 2d image into vector of values
 * \param An OpenCV Matrix
 * \return type of matrix
*/
string printMatType(const Mat input) {
    string type;
    switch(input.type()) {
        case CV_8U:
             type = "CV_8U";
             break;
        case CV_8S:
             type = "CV_8S";
             break;                   
        case CV_16S:
             type = "CV_16S";
             break; 
        case CV_16U:
             type = "CV_16U";
             break;         
        case CV_16F:
            type = "CV_16F";
            break;                                   
        case CV_32S:
            type = "CV_32S";
            break;    
        case CV_32F:
             type = "CV_32F";
             break;         
        case CV_64F:
            type = "CV_64F"; 
             break;         
        default:
             type = "other";
    }
    return type;
}

//!
 /*! 
 \brief This method saves an OpenCV matrix data to disk

kmeans will squish 2d image into vector of values
 * \param path path to save file on
 * \param fn name of file to save matrix data to
 * \param imageData OpenCV matrix data
 * \return if operation was successful or not
*/
bool imageSave(string path, string fn, Mat imageData) {
    /* check if path is valid */
    if ((path.size()) > 0 && (!filesystem::exists(path)))
        return false;

    /* check if filename contains anything*/
    if (fn.size() == 0) {
        return false;
    }

    /* Sanity checks are good, try writing image data out to 
        disk*/
    bool result = imwrite(path+fn, imageData);
    return result;
}