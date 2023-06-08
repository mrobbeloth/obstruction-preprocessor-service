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
    cout << "Got here in sharpen" << endl;
    Mat output(input.rows, input.cols, input.type());
    filter2D(input, output, CV_8U, kernelArray, anchor);
    cout << "Got here in sharpen 2" << endl;
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