#include "utility.h"
#include <cctype>
//!
/*!
 * Count the number of files in the directory
 * \param path -- where do I look w/ relative or absolute path
 * \return a count of the number of files in that directory
*/
/*size_t number_of_files_in_directory(filesystem::path path)
{
    using filesystem::directory_iterator;
    using fp = bool (*)( const filesystem::path&);
    return count_if(directory_iterator(path), directory_iterator{}, (fp)filesystem::is_regular_file);
}*/

//!
/*!
 * Find files in a directory given an extension pattern
 * \param directory -- where do I look w/ relative or absolute path
 * \param extension -- what type of file am I looking for
 * \return a vector of files from that directory that have the desired extension 
*/
/*vector<string> findFiles(const string& directory, 
                                   const string& extension)
{
    vector<string> pngFiles;
    
    for (const auto& entry : filesystem::directory_iterator(directory))
    {
        if (entry.is_regular_file() && entry.path().extension() == extension)
        {
            pngFiles.push_back(entry.path().filename().string());
        }
    }
    
    return pngFiles;
}*/

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
 /*! This method creates an evenly distributed set of initial center points
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

/*static path modifyFileName(path fn, string& appendStr, string& extension) {
    auto now = system_clock::now();
    auto now_ms = time_point_cast<milliseconds>(now);

    auto value = now_ms.time_since_epoch();
    long dt = value.count();
    
    string fnName;
    try {
        // find where the current extension starts
        string canonicalPath = fn.string();
        int lastIndex = canonicalPath.find_last_of(".");

        // Append string and extension or just extension
        if ((!appendStr.empty()) && (appendStr.length() > 0)) {
            fnName = canonicalPath.substr(0, lastIndex)
                        + "_"  + appendStr + "_" 
                        + "_" + to_string(dt) + "_." + extension; 
        }
        else {
            fnName = canonicalPath.substr(0, lastIndex) 
                        + "_" + to_string(dt) + "_." + extension;					
        }
    }
    catch(const exception& e) {
        cerr << e.what() << '\n';
        return path(NULL);
    }

    path outputFile(fnName);
    return outputFile;
}*/

/*bool writeImagesToDisk(Mat imageData, string path, string fn, string appendStr) {
    /* check if path is valid *//*
    if ((path.size()) > 0 && (!filesystem::exists(path)))
        return false;

    /* check if filename contains anything*//*
    if (fn.size() == 0) {
        return false;
    }

    /* Sanity checks are good, try writing image data out to 
        disk*//*
    bool result = imwrite(path + fn + appendStr + ".png", imageData);
    return result;
}*/

/*bool writeImagesToDisk(Mat imageData, string path, string fn, string appendStr, string format) {
    /* check if path is valid *//*
    if ((path.size()) > 0 && (!filesystem::exists(path)))
        return false;

    /* check if filename contains anything*//*
    if (fn.size() == 0) {
        return false;
    }

    /* get current date and time *//*
    auto now = system_clock::now();
    auto now_ms = time_point_cast<milliseconds>(now);

    auto value = now_ms.time_since_epoch();
    long dt = value.count();

    /* append to filename as needed *//*
    string fnName = path + fn;
    string extension = "_" + to_string(dt) + "_." + format;
    if ((!appendStr.empty()) && (appendStr.length() > 0)) {
		fnName.append("_"  + appendStr + "_" + extension);
	} 
    /* Nothing to add to the end of the original filename, 
	 * just add the new extension 
     *//*
    else {
        fnName.append(extension);
    }

    /* Do a different method to write jpeg images *//*
    bool result = false;
    string formatLower = "";
    for (char c : format) {
        formatLower.push_back(tolower(c));
    }
    if (formatLower == "jpeg" || formatLower == "jpg") {
        vector<int> compression_params;
        compression_params.push_back(IMWRITE_JPEG_QUALITY);
        compression_params.push_back(100);
        result = imwrite(fnName, imageData, compression_params);
        //result = (fnName, imageData);
    }
    else {
        /* Sanity checks are good, try writing image data out to 
        disk*//*
        result = imwrite(fnName, imageData);	
    }

    // bool result = imwrite(path + fn + appendStr + format, imageData, [IMWRITE_JPEG_QUALITY, 100])
    return result;
}*/

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




