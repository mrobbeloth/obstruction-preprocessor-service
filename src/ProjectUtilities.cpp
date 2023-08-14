#include "ProjectUtilities.h"

namespace ProjectUtilities {
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

    //!
    /*! 
    \brief Limited version of findInMat operator from MatLab. 
    Basically find k instances of a  non-zero entry in an input
    array starting from the beginning or end of the input array

    * \param input data to find non-zero values in
    * \param k nnumber of entries to find
    * \param direction start at beginning ("first") or end ("last")
    * \return the list of non-zero indeices from input data x=row, y=col
    */
    vector<Point> findInMat(Mat input, int k, string direction) {
        vector<Point> locNonZeroElements;

        if(direction == "first") {
            int rows = input.rows;
            int cols = input.cols;
            for(int i = 0; i < rows; i++) {
                for(int j = 0; j < cols; j++) {
                    int value = input.at<uint8_t>(i,j);
                    if (value != 0) {
                        Point p(i,j);
                        locNonZeroElements.push_back(p);
                    }

                    if (locNonZeroElements.size() == k) {
                        return locNonZeroElements;
                    }
                }
            }
        }
        else if (direction == "last") {
            int rows = input.rows;
            int cols = input.cols;
            for(int i = rows; i > 0; i--) {
                for(int j = cols; j > 0; j--) {
                    int value = input.at<uint8_t>(i,j);
                    if (value != 0) {
                        Point p(i,j);
                        locNonZeroElements.push_back(p);
                    }

                    if (locNonZeroElements.size() == k) {
                        return locNonZeroElements;
                    }
                }
            }        
        }

        cerr << "Invalid direction specified" << endl;
        return locNonZeroElements;
    }

    Mat* setInitialLabelsGrayscaler(int width, int height, int k) {
        int totalCells = width * height;
        int index = 0;
        int count = 0;
        int jump = totalCells/k;
        Mat* labels = new Mat(k,1,CV_32S);
        while (count < k) {
            index += jump;
            labels->at<int>(count,0) = index-1;
            count++;
        }
        return labels;
    }
 
    template <typename T>
    std::string currentTime() {
        int64_t currTime = std::chrono::duration_cast<T>(std::chrono::system_clock::now().time_since_epoch()).count();
        std::string currTimeStr = std::to_string(currTime);
        return currTimeStr;
    }
    template string 
    currentTime<milliseconds>();
    template string 
    currentTime<nanoseconds>();

    long convert_ns(long tic, long toc, string time_unit) {
        unsigned long multiplier;
        if (time_unit == "MILLISECONDS") {
            multiplier = pow(10, -6);
        }
        else if (time_unit == "SECONDS") {
            multiplier = pow(10, -9);
        }
        else {
            cout << "Incorrect multiplier specified\n defaulting to Millisecond conversion" << endl;
            multiplier = pow(10, -6);
        }
        unsigned long raw_ms = (toc - tic) * multiplier;
        return round(raw_ms);
    }

    long convert_ns(long time, string time_unit) {
        unsigned long multiplier;
        if (time_unit == "MILLISECONDS") {
            multiplier = pow(10, -6);
        }
        else if (time_unit == "SECONDS") {
            multiplier = pow(10, -9);
        }
        else {
            cout << "Incorrect multiplier specified\n defaulting to Millisecond conversion" << endl;
            multiplier = pow(10, -6);
        }
        unsigned long raw_ms = time * multiplier;
        return round(raw_ms);
    }

    long convert_mcs(long time, string time_unit) {
        unsigned long multiplier;
        if (time_unit == "MILLISECONDS") {
            multiplier = pow(10, -3);
        }
        else if (time_unit == "SECONDS") {
            multiplier = pow(10, -6);
        }
        else if (time_unit == "NANOSECONDS") {
            multiplier = pow(10, 3);
        }
        else {
            cout << "Incorrect multiplier specified\n defaulting to Millisecond conversion" << endl;
            multiplier = pow(10, -3);
        }
        unsigned long raw_ms = time * multiplier;
        return round(raw_ms);
    }

    long DetermineNodeSize(Mat border) {
		Mat nonZeroLocations;
		Mat temp;
		
		/* Find the number and location of all border pixels 
		 * we don't need much accuracy for this routine, 
		 * just non-zero values, should improve performance */
		if (border.type() != CV_8UC1) {
			border.convertTo(temp, CV_8UC1);	
		}
		else {
			temp = border.clone();
		}
		
		
		/* Each entry contains the row and column of a non-zero 
		 * pixel. Remember, some weird segments may have the 
		 * border running around the edge of the image, lots
		 * of x,1 locations at the beginning of the image */
		findNonZero(temp, nonZeroLocations);
		
		/* Changing type will not change the number of channels
		 * or depth of the image */
		nonZeroLocations.convertTo(nonZeroLocations, border.type());		
		
		/* total should give me the same result as countNonZero; 
		 * however, findnonzero gives me a two channel result
		 * and countNonZero expects one channel, sigh 
		 * 
		 * the same result as it's first channel * second (always = 1) */
		int n = (int)nonZeroLocations.total();
		
		/* Try to reduce burden on the research system and 
		 * still maintain a high level of accuracy 
		 * 
		 * a rule of thumb matrix here to handle large
		 * raster complex polygon borders */
		Mat reducednonZeroLocations;
		if (n > 1000) {
			reducednonZeroLocations = 
					returnEveryNthElement(
							nonZeroLocations, 100);	
		}		
		
		if (!reducednonZeroLocations.empty()) {
			nonZeroLocations = reducednonZeroLocations;
			n = (int)reducednonZeroLocations.total();
		}
	
		/* Determine the extents of the border region*/
		double xmin = DBL_MAX;
		double xmax = DBL_MIN;
		double ymin = DBL_MAX;
		double ymax = DBL_MIN;
		for (int i = 0; i < n; i++) {
			// double[] data = nonZeroLocations(i, 0);
            vector<double> data = nonZeroLocations.at<vector<double>>(i, 0);
			double xVal = data[0];
			double yVal = data[1];
			
			if (xVal < xmin) {
				xmin = xVal;
			}
			
			if (xVal > xmax) {
				xmax = xVal;
			}
			
			if (yVal < ymin) {
				ymin = yVal;
			}
			
			if (yVal > ymax) {
				ymax = yVal;
			}
		}
		
		
		/* this will give the full size of the image file, 
		 * not just the border part extents */
		long size = 0;
		
		/* So using the border pixels, each pixel being a vertex (not ideal,
		 * but this is a raster polygon, not a vector one, determine which 
		 * pixels in the extent area inside the polygon to determine
		 * the size of the polygon */
		int yMinInt = (int) ymin;
		int xMinInt = (int) xmin;
		int yMaxInt = (int) ymax;	
		int xMaxInt = (int) xmax;
		for (int i = yMinInt; i < yMaxInt; i++) {
			for (int j = xMinInt; j < xMaxInt; j++) {
				if (isInside(nonZeroLocations, n, Point(j, i))) {
					size++;
				}
			}
		}
		return size;
	}

    bool isInside(Mat nonZeroBorderPoints, int n, Point p) {
		// There must be at least 3 vertices in polygon[]
		if (n < 3) {
			return false;
		}
		
		Point extreme(10000, p.y);
		
	    // Count intersections of the above line with sides of polygon
	    int count = 0, i = 0;
	    
	    do {
	    	
	    	int next = (i+1)%n;
	    	
	    	/*Check if the line segment from 'p' to 'extreme' 
	    	 *intersects with the line segment from 'polygon[i]'
	    	 * to 'polygon[next]' */
	    	 vector<double> pairp1 = nonZeroBorderPoints.at<vector<double>>(i, 0);
	    	 Point p1(pairp1[0], pairp1[1]);
	    	 
	    	 vector<double> pairq1 = nonZeroBorderPoints.at<vector<double>>(next, 0);
	    	 Point q1(pairq1[0], pairq1[1]);
	    	 
	    	 //Point p1 = new Point 
	    	 if (doIntersect(p1, q1, p, extreme)) {
	    		 /*If the point 'p' is colinear with line segment 'i-next',
	    		  *then check if it lies on segment. If it lies, return true, 
	    		  *otherwise, false*/
	    		 if ((orientation(p1, p, q1) == 0)) {
	    			 return onSegment(p1, p, q1);
	    		 }
	    		 count++;
	    	 }
	    	 
	    	 i = next;
		} while (i != 0);
	    
	    return (count % 2 == 1);
	}

    Mat returnEveryNthElement(Mat p, int n) {
		if ((p.rows != 1) && p.cols != 1) { 
			return Mat();
		}
		
		Mat q;
		if (p.rows != 1) {
			q = (p.rows/n, 1, p.type());
		}
		else {
			q = (1, p.cols/n, p.type());
		}
		
		for (int i = 0; i < p.rows; i++) {
			for (int j = 0; j < p.cols; j++) {
				if (((i + 1) % n == 0) || ((j + 1) % n == 0)) {	
                    /*double[] value = p.get(i, j);					
					q.put(i, j, value[0], value[1]);*/
                    vector<double> value = p.at<vector<double>>(i, j);
                    q.at<vector<double>>(i, j)[0] = value[0];
                    q.at<vector<double>>(i, j)[1] = value[1];
                }					
			}
		}
		return q;
	}

    bool doIntersect(Point p1, Point q1, Point p2, Point q2) {
		
		/**
		 * Find the four orientations needed for general and
		 * special cases
		 */
	    int o1 = orientation(p1, q1, p2);
	    int o2 = orientation(p1, q1, q2);
	    int o3 = orientation(p2, q2, p1);
	    int o4 = orientation(p2, q2, q1);

	    // General case
	    if (o1 != o2 && o3 != o4) {
	    	return true;
	    }
	    
	    // Special Cases
     	// p1, q1 and p2 are colinear and p2 lies on segment p1q1
	    if (o1 == 0 && onSegment(p1, p2, q1)) {
	    	return true;
	    }
	    
	    // p1, q1 and p2 are colinear and q2 lies on segment p1q1
	    if (o2 == 0 && onSegment(p1, q2, q1))  {
	    	return true;
	    }
	    
	    // p2, q2 and p1 are colinear and p1 lies on segment p2q2
	    if (o3 == 0 && onSegment(p2, p1, q2)) {
	    	return true;
	    }
	    
	    // p2, q2 and q1 are colinear and q1 lies on segment p2q2
	    if (o4 == 0 && onSegment(p2, q1, q2)) {
	    	return true;
	    }
	    
		return false; // Doesn't fall in any of the above cases
	}

    int orientation(Point p, Point q, Point r) {
		
		double val = (q.y - p.y) * (r.x - q.x) - 
				      (q.x - p.x) * (r.y - q.y);
		if (val == 0) {
			// no slope between two other points, collinear
			return (int) 0;
		}
		else {
			// positive slope
			return (int) ((val > 0) ? 1 : 2);
		}
	}

    bool onSegment (Point p, Point q, Point r) {
		if ((q.x <= max(p.x, r.x)) && (q.x >= min(p.x, r.x)) && 
				(q.y <= max(p.y, r.y)) && (q.y >= min(p.y, r.y))) {
			return true;
		}						
		return false;
	}

    template <typename K, typename V>
    string unorderedMapToString(const unordered_map<K, V>& map) {
        ostringstream result;
        result << "{";

        bool first = true;
        for (const auto& entry : map) {
            if (!first) {
                result << ", ";
            }
            result << entry.first << "=" << entry.second;
            first = false;
        }

        result << "}";
        return result.str();
    }

    template <typename T>
    string vecToString(const vector<T>& vec) {
        ostringstream result;
        result << "[";

        bool first = true;
        for (const auto& element : vec) {
            if (!first) {
                result << ", ";
            }
            result << element;
            first = false;
        }

        result << "]";
        return result.str();
    }

    //template <typename T>
    string matToString(const Mat& mat) {
        ostringstream result;
        result << "[";

        bool firstRow = true;
        for (int i = 0; i < mat.rows; ++i) {
            if (!firstRow) {
                result << ", ";
            }
            result << "[";
            bool firstCol = true;
            for (int j = 0; j < mat.cols; ++j) {
                if (!firstCol) {
                    result << ", ";
                }
                result << mat.at<int>(i, j);
                firstCol = false;
            }
            result << "]";
            firstRow = false;
        }

        result << "]";
        return result.str();
    }
/*
    Mat autoCropGrayScaleImage(Mat segment, bool apply_threshold) {
		
		Mat original = segment.clone();
		Mat image = segment.clone();
		
	    // thresholding the image to make a binary image
		if (apply_threshold) {
			cv::threshold(image, image, 100, 255, THRESH_BINARY_INV);	
		}	    
		
	    // find the center of the image
	    vector<double> centers = {(double)image.rows/2, (double)image.cols/2};
	    Point image_center(centers[0], centers[1]);
	    
		// finding the contours
	    //vector<MatOfPoint> contours;
        vector<OutputArrayOfArrays> contours;
        
	    Mat hierarchy;
	    findContours(
	    		image, contours, hierarchy, 
	    		RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	    
	    // finding best bounding rectangle for a contour whose distance is closer to the image center that other ones
	    double d_min = DBL_MAX;
	    Rect rect_min;
	    for (OutputArrayOfArrays contour : contours) {
	        Rect rec = boundingRect(contour);
	        // find the best candidates
	        if ((rec.height > image.cols/2) && (rec.width > image.rows/2)) {
	        	Mat edges;
	        	Canny(image, edges, 0, 2);
	        	return edges.clone();
	        }
	             
	        Point pt1((double)rec.x, (double)rec.y);
	        Point center(rec.x+(double)(rec.width)/2, rec.y + (double)(rec.height)/2);
	        double d = sqrt(pow((double)(pt1.x-image_center.x),2) + pow((double)(pt1.y -image_center.y), 2));            
	        if (d < d_min)
	        {
	            d_min = d;
	            rect_min = rec;
	        }                   
	    }
	    // slicing the image for result region
	    int pad = 5;        
	    rect_min.x = rect_min.x - pad;
	    rect_min.y = rect_min.y - pad;
	    
	    if (rect_min.x <= 0) {
	    	rect_min.x = 1;
	    }

	    if (rect_min.y <= 0) {
	    	rect_min.y = 1;
	    }
	    
	    rect_min.width = rect_min.width + 2*pad;
	    rect_min.height = rect_min.height + 2*pad;

	    if ( rect_min.width <= 0) {
	    	rect_min.width = 1;
	    }
	    
	    if (rect_min.height <= 0) {
	    	rect_min.height = 1;
	    }
	    
	    if (rect_min.x >= original.rows) {
	    	rect_min.x = original.rows-1;
	    }
	    
	    if (rect_min.y >= original.cols) {
	    	rect_min.y = original.cols-1;
	    }
	    
	    if ((rect_min.x + rect_min.width) > original.rows) {
	    	rect_min.width -= (rect_min.x + rect_min.width) - original.rows;
	    }
	    
	    if ((rect_min.y + rect_min.height) > original.cols) {
	    	rect_min.height -= (rect_min.y + rect_min.height) - original.cols;
	    }
	    
	    // Size down the original
        Mat result = original(rect_min);
	    
	    // debug line
	    //Imgcodecs.imwrite("cropped_"+Math.abs(new Random().nextLong())+".jpg", result);
	    
	    // return the cropped image
	    return result.clone();
	}*/

    vector<double> convertMat1xn(vector<Mat> MatAL, bool duplicateBegEnd) {
		/* Find the total size of array needed for conversion*/
		int total = 0;
		int currentcnt = 0;
		for(Mat m : MatAL) {
			total += m.total();
		}
		
		/* perform conversion */
		vector<double> q;
		if (duplicateBegEnd) {
			q.resize(total);	
		}
		else {
			q.resize(total+1);
		}
		
		for (Mat m : MatAL) {
			int numElements = (int) m.total();
			Size s = m.size();
			if (s.height != 1) {
				/* ignore Mats not 1xn in size */
				continue;
			}
			for(int i = 0; i < numElements; i++) {
				q[currentcnt++] = m.at<int>(0, i, 0);
			}
		}
		
		if ((duplicateBegEnd) && (total - 1 >= 0)) {
			q[total-1] = q[0]; 
		}
		
		return q;
	}
}