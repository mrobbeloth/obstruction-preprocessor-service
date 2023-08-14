#include </usr/local/cuda/targets/x86_64-linux/include/cuda_runtime.h>

#include <iostream>
#include <stdlib.h>
#include <thread>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <time.h>
#include <iterator>
#include <list>
#include <set>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>


#include <opencv2/dnn.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/cvstd.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/cvdef.h>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/gapi/own/scalar.hpp>
#include <opencv2/imgcodecs.hpp> 
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/photo/photo.hpp>

#include <rapidfuzz/fuzz.hpp>
#include <parallel_hashmap/phmap.h>
#include <plplot/plplot.h>
#include <plplot/plstream.h>
#include <OpenXLSX/OpenXLSX.hpp>

// Includes to try and find minmaxloc library
#include <opencv2/core/cvdef.h>
#include <opencv2/core/base.hpp>
#include <opencv2/core/cvstd.hpp>
#include <opencv2/core/traits.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/persistence.hpp>

#include "BS_thread_pool.hpp"
#include "ProjectUtilities.h"
#include "LGNode.hpp"
#include "ChainCodingContainer.hpp"
#include "LineSegmentContainer.hpp"
#include "CurveLineSegmentMetaData.hpp"
#include "kMeansNGBContainer.hpp"

using namespace std;
using namespace cv;
using namespace filesystem;
using namespace chrono;
using namespace BS;
// using namespace cuda (if we choose to use cudnn.h)
using namespace ProjectUtilities;