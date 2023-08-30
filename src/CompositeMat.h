#include <vector>
#include <opencv2/core/mat.hpp>

using namespace std;
using namespace cv;

//!
 /*! 
    A composite object class where each container holds:
    1. An OpenCV matrix
    2. Time to process a matrix
 
 
 */
class CompositeMat {
    private:
        vector<Mat> listofMats;         // Segments from clusters
        Mat mat;                        // Statistics        
        long startingID;                  // first id recorded to datastore
        long lastID;                    // last id recorded to datastore
        string filename;                // image from which segments were generated
    public:
        CompositeMat();
        CompositeMat(vector<Mat> listofMats, Mat input);
        vector<Mat> getListofMats();
        Mat getMat();
        string getFilename();
        void setFileName(String filename);
        long getLastID();
        void setLastID(long lastID);
        long getStartingID();
        void setStartingID(long startingID);
        void setListofMat(vector<Mat> listofMats);        
};