#include "includes.h"

/**
 * A composite object class where each container holds:<br/>
 * 1. An OpenCV matrix<br/>
 * 2. Time to process a matrix
 * @author mrobbeloth
 */

class CompositeMat {
private:
	vector<Mat*> listofMats; // Segments from clusters
	Mat mat;					// Statistics
	long startingId;			// first id recorded to database
	long lastId;               // last used id recorded to database
	string filename;			// file from which segments were generated
	
public:
	CompositeMat() {
		setStartingId(0);
	}
	
	CompositeMat(vector<Mat*> listofMats, Mat mat) {
		auto it = begin(listofMats);
		// with use of native object, need to clone to ensure 
		// we safely get the underlying data, not pointers
		while (it != end(listofMats)) {
			advance(it, 1);
			Mat nextMat = **it;
			Mat* newMat = new Mat(nextMat.clone());
			listofMats.push_back(newMat);
		}
		mat = mat.clone();
		
		setStartingId(0);
	}

	~CompositeMat() {
		for (auto mat : listofMats) {
			delete mat;
		}
	}

	/* Get the actual raw cluster data */
	vector<Mat*> getListofMats() {
		return listofMats;
	}

	/* Get timing data for amount of time it took to scan each
	 * segment from a cluster */
	Mat getMat() {
		return mat;
	}

	long getStartingId() {
		return startingId;
	}

	void setStartingId(long startingId) {
		startingId = startingId;
	}

	string getFilename() {
		return filename;
	}

	void setFilename(string filename) {
		filename = filename;
	}

	long getLastId() {
		return lastId;
	}

	void setLastId(long lastId) {
		lastId = lastId;
	}
	
	void setListOfMat(vector<Mat> mats) {
		listofMats.resize(mats.size());
		auto it = begin(listofMats);
		// with use of native object, need to clone to ensure 
		// we safely get the underlying data, not pointers
		while (it != end(listofMats)) {
			advance(it, 1);
			Mat nextMat = **it;
			Mat* newMat = new Mat(nextMat.clone());
			listofMats.push_back(newMat);
		}
		mat = mat.clone();		
	}
	
	void addListofMat(vector<Mat> mats) {
		if (listofMats.empty()) {
			listofMats.resize(mats.size());
		}
		auto it = begin(listofMats);
		while (it != end(listofMats)) {
			advance(it, 1);
			Mat nextMat = **it;
			Mat* newMat = new Mat(nextMat.clone());
			listofMats.push_back(newMat);
		}
		mat = mat.clone();
	}
};