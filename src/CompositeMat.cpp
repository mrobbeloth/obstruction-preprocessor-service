#include "CompositeMat.h"

CompositeMat::CompositeMat() {
    listofMats = {};
    mat = Mat();
    startingID = 0;
}

CompositeMat::CompositeMat(vector<Mat> listofMats, Mat input) {
    for(Mat m : listofMats) {
        this->listofMats.push_back(m.clone());
    }
    this->mat = input.clone();
    this->startingID = 0;
}

vector<Mat> CompositeMat::getListofMats() {
    return listofMats;
}

Mat CompositeMat::getMat() {
    return mat;
}

string CompositeMat::getFilename() {
    return filename;
}

void CompositeMat::setFileName(String filename) {
    this->filename = filename;
}

long CompositeMat::getLastID() {
    return lastID;
}

void CompositeMat::setLastID(long lastID) {
    this->lastID = lastID;
}

void CompositeMat::setListofMat(vector<Mat> listofMats) {
    if ((&listofMats != nullptr) && (listofMats.size() > 0)) {
        for (Mat m: listofMats) {
            this->listofMats.push_back(m.clone());
        }
    }
}

long CompositeMat::getStartingID() {
    return startingID;
}

void CompositeMat::setStartingID(long startingID) {
    this->startingID = startingID;
}
