#pragma once
#include "includes.h"
#include "Direction.hpp"

class ChainCodingContainer {
private:
	Mat border;				// object contour pixel set
	long chain_time;		// time to produce chain code for seg in ns
	vector<double> cc;	// the chain code
	Point start;			// location of segment centroid
    

	/**
	 * Constructor for passing a complex object after running chain code 
	 * algorithm on some input data
	 * @param border -- data needed to draw the border of the region
	 * @param chain_time -- the time it took to generate the chain code 
	 * for thi region. 0 is south, 6 is east, 4 is north, and 2 is west
	 * numbers increment clockwise 0 to 7 (eight total directions)
	 * @param cc -- the chain code
	 * @param start -- start point of segment
	 */
	public:
    ChainCodingContainer(Mat border, long chain_time, 
			                     vector<double> cc, Point start) {
		this->border = border.clone();
		this->chain_time = chain_time;
		
		this->cc = vector<double>(cc.size());
		for (double dblValue : cc) {
			this->cc.emplace_back(dblValue);
		}
		this->start.x = start.x; 
        this->start.y = start.y;
	}

	/**
	 * Get the object border contour set
	 * @return
	 */
	public:
    Mat getBorder() {
		return border;
	}

	/**
	 * Return the amount of time it took to generate the chain code
	 * @return
	 */
	public:
    double getChain_time() {
		return chain_time;
	}

	/**
	 * Get the chain code for the segment -- direction of each line segment
	 * from a relative starting point
	 * @return
	 */
	public:
    vector<double> getCc() {
		return cc;
	}

	/**
	 * Get the centroid of the segment/region, the first non-zero element
	 * So, row, col --> i, j --> x, y
	 * @return
	 */
	public:
    Point getStart() {
		return start;
	}
	
	/**
	 * 
	 * @param border
	 */
	public:
    void setBorder(Mat border) {
		if (!border.empty()) {
			this->border = border.clone();
		}
	}
	
	/**
	 * Convert the matrix holding the chain code into a human readable
	 * chain code -- could be used for longest substring matching
	 * @return chain code in human readable format
	 */
	public:
    string chainCodeString() {
		string sb;
		for (double c : cc) {
			sb.append(to_string((int)c) + ",");
		}
		if (sb.length() > 0) {
			sb.pop_back();	
		}		
		return sb;
	}
	
	/* Provides human readable form of the chain code algorithm return
	 * container object
	 * (non-Javadoc)
	 * @see java.lang.Object#toString()
	 */
	public:
    string toString() {
		string sb;
		sb.append("Segment starts at ()" + to_string(start.x) + "," + to_string(start.y) + ")\n");
		sb.append("It took " + to_string(convert_ns((long)chain_time, "MILLISECONDS"))
				  + " ms to generate the segment \n");
		sb.append("Chain code is:");		
		sb.append("\n");
		int limiter = 0;
		for (double c : cc) {
			string cardinalDir = Direction::getEnumByString((int)c);
			sb.append(cardinalDir + ",");
			
			if (limiter >= 64) {
				break;
			}
			limiter++;
		}
		sb.pop_back();
		sb.append("\n");
		sb.append("Chain code length: " + cc.size());
		return sb;
	}
};