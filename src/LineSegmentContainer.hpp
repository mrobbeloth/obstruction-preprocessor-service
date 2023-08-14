#pragma once
#include "includes.h"
/**
 * A complex container class whose objects store<br/>
 * 1. x coordinates in segment <br/>
 * 2. y coordinates in segment <br/>
 * 3. segment time<br/>
 * 
 * @author mrobbeloth
 *
 */
class LineSegmentContainer {
	private:
		vector<Mat> segment_x;
		vector<Mat> segment_y;
		long segment_time;
	
	public:
		LineSegmentContainer(vector<Mat> segment_x, 
			                     vector<Mat> segment_y,
			                     long segment_time) {
		
		/*this->segment_x = new vector<Mat>();
		this->segment_y = new vector<Mat>();*/
		
		for(Mat m : segment_x) {
			if (!m.empty()) {
				this->segment_x.push_back(m.clone());	
			}			
		}
		
		for(Mat m : segment_y) {
			if (!m.empty()) {
				this->segment_y.push_back(m.clone());			
			}
		}
		
		this->segment_time = segment_time;
	}

	vector<Mat> getSegment_x() {
		return segment_x;
	}

	vector<Mat> getSegment_y() {
		return segment_y;
	}

	long getSegment_time() {
		return segment_time;
	}
	
	string toString() {
		string sb;
		sb.append("x entries (" + to_string(segment_x.size()) + " total): ");
		int nCnt = 0;
		for (Mat sx : segment_x) {
			for (int i  = 0; i < sx.cols; i++) {
				sb.append(sx.at<int>(0, i, 0) + ",");
			}

			if ((nCnt % 100) == 1) {
				sb.append("\n");
			}
			nCnt++;
			
		}
		sb.pop_back();
		sb.append("\n");
		sb.append("y entries (" + to_string(segment_y.size()) + " total): ");
		nCnt = 0;
		for (Mat sy : segment_y) {
			for (int i  = 0; i < sy.cols; i++) {
				sb.append(sy.at<int>(0,i , 0) + ",");
			}			

			if ((nCnt % 100) == 1) {
				sb.append("\n");
			}
			nCnt++;
		}
		sb.pop_back();
		sb.append("\n");
		sb.append("Line Segment Generation time: " 
					+ to_string(convert_ns(segment_time, "MILLISECONDS"))
		 			+ " ms\n");
				
		return sb;
	}
};