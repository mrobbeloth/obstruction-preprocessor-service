#pragma once
#include "includes.h"

/**
 * Based on SH = U{Ln_j, R^c_j,j+1, Ln_j+1}
 * with S = Ln_n1 . R^c_12, Ln_n2, ...
 * Lj = {sp, l, d, cu}
 * sp=starting point
 * l = length
 * d = orientation
 * cu = curvature
 * 
 * @author mrobbeloth
 *
 */
class CurveLineSegMetaData {
    private:
        Point sp;				// starting point of a curved line segment
        Point ep;				// ending point of a curved line segment
        double length;   		// length of curved line segment (pixels)
        double orientation;	// in degrees
        double curvature;		//
        unsigned long lineNumber;		// sequence number
        long totalTime;        // total time to calc curved line segment in ns
        vector<CurveLineSegMetaData> connList; 
		
    public:
	/**
	 * 
	 * @param sp	-- starting point of a curved line segment
	 * @param ep	-- ending point of a curved line segment
	 * @param length 	-- length of curved line segment (pixels?)
	 * @param orientation
	 * @param curvature
	 */
	CurveLineSegMetaData(Point sp, Point ep, 
			double length, double orientation,
			double curvature, long lineNumber) : CurveLineSegMetaData() {
		this->sp = sp;
		this->ep = ep;
		this->length = length;
		this->orientation = orientation;
		this->curvature = curvature;
		this->lineNumber = lineNumber;
		this->totalTime = 0;
	}
	
	CurveLineSegMetaData() {
		this->sp = Point(0, 0);
		this->ep = Point(0, 0);
		this->length = 0;
		this->orientation = 0;
		this->curvature = 0;
		this->lineNumber = 0;
		this->totalTime = 0;
	}
	
	long getLineNumber() {
		return lineNumber;
	}

	vector<CurveLineSegMetaData> getConnList() {
		return connList;
	}

	void setConnList(vector<CurveLineSegMetaData> connList) {
		this->connList = connList;
	}

	Point getSp() {
		return sp;
	}
	void setSp(Point sp) {
		this->sp = sp;
	}
	double getLength() {
		return length;
	}
	void setLength(double length) {
		this->length = length;
	}
	double getOrientation() {
		return orientation;
	}
	void setOrientation(double orientation) {
		this->orientation = orientation;
	}
	double getCurvature() {
		return curvature;
	}
	void setCurvature(double curvature) {
		this->curvature = curvature;
	}

	Point getEp() {
		return ep;
	}

    static inline long doubleToLongBits(double x) {
        long bits;
        memcpy(&bits, &x, sizeof bits);
        return bits;
    }

    static inline size_t pHashCode(Point p) {
        size_t prime = 31;
        size_t hash = prime * 31 + std::hash<int>{}(p.x);
        hash = hash * 31 + std::hash<int>{}(p.y);
        return hash;
    }

	int hashCode() {
		const int prime = 31;
		int result = 1;
		unsigned long temp;
		temp = doubleToLongBits(curvature);
		result = prime * result + (int) (temp ^ (temp >> 32));
		result = prime * result + ((ep == Point()) ? 0 : pHashCode(ep));
		temp = doubleToLongBits(length);
		result = prime * result + (int) (temp ^ (temp >> 32));
		result = prime * result + (int) (lineNumber ^ (lineNumber >> 32));
		temp = doubleToLongBits(orientation);
		result = prime * result + (int) (temp ^ (temp >> 32));
		result = prime * result + ((sp == Point()) ? 0 : pHashCode(sp));
		return result;
	}

    // Please pass by reference
	bool equals(CurveLineSegMetaData* obj) {
		if (this == obj) {
			return true;
        }
		if (obj == nullptr) {
			return false;
        }
		CurveLineSegMetaData other = *obj;
		if (doubleToLongBits(curvature) != doubleToLongBits(other.curvature))
			return false;
		if (ep.x != (other.ep.x) && ep.y != (other.ep.y))
			return false;
		if (doubleToLongBits(length) != doubleToLongBits(other.length))
			return false;
		if (lineNumber != other.lineNumber)
			return false;
		if (doubleToLongBits(orientation) != doubleToLongBits(other.orientation))
			return false;
		if (sp.x != (other.sp.x) && sp.y != (other.sp.y))
			return false;
		return true;
	}

    void setEp(Point ep) {
		this->ep = ep;
	}

    string toString() {
		string sb;
		sb.append("Curved Line Segment MetaData L" + to_string(lineNumber) + "[sp=(" + to_string(sp.x) + "," + to_string(sp.y) + ")" 
				+ ", ep=(" + to_string(ep.x) + "," + to_string(ep.y) + ")" + ", length=" + to_string(length)
				+ ", orientation=" + to_string(orientation) + ", curvature=" + to_string(curvature)
				+ "]\n");
		if (!connList.empty()) {
			sb.append("(");
			for(CurveLineSegMetaData l : connList) {
				sb.append("RC"+to_string(this->lineNumber)
                                         +","+to_string(l.getLineNumber())
                                         +"("+to_string(l.getEp().x)+","+to_string(l.getEp().y)+")"
                                         +",");
			}
			sb.pop_back();
			sb.append(")\n");			
		}
		sb.append("Total time to generate curved line segment " + 
				to_string(convert_mcs(totalTime, "NANOSECONDS")) + " us\n");    
        return sb;
	}

	/**
	 * value returned is in ns
	 * @return
	 */
	long getTotalTime() {
		return totalTime;
	}

	/**
	 * 
	 * @param totalTime -- time in ns
	 */
	void setTotalTime(long totalTime) {
		this->totalTime = totalTime;
	}	
};