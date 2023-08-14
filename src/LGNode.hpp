#pragma once
#include "includes.h"
//#include "CurveLineSegmentMetaData.hpp"

class LGNode {
	/* location of the node, correspondent to region's centroid 
	 * or the center of gravity */
	private:
        Point center;

        // Region stats
        unordered_map<string, double> stats;

        // local graph associated with this node (region)
        vector<CurveLineSegMetaData> L;  
        
        // object contour pixel set
        Mat border;

        // number of pixels belonging to this region
        long size;

        // Partitioning algorithm used
        Partitioning_Algorithm pa;

        // Node index for LG Algorithm processing
        int node_id;
	
	public:
	/* Contains average of all rows, cols, and intensity for region
	   Note there are no stats for regions created by region growing
	   process where small regions are inserted within the original
	   clusters */
    unordered_map<string, double> getStats() {
        return stats;
    }
	
	/**
	 * Default constructor
	 */
    LGNode() {
        // super();
        pa = Partitioning_Algorithm::NONE;
        node_id = -1;	
        center = Point();
        size = 0;
    }

    LGNode(Point& center, Mat& border, Partitioning_Algorithm pa, 
                int node_id) {
        LGNode();
        
        // copy the center for the region (segment)
        this->center = center;
        
        // copy the object contour pixel set
        this->border = border.clone();
        
        // Record the partitioning algorithm used
        this->pa = pa;
        
        /* total number of non-zero pixels in region */
        this->size = DetermineNodeSize(border);
        
        /* Record region (Node) identifier*/
        this->border = node_id;
    }
	
	LGNode(Point center, unordered_map<string, double> stats, 
			       Mat border, Partitioning_Algorithm pa,
			       int node_id) : LGNode(center, border, pa, node_id) {
		
		// now save stats
		this->stats = unordered_map<string, double>(stats.size());

        /*
		Set<String> keys = stats.keySet();
		for(String key : keys) {
			Double value = stats.get(key);
			this.stats.put(key, value);
		}
        */
        set<string> keys;
        transform(stats.begin(), stats.end(), inserter(keys, keys.begin()), [](const auto& pair) {
            return pair.first;
        });

        for(string key : keys) {
            this->stats[key] = stats[key];
        }
	}
	/**
	 * 
	 * @param center -- Location of the node, correspondent to region's centroid 
	 * or the center of gravity
	 * @param stats -- Region statistics
	 * @param border -- Object contour pixel set
	 * @param lmd -- Metadata of all the curved line segments in a region
	 * @param pa --  Partitioning algorithm
	 * @param node_id -- Node index for LG Algorithm processing
	 */
	LGNode(Point center, unordered_map<string, double> stats, 
			Mat border, vector<CurveLineSegMetaData> lmd, 
			Partitioning_Algorithm pa, 
			int node_id) : LGNode(center, stats, border, pa, node_id) {
		
		// now save local graph associated with this node/region/segment
		this->L = vector<CurveLineSegMetaData>(lmd.size());
		for(int i = 0; i < lmd.size(); i++) {
			this->L.push_back((lmd[i]));
		}
		
	}

    Point getCenter() {
		return center;
	}

    vector<CurveLineSegMetaData> getL() {
		return L;
	}

    Mat getBorder() {
		return border;
	}

    long getSize() {
		return size;
	}
	
    int getNodeID() {
		return node_id;
	}

    void setL(vector<CurveLineSegMetaData> lmd) {
		// now save local graph associated with this node/region/segment
		this->L = vector<CurveLineSegMetaData>(lmd.size());
		for(int i = 0; i < lmd.size(); i++) {
			this->L.push_back(lmd[i]);
		}
	}

    void setBorder(Mat border) {
		this->border = border;
	}

    void setSize(long size) {
		this->size = size;
	}

    string toString() {
		string sb;
		
		sb.append("LGNode [center=(" + to_string(center.x) + "," + to_string(center.y) + "), stats="
                + unorderedMapToString(stats)
                + ", L=" + vecToString(L) + ", border=" 
                + matToString(this->border)
				+ ", size=" + to_string(size) + "]\n");
		return sb;
	}
		
};