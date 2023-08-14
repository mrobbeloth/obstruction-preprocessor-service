/*
import info.debatty.java.stringsimilarity.Cosine;
import info.debatty.java.stringsimilarity.Damerau;
import info.debatty.java.stringsimilarity.JaroWinkler;
import info.debatty.java.stringsimilarity.LongestCommonSubsequence;
import info.debatty.java.stringsimilarity.MetricLCS;
import info.debatty.java.stringsimilarity.NGram;
import info.debatty.java.stringsimilarity.NormalizedLevenshtein;
import info.debatty.java.stringsimilarity.OptimalStringAlignment;
import info.debatty.java.stringsimilarity.QGram;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.stream.IntStream;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.TreeMap;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentSkipListMap;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import org.apache.poi.ss.usermodel.BorderStyle;
import org.apache.poi.ss.usermodel.CellType;
import org.apache.poi.xssf.usermodel.XSSFCell;
import org.apache.poi.xssf.usermodel.XSSFCellStyle;
import org.apache.poi.xssf.usermodel.XSSFFont;
import org.apache.poi.xssf.usermodel.XSSFRow;
import org.apache.poi.xssf.usermodel.XSSFSheet;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;
import org.apache.xmlbeans.impl.common.Levenshtein;
import org.opencv.core.Core;
import org.opencv.core.Core.MinMaxLocResult;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat6;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.core.TermCriteria;
import org.opencv.core.Rect;
import org.opencv.core.Point;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Subdiv2D;
import org.opencv.photo.Photo;
import plplot.core.*;
import static plplot.core.plplotjavacConstants.*;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.output.prediction.AbstractOutput;
import weka.classifiers.evaluation.output.prediction.InMemory;
import weka.classifiers.evaluation.output.prediction.InMemory.PredictionContainer;
import weka.classifiers.evaluation.output.prediction.PlainText;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.LMT;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
*/

// Most of the unfound libraries have been rewritten into other libraries
// Find out which libraries they were written into later
// We have chosen to ignore weka as it was designed for Java
// Use cudnn instead for more evolved ML capability
#include "includes.h"
#include "compositeMat.hpp"
#include "ProjectUtilities.h"
#include "kMeansNGBContainer.hpp"

using namespace ProjectUtilities;

/**
 * This class implements the Local-Global (LG) Algorithm based on a
 * highly derivative MATLAB code and since extensively modified for use
 * in a Java environment using OpenCV and other third party APIs
 *
 * @author Michael Robbeloth
 * @category Projects
 * @since 2/7/2015
 * @version 0.3
 * <br/><br/>
 * Class: CEG7900<br/>
 * <h2>Revision History</h2><br/>
 * <b>Date						Revision</b>
 *    7/18/2015					 (0.3) Place source into github
 *                                     closest approximation to source used
 *                                     at NAECON 2015 talk
 *
 *                                     Remove writing a second threshold
 *                                     copy to disk
 *
 *                                     Write hierarchy contour data to disk
 *                                     after contours are found, not before
 *
 *                                     fix problems with second set of contours
 *                                     being written to disk using the wrong
 *                                     data set
 *
 *                                     add documentation
 *
 *                                     remove writing copied data to images for
 *                                     verification of operations, I think it's
 *                                     safe to skip this...if something weird
 *                                     occurs it can easily be added back in
 *                                     as a single line piece of code
 *
 *                                     After removing the writing of threshold2
 *                                     images use threhold images the only diff
 *                                     with threshold2 was its use in ops whose
 *                                     output is separately wrote to disk
 *
 *     5/29/2016                 (0.4) Revision history is now in github logs
 *
 */
class LGAlgorithm {
private:
	inline static const string avRowsString = "Average Rows";
	inline static const string avColsString = "Average Columns";
	inline static const string avIntString = "Average Itensity";
	static CompositeMat ScanSegments(Mat* I, bool debug);
	static vector<Mat> regiongrowing(Mat* I, int x, int y, double reg_maxdist, bool debug);
	static kMeansNGBContainer opencv_kmeans_postProcess(Mat* data, Mat* labels, Mat* centers);
	
	// sheet names
public:
	inline static const string SUMMARY_SHEET = "Summary";
	inline static const string WEIGHTS_SHEET = "Weights";

	// summary sheet column positions
	inline static const short FILENAME_COLUMN_SUMMARY = 0;
	inline static const short Si_COLUMN_SUMMARY = 1;
	inline static const short Ci_COLUMN_SUMMARY = 2;
	inline static const short CSi_COLUMN_SUMMARY = 3;
	inline static const short LCSi_COLUMN_SUMMARY = 4;
	inline static const short SIMG_COLUMN_SUMMARY = 5;
	inline static const short WEKA_DELA_COLUMN_SUMMARY = 6;
	inline static const short Mj_COLUMN_SUMMARY = 7;

	// summary sheet column labels
	inline static const string FILENAME_COLUMN_NAME = "Filename";
	inline static const string Si_COLUMN_NAME = "Si";
	inline static const string Ci_COLUMN_NAME = "Ci";
	inline static const string CSi_COLUMN_NAME = "CSi";
	inline static const string LCSi_COLUMN_NAME = "LCSi";
	inline static const string SIMG_COLUMN_Name = "SimG";
	inline static const string WEKA_DELAUNAY_COLUMN_NAME = "WDC";
	inline static const string Mj_COLUMN_NAME = "Mj";

	// Weight Names
	inline static const string ALPHA = "alpha";
	inline static const string BETA = "beta";
	inline static const string GAMMA = "gamma";
	inline static const string EPLISON = "eplison";
	inline static const string ZETA = "zeta";
	inline static const string ETA = "eta";

	// Weight Values
	inline static const float ALPHA_W = 0.25;
	inline static const float BETA_W = 0.15;
	inline static const float GAMMA_W = 0.05;
	inline static const float EPLISON_W = 0.10;
	inline static const float ZETA_W = 0.20;
	inline static const float ETA_W = 0.25;

	/* This enumeration tells the LG Algorithm how to process the image */
	enum Mode
	{
		PROCESS_MODEL,
		PROCESS_SAMPLE
	};

	/**
	 * Local Global (LG) Graph Run Me Bootstrap Algorithm
	 * @param data -- input image
	 * @param K -- number of sets to partition data into
	 * @param clustered_data -- holder for data clusters
	 * @param criteria -- termination criteria
	 * @param attempts -- number of iterations to use in partitioning data
	 * @param flags -- special processing indicators (not used
	 * @param filename -- name of file that is being processed
	 * @param pa -- partitioning algorithm choice for OpenCV partitioning
	 * @param debug_flag -- calls to add extra output files or data where needed to verify correct operation
	 * @param imageTpe -- process image as standard (S), synthesis image (Y), rotated standard (R),
	 * rotated synthesis (Z), sample matching/not applicable (X)	 *
	 * @param imageRotation -- rotation of image (and subsequently segments)
	 * @param delaunay_calc -- perform the really expense Delaunay graph calculation
	 * @param classiferPref -- classifier to use for Weka ML Delaunay calcluations
	 * @return opencv matrix with timing data in a composite object
	 */
	static CompositeMat* LGRunME(Mat* data, int K, Mat* clustered_data,
									   TermCriteria criteria, int attempts,
									   int flags, string filename,
									   Partitioning_Algorithm pa,
									   Mode mode, bool debug_flag, char imageType,
									   short imageRotation, bool delaunay_calc,
									   String classiferPref)
	{
		// Deliverables
		Mat* labels = nullptr;

		// sanity check the number of clusters
		if (K < 2)
		{
			cerr << "The number of clusters must be greater than or equal to two." << endl;
			exit(1);
		}

		// sanity check that there is some data to work with
		if (data->total() == 0)
		{
			cerr << "There must be some input data to work with for analysis." << endl;
			exit(2);
		}

		/* Minimizing cpu/memory requirements to lower processing overhead
		 * Michael 2/27/2017 */
		Mat* converted_data_8U = new Mat(data->rows, data->cols, CV_8U);
		data->convertTo(*converted_data_8U, CV_8U);

		/* verify we have the actual full model image to work with
		 * at the beginning of the process */
		if (debug_flag)
		{
			imwrite("output/verify_full_image_in_ds_" + currentTime() + ".jpg",
							  *converted_data_8U);
		}

		if ((flags & KMEANS_USE_INITIAL_LABELS) == 0x1)
		{
			labels =
				setInitialLabelsGrayscaler(
					converted_data_8U->rows,
					converted_data_8U->cols, K);
					//converted_data_8U.height() instead of .cols originally
			cout << "Programming initial labels" << endl;
			cout << "Labels are:" << endl;
			cout << labels << endl;
		}
		else
		{
			/* labels = new Mat(); */
		}

		// start by smoothing the image -- let's get the obvious artifacts removed
		// start by smoothing the image -- let's get the obvious artifacts removed
		Mat* centers = new Mat();
		
		kMeansNGBContainer* container = nullptr;
		long tic = stol(currentTime<nanoseconds>());

		/* Aggressively sharpen and then remove noise */
		*converted_data_8U = sharpen(*converted_data_8U);
		if (debug_flag)
		{
			imwrite("output/" + filename.substr(filename.find_last_of('/') + 1, filename.find_last_of('.')) + "_sharpen.jpg",
							  *converted_data_8U);
		}
		/* the h parameter here is quite high, 85, to remove lots of detail that
		 * would otherwise generate extra segments from the clusters --
		 * we loose fine details, but processing times are lower */
		fastNlMeansDenoising(
			*converted_data_8U, *converted_data_8U, 85, 7, 21);
		imwrite("output/" + filename.substr(filename.find_last_of('/') + 1, filename.find_last_of('.')) + "_denoise.jpg",
						  *converted_data_8U);

		// after smoothing, let's partition the image
		/* produce the segmented image using NGB or OpenCV Kmeans algorithm */
		if (pa == Partitioning_Algorithm::OPENCV)
		{
			Mat* colVec = new Mat(converted_data_8U->reshape(
				1, converted_data_8U->rows * converted_data_8U->cols));
			Mat* colVecFloat = new Mat(colVec->rows, colVec->cols, colVec->type());
			colVec->convertTo(*colVecFloat, CV_32F);

			/* labels -- i/o integer array that stores the cluster indices
			 * for every sample
			 *
			 * centers --  Output matrix of the cluster centers, one row per
			 * each cluster center.
			 *
			 * Note this does not change the image data sent to the array, the
			 * clustering of image data itself has to be done in a
			 * post processing step */
			cout << "flags=" << flags << endl;
			double compatness = kmeans(*colVecFloat, K, *labels, criteria, attempts,
											flags, *centers);
			cout << "Compatness=" << to_string(compatness) << endl;
			Mat* labelsFromImg = new Mat(labels->reshape(1, converted_data_8U->rows));

			/* Map each pixel in image to the proper partition given its labeling assignment
			 * for a given cluster. so x1,y1 may have label 1, which is associated with
			 * center 100,100, etc. */
			*container = opencv_kmeans_postProcess(converted_data_8U, labelsFromImg, centers);
		}
		else if (pa == Partitioning_Algorithm::NGB)
		{
			// Deprecated
			/*data.convertTo(converted_data_8U, CV_8U);
			container = kmeansNGB(converted_data_8U, K, attempts);*/
		}
		else
		{
			cerr << "Paritioning algorithm not valid, returning" << endl;
			return nullptr;
		}

		// done with the converted data, so release this native memory
		cout << ("LGRunME(): Done with converted data, releasing native memory") << endl;
		if ((converted_data_8U != NULL) && (!converted_data_8U->empty()))
		{
			converted_data_8U->release();
		}
		else
		{
			cout << ("LGRunME(): Converted data is not present or is empty, not releasing") << endl;
		}

		*clustered_data = container->getClustered_data();
		long toc = stol(currentTime<nanoseconds>());
		cout << "Partitioning time: " << convert_ns(tic, toc, "MILLISECONDS") << " ms" << endl;

		// look at intermediate output from kmeans
		if (debug_flag && pa == Partitioning_Algorithm::OPENCV)
		{
			imwrite("output/opencv_" + currentTime<milliseconds>() + ".jpg",
							  *clustered_data);
		}
		else if (debug_flag && pa == Partitioning_Algorithm::NGB)
		{
			imwrite("output/kmeansNGB_" + currentTime<milliseconds>() + ".jpg",
							  *clustered_data);
		}

		// scan the image and produce one binary image for each segment
		if (debug_flag)
			cout << ("Calling ScanSegments") << endl;
		CompositeMat* cm = new CompositeMat(ScanSegments(clustered_data, false));
		if (debug_flag)
			cout << ("Finished ScanSegments") << endl;
		cm->setFilename(filename);
		vector<Mat*> cm_al_ms = cm->getListofMats();
		int segCnt = 0;
		for (auto m : cm_al_ms)
		{
			Mat* n = new Mat(m->rows, m->cols, m->type());
			if (m->type() != CV_8U)
			{
				n = new Mat(m->rows, m->cols, m->type());
				m->convertTo(*n, CV_8U);
			}
			else
			{
				n = m;
			}

			/* Just retain the edge pixels in white for each section for each
			 * segment there will be more segments as the user asks for more
			 * clusters -- no thresholds to get as many edges as possible,
			 * lots of extraenous details removed in preprocessing ops */
			Canny(*n, *n, 0, 0);

			/* Dilate edges to make them stand out better*/
			Mat* element = new Mat(getStructuringElement(MORPH_RECT,
														Size(2, 2),
														Point(1, 1)));
			dilate(*n, *n, *element);
			if (debug_flag)
			{
				imwrite("output/" + filename.substr(filename.find_last_of('/') + 1, filename.find_last_of('.')) +
									  "_segments_after_threshold" + to_string(++segCnt) + "_" + currentTime() + ".jpg",
								  *n);	
			}

			/* WARNING: Do not autocrop otherwise L-G Graph Algorithm
			 * calculations will be utterly wrong */
		}

		// Show time to scan each segment
		Mat* scanTimesPerSegment = new Mat(cm->getMat());
		int rowsSTPS = scanTimesPerSegment->rows;
		int colsSTPS = scanTimesPerSegment->cols;
		string sb = "Scan Times/Segment:";
		long totalTime = 0;
		if (rowsSTPS == 1)
		{
			sb.append("[");

			for (int i = 0; i < colsSTPS; i++)
			{
				double segScanTime = scanTimesPerSegment->at<int>(0, i, 0);
				long convertedSegScanTime = (long)(double)segScanTime;
				long time = convert_ns(convertedSegScanTime, "MILLISECONDS");

				sb.append(time + " ms ,");
				totalTime += (long)(double)segScanTime;
			}
			sb.append("]");
			sb.pop_back();
			sb.append("\n");
			sb.append("Average Scan Time/Segment: " +
					  convert_ns(totalTime/colsSTPS, "MILLISECONDS") +
					  string("ms\n"));
			

			sb.append("Total scan time: " + convert_ns(totalTime, "MILLISECONDS") + string(" ms") + "\n");
			cout << sb;
		}

		/* calculate the local global graph, specify string similarity method for now
		   maybe move up to user choice later on
		List<String> ssaChoices = Arrays.asList("QGram (Ukkonen) Distance",
												"Longest-Common-Subsequence");
		List<String> ssaChoices = Arrays.asList("all");

		measures use in rev 38 of dissertation, finding a most probably match
		List<String> ssaChoices = Arrays.asList(
				"QGram (Ukkonen) Distance",
				"Moments Similarity",
				"CC Segment Start Location",
				"Longest-Common-Subsequence",
				"Match Model Glb. Str. Angles",
				"Delaunay Weka Match");
						List<String> ssaChoices = Arrays.asList(
								  "NGram Distance");
				List<String> ssaChoices = Arrays.asList("Match Model Glb. Str. Angles")
		List<String> ssaChoices = Arrays.asList("Delaunay Weka Match"); */

		/*
		List<String> ssaChoices = Arrays.asList(
			"Moments Similarity",
			"Delaunay No ML");
		localGlobal_graph(cm_al_ms, container, filename,
						  pa, mode, debug_flag, cm, ssaChoices, imageType, imageRotation, delaunay_calc, classiferPref);

		return cm;
		*/

		list<string> ssaChoices = {
			"Moments Similarity",
			"Delaunay No ML"
		};
		localGlobal_graph(cm_al_ms, container, filename,
						  pa, mode, debug_flag, *cm, ssaChoices, imageType, imageRotation, delaunay_calc, classiferPref);
		return cm;

	}
	/**
	 * Use data from OpenCV kmeans algorithm to partition image data
	 * @param data -- input image
	 * @param labels -- i/o integer array that stores the cluster indices
	 * for every sample
	 * @param centers -- Output matrix of the cluster centers, one row per
	 * each cluster center.
	 * @return partitioned image
	 */
	static kMeansNGBContainer opencv_kmeans_postProcess(Mat* data, Mat* labels, Mat* centers)
	{
		if (data->channels() == 3)
		{
			data->reshape(3);
		}

		/* Setup data structure holding partitioned image data */
		Mat* clustered_data = new Mat(data->rows, data->cols,
									 data->type(), Scalar(0));
		unordered_map<string, Mat*> stats;

		/* Keep stats on partitioning process */
		// Map<Integer, Integer> counts = new HashMap<Integer, Integer>();
		unordered_map<int, int> counts;
		for (int i = 0; i < centers->rows; i++)
			counts[i] = 0;

		/* Run image against centroids and assign pixels to clusters */
		int data_height = data->rows;
		int data_width = data->cols;

		// MinMaxLocResult mmlr = minMaxLoc(labels);
		struct {
			double minVal, maxVal;
			Point minLoc, maxLoc;
		} typedef MinMaxLocResult;
		MinMaxLocResult mmlr;
    	cv::minMaxLoc(*labels, &mmlr.minVal, &mmlr.maxVal, &mmlr.minLoc, &mmlr.maxLoc);

		if (data->channels() == 3)
		{
			/* For each pixel in the image */
			for (int y = 0; y < data_height; y++)
			{
				for (int x = 0; x < data_width; x++)
				{
					/* Get the cluster the pixel is assigned to
					 * label is in 1D format*/
					int labelb = (int)labels->at<Vec3b>(y, x)[0];
					int labelg = (int)labels->at<Vec3b>(y, x)[1];
					int labelr = (int)labels->at<Vec3b>(y, x)[2];

					/* Update stats that this pixel will get assigned to
					 * label specified cluster */
					counts.at(labelb) = counts.at(labelb) + 1;

					/* Copy pixel into cluster data structure with the color
					 * of the specified centroid */
					/*
					clustered_data.put(y, x,
									   ((labelb + mmlr.minVal) / mmlr.maxVal) * 255,
									   ((labelg + mmlr.minVal) / mmlr.maxVal) * 255,
									   ((labelr + mmlr.minVal) / mmlr.maxVal) * 255);
					*/
					clustered_data->at<Vec3b>(y, x)[0] = ((labelb + mmlr.minVal) / mmlr.maxVal) * 255;
					clustered_data->at<Vec3b>(y, x)[1] = ((labelg + mmlr.minVal) / mmlr.maxVal) * 255;
					clustered_data->at<Vec3b>(y, x)[2] = ((labelr + mmlr.minVal) / mmlr.maxVal) * 255;
					
				}
			}
		}
		else if (data->channels() == 1)
		{
			/* For each pixel in the image */
			for (int y = 0; y < data_height; y++)
			{
				for (int x = 0; x < data_width; x++)
				{
					/* Get the cluster the pixel is assigned to
					 * label is in 1D format*/
					int label = (int)labels->at<int>(y, x, 0);

					/* Update stats that this pixel will get assigned to
					 * label specified cluster */
					counts[label] = counts[label] + 1;

					/* Copy pixel into cluster data structure with the color
					 * of the specified centroid */
					//clustered_data.put(y, x, ((label + mmlr.minVal) / mmlr.maxVal) * 255);
					clustered_data->at<int>(y, x) = ((label + mmlr.minVal) / mmlr.maxVal) * 255;
				}
			}
		}

		// Print counts map and get its keySet
		set<int> region_cnts;
		for (const auto& pair : counts) {
			cout << pair.first << " : " << pair.second << endl;
		};
		transform(counts.begin(), counts.end(), region_cnts.begin(), [](auto pair) {
			return pair.first;
		});
		
		for (int rc : region_cnts)
		{
			Mat m(1, 1, CV_32FC1);
			int cnt = counts[rc];
			m.at<int>(0, 0) = cnt;
			stats.at(to_string(rc)) = &m;
		}
		kMeansNGBContainer kmNGBCnt(*clustered_data, stats); //Problem
		clustered_data->release();
		
		return kmNGBCnt;
	}

	/**
	 * Calculate the local global graph -- this portion constitutes the
	 * image analysis and feature recognition portion of the system
	 *
	 * @param Segments   -- list of image segments from segmentation process
	 *                       with edge detection applied
	 * @param n -- data from application of kMeans algorithm
	 * @param filename   -- name of file being worked on
	 * @param pa         -- partitioning algorithm used
	 * @param mode       -- model or sample image
	 * @param debug_flag -- Whether or not to generate certain types of output
	 * @param cm         -- Composite Matrix of image data (contains unmodified
	 *                       image segments)
	 * to aid in verification or troubleshooting activities
	 * @param ssaChoices  -- String similarity algorithm choices
	 * @param imageType  -- process image as standard (S), synthesis image (Y), rotated standard (R),
	 * rotated synthesis (Z), sample matching/not applicable (X)
	 * @param imageRotation -- rotation of image (and subsequently segments)
	 * @param delaunay_calc -- preform Delaunay graph generation
	 * @parm classifierPref -- for Weka ML use of Delaunay graph, classifer to use
	 * @return the local global graph description of the image
	 */
	static vector<LGNode> localGlobal_graph(vector<Mat*> Segments,
											   kMeansNGBContainer* kMeansData,
											   string filename,
											   Partitioning_Algorithm pa,
											   Mode mode, bool debug_flag,
											   CompositeMat cm, list<String> ssaChoices,
											   char imageType, short imageRotation, bool delaunay_calc,
											   string classiferPref)
	{
		// Data structures for sample image
		unordered_map<int, string> sampleChains;
		unordered_map<int, Point> sampleMoments;
		vector<Point> sampleccStartPts;

		// Connect to database
		int segmentNumber = 1;
		int lastSegNumDb = -1;
		if (mode == PROCESS_MODEL)
		{
			// No :)
			lastSegNumDb = DatabaseModule.getLastId();
			cout << "localGlobal_graph(): Last used id: " <<
							   lastSegNumDb << endl;

			// Initialize database if necessary
			if (DatabaseModule.doesDBExist() != DatabaseModule.NUMBER_RELATIONS)
			{
				DatabaseModule.createModel();
			}
		}

		/* Initialize the Global Graph based on number of segments from
		   partitioning  */
		Mat* clustered_data = nullptr;
		if (kMeansData != nullptr)
		{
			*clustered_data = kMeansData->getClustered_data();
		}
		vector<LGNode> global_graph = vector<LGNode>(Segments.size());
		int n = Segments.size();
		cout << "The global graph says there are " << n << " total segments" << endl;

		//
		vector<double> t1;
		vector<double> t2;

		// Initialize array to keep track of time to generate L-G graph on segment
		vector<long> timing_array;

		/* Initialize array to hold centroids in L-G graph
		   called S array in original MATLAB code */
		vector<Point> centroid_array(n);

		/* Initialize array to hold generated chain codes for
		 * each image segment*/
		ChainCodingContainer* ccc = nullptr;

		// Make sure output directory exists
		/*
		File outputDir = new File("output/");
		if (!outputDir.exists())
		{
			outputDir.mkdirs();
		}
		*/
		directory_entry outputDir("output/");
		if (!outputDir.exists()) {
			bool success = create_directories(outputDir.path());
			if (!success) {
				cerr << "Error creating directory: " << outputDir.path() << endl;
			}
		}

		/* This section does the following two things:
		   1. Construct the local portion of the Local Global graph..
		   this portion focuses on the geometric description of the
			line segments that  define the segment under analysis

			big hairy for loop that follows

		   2. Build the overall Global part of the Local-Global graph
		   the global portion focuses on the establishment of the centroid
		   regions and the connection of the starting segment centroid to
		   the other centroids in the other segments as a part of creating
		   an overall geometrical description of a model or target image
		   */
		for (int i = 0; i < n; i++)
		{
			long tic = stol(currentTime<nanoseconds>());

			/* Generate a representation of the segment based upon how
			 * the various border connected pixels are connected to one another  */
			Mat segment = Segments.at(i)[0].clone();
			ccc = chaincoding1(segment);
			if (debug_flag) {
				cout << ccc << endl;
			}

			t1.push_back(ccc->getChain_time());
			vector<double> cc = ccc->getCc();
			Point start = ccc->getStart();

			/* Use the chain code description of the segment to create a
			 * border */
			Mat border = ccc->getBorder();
			Mat* convertedborder;

			/* Down sample border into unsigned 8 bit integer value,
			 * far less taxing on CPU and memory  */
			if (border.type() != CV_8U)
			{
				convertedborder = new Mat(
					border.rows, border.cols, border.type());
				border.convertTo(*convertedborder, CV_8U);
			}
			else
			{
				*convertedborder = border;
			}

			/* Invert the colors but don't remove any data, allow the
			 * entire "color" range to make it through */
			cv::threshold(*convertedborder, *convertedborder, 0, 255, THRESH_BINARY_INV);

			/* if needed, verify results for chain code to border image
			 * generation for researcher*/
			if (debug_flag)
			{
				imwrite("output/" + filename.substr(filename.find_last_of('/') + 1, filename.find_last_of('.')) +
									  "_converted_border_" + to_string(i + 1) + "_" + currentTime() + ".jpg",
								  *convertedborder);
			}

			Mat* croppedBorder =
				new Mat(autoCropGrayScaleImage(*convertedborder, true));
			if (debug_flag)
			{
				imwrite("output/" + filename.substr(filename.find_last_of('/') + 1, filename.find_last_of('.')) +
									  "_cropped_border_" + to_string(i + 1) + "_" + currentTime() + ".jpg",
								  *croppedBorder);
			}
			ccc->setBorder(*croppedBorder);
			cout << "original border area=" + convertedborder->size().area() << endl;
			cout << "cropped border area=" + croppedBorder->size().area() << endl;
			if (convertedborder->size().area() < croppedBorder->size().area())
			{
				cout << "Cropped image is larger, outlier";
				cout << " Redo the chain code, as canny or similar filter was applied" << endl;
				ccc = chaincoding1(croppedBorder);
				cc = ccc->getCc();
				cout << "New chain code length is " + cc.size() << endl;
			}

			// release native memory on converted border
			convertedborder->release();

			/* Using the chain code from the previous step, generate
			 * the line segments of the segment using the greatest
			 * possible sensitivity */
			LineSegmentContainer lsc =
				line_segment(cc, start, 1);
			if (debug_flag)
				cout << lsc.toString() << endl;

			/* Generate a pictoral representation of the line segments
			 * using plplot and save to disk */

			/* Convert segment arrays into a format suitable for
			 * plplot use */
			vector<Mat> segx = lsc.getSegment_x();
			vector<Mat> segy = lsc.getSegment_y();

			/* if needed, show plplot output of segment's border */
			if (debug_flag)
			{
				vector<double> x = convertMat1xn(segx, true);
				vector<double> y = convertMat1xn(segy, true);

				/* Determine limits to set in plot graph for plplot
				 * establish environment and labels of plot
				 * Add ten pixels of padding for border
				 * data is reversed coming out of line segment  */
				double xmin = *min_element(y.begin(), y.end());
				double xmax = *max_element(y.begin(), y.end());
				double ymin = *min_element(x.begin(), x.end());
				double ymax = *max_element(x.begin(), x.end());

				// Initialize plplot
				plstream pls;
				// Parse and process command line arguments
				pls.parseopts(0, nullptr, PL_PARSE_FULL | PL_PARSE_NOPROGRAM);
				pls.setopt("verbose", "verbose");
				pls.setopt("dev", "jpeg");
				pls.scolbg(255, 255, 255); // set background to white
				pls.scol0(15, 0, 0, 0);	   // axis color is black
				pls.setopt("o", (outputDir.path().string() + "/" 
							+ filename.substr(filename.find_last_of('/') 
							+ 1, filename.find_last_of('.')) 
							+ "_line_segment_" + to_string(i + 1) + "_" 
							+ currentTime() + ".jpg").c_str());

				/* Initialize plplot,
				 * use a ten pixel border using inverted y axis graph
				 * to match pixel arrangement of pc monitor,
				 * and set the title */
				pls.init();
				pls.env(xmin - 10, xmax + 10, ymax + 10, ymin - 10, 0, 0);
				pls.lab("x", "y", ("Rebuilt Segment " + to_string(i + 1) + " Using Chain Code").c_str());

				/* Plot the data that was prepared above.
				   Data comes out reversed from line segment construction */
				PLINT numPoints = static_cast<PLINT>(x.size());
				PLFLT* plX = new PLFLT[numPoints];
				PLFLT* plY = new PLFLT[numPoints];
				for (PLINT i = 0; i < numPoints; i++) {
					plX[i] = static_cast<PLFLT>(x[i]);
					plY[i] = static_cast<PLFLT>(y[i]);
				}
				pls.line(numPoints, plY, plX);

				delete plX, plY;

				// Close PLplot library
				pls.eop();
			}

			/* Derive the local graph shape description of segment
			 * under consideration */
			long tic2 = System.nanoTime();
			vector<CurveLineSegMetaData> lmd = shape_expression(segx, segy);
			long toc2 = System.nanoTime();
			long duration2 = toc2 - tic2;
			cout << "Shape Expression Took: " + to_string(convert_ns(duration2, "MILLISECONDS")) + " ms" << endl;

			if (debug_flag)
			{
				System.out.println("Shape expression of segment " + (i + 1) + ":");
				System.out.println(lmd);
			}

			if (lmd != null)
			{
				tic2 = System.nanoTime();
				determine_line_connectivity(lmd);
				toc2 = System.nanoTime();
				duration2 = toc2 - tic2;
				System.out.println("Determining Line Connectivity Took: " +
								   TimeUnit.MILLISECONDS.convert(
									   duration2, TimeUnit.NANOSECONDS) +
								   " ms");
			}
			else
			{
				lmd = new ArrayList<CurveLineSegMetaData>();
				lmd.add(new CurveLineSegMetaData());
			}

			/* Store the amount of time it took to generate SH for
			 * segment i, see (1) and (2) in 2008 paper */
			t2.add((double)lsc.getSegment_time());
			double lg_time = t1.get(i) + t2.get(i);

			/* call S(i)  = regionprops(Segments(:,:,i), 'centroid');
			   Note moments have not been exposed through JNI on opencv 3.0 yet
			   Moments are used as part of curve matching, in particular to find
			   the scale parameter of an object s = (mom'/mom)^(1/2)

				From Bourbakis paper, moments help us to find the center point
				of a region...

				Should hold up to translation and rotation on a candidate object */
			double[][] img = ProjectUtilities.convertMatToDoubleArray(segment);
			System.out.println("Raw Moment " + (i + 1) + ":" + Moments.getRawCentroid(img));
			Point centroid = Moments.getRawCentroid(img);

			/* keep a copy of centroids for use in the construction of the
			   global portion of the geometric/pictorial description of the image

			   this will aid in future matching of multiple regions using this
			   method (section 3.0 of 2008 Bourbakis paper) */
			if ((centroid.x != Double.NaN) && (centroid.y != Double.NaN))
			{
				centroid_array.add(centroid);
			}

			// store time to generate LG graph on segment
			long toc = System.nanoTime();
			timing_array.add(toc - tic);

			/* Build ith node containing local node description, which
			 * forms a part of the overall global geometric description
			 * of the image */
			HashMap<String, Mat> stats = null;
			HashMap<String, Double> segment_stats = null;
			if (pa.equals(ProjectUtilities.Partitioning_Algorithm.NGB))
			{
				stats = kMeansData.getStats();
				segment_stats = new HashMap<String, Double>(3);
				Mat avRows = stats.get(avRowsString);
				Double averageRows = null;
				if ((avRows != null) && (avRows.get(0, i) != null))
				{
					averageRows = avRows.get(0, i)[0];
				}
				else
				{
					System.out.println("WARNING: No Row stats to retrieve");
				}

				Mat avCols = stats.get(avColsString);
				Double averageColumns = null;
				if ((avCols != null) && (avCols.get(0, i) != null))
				{
					averageColumns = avCols.get(0, i)[0];
				}
				else
				{
					System.out.println("WARNING: No Column stats to retrieve");
				}
				Mat avIntensity = stats.get(avIntString);
				Double averageIntensity = null;
				if ((avIntensity != null) && (avIntensity.get(0, i) != null))
				{
					averageIntensity = avIntensity.get(0, i)[0];
				}
				else
				{
					System.out.println("WARNING: No Intensity stats to retrieve");
				}

				/* warning: segments added from region growing do not contain
				 * Statistical data, they will be null */
				segment_stats.put(avRowsString, averageRows);
				segment_stats.put(avColsString, averageColumns);
				segment_stats.put(avIntString, averageIntensity);
			}
			else if (pa.equals(ProjectUtilities.Partitioning_Algorithm.OPENCV))
			{
				if (kMeansData != null)
				{
					stats = kMeansData.getStats();
				}
				segment_stats = new HashMap<String, Double>();

				Set<String> statKeys = null;
				if (stats != null)
				{
					statKeys = stats.keySet();
				}
				else
				{
					statKeys = new HashSet<String>();
				}

				for (String s : statKeys)
				{
					Mat m = stats.get(s);
					Double d = m.get(0, 0)[0];
					segment_stats.put(s, d);
				}
			}

			/* Create the node */
			LGNode lgnode = new LGNode(centroid, segment_stats,
									   croppedBorder, lmd, pa, i);

			/* Add local region info to overall global description */
			global_graph.add(lgnode);

			// release memory for objects copied to node and not used again
			croppedBorder.release();

			/* Add entry into database if part of a model image */
			if (mode == Mode.PROCESS_MODEL)
			{
				int id = DatabaseModule.insertIntoModelDBLocalRelation(filename,
																	   segmentNumber++,
																	   ccc.chainCodeString(),
																	   start, imageType, imageRotation);

				System.out.println("Added id " + id + " into database ");
			}
			else
			{
				// add to data structure
				sampleChains.put(i, ccc.chainCodeString());
				sampleMoments.put(i, centroid_array.get(i));
				sampleccStartPts.add(start);
			}

			/* Debug -- show info about region to a human */
			if (debug_flag)
				System.out.println(lgnode.toString());
		} // end big hairy for loop on building local nodes 100s loc earlier

		// Initialize plplot stream object
		PLStream pls = new PLStream();

		// Parse and process command line arguments
		pls.parseopts(new String[]{""}, PL_PARSE_FULL | PL_PARSE_NOPROGRAM);
		pls.setopt("verbose", "verbose");
		pls.setopt("dev", "jpeg");
		pls.scolbg(255, 255, 255); // set background to white
		pls.setopt("o", "output/" + filename.substring(filename.lastIndexOf('/') + 1, filename.lastIndexOf('.')) +
							"_centroids_for_image" + "_" + System.currentTimeMillis() + ".jpg");

		// Initialize plplot engine
		pls.init();

		/* Convert Point objects into a format suitable for
		 * use by plplot
		 */
		int sizeConversion = centroid_array.size();
		double[] xValues = new double[centroid_array.size()];
		double[] yValues = new double[centroid_array.size()];
		int sizeForLines = centroid_array.size() * 2;
		double[] xValuePrime = new double[sizeForLines - 1];
		double[] yValuePrime = new double[sizeForLines - 1];
		double startingX = centroid_array.get(0).x;
		double startingY = centroid_array.get(0).y;
		for (int cnt = 0; cnt < sizeConversion; cnt++)
		{
			xValues[cnt] = centroid_array.get(cnt).x;
			yValues[cnt] = centroid_array.get(cnt).y;
		}
		int indexOtherArray = 1;
		for (int cnt = 0; indexOtherArray < sizeConversion; cnt += 2)
		{
			xValuePrime[cnt] = startingX;
			xValuePrime[cnt + 1] = centroid_array.get(indexOtherArray).x;
			yValuePrime[cnt] = startingY;
			yValuePrime[cnt + 1] = centroid_array.get(indexOtherArray++).y;
		}

		/* Determine limits to set in plot graph for plplot
		 * establish environment and labels of plot */
		double xmin = ProjectUtilities.findMin(xValues);
		double xmax = ProjectUtilities.findMax(xValues);
		double ymin = ProjectUtilities.findMin(yValues);
		double ymax = ProjectUtilities.findMax(yValues);
		pls.env(xmin, xmax, ymax, ymin, 0, 0);
		pls.lab("x", "y", "Centroids for image " + filename.substring(filename.lastIndexOf('/')));

		// Plot the data that was prepared above.
		// Symbol 25 is a medium sized circle glyph
		pls.poin(xValues, yValues, 25);
		pls.col0(3);
		pls.line(xValuePrime, yValuePrime);

		// Close PLplot library
		pls.end();

		// Display timing data for each segment in algorithm
		if (debug_flag)
		{
			Long T = 0l;
			int cntTs = 1;
			for (Long l : timing_array)
			{
				System.out.println("Time to generate segment " + cntTs++ + " is " +
								   TimeUnit.MILLISECONDS.convert(l,
																 TimeUnit.NANOSECONDS) +
								   " ms");
				T += l;
			}
		}

		/* Build the structures needed for the displaying of the LG
		 * graph over the segmented image */
		Mat C = new Mat(2, n, CvType.CV_64FC1);
		// Setup for getting directional vectors from centroids
		// C(:,i) = S(1,i).Centroid; and C = floor (C);
		// x's are in first row, y's in second
		for (int i = 0; i < n; i++)
		{
			Point Sp = centroid_array.get(i);
			C.put(0, i, Math.floor(Sp.x));
			C.put(1, i, Math.floor(Sp.y));
		}

		/* Carryover of code from matlab for generating line
		 * segments from the start node region's centroid to
		 * every other region's centroid ?*/
		long tic = System.nanoTime();
		double[][] DirVector1 = new double[1][2];
		double[][] DirVector2 = new double[1][2];
		double[] Angle = new double[n];
		/* Calculate the distance and angle of each line from start node
		 * centroid to local graph node centroid
		 *
		 *  Does this get used by plplot? -- TODO USE THIS IN MATCHING FRAMEWORK*/
		for (int i = 0; i < n - 1; i++)
		{
			if (i == n - 2)
			{
				// DirVector1 = C(:,1)' - C(:,1+i)';
				DirVector1[0][0] = (C.get(0, 0)[0] - C.get(0, i)[0]);
				DirVector1[0][1] = (C.get(1, 0)[0] - C.get(1, i)[0]);
				// DirVector2 = C(:,1)' - C(:,2)';
				DirVector2[0][0] = (C.get(0, 0)[0] - C.get(0, 2)[0]);
				DirVector2[0][1] = (C.get(1, 0)[0] - C.get(1, 2)[0]);
				/* Angle(i) =
				 * acos( dot(DirVector1,DirVector2)/norm(DirVector1)/norm(DirVector2));
				 */
				Mat DirVector1Mat =
					ProjectUtilities.convertDoubletoGrayscaleMat(DirVector1, 1, 2);
				Mat DirVector2Mat =
					ProjectUtilities.convertDoubletoGrayscaleMat(DirVector2, 1, 2);
				double dotProduct = DirVector1Mat.dot(DirVector2Mat);
				Angle[i] = Math.acos(dotProduct / Core.norm(DirVector1Mat) /
									 Core.norm(DirVector2Mat));
			}
			else
			{
				// DirVector1 = C(:,1)' - C(:,1+i)';
				DirVector1[0][0] = (C.get(0, 0)[0] - C.get(0, i)[0]);
				DirVector1[0][1] = (C.get(1, 0)[0] - C.get(1, i)[0]);
				// DirVector2 = C(:,1)' - C(:,2)';
				DirVector2[0][0] = (C.get(0, 0)[0] - C.get(0, 1)[0]);
				DirVector2[0][1] = (C.get(1, 0)[0] - C.get(1, 1 + i)[0]);
				/* Angle(i) =
				 * acos( dot(DirVector1,DirVector2)/norm(DirVector1)/norm(DirVector2));
				 */
				Mat DirVector1Mat =
					ProjectUtilities.convertDoubletoGrayscaleMat(DirVector1, 1, 2);
				Mat DirVector2Mat =
					ProjectUtilities.convertDoubletoGrayscaleMat(DirVector2, 1, 2);
				double dotProduct = DirVector1Mat.dot(DirVector2Mat);
				Angle[i] = Math.acos(dotProduct / Core.norm(DirVector1Mat) /
									 Core.norm(DirVector2Mat));
			}
		}
		long angle_time = System.nanoTime() - tic;
		System.out.println("Time to calculate angle_time: " +
						   TimeUnit.MICROSECONDS.convert(
							   angle_time, TimeUnit.NANOSECONDS) +
						   " us");

		/* Build the line segments, grab the coordinates of the centroids and
		 * clustered data and pass to the constructLines routine
		 *
		 *  This will connect each centroid to the start node centroid. This
		 *  is different from supplying the set of centroids coordinates to
		 *  generate the delaunay triangulation, from which the edges are needed
		 *  for the SimG and S_ANGSIM calcs, I believe.
		 *
		 *  Delaunay triangulation from set of centroids  gives us a a set of triangles
		 *  whose points corresponds to the centroids, so no centroid is in the circumcircle
		 *  of any circle that could drawn on the plane of the triangles. */
		Mat lined = null;
		if (clustered_data != null)
		{
			lined = clustered_data.clone();
		}
		else
		{
			lined = new Mat();
		}

		for (int i = 0; i < n; i++)
		{
			System.out.println("Building lines for segment " + i);

			// coords = [C(2,1) C(1,1);C(2,i+1) C(1,i+1)];
			// Get coordinates of start node and target node
			Mat coords = new Mat(2, 2, CvType.CV_64FC1);
			coords.put(0, 0, C.get(1, 0));
			coords.put(0, 1, C.get(0, 0));
			coords.put(1, 0, C.get(1, i));
			coords.put(1, 1, C.get(0, i));

			// lined = plotlines(lined, coords);
			/* Build plot line from source to target/dest node */
			lined = constructLines(lined, coords);
		}

		/*
		Mat border = null;
		if (ccc != null) {
			border = ccc.getBorder().clone();
		}

		// [y1] = zeros(size(border));
 * 		Mat y1 = null;
		if (border != null) {
			Size sz = border.size();
			y1 = new Mat((int)sz.height, (int)sz.width,
						 border.type(), Scalar.all(0));
		}*/
		/* highlight the starting and ending points
		 * connecting the segments with red circles
		 */

		// th = 0:pi/50:2*pi;
		Mat th = new Mat(1, 101, CvType.CV_64FC1);
		double angle = 0;
		for (int i = 0; i < n; i++, angle += (Math.PI) / 50)
		{
			th.put(0, i, angle);
		}

		/* DEBUG print the final moments for me to review
		 * Superimpose moments over clustered image data
		 * and write image data out to disk in an excel
		 * spreadsheet
		 *
		 *  Start by creating file itself */
		Mat clustered_data_clone = null;
		if (clustered_data != null)
		{
			clustered_data_clone = clustered_data.clone();
		}
		else
		{
			System.err.println("No clustered data to clone");
		}
		int index = 0;
		XSSFWorkbook workbook = new XSSFWorkbook();
		FileOutputStream fileOut = null;
		try
		{
			fileOut =
				new FileOutputStream("output/" +
									 filename.substring(filename.lastIndexOf('/') + 1,
														filename.lastIndexOf('.')) +
									 "_" + System.currentTimeMillis() + ".xlsx");
		}
		catch (FileNotFoundException e1)
		{
			System.err.println("File not found exception: " + e1.getMessage());
			e1.printStackTrace();
		}

		/* Create moments sheet/tab within spreadsheet file */
		XSSFSheet sheet = workbook.createSheet(
			filename.substring(filename.lastIndexOf('/') + 1,
							   filename.lastIndexOf('.')) +
			"_Moments" + "_" + System.currentTimeMillis());
		XSSFRow headerRow = sheet.createRow(index);
		XSSFCell headerCell = headerRow.createCell(0);
		headerCell.setCellValue("Moment");
		headerCell = headerRow.createCell(1);
		headerCell.setCellValue("X");
		headerCell = headerRow.createCell(2);
		headerCell.setCellValue("Y");
		headerCell = headerRow.createCell(3);
		headerCell.setCellValue("Distance (from Start)");
		Point startCentroid = centroid_array.get(0);

		/* writing to standard out, to image data structure,
		 * and spreadsheet */
		int caCnt = 0;
		for (Point p : centroid_array)
		{
			/* Write moment to standard output */
			System.out.println("Moment " + caCnt + ": " +
							   p.x + "," + p.y);
			caCnt++;

			/* Superimpose moment as a line from the starting
			 * region to the ith region center of mass */
			if (clustered_data_clone != null)
			{
				Imgproc.circle(
					clustered_data_clone, centroid_array.get(index), 5,
					new Scalar(25, 25, 112));
				Imgproc.line(clustered_data_clone, centroid_array.get(0),
							 centroid_array.get(index), new Scalar(25, 25, 112));
			}

			/* Fill in ith row of the spreadsheet with the ith
			 * moment */
			XSSFRow row = sheet.createRow(index + 1);
			XSSFCell cell = row.createCell(0);
			cell.setCellValue(index);
			cell = row.createCell(1);
			cell.setCellValue(p.x);
			cell = row.createCell(2);
			cell.setCellValue(p.y);
			double d = ProjectUtilities.distance(
				startCentroid.x, p.x, startCentroid.y, p.y);
			cell = row.createCell(3);
			cell.setCellValue(d);
			/* Work with next region (node) */
			index++;
		}
		if (clustered_data_clone != null)
		{
			boolean imWriteResult =
				Imgcodecs.imwrite("output/" + filename.substring(filename.lastIndexOf('/') + 1) + "_moments_over_clustered_data" + "_" + System.currentTimeMillis() + ".jpg",
								  clustered_data_clone);
			System.out.println("Result of merging centroids onto clustered image: " + imWriteResult);
		}

		/* Calculate angle threshold differences and write them out to
		 * the spreadsheet*/
		Mat angle_differences = calc_angle_differences(ccc.getStart(), centroid_array);
		XSSFSheet arc_sheet = workbook.createSheet("Global_Props" + "_" + System.nanoTime());
		XSSFRow headerRowarc = arc_sheet.createRow(0);
		XSSFCell headerCellarc = headerRowarc.createCell(0);
		headerCellarc.setCellValue("Node");
		headerCellarc = headerRowarc.createCell(1);
		headerCellarc.setCellValue("\u0398" + "1");
		headerCellarc = headerRowarc.createCell(2);
		headerCellarc.setCellValue("\u0398" + "2");
		headerCellarc = headerRowarc.createCell(3);
		headerCellarc.setCellValue("Size/Area (pixels)");
		headerCellarc = headerRowarc.createCell(4);
		for (int i = 0; i < angle_differences.rows(); i++)
		{

			// report angle thresholds for node
			XSSFRow row = arc_sheet.createRow(i + 1);
			XSSFCell cell = row.createCell(0);
			cell.setCellValue(i);
			cell = row.createCell(1);
			cell.setCellValue(angle_differences.get(i, 0)[0]);
			cell = row.createCell(2);
			cell.setCellValue(angle_differences.get(i, 1)[0]);
			cell = row.createCell(3);
			cell.setCellValue(global_graph.get(i).getSize());
		}

		if (mode == Mode.PROCESS_MODEL)
		{
			for (int i = 0; i < segmentNumber - 1; i++)
			{
				// copy into database global table
				double d = ProjectUtilities.distance(
					startCentroid.x, centroid_array.get(i).x,
					startCentroid.y, centroid_array.get(i).y);

				System.out.println("localGlobal_graph(): inserting into global table, working on segment: " + i);
				DatabaseModule.insertIntoModelDBGlobalRelation(
					(centroid_array.get(i) != null) ? centroid_array.get(i) : new Point(0, 0),
					d,
					(angle_differences.get(i, 0) != null) ? angle_differences.get(i, 0)[0] : 0.0,
					(angle_differences.get(i, 1) != null) ? angle_differences.get(i, 1)[0] : 0.0,
					(global_graph.get(i) != null) ? global_graph.get(i) : new LGNode());
			}
		}

		// Build Delaunay Triagulation from centroid set
		Mat delaunay_angle_differences = null;
		List<Point> convertedTriangleList = null;
		if (delaunay_calc)
		{
			System.out.println("Delaunay stuff");
			Rect r = ProjectUtilities.calcRectCoveringPts(centroid_array);
			System.out.println("Rect=" + r.width + " by " + r.height);
			Subdiv2D subdiv = new Subdiv2D(r);
			int ptCnt = 0;
			for (Point p : centroid_array)
			{
				if (!Double.isNaN(p.x) && !Double.isNaN(p.y))
				{
					System.out.println("Adding point " + ptCnt + ":" + p);
					subdiv.insert(p);
					ptCnt++;
				}
			}
			MatOfFloat6 triangleList = new MatOfFloat6();
			subdiv.getTriangleList(triangleList);
			/* Flatten the triad pairs or six values that make up each triangle into a format easier to use
			 * in drawing the Delaunay graph */
			convertedTriangleList = ProjectUtilities.convertMatOfFloat6(triangleList);
			Mat clustered_data_clone2 = null;
			if (clustered_data != null)
			{
				clustered_data_clone2 = clustered_data.clone();
				for (int i = 0; i < convertedTriangleList.size() - 1; i += 3)
				{
					/* Draw the three vertices of the Delaunay Triangle */
					Imgproc.circle(
						clustered_data_clone2, convertedTriangleList.get(i), 5,
						new Scalar(25, 25, 112));
					Imgproc.circle(
						clustered_data_clone2, convertedTriangleList.get(i + 1), 5,
						new Scalar(25, 25, 112));
					Imgproc.circle(
						clustered_data_clone2, convertedTriangleList.get(i + 2), 5,
						new Scalar(25, 25, 112));

					/* Connect the vertices of each Delaunay triangle*/
					Imgproc.line(clustered_data_clone2, convertedTriangleList.get(i),
								 convertedTriangleList.get(i + 1), new Scalar(25, 25, 112));
					Imgproc.line(clustered_data_clone2, convertedTriangleList.get(i + 1),
								 convertedTriangleList.get(i + 2), new Scalar(25, 25, 112));
					Imgproc.line(clustered_data_clone2, convertedTriangleList.get(i + 2),
								 convertedTriangleList.get(i), new Scalar(25, 25, 112));
				}
				Imgcodecs.imwrite("output/" + filename.substring(filename.lastIndexOf('/') + 1) + "_delaunay_tri_" + System.currentTimeMillis() + ".jpg", clustered_data_clone2);
			}
			triangleList.release();

			if (clustered_data_clone2 != null)
			{
				clustered_data_clone2.release();
			}

			delaunay_angle_differences = calc_angle_differences(convertedTriangleList);
			if (mode == Mode.PROCESS_MODEL)
			{
				// covert sample angle differences into suitable format for processing
				float[] upperSampleThresholds = new float[delaunay_angle_differences.rows()];
				float[] lowerSampleThresholds = new float[delaunay_angle_differences.rows()];
				for (int i = 0; i < delaunay_angle_differences.rows(); i++)
				{
					upperSampleThresholds[i] = (float)delaunay_angle_differences.get(i, 0)[0];
					lowerSampleThresholds[i] = (float)delaunay_angle_differences.get(i, 1)[0];
				}
				double simGModel = graphSimilarity(lowerSampleThresholds, upperSampleThresholds);
				System.out.println("SIM_G Score for Model Image: " + simGModel);

				DatabaseModule.insertIntoModelDBGlobaMetaRelation(filename, simGModel);
				delaunay_angle_differences.release();
			}

			// store delaunay graph in database for model images
			// NOTE: do not release delaunay angle differences here for sample image, it needs a separate
			// matching thread action below
			if (mode == Mode.PROCESS_MODEL)
			{
				DatabaseModule.insertIntoModelDBGblDelGraph(filename, convertedTriangleList);
				convertedTriangleList.clear();
			}
		}

		// Free up resources used for spreadsheet
		try
		{
			workbook.write(fileOut);
			fileOut.close();
			workbook.close();
		}
		catch (IOException e)
		{
			e.printStackTrace();
		}

		// if matching phase, call match method
		if (mode == Mode.PROCESS_SAMPLE)
		{
			XSSFWorkbook wkbkResults = new XSSFWorkbook();
			buildSummarySheet(wkbkResults);

			/* Chaincode matching methods */
			Thread levenshtein_thread = null;
			if (ssaChoices.contains("LevenShtein") || ssaChoices.contains("all"))
			{
				levenshtein_thread = new Thread("Levenshtein"){
					public void run(){
						System.out.println("Matching using Levenshtein measure");
				match_to_model_Levenshtein(sampleChains, wkbkResults);
			}
		};
		System.out.println("Running thread: " + levenshtein_thread.getName());
		levenshtein_thread.start();
	}

	Thread n_levenshtein_thread = null;
	if (ssaChoices.contains("Normalized Levenshtein") || ssaChoices.contains("all"))
	{
		n_levenshtein_thread = new Thread("Normalized Levenshtein"){
			public void run(){
				System.out.println("Matching using Normalized Levenshtein measure");
		match_to_model_Normalized_Levenshtein(sampleChains, wkbkResults);
	}
};
n_levenshtein_thread.start();
System.out.println("Running thread: " + n_levenshtein_thread.getName());
}

Thread damerau_levenshtein_thread = null;
if (ssaChoices.contains("Damerau Levenshtein") || ssaChoices.contains("all"))
{
	damerau_levenshtein_thread = new Thread("Damerau Levenshtein"){
		public void run(){
			System.out.println("Matching using Damerau-Levenshtein");
	match_to_model_Damerau_Levenshtein(sampleChains, wkbkResults);
}
}
;
damerau_levenshtein_thread.start();
System.out.println("Running thread: " + damerau_levenshtein_thread.getName());
}

Thread ost_thread = null;
if (ssaChoices.contains("Optimal String Alignment") || ssaChoices.contains("all"))
{
	ost_thread = new Thread("Optimal String Alignment"){
		public void run(){
			System.out.println("Optimal String Alignment");
	match_to_model_Opt_Str_Alignment(sampleChains, wkbkResults);
}
}
;
ost_thread.start();
System.out.println("Running thread: " + ost_thread.getName());
}

Thread jaro_thread = null;
if (ssaChoices.contains("Jaro-Winkler") || ssaChoices.contains("all"))
{
	jaro_thread = new Thread("Jaro-Winkler"){
		public void run(){
			System.out.println("Jaro-Winkler");
	match_to_model_Jaro_Winkler(sampleChains, wkbkResults);
}
}
;
jaro_thread.start();
System.out.println("Running thread: " + jaro_thread.getName());
}

/* this is equivalent to matching on string length for line segment properties
 * just that this is the long border chain code with fine resolution
 * string similarity match the chain codes in earlier experimental
 * runs, so let's do that here and add that into final equation match */
Thread lcs_thread = null;
if (ssaChoices.contains("Longest-Common-Subsequence") || ssaChoices.contains("all"))
{
	lcs_thread = new Thread("Longest-Common-SubSequence"){
		public void run(){
			System.out.println("Longest-Common-SubSequence");
	match_to_model_LCS(sampleChains, wkbkResults);
}
}
;
lcs_thread.start();
System.out.println("Running thread: " + lcs_thread.getName());
}

Thread mlcs_thread = null;
if (ssaChoices.contains("Metric Longest-Common-SubSequence") || ssaChoices.contains("all"))
{
	mlcs_thread = new Thread("Metric Longest-Common-SubSequence"){
		public void run(){
			System.out.println("Metric Longest-Common-SubSequence");
	match_to_model_MLCS(sampleChains, wkbkResults);
}
}
;
mlcs_thread.start();
System.out.println("Running thread: " + mlcs_thread.getName());
}

Thread ngram_thread = null;
if (ssaChoices.contains("NGram Distance") || ssaChoices.contains("all"))
{
	ngram_thread = new Thread("NGram Distance"){
		public void run(){
			System.out.println("NGram Distance");
	match_to_model_NGram_Distance(sampleChains, wkbkResults);
}
}
;
ngram_thread.start();
System.out.println("Running thread: " + ngram_thread.getName());
}

Thread qgram_thread = null;
if (ssaChoices.contains("QGram (Ukkonen) Distance") || ssaChoices.contains("all"))
{
	qgram_thread = new Thread("QGram (Ukkonen) Distance"){
		public void run(){
			System.out.println("QGram (Ukkonen) Distance");
	match_to_model_QGram_Distance(sampleChains, wkbkResults);
}
}
;
qgram_thread.start();
System.out.println("Running thread: " + qgram_thread.getName());
}

Thread cosSim_thread = null;
if (ssaChoices.contains("Cosine Similarity") || ssaChoices.contains("all"))
{
	cosSim_thread = new Thread("Cosine Similarity"){
		public void run(){
			System.out.println("Cosine Similarity");
	match_to_model_COS_Similarity(sampleChains, wkbkResults);
}
}
;
cosSim_thread.start();
System.out.println("Running thread: " + cosSim_thread.getName());
}

/* Ancillary match by moments */
Thread moments_thread = null;
if (ssaChoices.contains("Moments Similarity") || ssaChoices.contains("all"))
{
	moments_thread = new Thread("Moments Similarity"){
		public void run(){
			System.out.println("Moments Similarity");
	match_to_model_by_Moments(sampleMoments, wkbkResults, 0.10f);
}
}
;
moments_thread.start();
System.out.println("Running thread: " + moments_thread.getName());
}

Thread delaunay_no_ml_thread = null;
if (ssaChoices.contains("Delaunay No ML") || ssaChoices.contains("all"))
{
	final List<Point> copyConvertedTraingleList = new ArrayList<Point>(convertedTriangleList);
	delaunay_no_ml_thread = new Thread("Delaunay No ML"){
		public void run(){
			System.out.println("Delaunay No ML");
	match_to_model_by_Delaunay_Graph_NoML(wkbkResults, copyConvertedTraingleList, 0.03f);
}
}
;
delaunay_no_ml_thread.start();
System.out.println("Running thread: " + delaunay_no_ml_thread.getName());
}

/* Ancillary match by chain code start location */
Thread cc_segstart_thread = null;
if (ssaChoices.contains("CC Segment Start Location") || ssaChoices.contains("all"))
{
	cc_segstart_thread = new Thread("CC Segment Start Location"){
		public void run(){
			System.out.println("CC Segment Start Location");
	String matching_image_ccSegment =
		match_to_model_by_CC_Segment_Start(sampleccStartPts, wkbkResults);
	System.out.println("Segment Start was " + matching_image_ccSegment);
}
}
;
cc_segstart_thread.start();
System.out.println("Running thread: " + cc_segstart_thread.getName());
}

/*  match by global model similarity */
Thread matchGlbStrs_thread = null;
if (ssaChoices.contains("Match Model Glb. Str. Angles") || ssaChoices.contains("all"))
{
	matchGlbStrs_thread = new Thread("Match Model Glb. Str. Angles"){
		public void run(){
			System.out.println("Match Model Glb. Str. Angles");
	match_to_model_by_global_structure_angles(angle_differences, wkbkResults, "Sim_G Meas");
}
}
;
matchGlbStrs_thread.start();
System.out.println("Running thread: " + matchGlbStrs_thread.getName());
}

/* match by Delaunay model similarity */
Thread match_Delaunay_Weka_thread = null;
if (ssaChoices.contains("Delaunay Weka Match") || ssaChoices.contains("all"))
{
	final List<Point> copyConvertedTraingleList = new ArrayList<Point>(convertedTriangleList);
	match_Delaunay_Weka_thread = new Thread("Delaunay Weka Match"){
		public void run(){
			System.out.println("Delaunay Weka Match");
	match_to_model_by_Delaunay_Graph(wkbkResults, copyConvertedTraingleList, classiferPref);
}
}
;
match_Delaunay_Weka_thread.start();
System.out.println("Running thread: " + match_Delaunay_Weka_thread.getName());
}

try
{
	if (levenshtein_thread != null)
	{
		levenshtein_thread.join();
	}

	if (n_levenshtein_thread != null)
	{
		n_levenshtein_thread.join();
	}

	if (damerau_levenshtein_thread != null)
	{
		damerau_levenshtein_thread.join();
	}

	if (ost_thread != null)
	{
		ost_thread.join();
	}

	if (jaro_thread != null)
	{
		jaro_thread.join();
	}

	if (lcs_thread != null)
	{
		lcs_thread.join();
	}

	if (ngram_thread != null)
	{
		ngram_thread.join();
	}

	if (qgram_thread != null)
	{
		qgram_thread.join();
	}

	if (cosSim_thread != null)
	{
		cosSim_thread.join();
	}

	if (mlcs_thread != null)
	{
		mlcs_thread.join();
	}

	if (moments_thread != null)
	{
		moments_thread.join();
	}

	if (delaunay_no_ml_thread != null)
	{
		delaunay_no_ml_thread.join();
	}

	if (cc_segstart_thread != null)
	{
		cc_segstart_thread.join();
	}

	if (matchGlbStrs_thread != null)
	{
		matchGlbStrs_thread.join();
	}

	if (match_Delaunay_Weka_thread != null)
	{
		match_Delaunay_Weka_thread.join();
	}
}
catch (InterruptedException e1)
{
	// TODO Auto-generated catch block
	e1.printStackTrace();
}

/* Update summary sheet with final calcluations */
updateSummarySheet(wkbkResults);

/* Write results spreadsheet to disk */
FileOutputStream resultFile;
try
{
	resultFile = new FileOutputStream("output/match_of_" +
									  filename.substring(filename.lastIndexOf('/') + 1,
														 filename.lastIndexOf('.')) +
									  "_" + System.currentTimeMillis() + ".xlsx");
	synchronized(wkbkResults)
	{
		wkbkResults.write(resultFile);
		wkbkResults.close();
	}
}
catch (FileNotFoundException e)
{
	e.printStackTrace();
}
catch (IOException e)
{
	e.printStackTrace();
}
}

// finalize ids for image
cm.setStartingId(DatabaseModule.getStartId(filename));
cm.setLastId(DatabaseModule.getLastId(filename));
System.out.println(" IDs for image " + cm.getFilename() + " will be" +
				   cm.getStartingId() + " to " + cm.getLastId());

// return to caller
return global_graph;
}

/**
 * Update the summary sheet with the final probabilistic measures
 * @param wkbkResults
 */
private:
static void updateSummarySheet(XSSFWorkbook wkbkResults)
{
	XSSFSheet summarySheet = wkbkResults.getSheet(SUMMARY_SHEET);
	XSSFSheet weightSheet = wkbkResults.getSheet(WEIGHTS_SHEET);
	int lastRowNum = summarySheet.getLastRowNum();

	// for each row in the summary, don't forget to skip header row
	for (int i = summarySheet.getFirstRowNum() + 1; i < lastRowNum; i++)
	{
		XSSFRow row = summarySheet.getRow(i);
		int lastCellNum = row.getLastCellNum();

		// for each cell in ith row of the summary sheet, skip filename
		double total = 0;
		for (int j = 1; j < lastCellNum; j++)
		{
			XSSFCell cell = row.getCell(j);
			// get the weight
			XSSFRow weightRow = weightSheet.getRow(j);
			XSSFCell weightCell = null;
			if (weightRow != null)
			{
				weightCell = weightRow.getCell(1);
			}
			else
			{
				continue;
			}
			double weightValue = weightCell.getNumericCellValue();
			if (cell != null)
			{
				total += (cell.getNumericCellValue() * weightValue);
			}
		}

		// write final score Mj to cell
		XSSFCell totalScore = row.createCell(Mj_COLUMN_SUMMARY);
		totalScore.setCellValue(total);
	}
}

/**
 * Build the summary sheet page in the spreadsheet
 * This routine only builds the measure labels and the model filenames
 * You will need to fill in the values for each model for each type of
 * matching measure
 * @param wkbkResults -- the spreadsheet to work with
 */
private:
static void buildSummarySheet(XSSFWorkbook wkbkResults)
{
	XSSFSheet sheet = wkbkResults.createSheet(SUMMARY_SHEET);
	XSSFSheet weightSheet = wkbkResults.createSheet(WEIGHTS_SHEET);
	List<String> modelFilenames = DatabaseModule.getAllModelFileName();
	XSSFRow row = sheet.createRow(0);
	int colCounter = 1;
	XSSFCell cell = row.createCell(colCounter, CellType.STRING);
	cell.setCellValue(FILENAME_COLUMN_NAME);
	cell = row.createCell(colCounter++, CellType.STRING);
	cell.setCellValue(Si_COLUMN_NAME);
	cell = row.createCell(colCounter++, CellType.STRING);
	cell.setCellValue(Ci_COLUMN_NAME);
	cell = row.createCell(colCounter++, CellType.STRING);
	cell.setCellValue(CSi_COLUMN_NAME);
	cell = row.createCell(colCounter++, CellType.STRING);
	cell.setCellValue(LCSi_COLUMN_NAME);
	cell = row.createCell(colCounter++, CellType.STRING);
	cell.setCellValue(SIMG_COLUMN_Name);
	cell = row.createCell(colCounter++, CellType.STRING);
	cell.setCellValue(WEKA_DELAUNAY_COLUMN_NAME);
	cell = row.createCell(colCounter++, CellType.STRING);
	cell.setCellValue(Mj_COLUMN_NAME);
	cell = row.createCell(colCounter++, CellType.STRING);
	int i = 1;
	for (String model : modelFilenames)
	{
		row = sheet.createRow(i++);
		cell = row.createCell(0, CellType.STRING);
		cell.setCellValue(model);
	}

	// Build weights reference sheet -- weights should add to one
	row = weightSheet.createRow(0);
	cell = row.createCell(0, CellType.STRING);
	cell.setCellValue(ALPHA);
	cell = row.createCell(1, CellType.NUMERIC);
	cell.setCellValue(ALPHA_W);
	row = weightSheet.createRow(1);
	cell = row.createCell(0, CellType.STRING);
	cell.setCellValue(BETA);
	cell = row.createCell(1, CellType.NUMERIC);
	cell.setCellValue(BETA_W);
	row = weightSheet.createRow(2);
	cell = row.createCell(0, CellType.STRING);
	cell.setCellValue(GAMMA);
	cell = row.createCell(1, CellType.NUMERIC);
	cell.setCellValue(GAMMA_W);
	row = weightSheet.createRow(3);
	cell = row.createCell(0, CellType.STRING);
	cell.setCellValue(EPLISON);
	cell = row.createCell(1, CellType.NUMERIC);
	cell.setCellValue(EPLISON_W);
	row = weightSheet.createRow(4);
	cell = row.createCell(0, CellType.STRING);
	cell.setCellValue(ZETA);
	cell = row.createCell(1, CellType.NUMERIC);
	cell.setCellValue(ZETA_W);
	row = weightSheet.createRow(5);
	cell = row.createCell(0, CellType.STRING);
	cell.setCellValue(ETA);
	cell = row.createCell(1, CellType.NUMERIC);
	cell.setCellValue(ETA_W);
}

private:
static String match_to_model_by_CC_Segment_Start(ArrayList<Point> sampleccStartPts,
												 XSSFWorkbook wkbkResults)
{
	XSSFSheet sheet = null;
	synchronized(wkbkResults)
	{
		sheet = wkbkResults.createSheet("CCStartMeasure");
	}

	Map<String, Integer> modelFileCnts = new TreeMap<String, Integer>();

	/* for one chaincode starting segment in the sample image, find
	 * one matching model images
	 */
	for (Point ccStart : sampleccStartPts)
	{
		List<PointMatchContainer> pmcList = DatabaseModule.getImagesMatchingCCStart(ccStart);

		// if there any matches, count the number of times each model image was matched
		if (pmcList == null)
		{
			continue;
		}
		for (PointMatchContainer pmc : pmcList)
		{
			String filename = pmc.getMatch();
			if (modelFileCnts.containsKey(filename))
			{
				int cnt = modelFileCnts.get(filename);
				modelFileCnts.put(filename, ++cnt);
			}
			else
			{
				modelFileCnts.put(pmc.getMatch(), 1);
			}
		}
	}

	int rowNumber = 1;
	// build header
	synchronized(wkbkResults)
	{
		XSSFRow row = sheet.createRow(rowNumber);
		XSSFCell cell = row.createCell(0);
		cell.setCellValue("Model Image");
		cell = row.createCell(1);
		cell.setCellValue("Count");
		cell = row.createCell(2);
		cell.setCellValue("Match Prob.");
	}

	// find the key with the largest count since ordering is on keys
	Set<String> keys = modelFileCnts.keySet();
	String modelFilewithLargestCnt = "";
	int largestCnt = Integer.MIN_VALUE;
	for (String key : keys)
	{
		int fileCnt = modelFileCnts.get(key);
		synchronized(wkbkResults)
		{
			XSSFRow row = sheet.createRow(++rowNumber);
			XSSFCell cell = row.createCell(0);
			cell.setCellValue(key);
			cell = row.createCell(1);
			cell.setCellValue(fileCnt);
			cell = row.createCell(2);
			cell.setCellValue(((float)fileCnt) / sampleccStartPts.size());

			// update summary sheet as well for final calculation
			XSSFSheet summarySheet = wkbkResults.getSheet("Summary");
			int sumRowInt =
				ProjectUtilities.findRowInSpreadSheet(summarySheet, key);
			XSSFRow summaryRow = summarySheet.getRow(sumRowInt);
			XSSFCell summaryCell = summaryRow.createCell(CSi_COLUMN_SUMMARY, CellType.NUMERIC);
			summaryCell.setCellValue(((float)fileCnt) / sampleccStartPts.size());
		}
		System.out.println("Image " + key + " has " + fileCnt + " matching starting points");
		// sheet.createRow(arg0)
		if (fileCnt > largestCnt)
		{
			largestCnt = fileCnt;
			modelFilewithLargestCnt = key;
		}
	}

	// report and return the best model image for this type of match
	System.out.println("File with largest CC starting point count is " +
					   modelFilewithLargestCnt + " with a count of " + largestCnt);
	return modelFilewithLargestCnt;
}

/*
 * Match unknown to model by using centroids (raw momemnts) first order
 * @param sampleMoments -- centroids of unknown
 * @param wkbkResults -- place results into workbook
 * @param epsilon -- acceptable margin of error for approximate centroid match
 *
 * @return Updated spreadsheet with results
 */
private:
static void match_to_model_by_Moments(Map<Integer, Point> sampleMoments,
									  XSSFWorkbook wkbkResults, float epsilon)
{
	/* Simple Method:
	 * For each segment in Sample
	 *     Ask database for number of ids that match x and y moments
	 *
	 *  Announce best match
	 *
	 *   A More sophisticated method needs to look at partial regions
	 *   how close is close enough for a likely match or probable
	 *   match...
	 *
	 *   In a refined methodology:
	 *   We will use an approximate matching of centroids within +/- epsilon*/
	StringBuilder sb = new StringBuilder();
	int bestMatchesSz = 1;
	int cntMatchesSz = 1;
	if ((sampleMoments == null) || sampleMoments.size() == 0)
	{
		return;
	}
	else
	{
		bestMatchesSz = sampleMoments.size();
		cntMatchesSz = (int)(sampleMoments.size() * .1);
		if (cntMatchesSz < 1)
		{
			cntMatchesSz = 1;
		}
	}

	XSSFSheet sheet = null;
	synchronized(wkbkResults)
	{
		sheet = wkbkResults.createSheet("Moments");
	}

	Map<Integer, HashMap<Integer, Double>> bestMatches =
		new HashMap<Integer, HashMap<Integer, Double>>(
			sampleMoments.size(), (float)0.75);
	Map<String, Integer> cntMatches =
		new HashMap<String, Integer>(cntMatchesSz,
									 (float)0.90);

	Iterator<Integer> segments = sampleMoments.keySet().iterator();
	while (segments.hasNext())
	{
		Integer segment = segments.next();
		Point segmentMoment = sampleMoments.get(segment);
		sb.append("Working with sample segment Point " +
				  segment + " with coordinates (" + (int)segmentMoment.x + "," + (int)segmentMoment.y + ")" + "\n");
		ArrayList<String> names = DatabaseModule.getFilesWithMoment(
			(int)segmentMoment.x, (int)segmentMoment.y, epsilon);
		sb.append("Returned " + names.size() + " model image(s)" + "\n");
		for (String name : names)
		{
			Integer cnt = cntMatches.get(name);
			if (cnt == null)
			{
				cntMatches.put(name, 1);
			}
			else
			{
				cntMatches.put(name, ++cnt);
			}
		}
	}
	String bestMatch = null;
	Integer bestMatchCnt = Integer.MIN_VALUE;
	Set<String> models = cntMatches.keySet();
	int rowNumber = 1;

	// build header
	synchronized(wkbkResults)
	{
		XSSFRow row = sheet.createRow(rowNumber);
		XSSFCell cell = row.createCell(0);
		cell.setCellValue("Model Image");
		cell = row.createCell(1);
		cell.setCellValue("# Matching Moments");
		cell = row.createCell(2);
		cell.setCellValue("Match Prob.");
		cell = row.createCell(3);
		cell.setCellValue("Match Prob. Per.");
	}

	for (String model : models)
	{
		Integer cnt = cntMatches.get(model);

		/* record data in spreadsheet */
		synchronized(wkbkResults)
		{
			XSSFRow row = sheet.createRow(++rowNumber);

			XSSFCell cell = row.createCell(0);
			cell.setCellValue(model);

			cell = row.createCell(1);
			cell.setCellValue(cnt);

			cell = row.createCell(2, CellType.NUMERIC);
			cell.setCellValue(((double)cnt) / sampleMoments.size());

			cell = row.createCell(3, CellType.NUMERIC);
			cell.setCellValue((((double)cnt) / sampleMoments.size()) * 100);

			// update summary sheet as well for final calculation
			XSSFSheet summarySheet = wkbkResults.getSheet("Summary");
			int sumRowInt =
				ProjectUtilities.findRowInSpreadSheet(summarySheet, model);
			System.out.println("mtm_by_Moments(): Found lookup in row " + sumRowInt);

			XSSFRow summaryRow = null;
			if (summarySheet != null)
			{
				summaryRow = summarySheet.getRow(sumRowInt);
			}

			XSSFCell summaryCell = null;
			if (summaryRow != null)
			{
				summaryCell = summaryRow.createCell(Ci_COLUMN_SUMMARY, CellType.NUMERIC);
			}
			else
			{
				System.err.println("mtm_by_Moments(): " + "getRow called failed, cannot create cell");
			}

			if (summaryCell != null)
			{
				summaryCell.setCellValue(((double)cnt) / sampleMoments.size());
			}
			else
			{
				System.err.println("mtm_by_Moments(): " + "createCell called failed, cannot set cell value");
			}
		}

		if (cnt > bestMatchCnt)
		{
			bestMatchCnt = cnt;
			bestMatch = model;
		}
	}
	double percentageMatch = ((double)bestMatchCnt) / sampleMoments.size();

	/* Make sure the best results stands out from the other data */
	synchronized(wkbkResults)
	{
		XSSFRow bestRow = sheet.createRow(++rowNumber);
		XSSFCellStyle style = wkbkResults.createCellStyle();
		XSSFFont font = wkbkResults.createFont();
		style.setBorderBottom(BorderStyle.THICK);
		style.setBorderTop(BorderStyle.THICK);
		font.setFontHeightInPoints((short)14);
		font.setBold(true);
		style.setFont(font);
		bestRow.setRowStyle(style);

		/* Record data in row of spreadsheet */
		XSSFCell bestCellinRow = bestRow.createCell(0);
		bestCellinRow.setCellValue(bestMatch);
		bestCellinRow.setCellStyle(style);
		bestCellinRow = bestRow.createCell(1, CellType.NUMERIC);
		bestCellinRow.setCellValue(percentageMatch);
		bestCellinRow.setCellStyle(style);
		bestCellinRow = bestRow.createCell(2, CellType.NUMERIC);
		bestCellinRow.setCellValue(percentageMatch * 100);
		bestCellinRow.setCellStyle(style);
	}

	sb.append("Best match using contours (moments) is " +
			  bestMatch + " with " + bestMatchCnt +
			  " contours matching and " + (percentageMatch * 100) + "% level of confidence" + "\n");
	System.out.println(sb.toString());
	System.out.println("Done running thread");
}

private:
static void match_to_model_COS_Similarity(
	Map<Integer, String> sampleChains, XSSFWorkbook wkbkResults)
{
	/* The closer to zero, the more similar the two strings as the
	 * two angles have a large cosine value */
	/* 1. Take each segment of sample image
	 *    for each model image
	 *        for each segment in model image
	 *            apply java-string-similarity method
	 *            O(n)+O(m*n^2)+Runtime_Algorithm */
	StringBuilder sb = new StringBuilder();
	int bestMatchesSz = 1;
	int cntMatchesSz = 1;
	if ((sampleChains == null) || sampleChains.size() == 0)
	{
		return;
	}
	else
	{
		bestMatchesSz = sampleChains.size();
		cntMatchesSz = (int)(sampleChains.size() * .1);
		if (cntMatchesSz < 1)
		{
			cntMatchesSz = 1;
		}
	}
	XSSFSheet sheet = null;
	synchronized(wkbkResults)
	{
		sheet = wkbkResults.createSheet("COS_SIM");
	}

	Map<Integer, ConcurrentHashMap<Integer, Double>> bestMatches =
		new ConcurrentHashMap<Integer, ConcurrentHashMap<Integer, Double>>(
			sampleChains.size(), (float)0.75);
	Map<String, Integer> cntMatches =
		new ConcurrentHashMap<String, Integer>(cntMatchesSz,
											   (float)0.90);

	Iterator<Integer> segments = sampleChains.keySet().iterator();
	int lastEntryID = DatabaseModule.getLastId();
	while (segments.hasNext())
	{
		Integer segment = segments.next();
		String segmentChain = sampleChains.get(segment);
		sb.append("Working with sample segment " + segment + "\n");
		AtomicFloat bestSimSoFar = new AtomicFloat(Float.MIN_VALUE);
		AtomicInteger minID = new AtomicInteger(Integer.MIN_VALUE);
		IntStream.rangeClosed(0, lastEntryID).parallel().forEach((i)->{
			/* Get the ith chain code from the database */
			String modelSegmentChain = DatabaseModule.getChainCode(i);

			/* 1 - similarity where similarity is: /**
			 * the cosine of the angle between
			 * these two vectors representation.
			 * It is computed as V1 . V2 / (|V1| * |V2|)
			 **/
			Cosine c = new Cosine(5);
			if ((segmentChain == null) || (modelSegmentChain == null))
			{
				System.err.println("COS: modelSegment null for chain code" + i);
				return;
			}
			double similarity = c.distance(segmentChain, modelSegmentChain);

			/* We want measures as close to one as possible*/
			if (Float.compare((float)similarity, bestSimSoFar.get()) > 0)
			{
				bestSimSoFar.set((float)similarity);
				minID.set(i);
			}
		});

		ConcurrentHashMap<Integer, Double> hm =
			new ConcurrentHashMap<Integer, Double>(1, (float)0.75);
		hm.put(minID.get(), bestSimSoFar.doubleValue());
		bestMatches.put(segment, hm);

		/* For each segment of the sample, track which model image
		 * and which image model perspective provides the best match*/
		String modelOfInterest = DatabaseModule.getFileName(minID.get());
		Integer curCnt = cntMatches.get(modelOfInterest);
		if (curCnt == null)
		{
			cntMatches.put(modelOfInterest, 1);
		}
		else
		{
			cntMatches.put(modelOfInterest, ++curCnt);
		}
	}

	/* Display result */
	Iterator<Integer> bmIterator = bestMatches.keySet().iterator();
	while (bmIterator.hasNext())
	{
		Integer key = bmIterator.next();
		ConcurrentHashMap<Integer, Double> minValue = bestMatches.get(key);
		Iterator<Integer> ii = minValue.keySet().iterator();
		while (ii.hasNext())
		{
			Integer idmin = ii.next();
			String filenameOfID = DatabaseModule.getFileName(idmin);
			sb.append("Best COS_SIM Match for segment " + key + " is " +
					  idmin + " (" + filenameOfID + ") with " +
					  minValue.get(idmin) + " measure" + "\n");
		}
	}

	/* Tell user probability of matching various images based on how well
	 * sample segments matched to the database of model images */
	Iterator<String> cntIterator = cntMatches.keySet().iterator();
	double bestProbMatch = Double.MIN_VALUE;
	String nameOfModelMatch = null;
	int probsCnt = 0;
	while (cntIterator.hasNext())
	{
		String filename = cntIterator.next();
		Integer count = cntMatches.get(filename);
		double probMatch = ((double)count) / sampleChains.size();
		sb.append("Probablity of matching " + filename + " is :" + (probMatch * 100) + " %" + "\n");

		/* record data in spreadsheet */
		synchronized(wkbkResults)
		{
			XSSFRow row = sheet.createRow(probsCnt++);
			XSSFCell cell = row.createCell(0);
			cell.setCellValue(filename);
			cell = row.createCell(1);
			cell.setCellValue(probMatch);
		}

		/* Track most likely match*/
		if (probMatch > bestProbMatch)
		{
			bestProbMatch = probMatch;
			nameOfModelMatch = filename;
		}
	}

	/* Tell user most likely match and record in spreadsheet */
	sb.append("Best probable match is " + nameOfModelMatch +
			  " with probablity " + bestProbMatch + "\n");

	synchronized(wkbkResults)
	{
		XSSFRow bestRow = sheet.createRow(probsCnt);

		/* Make sure the best results stands out from the other data */
		XSSFCellStyle style = wkbkResults.createCellStyle();
		XSSFFont font = wkbkResults.createFont();
		style.setBorderBottom(BorderStyle.THICK);
		style.setBorderTop(BorderStyle.THICK);
		font.setFontHeightInPoints((short)14);
		font.setBold(true);
		style.setFont(font);
		bestRow.setRowStyle(style);

		/* Record data in row of spreadsheet */
		XSSFCell bestCellinRow = bestRow.createCell(0);
		bestCellinRow.setCellValue(nameOfModelMatch);
		bestCellinRow.setCellStyle(style);
		bestCellinRow = bestRow.createCell(1);
		bestCellinRow.setCellValue(bestProbMatch);
		bestCellinRow.setCellStyle(style);
	}

	System.out.println(sb.toString());
	System.out.println("Done running thread");
}

private:
static void match_to_model_QGram_Distance(
	Map<Integer, String> sampleChains, XSSFWorkbook wkbkResults)
{
	/* 1. Take each segment of sample image
	 *    for each model image
	 *        for each segmnent in model image
	 *            apply java-string-similarity method
	 *            O(n)+O(m*n^2)+Runtime_Algorithm */
	StringBuilder sb = new StringBuilder();
	int bestMatchesSz = 1;
	int cntMatchesSz = 1;
	if ((sampleChains == null) || sampleChains.size() == 0)
	{
		return;
	}
	else
	{
		bestMatchesSz = sampleChains.size();
		cntMatchesSz = (int)(sampleChains.size() * .1);
		if (cntMatchesSz < 1)
		{
			cntMatchesSz = 1;
		}
	}

	XSSFSheet sheet = null;
	synchronized(wkbkResults)
	{
		sheet = wkbkResults.createSheet("QGram");
	}

	Map<Integer, ConcurrentHashMap<Integer, Integer>> bestMatches =
		new ConcurrentHashMap<Integer, ConcurrentHashMap<Integer, Integer>>(
			sampleChains.size(), (float)0.75);
	Map<String, Integer> cntMatches =
		new ConcurrentHashMap<String, Integer>(cntMatchesSz,
											   (float)0.90);

	Iterator<Integer> segments = sampleChains.keySet().iterator();
	int lastEntryID = DatabaseModule.getLastId();
	while (segments.hasNext())
	{
		Integer segment = segments.next();
		String segmentChain = sampleChains.get(segment);
		sb.append("Working with sample segment " + segment + "\n");
		AtomicInteger minDistance = new AtomicInteger(Integer.MAX_VALUE);
		AtomicInteger minID = new AtomicInteger(Integer.MIN_VALUE);
		IntStream.rangeClosed(0, lastEntryID).parallel().forEach((i)->{
			/* Get the ith chain code from the database */
			String modelSegmentChain = DatabaseModule.getChainCode(i);

			QGram qg = new QGram(5);
			if ((segmentChain == null) || (modelSegmentChain == null))
			{
				System.err.println("QGram: modelSegment null for chain code" + i);
				return;
			}
			int distance = (int)qg.distance(segmentChain, modelSegmentChain);

			/* track entry with the small number of
			 * edits then report filename and segment of id entry */
			if (distance < minDistance.get())
			{
				minDistance.set(distance);
				minID.set(i);
			}
		});

		ConcurrentHashMap<Integer, Integer> hm =
			new ConcurrentHashMap<Integer, Integer>(1, (float)0.75);
		hm.put(minID.get(), minDistance.get());
		bestMatches.put(segment, hm);

		/* For each segment of the sample, track which model image
		 * and which image model perspective provides the best match*/
		String modelOfInterest = DatabaseModule.getFileName(minID.get());
		Integer curCnt = cntMatches.get(modelOfInterest);
		if (curCnt == null)
		{
			cntMatches.put(modelOfInterest, 1);
		}
		else
		{
			cntMatches.put(modelOfInterest, ++curCnt);
		}
	}

	/* Display result */
	Iterator<Integer> bmIterator = bestMatches.keySet().iterator();
	while (bmIterator.hasNext())
	{
		Integer key = bmIterator.next();
		ConcurrentHashMap<Integer, Integer> minValue = bestMatches.get(key);
		Iterator<Integer> ii = minValue.keySet().iterator();
		while (ii.hasNext())
		{
			Integer idmin = ii.next();
			String filenameOfID = DatabaseModule.getFileName(idmin);
			sb.append("Best QGram Match for segment " + key + " is " +
					  idmin + " (" + filenameOfID + ") with " +
					  minValue.get(idmin) + " mods needed to match" + "\n");
		}
	}

	// build header
	synchronized(wkbkResults)
	{
		XSSFRow row = sheet.createRow(0);
		XSSFCell cell = row.createCell(0);
		cell.setCellValue("Model Image");
		cell = row.createCell(1);
		cell.setCellValue("Match Prob.");
		cell = row.createCell(2);
		cell.setCellValue("Match Prob. Per.");
	}

	/* Tell user probably of matching various images based on how well
	 * sample segments matched to the database of model images */
	Iterator<String> cntIterator = cntMatches.keySet().iterator();
	float bestProbMatch = Float.MIN_NORMAL;
	String nameOfModelMatch = null;
	int probsCnt = 0;
	while (cntIterator.hasNext())
	{
		String filename = cntIterator.next();
		Integer count = cntMatches.get(filename);
		float probMatch = ((float)count) / sampleChains.size();
		sb.append("Probablity of matching " + filename + " is :" + (probMatch * 100) + " %" + "\n");

		/* record data in spreadsheet */
		synchronized(wkbkResults)
		{
			XSSFRow row = sheet.createRow(probsCnt++);
			XSSFCell cell = row.createCell(0);
			cell.setCellValue(filename);
			cell = row.createCell(1, CellType.NUMERIC);
			cell.setCellValue(probMatch);
			cell = row.createCell(2, CellType.NUMERIC);
			cell.setCellValue(probMatch * 100);

			// update summary sheet as well for final calculation
			XSSFSheet summarySheet = wkbkResults.getSheet(SUMMARY_SHEET);
			int sumRowInt =
				ProjectUtilities.findRowInSpreadSheet(summarySheet, filename);
			XSSFRow summaryRow = summarySheet.getRow(sumRowInt);
			XSSFCell summaryCell = summaryRow.createCell(Si_COLUMN_SUMMARY, CellType.NUMERIC);
			summaryCell.setCellValue(probMatch);
		}

		/* Track most likely match*/
		if (probMatch > bestProbMatch)
		{
			bestProbMatch = probMatch;
			nameOfModelMatch = filename;
		}
	}

	/* Tell user most likely match and record in spreadsheet */
	sb.append("Best probable match is " + nameOfModelMatch +
			  " with probablity " + bestProbMatch + "\n");
	synchronized(wkbkResults)
	{
		XSSFRow bestRow = sheet.createRow(probsCnt);

		/* Make sure the best results stands out from the other data */
		XSSFCellStyle style = wkbkResults.createCellStyle();
		XSSFFont font = wkbkResults.createFont();
		style.setBorderBottom(BorderStyle.THICK);
		style.setBorderTop(BorderStyle.THICK);
		font.setFontHeightInPoints((short)14);
		font.setBold(true);
		style.setFont(font);
		bestRow.setRowStyle(style);

		/* Record data in row of spreadsheet */
		XSSFCell bestCellinRow = bestRow.createCell(0);
		bestCellinRow.setCellValue(nameOfModelMatch);
		bestCellinRow.setCellStyle(style);
		bestCellinRow = bestRow.createCell(1, CellType.NUMERIC);
		bestCellinRow.setCellValue(bestProbMatch);
		bestCellinRow.setCellStyle(style);
		bestCellinRow = bestRow.createCell(2, CellType.NUMERIC);
		bestCellinRow.setCellValue(bestProbMatch * 100);
		bestCellinRow.setCellStyle(style);
	}

	System.out.println(sb.toString());
	System.out.println("Done running thread");
}

private:
static void match_to_model_NGram_Distance(
	Map<Integer, String> sampleChains, XSSFWorkbook wkbkResults)
{
	/* 1. Take each segment of sample image
	 *    for each model image
	 *        for each segmnent in model image
	 *            apply java-string-similarity method
	 *            O(n)+O(m*n^2)+Runtime_Algorithm */
	StringBuilder sb = new StringBuilder();
	int bestMatchesSz = 1;
	int cntMatchesSz = 1;
	if ((sampleChains == null) || sampleChains.size() == 0)
	{
		return;
	}
	else
	{
		bestMatchesSz = sampleChains.size();
		cntMatchesSz = (int)(sampleChains.size() * .1);
		if (cntMatchesSz < 1)
		{
			cntMatchesSz = 1;
		}
	}

	XSSFSheet sheet = null;
	synchronized(wkbkResults)
	{
		sheet = wkbkResults.createSheet("NGram");
	}

	Map<Integer, ConcurrentHashMap<Integer, Double>> bestMatches =
		new ConcurrentHashMap<Integer, ConcurrentHashMap<Integer, Double>>(
			sampleChains.size(), (float)0.75);
	Map<String, Integer> cntMatches =
		new HashMap<String, Integer>(cntMatchesSz,
									 (float)0.90);

	Iterator<Integer> segments = sampleChains.keySet().iterator();
	int lastEntryID = DatabaseModule.getLastId();
	while (segments.hasNext())
	{
		Integer segment = segments.next();
		String segmentChain = sampleChains.get(segment);
		sb.append("Working with sample segment " + segment + "\n");
		AtomicFloat minNormDistance = new AtomicFloat(Float.MIN_VALUE);
		AtomicInteger minID = new AtomicInteger(Integer.MIN_VALUE);
		IntStream.rangeClosed(0, lastEntryID).parallel().forEach((i)->{
			/* Get the ith chain code from the database */
			String modelSegmentChain = DatabaseModule.getChainCode(i);

			/* Convert strings into sets of n-grams */
			NGram ng = new NGram(5);
			double distance = ng.distance(segmentChain, modelSegmentChain);

			/* track entry with the small number of
			 * edits then report filename and segment of id entry */
			if (Float.compare((float)distance, minNormDistance.get()) > 0)
			{
				minNormDistance.set((float)distance);
				minID.set(i);
			}
		});

		ConcurrentHashMap<Integer, Double> hm =
			new ConcurrentHashMap<Integer, Double>(1, (float)0.75);
		hm.put(minID.get(), minNormDistance.doubleValue());
		bestMatches.put(segment, hm);

		/* For each segment of the sample, track which model image
		 * and which image model perspective provides the best match*/
		String modelOfInterest = DatabaseModule.getFileName(minID.get());
		Integer curCnt = cntMatches.get(modelOfInterest);
		if (curCnt == null)
		{
			cntMatches.put(modelOfInterest, 1);
		}
		else
		{
			cntMatches.put(modelOfInterest, ++curCnt);
		}
	}

	/* Display result */
	Iterator<Integer> bmIterator = bestMatches.keySet().iterator();
	while (bmIterator.hasNext())
	{
		Integer key = bmIterator.next();
		ConcurrentHashMap<Integer, Double> minValue = bestMatches.get(key);
		Iterator<Integer> ii = minValue.keySet().iterator();
		while (ii.hasNext())
		{
			Integer idmin = ii.next();
			String filenameOfID = DatabaseModule.getFileName(idmin);
			sb.append("Best NGram Match for segment " + key + " is chaincode identifier" +
					  idmin + " (" + filenameOfID + ") with " +
					  minValue.get(idmin) + " similarity measure" + "\n");
		}
	}

	/* Tell user probably of matching various images based on how well
	 * sample segments matched to the database of model images */
	Iterator<String> cntIterator = cntMatches.keySet().iterator();
	float bestProbMatch = Float.MIN_NORMAL;
	String nameOfModelMatch = null;
	int probsCnt = 0;
	while (cntIterator.hasNext())
	{
		String filename = cntIterator.next();
		Integer count = cntMatches.get(filename);
		float probMatch = ((float)count) / sampleChains.size();
		sb.append("Probablity of matching " + filename + " is :" + (probMatch * 100) + " %" + "\n");

		/* record data in spreadsheet */
		synchronized(wkbkResults)
		{
			XSSFRow row = sheet.createRow(probsCnt++);
			XSSFCell cell = row.createCell(0);
			cell.setCellValue(filename);
			cell = row.createCell(1);
			cell.setCellValue(probMatch);
		}

		/* Track most likely match*/
		if (probMatch > bestProbMatch)
		{
			bestProbMatch = probMatch;
			nameOfModelMatch = filename;
		}
	}

	/* Tell user most likely match and record in spreadsheet */
	sb.append("Best probable match is " + nameOfModelMatch +
			  " with probablity " + (bestProbMatch * 100) + "\n");
	synchronized(wkbkResults)
	{
		XSSFRow bestRow = sheet.createRow(probsCnt);

		/* Make sure the best results stands out from the other data */
		XSSFCellStyle style = wkbkResults.createCellStyle();
		XSSFFont font = wkbkResults.createFont();
		style.setBorderBottom(BorderStyle.THICK);
		style.setBorderTop(BorderStyle.THICK);
		font.setFontHeightInPoints((short)14);
		font.setBold(true);
		style.setFont(font);
		bestRow.setRowStyle(style);

		/* Record data in row of spreadsheet */
		XSSFCell bestCellinRow = bestRow.createCell(0);
		bestCellinRow.setCellValue(nameOfModelMatch);
		bestCellinRow.setCellStyle(style);
		bestCellinRow = bestRow.createCell(1);
		bestCellinRow.setCellValue(bestProbMatch);
		bestCellinRow.setCellStyle(style);
	}

	System.out.println(sb.toString());
	System.out.println("Done running thread");
}

private:
static void match_to_model_MLCS(Map<Integer, String> sampleChains, XSSFWorkbook wkbkResults)
{
	/* 1. Take each segment of sample image
	 *    for each model image
	 *        for each segmnent in model image
	 *            apply java-string-similarity method
	 *            O(n)+O(m*n^2)+Runtime_Algorithm */
	StringBuilder sb = new StringBuilder();
	int bestMatchesSz = 1;
	int cntMatchesSz = 1;
	if ((sampleChains == null) || sampleChains.size() == 0)
	{
		return;
	}
	else
	{
		bestMatchesSz = sampleChains.size();
		cntMatchesSz = (int)(sampleChains.size() * .1);
		if (cntMatchesSz < 1)
		{
			cntMatchesSz = 1;
		}
	}

	XSSFSheet sheet = null;
	synchronized(wkbkResults)
	{
		sheet = wkbkResults.createSheet("MLCS");
	}

	Map<Integer, ConcurrentHashMap<Integer, Double>> bestMatches =
		new ConcurrentHashMap<Integer, ConcurrentHashMap<Integer, Double>>(
			bestMatchesSz, (float)0.75);
	Map<String, Integer> cntMatches =
		new ConcurrentHashMap<String, Integer>(cntMatchesSz,
											   (float)0.90);

	Iterator<Integer> segments = sampleChains.keySet().iterator();
	int lastEntryID = DatabaseModule.getLastId();
	while (segments.hasNext())
	{
		Integer segment = segments.next();
		String segmentChain = sampleChains.get(segment);
		sb.append("Working with sample segment " + segment + "\n");
		AtomicFloat minDistance = new AtomicFloat(Float.MAX_VALUE);
		AtomicInteger minID = new AtomicInteger(Integer.MIN_VALUE);
		IntStream.rangeClosed(0, lastEntryID).parallel().forEach((i)->{
			/* Get the ith chain code from the database */
			String modelSegmentChain = DatabaseModule.getChainCode(i);

			/* */
			MetricLCS mlcs = new MetricLCS();
			if ((segmentChain == null) || (modelSegmentChain == null))
			{
				System.err.println("MetricLCS: modelSegment null for chain code" + i);
				return;
			}
			double distance = mlcs.distance(segmentChain, modelSegmentChain);

			/* track entry with the small number of
			 * edits then report filename and segment of id entry */
			if (Float.compare((float)distance, minDistance.get()) < 0)
			{
				minDistance.set((float)distance);
				minID.set(i);
			}
		});

		ConcurrentHashMap<Integer, Double> hm =
			new ConcurrentHashMap<Integer, Double>(1, (float)0.75);
		hm.put(minID.get(), minDistance.doubleValue());
		bestMatches.put(segment, hm);

		/* For each segment of the sample, track which model image
		 * and which image model perspective provides the best match*/
		String modelOfInterest = DatabaseModule.getFileName(minID.get());
		Integer curCnt = cntMatches.get(modelOfInterest);
		if (curCnt == null)
		{
			cntMatches.put(modelOfInterest, 1);
		}
		else
		{
			cntMatches.put(modelOfInterest, ++curCnt);
		}
	}

	/* Display result */
	Iterator<Integer> bmIterator = bestMatches.keySet().iterator();
	while (bmIterator.hasNext())
	{
		Integer key = bmIterator.next();
		ConcurrentHashMap<Integer, Double> minValue = bestMatches.get(key);
		Iterator<Integer> ii = minValue.keySet().iterator();
		while (ii.hasNext())
		{
			Integer idmin = ii.next();
			String filenameOfID = DatabaseModule.getFileName(idmin);
			sb.append("Best M.L.C.S Match for segment " + key + " is " +
					  idmin + " (" + filenameOfID + ") with " +
					  minValue.get(idmin) + " measure" + "\n");
		}
	}

	/* Tell user probably of matching various images based on how well
	 * sample segments matched to the database of model images */
	Iterator<String> cntIterator = cntMatches.keySet().iterator();
	float bestProbMatch = Float.MIN_NORMAL;
	String nameOfModelMatch = null;
	int probsCnt = 0;
	while (cntIterator.hasNext())
	{
		String filename = cntIterator.next();
		Integer count = cntMatches.get(filename);
		float probMatch = ((float)count) / sampleChains.size();
		sb.append("Probablity of matching " + filename + " is :" + (probMatch * 100) + " %" + "\n");

		/* record data in spreadsheet */
		synchronized(wkbkResults)
		{
			XSSFRow row = sheet.createRow(probsCnt++);
			XSSFCell cell = row.createCell(0);
			cell.setCellValue(filename);
			cell = row.createCell(1);
			cell.setCellValue(probMatch);
		}

		/* Track most likely match*/
		if (probMatch > bestProbMatch)
		{
			bestProbMatch = probMatch;
			nameOfModelMatch = filename;
		}
	}

	/* Tell user most likely match and record in spreadsheet */
	sb.append("Best probable match is " + nameOfModelMatch +
			  " with probablity " + bestProbMatch + "\n");
	XSSFRow bestRow = sheet.createRow(probsCnt);

	/* Make sure the best results stands out from the other data */
	synchronized(wkbkResults)
	{
		XSSFCellStyle style = wkbkResults.createCellStyle();
		XSSFFont font = wkbkResults.createFont();
		style.setBorderBottom(BorderStyle.THICK);
		style.setBorderTop(BorderStyle.THICK);
		font.setFontHeightInPoints((short)14);
		font.setBold(true);
		style.setFont(font);
		bestRow.setRowStyle(style);

		/* Record data in row of spreadsheet */
		XSSFCell bestCellinRow = bestRow.createCell(0);
		bestCellinRow.setCellValue(nameOfModelMatch);
		bestCellinRow.setCellStyle(style);
		bestCellinRow = bestRow.createCell(1);
		bestCellinRow.setCellValue(bestProbMatch);
		bestCellinRow.setCellStyle(style);
	}

	System.out.println(sb.toString());
	System.out.println("Done running thread");
}

/**
 * Match to model using Longest Common Subsequence of border chaincodes
 * @param sampleChains -- Model chaincodes
 * @param wkbkResults -- Spreadsheet containing results
 * @return
 */
private:
static String match_to_model_LCS(Map<Integer, String> sampleChains, XSSFWorkbook wkbkResults)
{
	/* 1. Take each segment of sample image
	 *    for each model image
	 *        for each segmnent in model image
	 *            apply java-string-similarity method
	 *            O(n)+O(m*n^2)+Runtime_Algorithm */
	StringBuilder sb = new StringBuilder();
	int bestMatchesSz = 1;
	int cntMatchesSz = 1;
	if ((sampleChains == null) || sampleChains.size() == 0)
	{
		return null;
	}
	else
	{
		bestMatchesSz = sampleChains.size();
		cntMatchesSz = (int)(sampleChains.size() * .1);
		if (cntMatchesSz < 1)
		{
			cntMatchesSz = 1;
		}
	}

	XSSFSheet sheet = null;
	synchronized(wkbkResults)
	{
		sheet = wkbkResults.createSheet("LCS");
	}

	Map<Integer, ConcurrentHashMap<Integer, Integer>> bestMatches =
		new ConcurrentHashMap<Integer, ConcurrentHashMap<Integer, Integer>>(
			bestMatchesSz, (float)0.75);
	Map<String, Integer> cntMatches =
		new ConcurrentHashMap<String, Integer>(cntMatchesSz,
											   (float)0.90);

	Iterator<Integer> segments = sampleChains.keySet().iterator();
	int lastEntryID = DatabaseModule.getLastId();

	while (segments.hasNext())
	{
		Integer segment = segments.next();
		String segmentChain = sampleChains.get(segment);
		sb.append("LCS(): Working with sample segment " + segment + "\n");
		AtomicInteger minDistance = new AtomicInteger(Integer.MAX_VALUE);
		AtomicInteger minID = new AtomicInteger(Integer.MIN_VALUE);

		IntStream.rangeClosed(0, lastEntryID).parallel().forEach((i)->{
			/* Get the ith chain code from the database */
			String modelSegmentChain = DatabaseModule.getChainCode(i);

			/* LCS measure is
			 * the minimum number of single-character edits
			 * (insertions, deletions or substitutions) required to
			 *  change one word into the other */
			LongestCommonSubsequence lcs = new LongestCommonSubsequence();
			int distance;
			if ((segmentChain != null) && (modelSegmentChain != null))
			{
				distance = (int)lcs.distance(segmentChain, modelSegmentChain);
			}
			else
			{
				/* if it is null, we cannot make a comparison, so
				 * the number of mods is essentially infinite
				 */
				distance = Integer.MAX_VALUE;
			}

			/* track entry with the small number of
			 * edits then report filename and segment of id entry
			 *
			 *  Remember that we are looking for the number of insertions,
			 *  deletions and substitutions to get the sample substring to look like
			 *  the model substring -- subsequence don't have to have consecutive
			 *  chars, unlike a common substring (hence subsequence)*/
			if (distance < minDistance.get())
			{
				minDistance.set(distance);
				minID.set(i);
			}
		});

		/* Keep track of the best match for the current segment */
		ConcurrentHashMap<Integer, Integer> hm =
			new ConcurrentHashMap<Integer, Integer>(1, (float)0.75);
		hm.put(minID.get(), minDistance.get());
		bestMatches.put(segment, hm);

		/* For each segment of the sample, track which model image
		 * and which image model perspective provides the best match*/
		String modelOfInterest = DatabaseModule.getFileName(minID.get());
		Integer curCnt = cntMatches.get(modelOfInterest);
		if (curCnt == null)
		{
			cntMatches.put(modelOfInterest, 1);
		}
		else
		{
			cntMatches.put(modelOfInterest, ++curCnt);
		}

		// Just trying to recover some memory here in a more aggressive way
		System.gc();
	}

	/* Display result */
	Iterator<Integer> bmIterator = bestMatches.keySet().iterator();
	while (bmIterator.hasNext())
	{
		Integer key = bmIterator.next();
		ConcurrentHashMap<Integer, Integer> minValue = bestMatches.get(key);
		Iterator<Integer> ii = minValue.keySet().iterator();
		while (ii.hasNext())
		{
			Integer idmin = ii.next();
			String filenameOfID = DatabaseModule.getFileName(idmin);
			sb.append("Best L.C.S Match for segment " + key + " is " +
					  idmin + " (" + filenameOfID + ") with " +
					  minValue.get(idmin) + " mods needed to match" + "\n");
		}
	}

	// build header
	synchronized(wkbkResults)
	{
		XSSFRow row = sheet.createRow(0);
		XSSFCell cell = row.createCell(0);
		cell.setCellValue("Model Image");
		cell = row.createCell(1);
		cell.setCellValue("Match Prob.");
		cell = row.createCell(2);
		cell.setCellValue("Match Prob. Per.");
	}

	/* Tell user probably of matching various images based on how well
	 * sample segments matched to the database of model images */
	Iterator<String> cntIterator = cntMatches.keySet().iterator();
	float bestProbMatch = Float.MIN_NORMAL;
	String nameOfModelMatch = null;
	int probsCnt = 1;
	while (cntIterator.hasNext())
	{
		String filename = cntIterator.next();
		Integer count = cntMatches.get(filename);
		float probMatch = ((float)count) / sampleChains.size();
		sb.append("Probablity of matching " + filename + " is :" + (probMatch * 100) + " %" + "\n");

		/* record data in spreadsheet */
		synchronized(wkbkResults)
		{
			XSSFRow row = sheet.createRow(probsCnt++);
			XSSFCell cell = row.createCell(0);
			cell.setCellValue(filename);
			cell = row.createCell(1, CellType.NUMERIC);
			cell.setCellValue(probMatch);
			cell = row.createCell(2, CellType.NUMERIC);
			cell.setCellValue(probMatch * 100);

			// update summary sheet as well for final calculation
			XSSFSheet summarySheet = wkbkResults.getSheet(SUMMARY_SHEET);
			int sumRowInt =
				ProjectUtilities.findRowInSpreadSheet(summarySheet, filename);
			XSSFRow summaryRow = summarySheet.getRow(sumRowInt);
			XSSFCell summaryCell = summaryRow.createCell(LCSi_COLUMN_SUMMARY, CellType.NUMERIC);
			summaryCell.setCellValue(probMatch);
		}

		/* Track most likely match*/
		if (probMatch > bestProbMatch)
		{
			bestProbMatch = probMatch;
			nameOfModelMatch = filename;
		}
	}

	/* Tell user most likely match and record in spreadsheet */
	sb.append("Best probable match with L.C.S. is " + nameOfModelMatch +
			  " with probablity " + bestProbMatch + " or " + (bestProbMatch * 100) + "%% \n");
	synchronized(wkbkResults)
	{
		XSSFRow bestRow = sheet.createRow(probsCnt);

		/* Make sure the best results stands out from the other data */

		XSSFCellStyle style = wkbkResults.createCellStyle();
		XSSFFont font = wkbkResults.createFont();
		style.setBorderBottom(BorderStyle.THICK);
		style.setBorderTop(BorderStyle.THICK);
		font.setFontHeightInPoints((short)14);
		font.setBold(true);
		style.setFont(font);
		bestRow.setRowStyle(style);

		/* Record data in row of spreadsheet */
		XSSFCell bestCellinRow = bestRow.createCell(0);
		bestCellinRow.setCellValue(nameOfModelMatch);
		bestCellinRow.setCellStyle(style);
		bestCellinRow = bestRow.createCell(1, CellType.NUMERIC);
		bestCellinRow.setCellValue(bestProbMatch);
		bestCellinRow.setCellStyle(style);
		bestCellinRow = bestRow.createCell(2, CellType.NUMERIC);
		bestCellinRow.setCellValue(bestProbMatch * 100);
		bestCellinRow.setCellStyle(style);
	}

	System.out.println(sb.toString());
	System.out.println("Done running thread");
	return nameOfModelMatch + "," + bestProbMatch;
}

private:
static void match_to_model_Jaro_Winkler(Map<Integer, String> sampleChains, XSSFWorkbook wkbkResults)
{
	/* 1. Take each segment of sample image
	 *    for each model image
	 *        for each segmnent in model image
	 *            apply java-string-similarity method
	 *            O(n)+O(m*n^2)+Runtime_Algorithm */
	StringBuilder sb = new StringBuilder();
	int bestMatchesSz = 1;
	int cntMatchesSz = 1;
	if ((sampleChains == null) || sampleChains.size() == 0)
	{
		return;
	}
	else
	{
		bestMatchesSz = sampleChains.size();
		cntMatchesSz = (int)(sampleChains.size() * .1);
		if (cntMatchesSz < 1)
		{
			cntMatchesSz = 1;
		}
	}
	Map<Integer, ConcurrentHashMap<Integer, Double>> bestMatches =
		new ConcurrentHashMap<Integer, ConcurrentHashMap<Integer, Double>>(
			bestMatchesSz, (float)0.75);
	Map<String, Integer> cntMatches =
		new ConcurrentHashMap<String, Integer>(cntMatchesSz,
											   (float)0.90);

	XSSFSheet sheet = null;
	synchronized(wkbkResults)
	{
		sheet = wkbkResults.createSheet("JaroWinkler");
	}

	Iterator<Integer> segments = sampleChains.keySet().iterator();
	int lastEntryID = DatabaseModule.getLastId();
	while (segments.hasNext())
	{
		Integer segment = segments.next();
		String segmentChain = sampleChains.get(segment);
		sb.append("Working with sample segment " + segment + "\n");
		AtomicFloat bestLvlOfMatch = new AtomicFloat(Float.MIN_VALUE);
		AtomicInteger bestID = new AtomicInteger(Integer.MIN_VALUE);
		IntStream.rangeClosed(0, lastEntryID).parallel().forEach((i)->{
			/* Get the ith chain code from the database */
			String modelSegmentChain = DatabaseModule.getChainCode(i);

			/* computes the similarity between 2 strings, and the returned value
			 * lies in the interval [0.0, 1.0]. It is (roughly) a variation of
			 * Damerau-Levenshtein, where the substitution of 2 close
			 * characters is considered less important then the substitution of
			 * 2 characters that a far from each other.*/
			JaroWinkler jw = new JaroWinkler();
			if ((segmentChain == null) || (modelSegmentChain == null))
			{
				System.err.println("JaroWinkler: modelSegment null for chain code" + i);
				return;
			}
			double similarity = jw.distance(segmentChain, modelSegmentChain);

			/* track entry with the small number of
			 * edits then report filename and segment of id entry */
			if (Float.compare((float)similarity, bestLvlOfMatch.get()) > 0)
			{
				bestLvlOfMatch.set((float)similarity);
				bestID.set(i);
			}
		});

		ConcurrentHashMap<Integer, Double> hm =
			new ConcurrentHashMap<Integer, Double>(1, (float)0.75);
		hm.put(bestID.get(), bestLvlOfMatch.doubleValue());
		bestMatches.put(segment, hm);

		/* For each segment of the sample, track which model image
		 * and which image model perspective provides the best match*/
		String modelOfInterest = DatabaseModule.getFileName(bestID.get());
		Integer curCnt = cntMatches.get(modelOfInterest);
		if (curCnt == null)
		{
			cntMatches.put(modelOfInterest, 1);
		}
		else
		{
			cntMatches.put(modelOfInterest, ++curCnt);
		}
	}

	/* Display result */
	Iterator<Integer> bmIterator = bestMatches.keySet().iterator();
	while (bmIterator.hasNext())
	{
		Integer key = bmIterator.next();
		ConcurrentHashMap<Integer, Double> minValue = bestMatches.get(key);
		Iterator<Integer> ii = minValue.keySet().iterator();
		while (ii.hasNext())
		{
			Integer idmin = ii.next();
			String filenameOfID = DatabaseModule.getFileName(idmin);
			sb.append("Best Jaro Winker match for segment " + key + " is " +
					  idmin + " (" + filenameOfID + ") with " +
					  minValue.get(idmin) + " similarity" + "\n");
		}
	}

	/* Tell user probably of matching various images based on how well
	 * sample segments matched to the database of model images */
	Iterator<String> cntIterator = cntMatches.keySet().iterator();
	float bestProbMatch = Float.MIN_NORMAL;
	String nameOfModelMatch = null;
	int probsCnt = 0;
	while (cntIterator.hasNext())
	{
		String filename = cntIterator.next();
		Integer count = cntMatches.get(filename);
		float probMatch = ((float)count) / sampleChains.size();
		sb.append("Probablity of matching " + filename + " is :" + (probMatch * 100) + " %" + "\n");

		/* record data in spreadsheet */
		synchronized(wkbkResults)
		{
			XSSFRow row = sheet.createRow(probsCnt++);
			XSSFCell cell = row.createCell(0);
			cell.setCellValue(filename);
			cell = row.createCell(1);
			cell.setCellValue(probMatch);
		}

		/* Track most likely match*/
		if (probMatch > bestProbMatch)
		{
			bestProbMatch = probMatch;
			nameOfModelMatch = filename;
		}
	}

	/* Tell user most likely match and record in spreadsheet */
	sb.append("Best probable match is " + nameOfModelMatch +
			  " with probablity " + bestProbMatch + "\n");
	synchronized(wkbkResults)
	{
		XSSFRow bestRow = sheet.createRow(probsCnt);

		/* Make sure the best results stands out from the other data */
		XSSFCellStyle style = wkbkResults.createCellStyle();
		XSSFFont font = wkbkResults.createFont();
		style.setBorderBottom(BorderStyle.THICK);
		style.setBorderTop(BorderStyle.THICK);
		font.setFontHeightInPoints((short)14);
		font.setBold(true);
		style.setFont(font);
		bestRow.setRowStyle(style);

		/* Record data in row of spreadsheet */
		XSSFCell bestCellinRow = bestRow.createCell(0);
		bestCellinRow.setCellValue(nameOfModelMatch);
		bestCellinRow.setCellStyle(style);
		bestCellinRow = bestRow.createCell(1);
		bestCellinRow.setCellValue(bestProbMatch);
		bestCellinRow.setCellStyle(style);
	}

	System.out.println(sb.toString());
	System.out.println("Done running thread");
}

private:
static void match_to_model_Opt_Str_Alignment(Map<Integer, String> sampleChains, XSSFWorkbook wkbkResults)
{
	/* 1. Take each segment of sample image
	 *    for each model image
	 *        for each segment in model image
	 *            apply java-string-similarity method
	 *            O(n)+O(m*n^2)+Runtime_Algorithm */
	StringBuilder sb = new StringBuilder();
	int bestMatchesSz = 1;
	int cntMatchesSz = 1;
	if ((sampleChains == null) || sampleChains.size() == 0)
	{
		return;
	}
	else
	{
		bestMatchesSz = sampleChains.size();
		cntMatchesSz = (int)(sampleChains.size() * .1);
		if (cntMatchesSz < 1)
		{
			cntMatchesSz = 1;
		}
	}

	XSSFSheet sheet = null;
	synchronized(wkbkResults)
	{
		sheet = wkbkResults.createSheet("OSA");
	}

	Map<Integer, ConcurrentHashMap<Integer, Integer>> bestMatches =
		new ConcurrentHashMap<Integer, ConcurrentHashMap<Integer, Integer>>(
			bestMatchesSz, (float)0.75);
	Map<String, Integer> cntMatches =
		new ConcurrentHashMap<String, Integer>(cntMatchesSz,
											   (float)0.90);

	Iterator<Integer> segments = sampleChains.keySet().iterator();
	int lastEntryID = DatabaseModule.getLastId();
	while (segments.hasNext())
	{
		Integer segment = segments.next();
		String segmentChain = sampleChains.get(segment);
		sb.append("Working with sample segment " + segment + "\n");
		AtomicInteger minDistance = new AtomicInteger(Integer.MAX_VALUE);
		AtomicInteger minID = new AtomicInteger(Integer.MIN_VALUE);
		IntStream.rangeClosed(0, lastEntryID).parallel().forEach((i)->{
			/* Get the ith chain code from the database */
			String modelSegmentChain = DatabaseModule.getChainCode(i);

			/* the number of edit operations needed to make the strings
			 *  equal under the condition that no substring is edited
			 *  more than once*/
			OptimalStringAlignment osa = new OptimalStringAlignment();
			if ((segmentChain == null) || (modelSegmentChain == null))
			{
				System.err.println("OSA: modelSegment null for chain code" + i);
				return;
			}
			int distance = (int)osa.distance(
				segmentChain, modelSegmentChain);

			/* track entry with the small number of
			 * edits then report filename and segment of id entry */
			if (distance < minDistance.get())
			{
				minDistance.set(distance);
				minID.set(i);
			}
		});

		ConcurrentHashMap<Integer, Integer> hm =
			new ConcurrentHashMap<Integer, Integer>(1, (float)0.75);
		hm.put(minID.get(), minDistance.get());
		bestMatches.put(segment, hm);

		/* For each segment of the sample, track which model image
		 * and which image model perspective provides the best match*/
		String modelOfInterest = DatabaseModule.getFileName(minID.get());
		Integer curCnt = cntMatches.get(modelOfInterest);
		if (curCnt == null)
		{
			cntMatches.put(modelOfInterest, 1);
		}
		else
		{
			cntMatches.put(modelOfInterest, ++curCnt);
		}
	}

	/* Display result */
	Iterator<Integer> bmIterator = bestMatches.keySet().iterator();
	while (bmIterator.hasNext())
	{
		Integer key = bmIterator.next();
		ConcurrentHashMap<Integer, Integer> minValue = bestMatches.get(key);
		Iterator<Integer> ii = minValue.keySet().iterator();
		while (ii.hasNext())
		{
			Integer idmin = ii.next();
			String filenameOfID = DatabaseModule.getFileName(idmin);
			sb.append("Best O.S.A. for segment " + key + " is " +
					  idmin + " (" + filenameOfID + ") with " +
					  minValue.get(idmin) + " mods needed to match" + "\n");
		}
	}

	/* Tell user probably of matching various images based on how well
	 * sample segments matched to the database of model images */
	Iterator<String> cntIterator = cntMatches.keySet().iterator();
	float bestProbMatch = Float.MIN_NORMAL;
	String nameOfModelMatch = null;
	int probsCnt = 0;
	while (cntIterator.hasNext())
	{
		String filename = cntIterator.next();
		Integer count = cntMatches.get(filename);
		float probMatch = ((float)count) / sampleChains.size();
		sb.append("Probablity of matching " + filename + " is :" + (probMatch * 100) + " %" + "\n");

		/* record data in spreadsheet */
		synchronized(wkbkResults)
		{
			XSSFRow row = sheet.createRow(probsCnt++);
			XSSFCell cell = row.createCell(0);
			cell.setCellValue(filename);
			cell = row.createCell(1);
			cell.setCellValue(probMatch);
		}

		/* Track most likely match*/
		if (probMatch > bestProbMatch)
		{
			bestProbMatch = probMatch;
			nameOfModelMatch = filename;
		}
	}

	/* Tell user most likely match and record in spreadsheet */
	sb.append("Best probable match is " + nameOfModelMatch +
			  " with probablity " + bestProbMatch + "\n");
	synchronized(wkbkResults)
	{
		XSSFRow bestRow = sheet.createRow(probsCnt);

		/* Make sure the best results stands out from the other data */
		XSSFCellStyle style = wkbkResults.createCellStyle();
		XSSFFont font = wkbkResults.createFont();
		style.setBorderBottom(BorderStyle.THICK);
		style.setBorderTop(BorderStyle.THICK);
		font.setFontHeightInPoints((short)14);
		font.setBold(true);
		style.setFont(font);
		bestRow.setRowStyle(style);

		/* Record data in row of spreadsheet */
		XSSFCell bestCellinRow = bestRow.createCell(0);
		bestCellinRow.setCellValue(nameOfModelMatch);
		bestCellinRow.setCellStyle(style);
		bestCellinRow = bestRow.createCell(1);
		bestCellinRow.setCellValue(bestProbMatch);
		bestCellinRow.setCellStyle(style);
	}

	System.out.println(sb.toString());
	System.out.println("Done running thread");
}

private:
static void match_to_model_Damerau_Levenshtein(
	Map<Integer, String> sampleChains, XSSFWorkbook wkbkResults)
{
	/* 1. Take each segment of sample image
	 *    for each model image
	 *        for each segment in model image
	 *            apply java-string-similarity method
	 *            O(n)+O(m*n^2)+Runtime_Algorithm */
	StringBuilder sb = new StringBuilder();
	int bestMatchesSz = 1;
	int cntMatchesSz = 1;
	if ((sampleChains == null) || sampleChains.size() == 0)
	{
		return;
	}
	else
	{
		bestMatchesSz = sampleChains.size();
		cntMatchesSz = (int)(sampleChains.size() * .1);
		if (cntMatchesSz < 1)
		{
			cntMatchesSz = 1;
		}
	}

	XSSFSheet sheet = null;
	synchronized(wkbkResults)
	{
		sheet = wkbkResults.createSheet("Damerau");
	}

	Map<Integer, ConcurrentHashMap<Integer, Integer>> bestMatches =
		new ConcurrentHashMap<Integer, ConcurrentHashMap<Integer, Integer>>(
			sampleChains.size(), (float)0.75);
	Map<String, Integer> cntMatches =
		new ConcurrentHashMap<String, Integer>(cntMatchesSz,
											   (float)0.90);

	Iterator<Integer> segments = sampleChains.keySet().iterator();
	int lastEntryID = DatabaseModule.getLastId();
	while (segments.hasNext())
	{
		Integer segment = segments.next();
		String segmentChain = sampleChains.get(segment);
		sb.append("Working with sample segment " + segment + "\n");
		AtomicInteger minDistance = new AtomicInteger(Integer.MAX_VALUE);
		AtomicInteger minID = new AtomicInteger(-Integer.MIN_VALUE);
		IntStream.rangeClosed(0, lastEntryID).parallel().forEach((i)->{
			/* Get the ith chain code from the database */
			String modelSegmentChain = DatabaseModule.getChainCode(i);

			/* Levenshtein measure is
			 * the minimum number of single-character edits
			 * (insertions, deletions or substitutions) required to
			 *  change one word into the other */
			Damerau d = new Damerau();
			if ((segmentChain == null) || (modelSegmentChain == null))
			{
				System.err.println("Damerau: modelSegment null for chain code" + i);
				return;
			}
			int distance = (int)d.distance(segmentChain, modelSegmentChain);

			/* track entry with the small number of
			 * edits then report filename and segment of id entry */
			if (distance < minDistance.get())
			{
				minDistance.set(distance);
				minID.set(i);
			}
		});

		ConcurrentHashMap<Integer, Integer> hm =
			new ConcurrentHashMap<Integer, Integer>(1, (float)0.75);
		hm.put(minID.get(), minDistance.get());
		bestMatches.put(segment, hm);

		/* For each segment of the sample, track which model image
		 * and which image model perspective provides the best match*/
		String modelOfInterest = DatabaseModule.getFileName(minID.get());
		Integer curCnt = cntMatches.get(modelOfInterest);
		if (curCnt == null)
		{
			cntMatches.put(modelOfInterest, 1);
		}
		else
		{
			cntMatches.put(modelOfInterest, ++curCnt);
		}
	}

	/* Display result */
	Iterator<Integer> bmIterator = bestMatches.keySet().iterator();
	while (bmIterator.hasNext())
	{
		Integer key = bmIterator.next();
		ConcurrentHashMap<Integer, Integer> minValue = bestMatches.get(key);
		Iterator<Integer> ii = minValue.keySet().iterator();
		while (ii.hasNext())
		{
			Integer idmin = ii.next();
			String filenameOfID = DatabaseModule.getFileName(idmin);
			sb.append("Best D-L Match for segment " + key + " is " +
					  idmin + " (" + filenameOfID + ") with " +
					  minValue.get(idmin) +
					  " mods needed to match" + "\n");
		}
	}

	/* Tell user probably of matching various images based on how well
	 * sample segments matched to the database of model images */
	Iterator<String> cntIterator = cntMatches.keySet().iterator();
	float bestProbMatch = Float.MIN_NORMAL;
	String nameOfModelMatch = null;
	int probsCnt = 0;
	while (cntIterator.hasNext())
	{
		String filename = cntIterator.next();
		Integer count = cntMatches.get(filename);
		float probMatch = ((float)count) / sampleChains.size();
		sb.append("Probablity of matching " + filename + " is :" + (probMatch * 100) + " %" + "\n");

		/* record data in spreadsheet */
		synchronized(wkbkResults)
		{
			XSSFRow row = sheet.createRow(probsCnt++);
			XSSFCell cell = row.createCell(0);
			cell.setCellValue(filename);
			cell = row.createCell(1);
			cell.setCellValue(probMatch);
		}

		/* Track most likely match*/
		if (probMatch > bestProbMatch)
		{
			bestProbMatch = probMatch;
			nameOfModelMatch = filename;
		}
	}

	/* Tell user most likely match and record in spreadsheet */
	sb.append("Best probable match is " + nameOfModelMatch +
			  " with probablity " + bestProbMatch + "\n");
	synchronized(wkbkResults)
	{
		XSSFRow bestRow = sheet.createRow(probsCnt);

		/* Make sure the best results stands out from the other data */
		XSSFCellStyle style = wkbkResults.createCellStyle();
		XSSFFont font = wkbkResults.createFont();
		style.setBorderBottom(BorderStyle.THICK);
		style.setBorderTop(BorderStyle.THICK);
		font.setFontHeightInPoints((short)14);
		font.setBold(true);
		style.setFont(font);
		bestRow.setRowStyle(style);

		/* Record data in row of spreadsheet */
		XSSFCell bestCellinRow = bestRow.createCell(0);
		bestCellinRow.setCellValue(nameOfModelMatch);
		bestCellinRow.setCellStyle(style);
		bestCellinRow = bestRow.createCell(1);
		bestCellinRow.setCellValue(bestProbMatch);
		bestCellinRow.setCellStyle(style);
	}

	System.out.println(sb.toString());
	System.out.println("Done running thread");
}

private:
static void match_to_model_Normalized_Levenshtein(
	Map<Integer, String> sampleChains, XSSFWorkbook wkbkResults)
{
	/* 1. Take each segment of sample image
	 *    for each model image
	 *        for each segmnent in model image
	 *            apply java-string-similarity method
	 *            O(n)+O(m*n^2)+Runtime_Algorithm */
	StringBuilder sb = new StringBuilder();
	int bestMatchesSz = 1;
	int cntMatchesSz = 1;
	if ((sampleChains == null) || sampleChains.size() == 0)
	{
		return;
	}
	else
	{
		bestMatchesSz = sampleChains.size();
		cntMatchesSz = (int)(sampleChains.size() * .1);
		if (cntMatchesSz < 1)
		{
			cntMatchesSz = 1;
		}
	}

	XSSFSheet sheet = null;
	synchronized(wkbkResults)
	{
		sheet = wkbkResults.createSheet("NLevenshtein");
	}

	Map<Integer, ConcurrentHashMap<Integer, Double>> bestMatches =
		new ConcurrentHashMap<Integer, ConcurrentHashMap<Integer, Double>>(
			bestMatchesSz, (float)0.75);
	Map<String, Integer> cntMatches =
		new ConcurrentHashMap<String, Integer>(cntMatchesSz,
											   (float)0.90);

	Iterator<Integer> segments = sampleChains.keySet().iterator();
	int lastEntryID = DatabaseModule.getLastId();
	sb.append("N.Lev. lastEntryID = " + lastEntryID);

	// run through all the segments in the sample image
	while (segments.hasNext())
	{

		// get the chaincode for the current segment
		Integer segment = segments.next();
		String segmentChain = sampleChains.get(segment);
		sb.append("Working with sample segment " + segment + "\n");
		System.out.println("Ẅorking with segment" + segment);

		AtomicFloat bestLvlOfMatch = new AtomicFloat(Float.MIN_VALUE);
		AtomicInteger bestID = new AtomicInteger(Integer.MIN_VALUE);

		// run through all the chaincodes in the database
		IntStream.rangeClosed(0, lastEntryID).parallel().forEach((i)->{
			/* Get the ith chain code from the database */
			String modelSegmentChain = DatabaseModule.getChainCode(i);

			/* Levenshtein measure is
			 * the minimum number of single-character edits
			 * (insertions, deletions or substitutions) required to
			 *  change one word into the other */
			NormalizedLevenshtein nl = new NormalizedLevenshtein();
			if ((segmentChain == null) || (modelSegmentChain == null))
			{
				System.err.println("NormalizedLevenshtein: modelSegment null for chain code" + i);
				return;
			}
			double similarity = nl.distance(segmentChain, modelSegmentChain);

			/* track entry with the small number of
			 * edits then report filename and segment of id entry */
			if (Float.compare((float)similarity, bestLvlOfMatch.get()) > 0)
			{
				bestLvlOfMatch.set((float)similarity);
				bestID.set(i);
			}
		});

		ConcurrentHashMap<Integer, Double> hm =
			new ConcurrentHashMap<Integer, Double>(1, (float)0.75);
		hm.put(bestID.get(), bestLvlOfMatch.doubleValue());
		bestMatches.put(segment, hm);

		/* For each segment of the sample, track which model image
		 * and which image model perspective provides the best match*/
		String modelOfInterest = DatabaseModule.getFileName(bestID.get());
		Integer curCnt = cntMatches.get(modelOfInterest);
		if (curCnt == null)
		{
			cntMatches.put(modelOfInterest, 1);
		}
		else
		{
			cntMatches.put(modelOfInterest, ++curCnt);
		}
	}

	/* Display result, segment by segment */
	Iterator<Integer> bmIterator = bestMatches.keySet().iterator();
	while (bmIterator.hasNext())
	{
		// get segment number
		Integer key = bmIterator.next();
		// get hm essentially from segment
		ConcurrentHashMap<Integer, Double> minValue = bestMatches.get(key);
		// display similarity 0 to 1 score (this value * 100 for percent prob match)
		Iterator<Integer> ii = minValue.keySet().iterator();
		while (ii.hasNext())
		{
			Integer idmin = ii.next();
			String filenameOfID = DatabaseModule.getFileName(idmin);
			sb.append("N.Lev. Best Match for segment " + key + " is " +
					  idmin + " (" + filenameOfID + ") with " +
					  minValue.get(idmin) + " similarity" + "\n");
		}
	}

	/* Tell user probably of matching various images based on how well
	 * sample segments matched to the database of model images */
	Iterator<String> cntIterator = cntMatches.keySet().iterator();
	float bestProbMatch = Float.MIN_VALUE;
	String nameOfModelMatch = null;
	int probsCnt = 0;
	while (cntIterator.hasNext())
	{
		String filename = cntIterator.next();
		Integer count = cntMatches.get(filename);
		float probMatch = ((float)count) / sampleChains.size();
		sb.append("Probablity of matching " + filename + " is :" + (probMatch * 100) + " %" + "\n");

		/* record data in spreadsheet */
		synchronized(wkbkResults)
		{
			XSSFRow row = sheet.createRow(probsCnt++);
			XSSFCell cell = row.createCell(0);
			cell.setCellValue(filename);
			cell = row.createCell(1);
			cell.setCellValue(probMatch);
		}

		/* Track most likely match*/
		if (probMatch > bestProbMatch)
		{
			bestProbMatch = probMatch;
			nameOfModelMatch = filename;
		}
	}

	/* Tell user most likely match and record in spreadsheet */
	sb.append("N.Lev. Best probable match is " + nameOfModelMatch +
			  " with probablity " + bestProbMatch +
			  "\n");
	synchronized(wkbkResults)
	{
		XSSFRow bestRow = sheet.createRow(probsCnt);

		/* Make sure the best results stands out from the other data */
		XSSFCellStyle style = wkbkResults.createCellStyle();
		XSSFFont font = wkbkResults.createFont();
		style.setBorderBottom(BorderStyle.THICK);
		style.setBorderTop(BorderStyle.THICK);
		font.setFontHeightInPoints((short)14);
		font.setBold(true);
		style.setFont(font);
		bestRow.setRowStyle(style);

		/* Record data in row of spreadsheet */
		XSSFCell bestCellinRow = bestRow.createCell(0);
		bestCellinRow.setCellValue(nameOfModelMatch);
		bestCellinRow.setCellStyle(style);
		bestCellinRow = bestRow.createCell(1);
		bestCellinRow.setCellValue(bestProbMatch);
		bestCellinRow.setCellStyle(style);
	}
	System.out.println(sb.toString());
	System.out.println("Done running thread");
}

private:
static void match_to_model_Levenshtein(
	Map<Integer, String> sampleChains, XSSFWorkbook wkbkResults)
{
	/* 1. Take each segment of sample image
	 *    for each model image
	 *        for each segment in model image
	 *            apply java-string-similarity method
	 *            O(n)+O(m*n^2)+Runtime_Algorithm */
	StringBuilder sb = new StringBuilder();
	int bestMatchesSz = 1;
	int cntMatchesSz = 1;
	if ((sampleChains == null) || sampleChains.size() == 0)
	{
		return;
	}
	else
	{
		bestMatchesSz = sampleChains.size();
		cntMatchesSz = (int)(sampleChains.size() * .1);
		if (cntMatchesSz < 1)
		{
			cntMatchesSz = 1;
		}
	}

	XSSFSheet sheet = null;
	synchronized(wkbkResults)
	{
		sheet = wkbkResults.createSheet("Levenshtein");
	}

	Map<Integer, ConcurrentHashMap<Integer, Integer>> bestMatches =
		new ConcurrentHashMap<Integer, ConcurrentHashMap<Integer, Integer>>(
			bestMatchesSz, (float)0.75);
	Map<String, Integer> cntMatches =
		new ConcurrentHashMap<String, Integer>(cntMatchesSz,
											   (float)0.90);

	Iterator<Integer> segments = sampleChains.keySet().iterator();
	int lastEntryID = DatabaseModule.getLastId();
	sb.append("Last ID=" + lastEntryID + "\n");
	while (segments.hasNext())
	{
		Integer segment = segments.next();
		String segmentChain = sampleChains.get(segment);
		sb.append("Working with sample segment " + segment + "\n");
		AtomicInteger minDistance = new AtomicInteger(Integer.MAX_VALUE);
		AtomicInteger minID = new AtomicInteger(Integer.MIN_VALUE);
		IntStream.rangeClosed(0, lastEntryID).parallel().forEach((i)->{
			/* Get the ith chain code from the database */
			String modelSegmentChain = DatabaseModule.getChainCode(i);

			/* Levenshtein measure is
			 * the minimum number of single-character edits
			 * (insertions, deletions or substitutions) required to
			 *  change one word into the other */
			if ((segmentChain == null) || (modelSegmentChain == null))
			{
				System.err.println("Levenshtein: modelSegment null for chain code" + i);
				return;
			}
			int distance = Levenshtein.distance(segmentChain, modelSegmentChain);

			/* track entry with the small number of
			 * edits then report filename and segment of id entry */
			if (distance < minDistance.get())
			{
				minDistance.set(distance);
				minID.set(i);
			}
		});

		/* Track which model segment provides the
		 * fewest modifications to a match */
		ConcurrentHashMap<Integer, Integer> hm =
			new ConcurrentHashMap<Integer, Integer>(1, (float)0.75);
		hm.put(minID.get(), minDistance.get());
		bestMatches.put(segment, hm);

		/* For each segment of the sample, track which model image
		 * and which image model perspective provides the best match*/
		String modelOfInterest = DatabaseModule.getFileName(minID.get());
		Integer curCnt = cntMatches.get(modelOfInterest);
		if (curCnt == null)
		{
			cntMatches.put(modelOfInterest, 1);
		}
		else
		{
			cntMatches.put(modelOfInterest, ++curCnt);
		}
	}

	/* Display result */
	Iterator<Integer> bmIterator = bestMatches.keySet().iterator();
	while (bmIterator.hasNext())
	{
		Integer key = bmIterator.next();
		ConcurrentHashMap<Integer, Integer> minValue = bestMatches.get(key);
		Iterator<Integer> ii = minValue.keySet().iterator();
		while (ii.hasNext())
		{
			Integer idmin = ii.next();
			String filenameOfID = DatabaseModule.getFileName(idmin);
			sb.append("Best L. Match for segment " + key + " is " +
					  idmin + " (" + filenameOfID + ") with " +
					  minValue.get(idmin) + " mods needed to match" + "\n");
		}
	}

	/* Tell user probably of matching various images based on how well
	 * sample segments matched to the database of model images */
	Iterator<String> cntIterator = cntMatches.keySet().iterator();
	float bestProbMatch = Float.MIN_NORMAL;
	String nameOfModelMatch = null;
	int probsCnt = 0;
	while (cntIterator.hasNext())
	{
		String filename = cntIterator.next();
		Integer count = cntMatches.get(filename);
		float probMatch = ((float)count) / sampleChains.size();
		sb.append("Probablity of matching " + filename + " is :" + (probMatch * 100) + " %" + "\n");

		/* record data in spreadsheet */
		synchronized(wkbkResults)
		{
			XSSFRow row = sheet.createRow(probsCnt++);
			XSSFCell cell = row.createCell(0);
			cell.setCellValue(filename);
			cell = row.createCell(1);
			cell.setCellValue(probMatch);
		}

		/* Track most likely match*/
		if (probMatch > bestProbMatch)
		{
			bestProbMatch = probMatch;
			nameOfModelMatch = filename;
		}
	}

	/* Tell user most likely match and record in spreadsheet */
	sb.append("Best probable match with Levenshetin is " + nameOfModelMatch +
			  " with probablity " + bestProbMatch + "\n");
	synchronized(wkbkResults)
	{
		XSSFRow bestRow = sheet.createRow(probsCnt);

		/* Make sure the best results stands out from the other data */
		XSSFCellStyle style = wkbkResults.createCellStyle();
		XSSFFont font = wkbkResults.createFont();
		style.setBorderBottom(BorderStyle.THICK);
		style.setBorderTop(BorderStyle.THICK);
		font.setFontHeightInPoints((short)14);
		font.setBold(true);
		style.setFont(font);
		bestRow.setRowStyle(style);

		/* Record data in row of spreadsheet */
		XSSFCell bestCellinRow = bestRow.createCell(0);
		bestCellinRow.setCellValue(nameOfModelMatch);
		bestCellinRow.setCellStyle(style);
		bestCellinRow = bestRow.createCell(1);
		bestCellinRow.setCellValue(bestProbMatch);
		bestCellinRow.setCellStyle(style);
	}
	System.out.println(sb.toString());
	System.out.println("Done running thread");
}

/**
 * Calculate the angle thresholds </br>
 * theta1 is the lower bound of angle difference </br>
 * theta2 is the upper bound of angle difference </br>
 * <br/>
 * Reference IJAI tools 2008 paper, leading to equation 13,14
 * section 4.0
 * <br/>
 * @param start -- Fixed point from which all centroids are connected
 * @param s -- list of centroids
 * @return thresholds -- an opencv matrix with the lower and upper
 * thresholds from the start node to each centroid in the image
 * 	0 [ theta_1, theta_2 ]
 *  1 [ theta_1, theta_2 ]
 *  ...
 *  n-1 [theta_1, theta_2]
 */
private:
static Mat calc_angle_differences(Point start, List<Point> s)
{
	Mat thresholds = new Mat(s.size() - 1, 2, CvType.CV_64FC1);
	for (int i = 0; i < s.size() - 1; i++)
	{
		Point p1 = s.get(i);
		Point p2 = s.get(i + 1);
		double theta1 = Math.atan2(p1.y - start.y, p1.x - start.x);
		double theta2 = Math.atan2(p2.y - start.y, p2.x - start.x);
		thresholds.put(i, 0, Math.toDegrees(theta1));
		thresholds.put(i, 1, Math.toDegrees(theta2));
	}
	return thresholds;
}

/**
 * Overloaded method that uses an array of point w/o considering a start point
 * Trying this with a Delaunay triangulation
 * @param s -- set of points, such as a Delaunay triangulation or Voroni tesslation
 * @return
 */
private:
static Mat calc_angle_differences(List<Point> s)
{
	Mat thresholds = new Mat(s.size() - 1, 2, CvType.CV_64FC1);
	for (int i = 0; i < s.size() - 1; i++)
	{
		Point p1 = s.get(i);
		Point p2 = s.get(i + 1);
		double theta1 = Math.atan2(p2.y - p1.y, p2.x - p1.x);
		double theta2 = Math.atan2(p1.y - p2.y, p1.x - p2.x);
		thresholds.put(i, 0, Math.toDegrees(theta1));
		thresholds.put(i, 0, Math.toDegrees(theta2));
	}
	return thresholds;
}

/**
 * <br/>
 * Reference IJAI tools 2008 paper, leading to equation 13,14
 * section 4.0
 * <br/>
 * Compute the angle similarity measure between two points </br>
 * @param theta_i0 -- model (exemplar0 base angle, first arc associated with node i (start node)
 * @param theta_1 -- model (exemplar) threshold angle
 * @param theta_2 -- sample (obstructed/full) threshold angle	 *
 * S_ANGSIM (deltatheta) = 1, deltatheta < theta_1 <br/>
 *                         ((theta_2 - deltatheta)/(theta_2-theta_1)) <br/>
 *                         0, deltatheta > theta2
 * @return angle of similarity from start to target points
 */
private:
static double angleSimilarity(double theta_i0, double theta_1, double theta_2)
{
	double deltatheta = theta_i0 - theta_1;
	if (deltatheta < theta_1)
	{
		return 1.0;
	}
	else if ((deltatheta <= theta_2) && (deltatheta >= 1))
	{
		return ((theta_2 - deltatheta) / (theta_2 - theta_1));
	}
	else
	{ // deltatheta > theta_2
		return 0.0;
	}
}

/**
 * Calc total similarity between two graphs
 * @param lwrAngThrshlds -- lower angle thresholds from  image
 * @param uprAngThreshlds -- upper angle thresholds from image
 * SIM_G = 1/N * sum_(i=1..n) * sum(j=1..N)[E(i,j) * S_ANGSIM(theta_ij - theta i0)
 * @return total similarity between two graphs
 */
private:
static double graphSimilarity(double[] lwrAngThrshlds,
							  double[] uprAngThreshlds)
{
	/* How to handle obstrucitons:
	 * Sample length <= model length so
	 * Keep list of best model choices for score
	 *  for each sample diff
	 *      for each model diff
	 *          angle sim calc w/ sample and model values
	 *          take score and place into sorted list
	 *   do the simG summation last with the sorted list */
	double simG = 0.0;
	for (int i = 0; i < uprAngThreshlds.length; i++)
	{
		for (int j = 0; j < lwrAngThrshlds.length; j++)
		{
			double angSim = angleSimilarity(lwrAngThrshlds[0], lwrAngThrshlds[j], uprAngThreshlds[i]);
			simG += angSim;
		}
	}

	/* with only partial info, N is limited to subset of model nodes for comparison */
	simG *= (1.0 / (uprAngThreshlds.length * lwrAngThrshlds.length));
	return simG;
}

/**
 * Calc total similarity between two graphs
 * @param lwrAngThrshlds -- lower angle thresholds from  image
 * @param uprAngThreshlds -- upper angle thresholds from image
 * SIM_G = 1/N * sum_(i=1..n) * sum(j=1..N)[E(i,j) * S_ANGSIM(theta_ij - theta i0)
 * @return total similarity between two graphs
 */
private:
static double graphSimilarity(float[] lwrAngThrshlds,
							  float[] uprAngThreshlds)
{
	/* How to handle obstrucitons:
	 * Sample length <= model length so
	 * Keep list of best model choices for score
	 *  for each sample diff
	 *      for each model diff
	 *          angle sim calc w/ sample and model values
	 *          take score and place into sorted list
	 *   do the simG summation last with the sorted list */
	float simG = (float)0.0;
	Map<Integer, Float> angSimValues = new HashMap<Integer, Float>(uprAngThreshlds.length * lwrAngThrshlds.length);
	for (int i = 0; i < uprAngThreshlds.length; i++)
	{
		for (int j = 0; j < lwrAngThrshlds.length; j++)
		{
			double angSim = angleSimilarity(lwrAngThrshlds[0], lwrAngThrshlds[j], uprAngThreshlds[i]);
			simG += angSim;
		}
	}

	/* with only partial info, N is limited to subset of model nodes for comparison */
	simG *= (1.0 / (uprAngThreshlds.length * lwrAngThrshlds.length));
	return simG;
}

private:
static void match_to_model_by_global_structure_angles(Mat sampleModelAngDiffs,
													  XSSFWorkbook wkbkResults,
													  String workbook_page_name)
{
	/*1. For model i (get all model filenames)		 *
	 *       1.1 Get upper thresholds (theta1)
	 *           1.1.1 Get first id of model image
	 *           1.1.2 Get last id of model image
	 *           1.1.3 Based on first and last id of model image get theta1s
	 *       1.2 Call graphSimilarity with upper thresholds of model and sample
	 *           1.2.1 place result in ordered list
	 *       1.3 Repeat 1.1 and 1.2 with lower thresholds (theta2)
	 *2. Rank upper and lower thresholds
	 *3. Report best upper and lower threshold match
	 * */
	ConcurrentSkipListMap<String, Double> modelSimGScores =
		new ConcurrentSkipListMap<String, Double>();
	List<String> modelNames = DatabaseModule.getAllModelFileName();

	// covert sample angle differences into suitable format for processing
	double[] upperSampleThresholds = new double[sampleModelAngDiffs.rows()];
	double[] lowerSampleThresholds = new double[sampleModelAngDiffs.rows()];
	for (int i = 0; i < sampleModelAngDiffs.rows(); i++)
	{
		upperSampleThresholds[i] = sampleModelAngDiffs.get(i, 0)[0];
		lowerSampleThresholds[i] = sampleModelAngDiffs.get(i, 1)[0];
	}
	double simGSample = graphSimilarity(lowerSampleThresholds, upperSampleThresholds);

	// parallel process the global model similarity measures
	modelNames.parallelStream().forEach(s->{
		s = s.replace(':', '/');
		// System.out.println("retrieving simg score for " + s);
		double simG = DatabaseModule.getSimGScore(s);
		modelSimGScores.put(s, simG);
	});

	// print the results to screen/file
	System.out.println("SimG-Delaunay score of sample is: " + simGSample);
	modelSimGScores.forEach((k, v)->{
		System.out.println("Model " + k + " has SimG score " + v);
	});

	// find lowest difference between models and sample
	Iterator<String> allTheModels = modelSimGScores.keySet().iterator();
	double lowestSimGDiff = Double.MAX_VALUE;
	String lowestSimGDiffModel = "No Match";
	while (allTheModels.hasNext())
	{
		String m = allTheModels.next();
		double mSimGScore = modelSimGScores.get(m);
		// System.out.println("Getting image " + m + " with simg score " + mSimGScore);
		double mSimGDiff = Math.abs(simGSample - mSimGScore);
		if (mSimGDiff < lowestSimGDiff)
		{
			lowestSimGDiff = mSimGDiff;
			lowestSimGDiffModel = m;
		}
	}

	System.out.println("Best simG match is: " + lowestSimGDiffModel +
					   " with difference " + lowestSimGDiff);

	// store the results in the spreadsheet
	XSSFSheet sheet = null;
	synchronized(wkbkResults)
	{
		sheet = wkbkResults.createSheet(workbook_page_name);
		XSSFRow row = sheet.createRow(0);
		XSSFCell cell = row.createCell(0);
		cell.setCellValue("Filename");
		cell = row.createCell(1);
		cell.setCellValue("SimG");
		cell = row.createCell(2);
		cell.setCellValue("Diff. Sample");
		cell = row.createCell(3);
		cell.setCellValue("Prob. Match");

		row = sheet.createRow(1);
		cell = row.createCell(0);
		cell.setCellValue("Sample");
		cell = row.createCell(1);
		cell.setCellValue(simGSample);

		Iterator<String> modelSimGIter = modelSimGScores.keySet().iterator();
		int simGCnt = 2;
		while (modelSimGIter.hasNext())
		{
			String simGModelName = modelSimGIter.next();
			Double simGValue = modelSimGScores.get(simGModelName);
			row = sheet.createRow(simGCnt);
			cell = row.createCell(0);
			cell.setCellValue(simGModelName);
			cell = row.createCell(1);
			cell.setCellValue(simGValue);
			cell = row.createCell(2);
			cell.setCellValue(Math.abs(simGValue - simGSample));
			cell = row.createCell(3);
			cell.setCellValue(1 - ((Math.atan(Math.abs(simGValue - simGSample)) / (Math.PI / 2))));
			simGCnt++;

			// update summary sheet as well for final calculation
			XSSFSheet summarySheet = wkbkResults.getSheet(SUMMARY_SHEET);
			simGModelName = simGModelName.replace('/', ':');
			int sumRowInt =
				ProjectUtilities.findRowInSpreadSheet(summarySheet, simGModelName);
			XSSFRow summaryRow = summarySheet.getRow(sumRowInt);
			XSSFCell summaryCell = summaryRow.createCell(SIMG_COLUMN_SUMMARY, CellType.NUMERIC);
			double probMatch = 1 - ((Math.atan(Math.abs(simGValue - simGSample)) / (Math.PI / 2)));
			if (Double.isNaN(probMatch) || Double.isInfinite(probMatch))
			{
				summaryCell.setCellValue(0.0);
			}
			else
			{
				summaryCell.setCellValue(probMatch);
			}
		}

		row = sheet.createRow(simGCnt);
		cell = row.createCell(0);
		cell.setCellValue("Best simG Value");
		cell = row.createCell(1);
		cell.setCellValue(lowestSimGDiffModel);
		cell = row.createCell(2);
		cell.setCellValue(lowestSimGDiff);
	}
}

@SuppressWarnings("unused") private: 
static void match_to_model_by_global_structure_angles2(Mat sampleModelAngDiffs,
																						   XSSFWorkbook wkbkResults,
																						   String workbook_page_name)
{
	/*1. For model i (get all model filenames)		 *
	 *       1.1 Get upper thresholds (theta1)
	 *           1.1.1 Get first id of model image
	 *           1.1.2 Get last id of model image
	 *           1.1.3 Based on first and last id of model image get theta1s
	 *       1.2 Call graphSimilarity with upper thresholds of model and sample
	 *           1.2.1 place result in ordered list
	 *       1.3 Repeat 1.1 and 1.2 with lower thresholds (theta2)
	 *2. Rank upper and lower thresholds
	 *3. Report best upper and lower threshold match
	 * */
	ConcurrentSkipListMap<String, Double> modelSimGScores =
		new ConcurrentSkipListMap<String, Double>();
	List<String> modelNames = DatabaseModule.getAllModelFileName();

	// covert sample angle differences into suitable format for processing
	double[] upperSampleThresholds = new double[sampleModelAngDiffs.rows()];
	double[] lowerSampleThresholds = new double[sampleModelAngDiffs.rows()];
	for (int i = 0; i < sampleModelAngDiffs.rows(); i++)
	{
		upperSampleThresholds[i] = sampleModelAngDiffs.get(i, 0)[0];
		lowerSampleThresholds[i] = sampleModelAngDiffs.get(i, 1)[0];
	}
	double simGSample = graphSimilarity(lowerSampleThresholds, upperSampleThresholds);

	// parallel process the global model similarity measures
	modelNames.parallelStream().forEach(s->{
		double simG = DatabaseModule.getSimGScore(s);
		modelSimGScores.put(s, simG);
	});

	// print the results to screen/file
	System.out.println("SimG-Delaunay score of sample is: " + simGSample);
	modelSimGScores.forEach((k, v)->{
		System.out.println("Model " + k + " has SimG score " + v);
	});

	// find lowest difference between models and sample
	Iterator<String> allTheModels = modelSimGScores.keySet().iterator();
	double lowestSimGDiff = Double.MAX_VALUE;
	String lowestSimGDiffModel = "No Match";
	while (allTheModels.hasNext())
	{
		String m = allTheModels.next();
		double mSimGScore = modelSimGScores.get(m);
		double mSimGDiff = Math.abs(simGSample - mSimGScore);
		if (mSimGDiff < lowestSimGDiff)
		{
			lowestSimGDiff = mSimGScore;
			lowestSimGDiffModel = m;
		}
	}

	System.out.println("Best simG-Delaunay match is: " + lowestSimGDiffModel +
					   "with difference " + lowestSimGDiff);

	// store the results in the spreadsheet
	XSSFSheet sheet = null;
	synchronized(wkbkResults)
	{
		sheet = wkbkResults.createSheet(workbook_page_name);
		XSSFRow row = sheet.createRow(0);
		XSSFCell cell = row.createCell(0);
		cell.setCellValue("Filename");
		cell = row.createCell(1);
		cell.setCellValue("SimG-Delaunay");

		row = sheet.createRow(1);
		cell = row.createCell(0);
		cell.setCellValue("Sample");
		cell = row.createCell(1);
		cell.setCellValue(simGSample);

		Iterator<String> modelSimGIter = modelSimGScores.keySet().iterator();
		int simGCnt = 2;
		while (modelSimGIter.hasNext())
		{
			String simGModelName = modelSimGIter.next();
			Double simGValue = modelSimGScores.get(simGModelName);
			row = sheet.createRow(simGCnt);
			cell = row.createCell(0);
			cell.setCellValue(simGModelName);
			cell = row.createCell(1);
			cell.setCellValue(simGValue);
			simGCnt++;
		}

		row = sheet.createRow(simGCnt);
		cell = row.createCell(0);
		cell.setCellValue("Best simG-Delauany Value");
		cell = row.createCell(1);
		cell.setCellValue(lowestSimGDiffModel);
		cell = row.createCell(2);
		cell.setCellValue(lowestSimGDiff);
	}
}

/**
 *
 * @param wkbkResults -- spreadsheet to record results
 * @param convertedTriangleList -- Delaunay triangulation
 * @param epsilon -- acceptable error in matching results
 */
private:
static void match_to_model_by_Delaunay_Graph_NoML(XSSFWorkbook wkbkResults, List<Point> convertedTriangleList,
												  float epsilon)
{
	/* Take each triad in unknown and compare to each known from the delaunay relation
	 * 1. Get all the model images
	 * 2. For each model image
	 *    2.1 Ask the database for all the triads for the model image
	 *    2.2 For each model image triad vertice, run through all the unknown model image triad vertices
	 *        2.2.1. If there is a match, inc the count for that model image
	 * 3. Take ds holding total matches and calc probability of match
	 * 4. Record results in spreadsheet */

	// Store matching results
	Map<String, Integer> cnts = new ConcurrentHashMap<>();
	Map<String, Double> contributions = new ConcurrentHashMap<>();
	Map<String, Map<Double, Double>> trackdupsAllModel = new ConcurrentHashMap<>();
	Map<Double, Double> trackdupsSample = new ConcurrentHashMap<>();

	// 1. Get all the model images
	List<String> modelFileNames = DatabaseModule.getAllModelFileName();
	System.out.println("Delaunay_Graph_NoML: Working with " + modelFileNames.size() + " models ");

	//  2. For each model image
	for (String model : modelFileNames)
	{
		// 2.1 Ask the database for all the triads for the model image
		// System.out.println("Delaunay_Graph_NoML: Working with " + model + " model ");
		List<Point> delaunay_model = DatabaseModule.getTriads(model.replace(':', '/'));
		// System.out.println("Delaunay_Graph_NoML: There are " + (delaunay_model.size()/3) + " triads to work with.");

		// 2.2 For each model image triad, run through all the unknown model image triads vertices
		Map<Double, Double> trackdupsModel = new ConcurrentHashMap<>();
		for (int i = 0; i < delaunay_model.size() - 1; i++)
		{
			Point m1 = delaunay_model.get(i);

			Double dupResult = trackdupsModel.put(m1.x, m1.y);
			if (dupResult != null)
			{
				continue;
			}

			double m1minx = delaunay_model.get(i).x;
			m1minx = m1minx - (m1minx * epsilon);
			double m1miny = delaunay_model.get(i).y;
			m1miny = m1miny - (m1miny * epsilon);
			double m1maxx = delaunay_model.get(i).x;
			m1maxx = m1maxx + (m1maxx * epsilon);
			double m1maxy = delaunay_model.get(i).y;
			m1maxy = m1maxy + (m1maxy * epsilon);

			for (int j = 0; j < convertedTriangleList.size() - 1; j++)
			{
				Point u1 = convertedTriangleList.get(j);

				trackdupsSample.put(u1.x, u1.y);

				// 2.2.1. If there is a match, inc the count for that model image

				/*
				 * An obstruction that does not outright remove a section may cause the vertices
				 * of intact regions to move, perhaps significantly (above and beyond any low
				 * epsilon value). This will cause parts of the Delaunay triangulation to shift as
				 * well due to the impact of segmentation with obstruction provided canvas size is
				 * maintained between exemplar and sample. In other words, triads may be left in part at
				 * certain node points and shifted at other node points, impacting the placement of other
				 * triads. If you sample just the slice, then you need to adjust the coordinate systems,but
				 * the questions becomes how much translation to apply for a system where exemplar and sample
				 * image may be taken under different conditions is
				 * likely needed (how to quantify if not passed as parameters into matching routine)
				 */

				if ((((u1.x >= m1minx) && (u1.x <= m1maxx) && (u1.y >= m1miny) && (u1.y <= m1maxy))))
				{
					System.out.println("Woohoo I found a match with " + model + " with node " + m1 + " and sample having node" + u1);
					if (cnts.get(model) == null)
					{
						cnts.put(model, 1);
					}
					else
					{
						int curCnt = cnts.get(model).intValue();
						curCnt++;
						cnts.put(model, curCnt);
					}
				}
			}
		}

		trackdupsAllModel.put(model, trackdupsModel);
	}

	// 3. Take ds holding total matches and calc probability of match
	Set<String> models = cnts.keySet();
	System.out.println("Del.Graph_NoML - Number of Model Images Matching: " + models.size());
	for (String model : models)
	{
		int theCount = cnts.get(model).intValue();
		Map<Double, Double> trackdupsModel = trackdupsAllModel.get(model);
		double contributionValue = theCount / ((double)(trackdupsModel.size()));
		System.out.println("Model " + model + " contributed " + contributionValue + " or " + (contributionValue * 100) + " to the overall match");
		contributions.put(model, contributionValue);
	}

	// 4. Record results in spreadsheet
	XSSFSheet sheet = null;
	synchronized(wkbkResults)
	{
		sheet = wkbkResults.createSheet("Delaunay-noML");
		XSSFRow row = sheet.createRow(0);
		XSSFCell cell = row.createCell(0);
		cell.setCellValue("Model");
		cell = row.createCell(1);
		cell.setCellValue("Contribution Count");
		cell = row.createCell(2);
		cell.setCellValue("Prob. Match ");
		cell = row.createCell(3);
		cell.setCellValue("Prob. Match Percentage");

		models = cnts.keySet();
		int rowCnt = 1;
		for (String model : models)
		{
			row = sheet.createRow(rowCnt++);
			cell = row.createCell(0);
			cell.setCellValue(model);
			cell = row.createCell(1);
			cell.setCellValue(cnts.get(model).intValue());
			cell = row.createCell(2);
			cell.setCellValue(contributions.get(model).doubleValue());
			cell = row.createCell(3);
			cell.setCellValue(contributions.get(model).doubleValue() * 100);
		}
	}
}

/**
 * Using Weka ML, match a Delaunay triangulation of an unknown incomplete image to a database of samples
 * @param wkbkResults -- spreadsheet to record results
 * @param convertedTriangleList -- Delaunay triangulation
 * @param classifierPref -- which Weka ML algorithm to use
 */
private:
static void match_to_model_by_Delaunay_Graph(XSSFWorkbook wkbkResults, List<Point> convertedTriangleList,
											 String classifierPref)
{
	/**
	 *    Using Weka to look at the Delaunay graphs, perform supervised learning
	 * 1. Build training data w/ Weka objects
	 *    1.1 Ask database for Delaunay graph
	 * 2. Build sample data w/ Weka objects
	 * 3. Match
	 * 4. Write results to spreadsheet
	 */

	// Build training data with Weka objects
	// Ask database for Delaunay graph

	// build training database for all model images
	List<String> modelFileNames = DatabaseModule.getAllModelFileName();

	// declare objects to aid spreadsheet writing
	Map<String, Integer> imgNodeCnt = new ConcurrentHashMap<String, Integer>(modelFileNames.size());

	// create attributes
	ArrayList<Attribute> attributes = new ArrayList<Attribute>();
	attributes.add(new Attribute(DatabaseModule.TRIAD_X1, 0));
	attributes.add(new Attribute(DatabaseModule.TRIAD_Y1, 1));
	attributes.add(new Attribute(DatabaseModule.TRIAD_X2, 2));
	attributes.add(new Attribute(DatabaseModule.TRIAD_Y2, 3));
	attributes.add(new Attribute(DatabaseModule.TRIAD_X3, 4));
	attributes.add(new Attribute(DatabaseModule.TRIAD_Y3, 5));
	attributes.add(new Attribute(DatabaseModule.FILENAME_COLUMN, modelFileNames, 6));

	for (Attribute a : attributes)
	{
		System.out.println("Attribute Information");
		System.out.println(a);
		System.out.println("");
	}

	// create instances object, set initial? capacity to number of models and each rotation
	Instances training = new Instances("Training", attributes, modelFileNames.size() * 8);

	// need to determine which attribute and its index will hold the labels
	training.setClassIndex(training.numAttributes() - 1);

	// work through each model
	for (String model : modelFileNames)
	{
		System.out.println("Working with model " + model);
		List<Point> modelPointsForTraining = DatabaseModule.getDelaunayGraph(model);

		if (modelPointsForTraining == null)
		{
			System.out.println(" Model " + model + " has no valid data for training ");
			continue;
		}

		// go tuple-by-tuple for each model image
		int graphSize = modelPointsForTraining.size();
		for (int i = 0; i < graphSize; i += 3)
		{

			// take graph data row from database and transform into training instance
			Instance inst = new DenseInstance(attributes.size());
			inst.setValue(attributes.get(0), modelPointsForTraining.get(i).x);
			inst.setValue(attributes.get(1), modelPointsForTraining.get(i).y);
			inst.setValue(attributes.get(2), modelPointsForTraining.get(i + 1).x);
			inst.setValue(attributes.get(3), modelPointsForTraining.get(i + 1).y);
			inst.setValue(attributes.get(4), modelPointsForTraining.get(i + 2).x);
			inst.setValue(attributes.get(5), modelPointsForTraining.get(i + 2).y);

			/* To prevent UnsignedDataSetExeception, set the instance dataset to the
			   instances object that you are adding the instance to, seems circular
			   to me */
			inst.setDataset(training);

			// does this attach a label to this instance, need to associate the
			// filename, model name
			inst.setClassValue(model);

			// add instance to training data instances
			boolean result = training.add(inst);
			if (result == false)
			{
				System.out.println("Instance " + inst.toString() + " not added ");
			}
		}
		// remove any attributes you don't want w/ filter, not applicable, yet
	}

	// remove any possible wasted space from declaration
	training.compactify();

	// now work on sample data
	Instances sample = new Instances("Sample", attributes, convertedTriangleList.size());
	sample.setClassIndex(sample.numAttributes() - 1);

	int graphSize = convertedTriangleList.size();
	for (int i = 0; i < graphSize; i += 3)
	{
		// take graph data row from database and transform into sample instance
		Instance inst = new DenseInstance(attributes.size());
		inst.setValue(attributes.get(0), convertedTriangleList.get(i).x);
		inst.setValue(attributes.get(1), convertedTriangleList.get(i).y);
		inst.setValue(attributes.get(2), convertedTriangleList.get(i + 1).x);
		inst.setValue(attributes.get(3), convertedTriangleList.get(i + 1).y);
		inst.setValue(attributes.get(4), convertedTriangleList.get(i + 2).x);
		inst.setValue(attributes.get(5), convertedTriangleList.get(i + 2).y);

		/* To prevent UnsignedDataSetExeception, set the instance dataset to the
		   instances object that you are adding the instance to, seems circular
		   to me */
		inst.setDataset(sample);

		// don't label these instances with anything, we want alg to guess here

		// add sample data to compare against model data
		sample.add(inst);
	}

	// remove any possible wasted space from declaration
	sample.compactify();

	// Initialize a classifier based on user preference
	// start with tree-based ML classifier given multi-class nature of data
	Classifier classifier = null;
	switch (classifierPref)
	{
	case "J48":
		classifier = new J48();
		break;
	case "LMT":
		classifier = new LMT();
		break;
	default:
		classifier = new J48();
	}
	try
	{
		classifier.buildClassifier(training);
	}
	catch (Exception e)
	{
		// TODO Auto-generated catch block
		e.printStackTrace();
	}

	/* match unknown to best model image, store matches
	   into an inmemory object for later processing */
	Evaluation eval = null;
	AbstractOutput outcomes = new InMemory();
	AbstractOutput outcomesHuman = new PlainText();
	StringBuffer outcomeStr = new StringBuffer();
	StringBuffer outcomeStrHuman = new StringBuffer();

	((PlainText)outcomesHuman).setBuffer(outcomeStrHuman);
	((PlainText)outcomesHuman).setAttributes("7");
	((PlainText)outcomesHuman).setHeader(sample);

	((InMemory)outcomes).setBuffer(outcomeStr);
	((InMemory)outcomes).setAttributes("7");
	((InMemory)outcomes).setHeader(sample);

	/* this will create the arraylist of predictions needed for later processing
	   into the spreadsheet */
	outcomes.printHeader();
	outcomesHuman.printHeader();

	try
	{
		eval = new Evaluation(training);
		eval.evaluateModel(classifier, sample, outcomes, outcomesHuman);
		System.out.println("kappa=" + eval.kappa());
		System.out.println(eval.toSummaryString("\nResults\n======\n", true));
		List<PredictionContainer> predictions = ((InMemory)outcomes).getPredictions();
		int i = 0;
		for (PredictionContainer pred : predictions)
		{
			System.out.println("\nContainer #" + i);
			System.out.println("- instance:\n" + pred.instance);
			System.out.println("- prediction:\n" + pred.prediction);
			double predValue = pred.prediction.predicted();
			System.out.println("- prediction image class=" + predValue + " and filename:\n" + modelFileNames.get((int)predValue));
			Integer cnt = imgNodeCnt.get(modelFileNames.get((int)predValue));
			if (cnt == null)
			{
				imgNodeCnt.put(modelFileNames.get((int)predValue), 1);
			}
			else
			{
				imgNodeCnt.put(modelFileNames.get((int)predValue), ++cnt);
			}
			Set<String> imageFNStrs = imgNodeCnt.keySet();
			for (String fn : imageFNStrs)
			{
				System.out.println(fn + "=" + imgNodeCnt.get(fn));
			}

			i++;
		}

		StringBuffer sb = ((PlainText)outcomesHuman).getBuffer();
		System.out.println(sb);

		outcomes.printFooter();
		outcomesHuman.printFooter();
	}
	catch (Exception e)
	{
		// TODO Auto-generated catch block
		e.printStackTrace();
	}

	// store the results in the spreadsheet
	XSSFSheet sheet = null;
	String bestModel = "NoModel";
	int bestModelCnt = Integer.MIN_VALUE;
	synchronized(wkbkResults)
	{
		sheet = wkbkResults.createSheet("DelGraphWeka");
		XSSFRow row = sheet.createRow(0);
		XSSFCell cell = row.createCell(0);
		cell.setCellValue("Model");
		cell = row.createCell(1);
		cell.setCellValue("Count");
		cell = row.createCell(2);
		cell.setCellValue("Probability Match");

		Iterator<String> imgNdIt = imgNodeCnt.keySet().iterator();
		int sprRowCnt = 2;
		Integer predCnt = ((InMemory)outcomes).getPredictions().size();
		while (imgNdIt.hasNext())
		{
			String daModel = imgNdIt.next();
			Integer daCount = imgNodeCnt.get(daModel);
			row = sheet.createRow(sprRowCnt);
			cell = row.createCell(0);
			cell.setCellValue(daModel);
			cell = row.createCell(1);
			cell.setCellType(CellType.NUMERIC);
			cell.setCellValue(daCount);
			cell = row.createCell(2);
			cell.setCellType(CellType.NUMERIC);
			cell.setCellValue((((float)daCount) / predCnt) * 100.0);
			sprRowCnt++;

			if (daCount > bestModelCnt)
			{
				bestModel = daModel;
				bestModelCnt = daCount;
			}

			// update summary sheet as well for final calculation
			XSSFSheet summarySheet = wkbkResults.getSheet(SUMMARY_SHEET);
			daModel = daModel.replace('/', ':');
			int sumRowInt =
				ProjectUtilities.findRowInSpreadSheet(summarySheet, daModel);
			XSSFRow summaryRow = summarySheet.getRow(sumRowInt);
			XSSFCell summaryCell = summaryRow.createCell(WEKA_DELA_COLUMN_SUMMARY, CellType.NUMERIC);
			summaryCell.setCellValue((((float)daCount) / predCnt) * 100.0);
		}

		// note best result
		XSSFCellStyle style = wkbkResults.createCellStyle();
		XSSFFont font = wkbkResults.createFont();
		style.setBorderBottom(BorderStyle.THICK);
		style.setBorderTop(BorderStyle.THICK);
		font.setFontHeightInPoints((short)14);
		font.setBold(true);
		style.setFont(font);

		// record the best row
		sprRowCnt++;
		row = sheet.createRow(sprRowCnt);
		cell = row.createCell(0);
		cell.setCellStyle(style);
		cell.setCellValue(bestModel);
		cell = row.createCell(1);
		cell.setCellStyle(style);
		cell.setCellType(CellType.NUMERIC);
		cell.setCellValue(bestModelCnt);
		cell = row.createCell(2);
		cell.setCellType(CellType.NUMERIC);
		cell.setCellStyle(style);
		cell.setCellValue((((float)bestModelCnt) / predCnt) * 100.0);
	}

	return;
}

private:
static void determine_line_connectivity(ArrayList<CurveLineSegMetaData> lmd)
{
	ArrayList<CurveLineSegMetaData> cList;

	for (CurveLineSegMetaData line : lmd)
	{
		cList = new ArrayList<CurveLineSegMetaData>();
		Point p1 = line.getSp();
		int lineCnt = 0;
		for (CurveLineSegMetaData line2 : lmd)
		{

			// skip the first line
			if (lineCnt == 0)
			{
				lineCnt++;
				continue;
			}
			else
			{
				lineCnt++;
			}

			Point p2 = line2.getSp();
			if (!p1.equals(p2))
			{
				continue;
			}
			else
			{
				Point ep = line2.getEp();

				// don't connect a line that is really a point
				if (p2.equals(ep))
				{
					continue;
				}
				cList.add(line2);
			}
		}
		line.setConnList(cList);
	}
}

/**
 * Generate the shape description for a region<br/>
 * <ul>
 * <li> Line length is based on line distance formula from beg.
 * to end of a curved line segment </li>
 * <li> Line orientation is based on the derivative of the line
 * and relative to the first sp of the first curved line segment </li>
 * <li> Line curvature is the amount by which a line deviates from
 * being straight or how much of a curve it is </li>
 * </ul>
 * @param segx -- x entries from line segment generation
 * @param segy -- y entries from line segment generation
 */
private:
static ArrayList<CurveLineSegMetaData> shape_expression(ArrayList<Mat> segx,
														ArrayList<Mat> segy)
{
	int sz = segx.size();
	ArrayList<CurveLineSegMetaData> lmd = new ArrayList<CurveLineSegMetaData>(sz);
	long lineNumber = 0;

	// Sanity check
	if ((segx.size() == 0) || (segy.size() == 0))
	{
		System.out.println("WARNING: No segment data to generate " +
						   "line shape expression from for image " +
						   "analysis");
		return null;
	}

	Mat segx1Mat = segx.get(0);
	Mat segy1Mat = segy.get(0);
	int startingElement = 0;
	Size szFirst = segx1Mat.size();
	while ((startingElement < sz - 1) &&
		   ((szFirst.width == 0) || (szFirst.height == 0)))
	{
		startingElement++;
		segx1Mat = segx.get(startingElement);
		segy1Mat = segy.get(startingElement);
	}
	if (startingElement >= sz - 1)
	{
		return null;
	}
	double x1 = segx1Mat.get(0, 0)[0];
	double y1 = segy1Mat.get(0, 0)[0];
	double spX1 = segx1Mat.get(0, 0)[0];
	double spY1 = segy1Mat.get(0, 0)[0];
	double x1C = segx1Mat.get(0, 0)[0];
	double y1C = segy1Mat.get(0, 0)[0];
	// store basic line information including length
	for (int i = startingElement + 1; i < sz; i++)
	{
		long tic = System.nanoTime();
		Mat segx2Mat = segx.get(i);
		Mat segy2Mat = segy.get(i);

		double x2 = segx2Mat.get(0, 0)[0];
		double y2 = segy2Mat.get(0, 0)[0];

		// distance calculation in pixels
		double distance = Math.sqrt(
			Math.pow((x1 - x2), 2) +
			Math.pow((y1 - y2), 2));

		// orientation calculation in degrees
		double dy = y2 - spY1;
		double dx = x2 - spX1;
		double orientation = Math.atan2(dy, dx);
		orientation = Math.toDegrees(orientation);
		if (orientation < 0)
		{
			orientation += 360;
		}

		/* calculate line curvature -- note that there is
		  no curvature between two lines so what does
		  Bourbakis's older work mean when they talk about
		  this -- over two line segments with the first one
		  zero? */
		double curvature = 0;
		if (i == 1)
		{
			curvature = 0;
		}
		else
		{
			double Cdx = x2 - x1C;
			double Cdy = y2 - y1C;
			curvature = Math.atan2(Cdy, Cdx) / Math.hypot(Cdy, Cdx);

			/* Note for the entire region it might be
			 * dx = gradient(seg_x);
			 * dy = graident(seg_y);
			 * curv = gradietn(atan2(dy,dx)/hypot(dx,dy)*/
		}

		// given good values, let's save this curved line segment
		if (distance > 0)
		{
			CurveLineSegMetaData lmdObj = new CurveLineSegMetaData(
				new Point(x1, y1),
				new Point(x2, y2),
				distance, orientation, curvature, ++lineNumber);

			/* calc time to determine this curved line segments,
			   us to low ms probably, store result*/
			long toc = System.nanoTime();
			long totalTime = toc - tic;
			lmdObj.setTotalTime(totalTime);

			/* add curve line segment to data structure for all curved
			 * line segment for the segmented region of the image */
			lmd.add(lmdObj);
		}
		segx1Mat = segx2Mat.clone();
		segy1Mat = segy2Mat.clone();

		/* starting point of next curved line segment is end
		 * point of the previous segment */
		x1C = x2;
		y1C = y2;
		spX1 = x2;
		spY1 = y2;
		x1 = x2;
		y1 = y2;
	}
	return lmd;
}

/***
 * Create lines connecting the segments
 * @param labels
 * @param coords
 * @return
 */
private:
static Mat constructLines(Mat labels, Mat coords)
{
	if (labels == null)
	{
		System.err.println("constructLines(): WARNING: labels is null");
		return null;
	}

	if (coords == null)
	{
		System.err.println("constructLines(): WARNING: coords is null");
		return null;
	}

	if (coords.empty())
	{
		System.err.println("constructLines(): WARNING: coords is empty");
		return null;
	}

	// total number of points to generate between a and b
	int n = 1000;

	// points from x1 to x2
	double[] x1 = coords.get(0, 0);
	double[] x2 = coords.get(1, 0);
	Mat cpts = ProjectUtilities.linspace_Mat(x1[0], x2[0], n);

	// points from y1 to y2
	double[] y1 = coords.get(0, 1);
	double[] y2 = coords.get(1, 1);
	Mat rpts = ProjectUtilities.linspace_Mat(y1[0], y2[0], n);

	int rows = labels.rows();
	int cols = labels.cols();
	// index = sub2ind([r c],round(cpts),round(rpts));
	// Convert all the 2d subscripts to linear indices
	Mat index = new Mat(1, n, cpts.type(), Scalar.all(0));
	for (int i = 0; i < rows; i++)
	{
		double[] colPtArray = cpts.get(0, i);
		double[] rowPtArray = rpts.get(0, i);
		if ((colPtArray == null) || (rowPtArray == null) ||
			(colPtArray.length == 0) || (rowPtArray.length == 0))
		{
			System.err.println("Part of row " + i + " of total number of rows " + rows + " has no data");
			continue;
		}
		int rowSub = (int)Math.round(colPtArray[0]);
		int colSub = (int)Math.round(rowPtArray[0]);
		int value = ProjectUtilities.sub2ind(rowSub, colSub,
											 rows - 1, cols - 1);
		index.put(0, i, value);
	}

	//  bbw(index) = 1;
	int size = index.cols();
	for (int i = 0; i < size; i++)
	{
		double ind = index.get(0, i)[0];
		Mat m = ProjectUtilities.ind2sub((int)ind, rows, cols);
		int rowLabel = (int)m.get(0, 0)[0];
		int colLabel = (int)m.get(0, 1)[0];
		if ((rowLabel >= 0) && (colLabel >= 0))
		{
			labels.put(rowLabel, colLabel, 1);
		}
		m.release();
	}

	/* Allow the column and row points matrices to be release
	 * back to memory */
	cpts.release();
	rpts.release();

	return labels.clone();
}

/**
 * Generate the line segments of a region
 * @param cc -- chain code that aids in calculating line lengths and in
 * determining how the direction changes from segment to segment
 * @param start -- relative anchor point to use as starting point in
 * generating line segments, for circular regions, you would see the
 * last line segment connect back to the starting point
 * @param sensitivity -- the fineness with which line segments are
 * generated to more smoothly or roughly define a geometric area
 * the higher the value the more change or rougher the generated
 * line segments will appear
 * @return a composite object with
 * the list of x coordinates of all line segments, the list of
 * y coordinates of all line segments, and the time to generate the
 * line segments
 */
private:
static LineSegmentContainer line_segment(ArrayList<Double> cc,
										 Point start,
										 int sensitivity)
{

	// sanity checks
	if ((cc == null) && (start == null))
	{
		System.err.println("WARNING: No chain code or start point");
		return null;
	}
	else if ((cc == null) && (start != null))
	{
		System.err.println("WARNING: Segment is only a point");
		return null;
	}
	else if ((cc != null) && (start == null))
	{
		System.err.println("WARNING: start point not defined");
		return null;
	}

	System.out.println("cc length: " + cc.size());

	/* offsets for visiting the eight neighbor pixels
	 * of the current pixel under analysis */
	int[][] directions = new int[][]{
		{1, 0},
		{1, -1},
		{0, -1},
		{-1, -1},
		{-1, 0},
		{-1, 1},
		{0, 1},
		{1, 1}};

	/* All the points in x and y directions making up the line
	 * segments of a region */
	ArrayList<Mat> segment_x = new ArrayList<Mat>();
	ArrayList<Mat> segment_y = new ArrayList<Mat>();

	long tic = System.nanoTime();

	Point coords = start.clone();
	Point startCoordinate = start.clone();

	Mat newMatx = new Mat(1, 1, CvType.CV_32FC1,
						  Scalar.all(0));
	Mat newMaty = new Mat(1, 1, CvType.CV_32FC1,
						  Scalar.all(0));
	newMatx.put(0, 0, coords.x);
	newMaty.put(0, 0, coords.y);

	segment_x.add(newMatx.clone());
	segment_y.add(newMaty.clone());

	// Move through each value in the chain code
	int limit = cc.size() - 1;
	for (int i = 1; i < limit; i++)
	{
		Point newCoordinate = new Point(coords.x + directions[(int)(cc.get(i).intValue())][0],
										coords.y + directions[(int)(cc.get(i).intValue())][1]);
		double distMeasure = Math.sqrt(Math.pow(newCoordinate.x - startCoordinate.x, 2) +
									   Math.pow(newCoordinate.y - startCoordinate.y, 2));
		if (distMeasure >= sensitivity)
		{
			newMatx = new Mat(1, 1, CvType.CV_32FC1,
							  Scalar.all(0));
			newMaty = new Mat(1, 1, CvType.CV_32FC1,
							  Scalar.all(0));
			newMatx.put(0, 0, coords.x);
			newMaty.put(0, 0, coords.y);

			segment_x.add(newMatx.clone());
			segment_y.add(newMaty.clone());

			startCoordinate.x = newCoordinate.x;
			startCoordinate.y = newCoordinate.y;
		}

		coords.x = coords.x + directions[(int)(cc.get(i).intValue())][0];
		coords.y = coords.y + directions[(int)(cc.get(i).intValue())][1];
	}

	// how long in ns did it take for us to generate the line segments
	long segment_time = System.nanoTime() - tic;

	/* package all the line segment coordinates and times into a
	   composite object */
	System.out.println("segment_x size=" + segment_x.size());
	System.out.println("segment_y size=" + segment_y.size());
	LineSegmentContainer lsc = new LineSegmentContainer(
		segment_x, segment_y,
		segment_time);
	return lsc;
}

/**
 * Generate an encoding for the input image
 *
 * the chain code uses a compass metaphor with numbers 0 to 7
 * incrementing in a clock wise fashion. South is 0, North is
 * 4, East is 6, and West is 2
 *
 * NOTE: chain code seems to end with the first break, pixels
 * have to be part of one continuous border
 *
 * NOTE: not scale invariant
 * NOTE: to be rotation invariant, needs difference coding
 *
 * @param img -- input image
 * @return a composite object consisting of the image border,
 * the list of times it took to generate each chain, the chain
 * code itself, and the starting location for the chain code
 */
private:
static ChainCodingContainer chaincoding1(Mat img)
{
	int[][] directions = new int[][]{
		{1, 0},
		{1, -1},
		{0, -1},
		{-1, -1},
		{-1, 0},
		{-1, 1},
		{0, 1},
		{1, 1}};

	long tic = System.nanoTime();
	ArrayList<Point> pts = ProjectUtilities.findInMat(img, 1, "first");

	/* Verify there is data to process in segment if not return
	 * empty chain code container */
	if (pts.size() == 0)
	{
		ArrayList<Double> noData = new ArrayList<Double>(1);
		noData.add(0.0);
		ChainCodingContainer ccc =
			new ChainCodingContainer(
				img, System.nanoTime() - tic, noData, new Point(0, 0));
		return ccc;
	}

	/* Get the number of rows and columns in segment to process over the whole
	 * of it */
	int rows = img.rows();
	int cols = img.cols();

	// The chain code
	ArrayList<Double> cc = new ArrayList<Double>();
	ArrayList<Double> allD = new ArrayList<Double>();

	// Coordinates of the current pixel
	Point coord = pts.get(0);
	Point start = coord.clone();

	// The starting direction
	int dir = 1;
	ArrayList<Point> coordsLookedAt = new ArrayList<Point>();
	coordsLookedAt.add(start);
	while (true)
	{
		Point newcoord = new Point(coord.x + directions[dir][0],
								   coord.y + directions[dir][1]);
		coordsLookedAt.add(newcoord);

		double[] value = img.get((int)newcoord.x, (int)newcoord.y);
		if (((int)newcoord.x < rows) &&
			((int)newcoord.y < cols) &&
			(value != null) && (value[0] != 0.0))
		{
			// not sure about this line cc = [cc, dir] from matlab code
			cc.add(Double.valueOf(dir));
			coord = newcoord.clone();
			dir = Math.floorMod(dir + 2, 8);
		}
		else
		{
			dir = Math.floorMod(dir - 1, 8);
		}
		allD.add((double)dir);

		// Back to starting situation
		if (((int)coord.x == start.x) &&
			((int)coord.y == start.y) &&
			(dir == 1))
		{
			break;
		}
	}

	/* Line segment generation using generated line code, set cells to
	 * almost total black */
	Mat border = new Mat(rows, cols, img.type(), Scalar.all(0));
	Point coords = start.clone();
	for (int i = 0; i < cc.size(); i++)
	{
		border.put((int)coords.x, (int)coords.y, new double[]{1.0});

		// coords = coords + directions(cc(ii)+1,:);
		coords.x = coords.x + directions[(int)(cc.get(i).intValue())][0];
		coords.y = coords.y + directions[(int)(cc.get(i).intValue())][1];
	}

	long chain_time = System.nanoTime() - tic;

	ChainCodingContainer ccc =
		new ChainCodingContainer(border, chain_time, cc, start);

	return ccc;
}

/**
 * Partition the input image data into clusters using NGB provided method
 * <br/> NOTE: opencv kmeans does not present the data in a useful form w/o
 * additional post-processing, results are generally different
 * @param data -- input image (signal data or n observations)
 * @param nclusters -- number of sets to partition data into
 * @param niterations -- number of times to attempt partitioning
 * @return container with clustered data, stats
 */
private:
static kMeansNGBContainer
kmeansNGB(Mat data, int nclusters, int niterations)
{
	// adjust input to double precision floating
	kMeansNGBContainer container = null;
	Mat input = new Mat(data.rows(), data.cols(), data.type());
	data.convertTo(input, data.type(), 1.0 / 255.0);

	// create return matrix
	Mat Label = new Mat(data.rows(), data.cols(), data.type(),
						Scalar.all(0.0));

	int nrows = data.rows();
	int ncols = data.cols();

	// random seed
	Mat Temprows = new Mat(1, nclusters, CvType.CV_32SC1);
	Mat Tempcols = new Mat(1, nclusters, CvType.CV_32SC1);

	// test data for cell2.pgm with 16 clusters with 16 iterations
	/* Temprows.put(0, 0, 29);
	Temprows.put(0, 1, 114);
	Temprows.put(0, 2, 15);
	Temprows.put(0, 3, 25);
	Temprows.put(0, 4, 171);
	Temprows.put(0, 5, 79);
	Temprows.put(0, 6, 108);
	Temprows.put(0, 7, 168);
	Temprows.put(0, 8, 179);
	Temprows.put(0, 9, 27);
	Temprows.put(0, 10, 69);
	Temprows.put(0, 11, 52);
	Temprows.put(0, 12, 122);
	Temprows.put(0, 13, 101);
	Temprows.put(0, 14, 36);
	Temprows.put(0, 15, 48);

	Tempcols.put(0, 0, 116);
	Tempcols.put(0, 1, 34);
	Tempcols.put(0, 2, 16);
	Tempcols.put(0, 3, 4);
	Tempcols.put(0, 4, 243);
	Tempcols.put(0, 5, 44);
	Tempcols.put(0, 6, 189);
	Tempcols.put(0, 7, 212);
	Tempcols.put(0, 8, 167);
	Tempcols.put(0, 9, 246);
	Tempcols.put(0, 10, 61);
	Tempcols.put(0, 11, 25);
	Tempcols.put(0, 12, 12);
	Tempcols.put(0, 13, 148);
	Tempcols.put(0, 14, 91);
	Tempcols.put(0, 15, 141); */

	Core.randu(Temprows, 0, input.rows());
	Core.randu(Tempcols, 0, input.cols());
	Mat Indrows = Temprows.clone();
	Mat Indcolumns = Tempcols.clone();

	// determine average intensity of randomly chosen clusters
	int counter = 0;
	Mat avItensity = new Mat(1, nclusters, CvType.CV_64FC1, Scalar.all(0));
	Mat avRows = new Mat(1, nclusters, CvType.CV_64FC1,
						 Scalar.all(0.0));
	Mat avCols = new Mat(1, nclusters, CvType.CV_64FC1,
						 Scalar.all(0.0));
	for (int k = 0; k < nclusters; k++)
	{
		int rowToRetrieve = (int)Indrows.get(0, k)[0];
		int colToRetrieve = (int)Indcolumns.get(0, k)[0];
		double[] value = input.get(rowToRetrieve, colToRetrieve);
		avItensity.put(0, k, value[0]);
		// System.out.println(avItensity.get(k, 0)[0]);
		// System.out.println(avItensity.dump());
	}
	// System.out.println(avItensity.dump());

	Mat ClusterCenter = new Mat(nclusters, 3, CvType.CV_64FC1, Scalar.all(0.0));
	Mat count = new Mat(1, nclusters, CvType.CV_64FC1, Scalar.all(0.0));
	Mat sumInt = new Mat(1, nclusters, CvType.CV_64FC1, Scalar.all(0.0));
	Mat sumy = new Mat(1, nclusters, CvType.CV_64FC1, Scalar.all(0.0));
	Mat sumx = new Mat(1, nclusters, CvType.CV_64FC1, Scalar.all(0.0));
	while (counter < niterations)
	{
		// assign the cluster center
		for (int k = 0; k < nclusters; k++)
		{
			ClusterCenter.put(k, 0, Indrows.get(0, k));
			ClusterCenter.put(k, 1, Indcolumns.get(0, k));
			ClusterCenter.put(k, 2, avItensity.get(0, k));
			count.put(0, k, 0);
			sumInt.put(0, k, 0);
			sumy.put(0, k, 0);
			sumx.put(0, k, 0);
		}

		// assign the pixel to clusters
		Mat distance = new Mat(1, nclusters, CvType.CV_64FC1, Scalar.all(0));
		for (int i = 0; i < nrows; i++)
		{
			for (int j = 0; j < ncols; j++)
			{
				for (int k = 0; k < nclusters; k++)
				{
					double value =
						Math.pow((i - Indrows.get(0, k)[0]), 2) + Math.pow((j - Indcolumns.get(0, k)[0]), 2) + Math.pow(((255 * input.get(i, j)[0]) - ClusterCenter.get(k, 2)[0]), 2);
					value = Math.sqrt(value);
					distance.put(0, k, value);
				}
				MinMaxLocResult minmaxlocs = Core.minMaxLoc(distance);
				Point cluster = minmaxlocs.minLoc;

				/* this gives pixels in the same area that are assigned to
				 * the same cluster a standard color */
				double intensity = minmaxlocs.minLoc.x;
				double length = 255 / (double)nclusters;

				// place the pixel in its bucket with its artificial color
				Label.put(i, j, length * intensity);

				// update stat counts
				double cnt = count.get(0, (int)cluster.x)[0];
				count.put(0, (int)cluster.x, ++cnt);
				double sumVal = sumInt.get(0, (int)cluster.x)[0] + input.get(i, j)[0];
				sumInt.put(0, (int)cluster.x, sumVal);
				double sumyVal = sumy.get(0, (int)cluster.x)[0] + i;
				sumy.put(0, (int)cluster.x, (sumyVal == 0) ? 1 : sumyVal);
				double sumxVal = sumx.get(0, (int)cluster.x)[0] + j;
				sumx.put(0, (int)cluster.x, (sumxVal == 0) ? 1 : sumxVal);
			}
		}

		// update stats
		Core.divide(sumy, count, avRows);
		avRows = ProjectUtilities.round(avRows);
		Core.divide(sumx, count, avCols);
		avCols = ProjectUtilities.round(avCols);
		Core.divide(sumInt, count, avItensity);
		avItensity = ProjectUtilities.multiplyScalar(avItensity, 255.0);
		avItensity = ProjectUtilities.round(avItensity);

		Indrows = avRows.clone();
		Indcolumns = avCols.clone();

		// Show any intermediate temp values here
		System.out.println("Iteration: " + (counter + 1));
		System.out.println(ClusterCenter.dump());
		System.out.println("Percent complete: " +
						   ((((float)counter + 1) / (float)niterations)) * 100.0 + " %");
		counter++;
	}

	/* Adjusting for a future border of width three,
	 * keep count of pixels' values that are identical
	 * to each other */
	for (int i = 2; i < (nrows - 3); i++)
	{
		for (int j = 2; j < (ncols - 3); j++)
		{
			int count_pixel = 0;
			for (int l = (i - 2); l <= (i + 2); l++)
			{
				for (int k = (j - 2); k <= (j + 2); k++)
				{
					double v1 = Label.get(l, k)[0];
					double v2 = Label.get(i, j)[0];
					if (v1 == v2)
					{
						count_pixel++;
					}
				}
			}
			/* if the pixel count is above some arbitrary threshold
			 * for some row copy the other row to here, what is the
			 * theoretical motivation for this? */
			if (count_pixel > 18)
			{
				for (int l = (i - 1); l <= (i + 1); l++)
				{
					for (int k = (j - 1); k <= (j + 1); k++)
					{
						Label.put(l, k, Label.get(i, j)[0]);
					}
				}
			}
		}
	}
	HashMap<String, Mat> stats = new HashMap<String, Mat>(3);
	stats.put(avRowsString, avRows);
	stats.put(avColsString, avCols);
	stats.put(avIntString, avItensity);
	container = new kMeansNGBContainer(Label, stats);
	return container;
}

/**
 * Generate binary image segments from clustered binary image
 * <br/> <b> Prerequisite:</b> kmeans clustering and thresholding
 * to generate global binary image
 * @param I -- clustered image with thresholding applied
 * @return binary image segments and a list of times to generate each
 * segment
 */
static CompositeMat ScanSegments(Mat* I, bool debug)
{
	vector<long> ScanTimes; // = new ArrayList<Long>()
	Mat* Temp = NULL;

	// find how many regions we need to segment?
	int rows = I->rows;
	int cols = I->cols;
	if (debug)
	{
		cout << "ScanSegments(): Rows=" << rows << "Cols=" << cols << endl;
	}

	vector<Mat> Segment; //= new ArrayList<Mat>();

	// Create a matrix with all rows and all columns of a label
	if (debug)
	{
		cout << ("ScanSegments(): Setting up labels") << endl;
	}

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			vector<double> labelData = I->at<vector<double>>(i, j);
			/* A label is a dark pixel starting point from which we
			 * will grow a region */
			if (labelData[0] == 0)
			{
				//I.put(i, j, 1.0);
				I->at<int>(i, j) = 1.0;
			}
		}
	}

	// convert the input image to double precision
	Temp = new Mat();
	I->convertTo(*Temp, I->type());
	// System.out.println("labels="+labels.dump());
	// System.out.println("Temp="+Temp.dump());

	// find first non-zero location
	vector<Point>* points = &(findInMat(*Temp, 1, "first"));

	int n = 1;
	int indx = -1;
	int indy = -1;
	if (points != NULL)
	{
		indx = (int)points->at(0).x;
		indy = (int)points->at(0).y;
	}

	// keep going while we still have regions to process
	if (debug)
	{
		cout << ("ScanSegments():starting to process regions") << endl;
	}
	while (points != NULL)
	{
		// get the next set of nonzero indices that is pixel of region
		int i = indx;
		int j = indy;

		// Start timing code for segment
		long tic = stol(currentTime<nanoseconds>());

		/* pass the image segment to the region growing code along with
		 * the coordinates of the seed and max intensity distance of
		 * 1x10e-5
		 *
		 * This tends to eat the k-means segmented image starting at the
		 * start pixel. When the original segmented image is consumed, then
		 * we are done scanning for segments */
		double max_intensity_distance = 0.00001;
		if (debug)
		{
			cout << ("ScanSegments(): Calling region growing code") << endl;
		}
		vector<Mat> JAndTemp =
			regiongrowing(Temp, i, j, max_intensity_distance, false);
		if (debug)
		{
			cout << ("ScanSegments(): Returned from region growing code") << endl;
		}
		Mat* output_region_image = &(JAndTemp.at(0));
		// System.out.println("output_region_image="+output_region_image.dump());
		*Temp = JAndTemp.at(1);
		// System.out.print("Temp="+Temp.dump());

		/* pad the array and copy the extracted image segment with its
		   grown region into it */
		Mat* padded = new Mat();
		int padding = 3;
		if (Temp != NULL)
		{
			padded->create(output_region_image->rows + 2 * padding, output_region_image->cols + 2 * padding,
						  output_region_image->type());
			padded->setTo(*(new Scalar(0)));
			Rect rect(padding, padding, output_region_image->cols, output_region_image->rows);
			Mat* paddedPortion = new Mat(*padded, rect); //Mat paddedPortion = padded->submat(rect);
			output_region_image->copyTo(*paddedPortion);
			// System.out.println("");
			// System.out.println("paddedPortion="+paddedPortion.dump());

			/* Assign padded array to Segment structure that gets
			   returned to caller */
			// System.out.println("padded="+padded.dump());

			//Segment.add(padded);
			Segment.emplace_back(padded);
			delete padded;
			delete paddedPortion;
		}

		// increment for storing next image segment
		n++;
		if (debug)
		{
			cout << ("ScanSegments(): Preparing for segment " + n) << endl;
		}

		// finish timing work on current segment
		long toc = stol(currentTime<nanoseconds>());
		ScanTimes.emplace_back(toc - tic);
		if (debug)
		{
			cout << ("QUICK TEST MICHAEL REMOVE: Added scan time : " + (toc - tic)) << endl;
		}
		// find next non-zero point to grow
		*points = findInMat(*Temp, 1, "first");
		if ((points != NULL) && (points->size() > 0))
		{
			indx = (int)points->at(0).x;
			indy = (int)points->at(0).y;
		}
		else if (points->size() == 0)
		{
			points = NULL;
		}

		/* Generates huge number of files
		Imgcodecs.imwrite("output/padded"+n+".jpg", padded);
		Imgcodecs.imwrite("output/temp"+n+".jpg", Temp);
		*/
		delete output_region_image;
	}
	if (debug)
	{
		cout << ("ScanSegments(): Done with processing regions") << endl;
	}

	Mat* allScanTimes = new Mat(1, ScanTimes.size(), CV_32FC1);
	for (int i = 0; i < ScanTimes.size(); i++)
	{
		allScanTimes->at<int>(0, i) = ScanTimes.at(i);
	}
	CompositeMat compositeSetMats(Segment, *allScanTimes);
	delete allScanTimes;
	delete Temp;
	delete points;
	return compositeSetMats;
}

/**
 * Region based image segmentation method.  This method performs region
 * growing in an image from a specified seedpoint
 *
 * The region is iteratively grown by comparing all unallocated neigh-
 * boring pixels to the region. The difference between a pixel's intensity
 * value and the region's mean is used as a measure of similarity. The
 * pixel with the smallest difference measured this way is allocated to the
 * respective region. This process continues until the intensity
 * difference between region mean and new pixel become larger than a
 * certain threshold (t)
 *
 * Properties:
 * All pixels must be in a region
 * Pixels must be connected
 * Regions should be disjoint (share border?)
 * Pixels have approximately same grayscale
 * Some predicate determines how two pixels are different (intensity
 * differences, see above)
 *
 * Points to remember:
 * Selecting seed points is important
 * Helps to have connectivity or pixel adjacent information
 * Minimum area threshold (min size of segment) could be tweaked
 * Similarity threshold value -- if diff of set of pixels is less than
 * some value, all part of same region
 *
 * @param I -- input matrix or image
 * @param x -- x coordinate of seedpoint
 * @param y -- y coordinate of seedpoint
 * @param reg_maxdist
 * @return logical output image of region (J in the original matlab code)
 */
static ArrayList<Mat> regiongrowing(Mat I, int x, int y, double reg_maxdist, boolean debug)
{
	// Local neighbor class to aid in region growing
	class Neighbor
	{
	public:
		Point pt;
		vector<double> px;
		Neighbor(Point pt, vector<double> px)
		{
			pt = pt;
			px = px;
		}

	};

	// Sanity check 1
	if (reg_maxdist == 0.0)
	{
		reg_maxdist = 0.2;
	}

	// Sanity check 2
	if (I == nullptr)
	{
		cout << ("regiongrowing(): input matrix is null, bad things will happen now") << endl;
	}

	// Sanity check 2
	/* in the Kroon code, the user will select a non-zero point to use that
	 * then gets rounded this is really hard to do in this code at this
	 * time, will defer implementation
	 *
	 * if(exist('y','var')==0), figure, imshow(I,[]); [y,x]=getpts;
	 * y=round(y(1)); x=round(x(1)); end*/
	/*		if (y == 0) {
				return null;
			}*/
	// System.out.println("I at beginning (region_growing)="+I.dump());
	// Create output image
	Scalar scalar;
	Mat* J = new Mat(I->size(), I->type(), scalar.all(0));     

	// get dimensions of input image
	int rows = I->rows;
	int cols = I->cols;

	// get the mean of the segmented image
	// for get and put use see:
	// http://answers.opencv.org/question/14961/using-get-and-put-to-access-pixel-values-in-java/
	double reg_mean = I->at<int>(x, y, 0);

	// set the number of pixels in the region
	int reg_size = 1;

	// Free memory to store neighbors of the segmented region
	int neg_pos = 0;

	vector<Neighbor>* neg_list = new vector<Neighbor>(neg_free);
	// Neighbor[][] neg_list = new Neighbor[neg_free][neg_free];

	// Distance of the region newest pixel to the region mean
	double pixdist = 0;

	// Neighbor locations (footprint)
	if (debug)
	{
		cout << ("regiongrowing(): Starting neighbor pixel processing") << endl;
	}
	int neigb[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
	while ((pixdist < reg_maxdist) && (reg_size < I->total()))
	{
		for (int j = 0; j < 4; j++)
		{
			// Add new neighbors pixels
			int xn = x + neigb[j][0];
			int yn = y + neigb[j][1];

			// Calculate the neighbor coordinate
			bool ins = (xn >= 0) && (yn >= 0) && (xn < rows) && (yn < cols);

			// Check if neighbor is inside or outside the image
			// only checks one band here, may need to adjust
			double outputPt[1];
			if (ins && (J->at<int>(xn, yn) != NULL))
			{
				outputPt[0] = J->at<int>(xn, yn, 0);
			}
			else
			{
				// System.out.println("J["+xn+","+yn+"]not available");
				continue;
			}
			if (ins && (outputPt[0] == 0))
			{

				// Add neighbor if inside and not already part of the segmented area
				neg_pos++;
				Point p(xn, yn);
				Neighbor* nObj = new Neighbor(p, I->at<vector<double>>(xn, yn));
				neg_list->emplace_back(nObj);
				J->at<int>(xn, yn) = 1.0;
			}
		}

		// Add pixel with intensity nearest to the mean of the region
		double min_dist = DBL_MAX;
		Neighbor* minNeighbor = nullptr;
		Neighbor* curNeighbor = nullptr;
		if (debug)
		{
			cout << ("regiongrowing(): add pixel with intensity nearest mean of region") << endl;
		}
		for (int neg_pos_cnt = 0; neg_pos_cnt < neg_pos; neg_pos_cnt++)
		{
			if (neg_pos_cnt < neg_list->size())
			{
				*curNeighbor = neg_list->at(neg_pos_cnt);
			}
			else
			{
				cerr << "regiongrowing(): neg_list position not available, continuing" << endl;
				continue;
			}
			vector<double> value;
			if (curNeighbor != nullptr)
			{
				value.push_back(curNeighbor->px[0]);
			}
			else
			{
				cerr << "regiongrowing(): cur neighbor was null, setting value to zero" << endl;
				value.push_back(0.0);
			}
			double dist = abs(value[0] - reg_mean);
			if (dist < min_dist)
			{
				min_dist = dist;
				minNeighbor = curNeighbor;
			}
		}
		if (debug)
		{
			cout << "regiongrowing(): done adding pixel with intensity nearest mean of region" << endl;
		}
		J->at<int>(x, y) = 2.0;
		reg_size++;

		// Calculate the new mean of the region
		if (minNeighbor != nullptr)
		{
			// update best min pixel distance
			pixdist = min_dist;

			reg_mean = ((reg_mean * reg_size) +
						minNeighbor->px[0]) /
					   (reg_size + 1);

			/*  Save the x and y coordinates of the pixel
			 *  (for the neighbour add proccess) */
			Point pForUpdate = minNeighbor->pt;
			x = (int)pForUpdate.x;
			y = (int)pForUpdate.y;

			// Remove the pixel from the neighbor (check) list
			auto it = find(neg_list->begin(), neg_list->end(), minNeighbor);
			if (it != neg_list->end()) {
				neg_list->erase(it);
			}
			neg_pos--;
		}
	}
	if (debug)
	{
		cout << ("regiongrowing(): Done with neighbor pixel processing") << endl;
	}

	// Return the segmented area as logical matrix
	// J=J>1;
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			if (J->at<int>(i, j, 0) > 1)
			{
				J->at<int>(i, j) = 1;
			}
			else
			{
				J->at<int>(i, j) = 0;
			}
		}
	}

	// Remove pixels from region image that been processed
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			if (J->at<int>(i, j, 0) == 1)
			{
				I->at<int>(i, j) = 0.0;
			}
		}
	}

	// Package data structures since Java can only return 1 value
	vector<Mat> JAndTmp;
	// Temp = I
	Mat Temp = I->clone();
	// System.out.println("I(region_growing)="+I.dump());
	// System.out.println("J(region_growing)="+J.dump());
	// System.out.println("Temp(region_growing)="+Temp.dump());
	JAndTmp.emplace_back(J->clone());
	JAndTmp.emplace_back(Temp.clone());


	// Free pointer
	delete[] neg_list;
	delete J;

	return JAndTmp;
}

/**
 * The Synthesize method is intended to join together separate segments into
 * a larger subcomponent assembly to provide better matching against obstucted
 * images
 *
 * The database will be updated in the same manner as during LGRunME except that
 * instead of rows containing single segments, they will contain subcomponents
 * (e.g., two or more regions of processed image data as a single unit)
 *
 * @param cm -- the set of matrices used from the initial LGRunME processing
 * @param debug -- write out extra debug data
 * @return an update set of matrices
 */
public:
static CompositeMat Synthesize(CompositeMat cm, boolean debug)
{
	/* Give the database holds many images and views of those images, it is
	 * important to find the starting and end points for the model image that
	 * was just process and calculate the total number of ids to move through*/
	String filename = cm.getFilename();
	long startingID = cm.getStartingId();
	long lastID = cm.getLastId();
	long totalIDs = lastID - startingID + 1;
	long dbTotalIDs = DatabaseModule.cntSegmentsForFile(filename);
	filename = filename.replace('/', ':');
	int dbFirstID = DatabaseModule.getStartId(filename);
	int dbLastID = DatabaseModule.getLastId(filename);
	System.out.println("CM would retrive segments for " + filename +
					   " between IDs " + startingID + " and " + lastID + " with total " + totalIDs);
	System.out.println("Database would retrive segments for " + filename +
					   " between IDs " + dbFirstID + " and " + dbLastID + " with total " + dbTotalIDs);
	String dbFileNameStart = DatabaseModule.getFileName((int)startingID);
	String dbFileNameEnd = DatabaseModule.getFileName((int)lastID);
	Point startingSegmentMoment = DatabaseModule.getMoment((int)startingID);
	TreeMap<Double, Integer> distances =
		new TreeMap<Double, Integer>();
	double newSize = Math.pow(cm.getListofMats().size(), 2.0);
	ArrayList<Mat> cmsToInsert = new ArrayList<Mat>((int)newSize + 1);
	CompositeMat scm = new CompositeMat();

	// Sanity checks
	if ((startingID != dbFirstID) || (lastID != dbLastID))
	{
		System.err.println("ID Mismatch between segments and database");
		System.exit(500);
	}

	if ((dbFileNameStart == null) || (!dbFileNameStart.equalsIgnoreCase(filename)))
	{
		System.err.println("Filename mismatch between starting " + "segments and database");
		System.exit(501);
	}

	if ((dbFileNameStart == null) || (!dbFileNameEnd.equalsIgnoreCase(filename)))
	{
		System.err.println("Filename mismatch between ending " + "segments and database");
		System.exit(502);
	}

	if (dbTotalIDs != totalIDs)
	{
		System.err.println("Mismatch on total number of segments");
		System.exit(503);
	}

	/* Calculate distances from ith segment to all other segments
	   took out lastID for now, just three iterations due to the
	   heavy computational burden this highly unoptimized code
	   is placing on the system */
	for (long i = startingID; i < (startingID + 4); i++)
	{
		long counter = 0;
		long strtSegment = i;

		/* Move through all the other segments relative to the ith
		 * segment */
		long c1 = 0;
		while (counter < totalIDs)
		{

			Point curSegMoment = DatabaseModule.getMoment((int)(strtSegment + counter));
			if (curSegMoment == null)
			{
				System.err.println("null moment encountered");
				counter++;
				continue;
			}
			double distance =
				ProjectUtilities.distance(startingSegmentMoment, curSegMoment);
			System.out.println("Distance from " + strtSegment + " to " + (strtSegment + counter) + " is " + distance);

			/* since distances serve as keys, you may have two or more
			   calculations that come out the same, so this handles the collision */
			boolean distancesHasKey = distances.containsKey(distance);
			if (distancesHasKey)
			{
				System.err.println("There was a previous value associated " + " with the key " + distance + " and value counter=" + (counter - 1));
				System.err.println("Adjusting calculation slightly to include entry");
				Random rnd = new Random();
				distance += (rnd.nextDouble() * 0.001);
				distancesHasKey = distances.containsKey(distance);
				while (distancesHasKey)
				{
					distance += (rnd.nextDouble() * 0.001);
					distancesHasKey = distances.containsKey(distance);
				}
			}
			distances.put(distance, (int)(strtSegment + counter));
			counter++;
			c1++;
		}

		// display sorted distances if debug mode is on
		Set<Double> keys = distances.keySet();
		Iterator<Double> kIt = keys.iterator();
		long c2 = 0;
		while ((debug == true) && (kIt.hasNext()))
		{
			Double key = kIt.next();
			System.out.println(c2 + ".) Sorted distance " + key + " from " + (strtSegment + c2) + " to  " +
							   distances.get(key));
			c2++;
		}

		/* see http://docs.opencv.org/2.4/doc/tutorials/core/adding_images/adding_images.html
		   for reference

		   Base segment is an intermediate segment, just the trivial
		   case */
		counter = 0;
		Mat baseSegment = cm.getListofMats().get((int)counter);
		kIt = keys.iterator();
		long c3 = 0;

		/*Synthesize intermediates in a progressive manner
		 * based on calculated distances from start segment
		 * moment to target segment moment */
		while (kIt.hasNext())
		{
			Double key = kIt.next();
			int relativekey = (int)(distances.get(key) - startingID);
			System.out.println("Merging " + distances.get(key) + " or relative segment " +
							   relativekey);
			Mat mergingSegment =
				cm.getListofMats().get(relativekey);

			/* dst = alpha(src1) + beta(src2) + gamma */
			if (debug == true)
			{
				Imgcodecs.imwrite("output/baseSegment" + filename + "_" + (c3) + ".jpg",
								  baseSegment);
				Imgcodecs.imwrite("output/mergingSegment" + filename + (c3) + ".jpg",
								  mergingSegment);
			}

			Core.addWeighted(baseSegment, 0.5,
							 mergingSegment, 0.5, 0.0, baseSegment);

			/* Due to 50% weighting when merging segments, use a threshold
			 * operator to strength or refresh border pixels */
			Imgproc.threshold(baseSegment, baseSegment,
							  1, 255, Imgproc.THRESH_BINARY);

			/* Add synthesize segment into list of segments */
			cmsToInsert.add(baseSegment.clone());
			if (debug == true)
			{
				Imgcodecs.imwrite("output/mergedSegment_" + strtSegment + filename + "_" + (c3) + ".jpg",
								  baseSegment);
			}

			/* Imgcodecs.imwrite("output/mergedSegment_"+strtSegment+"_"+(distances.get(key))+".jpg",
							   baseSegment); */
			c3++;
		}
		System.out.println("c1=" + c1 + " and c2=" + c2 + " and c3=" + c3);
		scm.addListofMat(cmsToInsert);

		// initialize values for next loop
		cmsToInsert = new ArrayList<Mat>((int)newSize + 1);
		startingSegmentMoment = DatabaseModule.getMoment((int)i + 1);
		distances = new TreeMap<Double, Integer>();
	}
	return scm;
}

public:
static CompositeMat Synthesize_sequential(CompositeMat cm, boolean debug)
{
	/* Give the database holds many images and views of those images, it is
	 * important to find the starting and end points for the model image that
	 * was just process and calculate the total number of ids to move through*/
	String filename = cm.getFilename();
	long startingID = cm.getStartingId();
	long lastID = cm.getLastId();
	long totalIDs = lastID - startingID + 1;
	filename = filename.replace('/', ':');
	int dbFirstID = DatabaseModule.getStartId(filename);
	int dbLastID = DatabaseModule.getLastId(filename);
	long dbTotalIDs = DatabaseModule.cntSegmentsForFile(filename);
	System.out.println("CM would retrive segments for " + filename +
					   " between IDs " + startingID + " and " + lastID + " with total " + totalIDs);
	System.out.println("Database would retrive segments for " + filename +
					   " between IDs " + dbFirstID + " and " + dbLastID + " with total " + dbTotalIDs);
	System.out.println("Composte Matrices Object says there is/are " + cm.getListofMats().size() + " matrice(s) available");
	;
	String dbFileNameStart = DatabaseModule.getFileName((int)startingID);
	String dbFileNameEnd = DatabaseModule.getFileName((int)lastID);
	double newSize = Math.pow(cm.getListofMats().size(), 2.0);
	ArrayList<Mat> cmsToInsert = new ArrayList<Mat>((int)newSize + 1);
	CompositeMat scm = new CompositeMat();
	scm.setFilename(cm.getFilename());

	// Sanity checks
	if ((startingID != dbFirstID) || (lastID != dbLastID))
	{
		System.err.println("ID Mismatch between segments and database");
		System.exit(500);
	}

	if ((dbFileNameStart == null) || (!dbFileNameStart.equalsIgnoreCase(filename)))
	{
		System.err.println("Filename mismatch between starting " + "segments and database");
		DatabaseModule.shutdown();
		System.exit(501);
	}

	if ((dbFileNameStart == null) || (!dbFileNameEnd.equalsIgnoreCase(filename)))
	{
		System.err.println("Filename mismatch between ending " + "segments and database");
		DatabaseModule.shutdown();
		System.exit(502);
	}

	if (dbTotalIDs != totalIDs)
	{
		System.err.println("Mismatch on total number of segments");
		DatabaseModule.shutdown();
		System.exit(503);
	}

	if (cm.getListofMats().size() == 0)
	{
		System.err.println("No matrices to work with");
		DatabaseModule.shutdown();
		System.exit(504);
	}

	/* see http://docs.opencv.org/2.4/doc/tutorials/core/adding_images/adding_images.html
	   for reference

	   Base segment is an intermediate segment, just the trivial
	   case */
	int counter = 0;
	Mat baseSegment = cm.getListofMats().get(counter);

	for (counter = 0; (counter < totalIDs - 1) || (counter < cm.getListofMats().size()); counter++)
	{
		System.out.println("Synthesize_sequential(): counter=" +
						   counter + " or " + ((counter / (float)totalIDs) * 100) + " percent done");
		Mat mergingSegment = cm.getListofMats().get(counter);
		/* dst = alpha(src1) + beta(src2) + gamma */
		if (debug == true)
		{
			Imgcodecs.imwrite("output/baseSegment" + filename.replace(':', '_') + "_" + (counter) + ".jpg",
							  baseSegment);
			Imgcodecs.imwrite("output/mergingSegment" + filename.replace(':', '_') + "_" + (counter) + ".jpg",
							  mergingSegment);
		}

		Core.addWeighted(baseSegment, 0.5,
						 mergingSegment, 0.5, 0.0, baseSegment);

		/* Due to 50% weighting when merging segments, use a threshold
		 * operator to strength or refresh border pixels */
		Imgproc.threshold(baseSegment, baseSegment,
						  1, 255, Imgproc.THRESH_BINARY);

		/* Add synthesize segment into list of segments */
		cmsToInsert.add(baseSegment.clone());
		if (debug == true)
		{
			Imgcodecs.imwrite("output/mergedSegment_" + filename.replace(':', '_') + "_" + (counter) + ".jpg",
							  baseSegment);
		}

		scm.addListofMat(cmsToInsert);

		// initialize values for next loop
		cmsToInsert.clear();
	}

	// return final result
	return scm;
}
};