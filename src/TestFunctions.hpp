#include "includes.h"

static kMeansNGBContainer opencv_kmeans_postProcess(Mat* data, Mat* labels, Mat* centers) {
    /*if (data->channels() == 3)
    {
        data->reshape(3);
    }*/

    /* Setup data structure holding partitioned image data */
    Mat* clustered_data = new Mat(data->rows, data->cols,
                                data->type(), Scalar(0));
    unordered_map<string, Mat*> stats;

    centers->convertTo(*centers, CV_8U); 
    Mat res(labels->rows, labels->cols, CV_8U);

    /* Map each label to a cluster center */
    for (int i = 0; i < res.rows;  i++) {
            int label = labels->at<int>(i,0);
            res.at<uint8_t>(i,0) = centers->at<uint8_t>(label,0);
    }
    
    /* Turn partitioned data back into a format suitable as an 
       image*/
    Mat res2 = res.clone().reshape(1,data->rows);
    kMeansNGBContainer res3(res2);

    /* return partitioned image */
    return res3;

    /* Keep stats on partitioning process */
    // Map<Integer, Integer> counts = new HashMap<Integer, Integer>();
    //unordered_map<int, int> counts;
    //for (int i = 0; i < centers->rows; i++)
    //    counts[i] = 0;

    /* Run image against centroids and assign pixels to clusters */
    /*int data_height = data->rows;
    int data_width = data->cols;

    // MinMaxLocResult mmlr = minMaxLoc(labels);
    struct {
        double minVal, maxVal;
        Point minLoc, maxLoc;
    } typedef MinMaxLocResult;
    MinMaxLocResult mmlr;
    cv::minMaxLoc(*labels, &mmlr.minVal, &mmlr.maxVal, &mmlr.minLoc, &mmlr.maxLoc);

    if (data->channels() == 3)
    {*/
        /* For each pixel in the image */
        /*for (int y = 0; y < data_height; y++)
        {
            for (int x = 0; x < data_width; x++)
            {*/
                /* Get the cluster the pixel is assigned to
                * label is in 1D format*/
               // int labelb = (int)labels->at<Vec3b>(y, x)[0];
               // int labelg = (int)labels->at<Vec3b>(y, x)[1];
                //int labelr = (int)labels->at<Vec3b>(y, x)[2];

                /* Update stats that this pixel will get assigned to
                * label specified cluster */
                //counts.at(labelb) = counts.at(labelb) + 1;

                /* Copy pixel into cluster data structure with the color
                * of the specified centroid */
                /*
                clustered_data.put(y, x,
                                ((labelb + mmlr.minVal) / mmlr.maxVal) * 255,
                                ((labelg + mmlr.minVal) / mmlr.maxVal) * 255,
                                ((labelr + mmlr.minVal) / mmlr.maxVal) * 255);
                */
                /*clustered_data->at<Vec3b>(y, x)[0] = ((labelb + mmlr.minVal) / mmlr.maxVal) * 255;
                clustered_data->at<Vec3b>(y, x)[1] = ((labelg + mmlr.minVal) / mmlr.maxVal) * 255;
                clustered_data->at<Vec3b>(y, x)[2] = ((labelr + mmlr.minVal) / mmlr.maxVal) * 255;
                
            }
        }
    }
    else if (data->channels() == 1)
    {*/




        //centers->convertTo(*centers, CV_8U); 
        //Mat res(labels->rows, labels->cols, CV_8U);

        /* Map each label to a cluster center */
        /*for(int i = 0; i < res.rows;  i++) {
            //int label = labels->at<int>(i,0);
            //res.at<uint8_t>(i,0) = centers->at<uint8_t>(label,0);
            int label = (int)labels->at<int>(i, 0);
            res.at<uint8_t>(i,0) = centers->at<uint8_t>(label,0);
            clustered_data->at<int>(i, 0) = ((res.at<uint8_t>(i, 0) + mmlr.minVal) / mmlr.maxVal) * 255;
        }*/

        /* Turn partitioned data back into a format suitable as an 
        image*/
        //Mat res2 = res.clone().reshape(1,data->rows);
        //*clustered_data = clustered_data->clone().reshape(1,data->rows);

        /* return partitioned image */
        //return res2; // <- Now we're returning/using clustered_data






        ///* For each pixel in the image */
        //for (int y = 0; y < data_height; y++)
        //{
        //    for (int x = 0; x < data_width; x++)
        //    {
        //        /* Get the cluster the pixel is assigned to
        //        * label is in 1D format*/
        //       int label = (int)labels->at<int>(y, x, 0);
        //
        //        /* Update stats that this pixel will get assigned to
        //        * label specified cluster */
        //        counts[label] = counts[label] + 1;
        //
        //        /* Copy pixel into cluster data structure with the color
        //        * of the specified centroid */
        //        //clustered_data.put(y, x, ((label + mmlr.minVal) / mmlr.maxVal) * 255);
        //        clustered_data->at<int>(y, x) = ((label + mmlr.minVal) / mmlr.maxVal) * 255;
        //    }
        //}
    //}

    // Print counts map and get its keySet
    /*set<int> region_cnts;
    for (const auto& pair : counts) {
        cout << pair.first << " : " << pair.second << endl;
    };
    std::transform(counts.begin(), counts.end(), inserter(region_cnts, region_cnts.begin()), [](const auto& pair) {
        return pair.first;
    });

    for (int rc : region_cnts)
    {
        Mat m(1, 1, CV_32FC1);
        int cnt = counts[rc];
        m.at<int>(0, 0) = cnt;
        stats.at(to_string(rc)) = &m;
    }*/
    /*cout << matToString(*clustered_data) << endl;
    kMeansNGBContainer kmNGBCnt(*clustered_data, stats); 
    clustered_data->release();

    return kmNGBCnt;*/
};
