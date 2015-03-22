#include "edge_based_presegmenter.h"
#include "v4r/OREdgeDetector/organized_edge_detection.h"
#include <pcl/visualization/pcl_visualizer.h>
#include <opencv2/opencv.hpp>
#include <pcl/io/pcd_io.h>

#define OCTDS_NAN_VALUES 0
#define OCTDS_EDGE_REGIONS 1
#define OCTDS_OTHER_REGIONS 2

template<typename PointT>
v4rOCTopDownSegmenter::EdgeBasedPreSegmenter<PointT>::EdgeBasedPreSegmenter()
{
    label_colors_.clear();
}

//labels are 0 for NaN values, 1 for edges and 2 for the rest

template<typename PointT>
void v4rOCTopDownSegmenter::EdgeBasedPreSegmenter<PointT>::edgesFromCloudUckermann(cv::Mat & labels)
{

    cv::Mat dot_p(cloud_->height, cloud_->width, CV_32F);

    int kernel_width = 5;
    float min_val = 0.f;

    for(int r=0; r < (int)cloud_->height; r++)
    {
        for(int c=0; c < (int)cloud_->width; c++)
        {

            Eigen::Vector3f n_rc = surface_normals_->at(c, r).getNormalVector3fMap();
            if(!pcl_isfinite(n_rc[0]))
            {
                dot_p.at<float>(r,c) = min_val;
                continue;
            }

            float avg_dot = 0.f;
            int num_valid = 0;
            for(int u=std::max(0, r - kernel_width); u <= std::min((int)(cloud_->height - 1), r + kernel_width); u++)
            {
                for(int v=std::max(0, c - kernel_width); v <= std::min((int)(cloud_->width - 1), c + kernel_width); v++)
                {
                    if(u == r && v == c)
                        continue;

                    int u_diff = std::abs(r - u);
                    int v_diff = std::abs(c - v);
                    if( !( (u_diff == v_diff) || (u_diff == 0) || (v_diff == 0) ) )
                        continue;

                    Eigen::Vector3f n_uv = surface_normals_->at(v, u).getNormalVector3fMap();

                    if(!pcl_isfinite(n_uv[0]))
                       continue;

                    float dot = n_rc.dot(n_uv);
                    avg_dot += dot;
                    num_valid++;
                }
            }


            if(num_valid != 0)
            {
                dot_p.at<float>(r,c) = std::max(min_val, avg_dot / num_valid);
                //if( (avg_dot / num_valid) < 0.9f )
                //std::cout << avg_dot / num_valid << " " << num_valid << std::endl;
            }
            else
            {
                dot_p.at<float>(r,c) = 0.5f;
            }
        }
    }

    labels = cv::Mat(cloud_->height, cloud_->width, CV_32FC1);

    for(int r=0; r < (int)cloud_->height; r++)
    {
        for(int c=0; c < (int)cloud_->width; c++)
        {
            if(dot_p.at<float>(r,c) <= 0.95f)
            {
                if(pcl_isfinite(cloud_->at(c, r).z))
                {
                    labels.at<float>(r,c) = 1;
                }
                else
                {
                    labels.at<float>(r,c) = 0;
                }
            }
            else
            {
                labels.at<float>(r,c) = 2;
            }
        }
    }

    cv::imshow("labels", labels);
    cv::waitKey(0);
}

template<typename PointT>
int v4rOCTopDownSegmenter::EdgeBasedPreSegmenter<PointT>::connectedComponents
    (cv::Mat & initial_labels,
     int good_label,
     cv::Mat & label_image,
     int min_area)
{

    label_image = initial_labels.clone();

    int label_count = OCTDS_OTHER_REGIONS; // starts at 2 because 0 (NaNs), 1 (edges) are used already

    for(int y=0; y < initial_labels.rows; y++) {
        for(int x=0; x < initial_labels.cols; x++) {
            if( (int)(label_image.at<float>(y,x)) != good_label)
                continue;

            cv::Rect rect;
            int area = cv::floodFill(label_image, cv::Point(x,y), cv::Scalar(label_count), &rect, cv::Scalar(0), cv::Scalar(0), 4);

            if(area < min_area)
            {
                for(int i=rect.y; i < (rect.y+rect.height); i++) {
                   for(int j=rect.x; j < (rect.x+rect.width); j++) {
                       if((int)label_image.at<float>(i,j) != label_count) {
                           continue;
                       }

                       label_image.at<float>(i,j) = (float)OCTDS_EDGE_REGIONS;
                   }
               }
            }

            label_count++;
        }
    }

    return label_count;

}

template<typename PointT>
void v4rOCTopDownSegmenter::EdgeBasedPreSegmenter<PointT>::visualizeCCCloud(cv::Mat & connected, int num_labels)
////////////////////////////////////////////////////////////
//visualize connected components
{
    if(label_colors_.size() != num_labels)
    {
        int max_label = num_labels;
        label_colors_.reserve (max_label + 1);
        srand (static_cast<unsigned int> (time (0)));
        while (label_colors_.size () <= max_label )
        {
            uint8_t r = static_cast<uint8_t>( (rand () % 256));
            uint8_t g = static_cast<uint8_t>( (rand () % 256));
            uint8_t b = static_cast<uint8_t>( (rand () % 256));
            label_colors_.push_back (static_cast<uint32_t>(r) << 16 | static_cast<uint32_t>(g) << 8 | static_cast<uint32_t>(b));
        }

        label_colors_[1] = (static_cast<uint32_t>(125) << 16 | static_cast<uint32_t>(125) << 8 | static_cast<uint32_t>(125));
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cc(new pcl::PointCloud<pcl::PointXYZRGB>(*cloud_));

    for(int r=0; r < (int)cloud_->height; r++)
    {
        for(int c=0; c < (int)cloud_->width; c++)
        {
            cloud_cc->at(c, r).rgb = label_colors_[(int)connected.at<float>(r,c)];
        }
    }

    pcl::visualization::PCLVisualizer vis("CC");

    {
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler(cloud_cc);
        vis.addPointCloud(cloud_cc, handler, "cloud");
    }

    vis.spin();
}

template<typename PointT>
void v4rOCTopDownSegmenter::EdgeBasedPreSegmenter<PointT>::dilateRegionsIteratively
        (cv::Mat & label_image)
{
    //assuming edge regions are much smaller than other regions, this is equivalent (and more efficient)
    //to eroding edge_regions

    int kw = 1;
    bool converged = false;
    while(!converged)
    {
        cv::Mat label_image_iteration = label_image.clone();

        int n_changes = 0;

        for(int r=0; r < label_image.rows; r++)
        {
            for(int c=0; c < label_image.cols; c++)
            {
                Eigen::Vector3f p = cloud_->at(c,r).getVector3fMap();

                if((int)label_image_iteration.at<float>(r,c) == OCTDS_EDGE_REGIONS)
                {
                    //erode
                    int max_label = -1;

                    for(int u=std::max(0, r - kw); u <= std::min((int)(cloud_->height - 1), r + kw); u++)
                    {
                        for(int v=std::max(0, c - kw); v <= std::min((int)(cloud_->width - 1), c + kw); v++)
                        {

                            Eigen::Vector3f puv = cloud_->at(v,u).getVector3fMap();
                            if( (puv - p).norm() > 0.01f)
                                continue;

                            if( (int)label_image_iteration.at<float>(u,v) > max_label)
                            {
                                max_label = (int)label_image_iteration.at<float>(u,v);
                            }
                        }
                    }

                    if( (int)label_image.at<float>(r,c) != max_label )
                    {
                        label_image.at<float>(r,c) = (float)max_label;
                        n_changes++;
                    }
                }
            }
        }

        if(n_changes == 0)
            converged = true;
    }
}

template<typename PointT>
void v4rOCTopDownSegmenter::EdgeBasedPreSegmenter<PointT>::process()
{
    //1. compute edges
    cv::Mat labels;
    edgesFromCloudUckermann(labels);

    //2. compute connected components
        //remove small regions (DONE)
    cv::Mat connected;
    int num_labels = connectedComponents(labels, OCTDS_OTHER_REGIONS, connected, 100);

    visualizeCCCloud(connected, num_labels);

    //3. label rest of edge regions (label 1) by dilating valid regions iteratively until convergence
        //i.e. all valid pixels have been labeled
        //use max_depth_jump when dilating (DONE)
        //use color edges to avoid propagating labels across colors (TODO?)
    dilateRegionsIteratively(connected);

    visualizeCCCloud(connected, num_labels);
}

template class v4rOCTopDownSegmenter::EdgeBasedPreSegmenter<pcl::PointXYZRGB>;
//template class v4rOCTopDownSegmenter::EdgeBasedPreSegmenter<pcl::PointXYZRGBA>;
