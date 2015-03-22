#include "pre_segmenter.h"
#include "opencv2/opencv.hpp"

namespace v4rOCTopDownSegmenter
{
    //follows the presegmentation strategy from Ückermann
    template<typename PointT>
    class EdgeBasedPreSegmenter : public PreSegmenter<PointT>
    {
        private:
            pcl::PointCloud<pcl::Normal>::Ptr surface_normals_;
            using PreSegmenter<PointT>::cloud_;

            void edgesFromCloudUckermann(cv::Mat & labels);

            int connectedComponents(cv::Mat & initial_labels, int good_label, cv::Mat & label_image, int min_area = -1);

            void visualizeCCCloud(cv::Mat & connected, int num_labels);

            void dilateRegionsIteratively(cv::Mat & label_image);

            std::vector<uint32_t> label_colors_;

        public:
            EdgeBasedPreSegmenter();

            //compute a pre-segmentation of the scene based on depth discontinuity edges and surface normal edges
            void process();

            void setSurfaceNormals(pcl::PointCloud<pcl::Normal>::Ptr & normals)
            {
                surface_normals_ = normals;
            }
    };
}
