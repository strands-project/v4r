#include <pcl/common/common.h>
#ifndef OCTOPDPS_PRESEGMENTER_H
#define OCTOPDPS_PRESEGMENTER_H

namespace v4rOCTopDownSegmenter
{
    template<typename PointT>
    class PreSegmenter
    {

        protected:
            typename pcl::PointCloud<PointT>::Ptr cloud_;
            std::vector<std::vector<int> > segment_indices_;

        public:
            virtual void process() = 0;
            void setInputCloud(typename pcl::PointCloud<PointT>::Ptr & cloud)
            {
                cloud_ = cloud;
            }

            std::vector<std::vector<int> > getSegmentIndices()
            {
                return segment_indices_;
            }
    };
}

#endif
