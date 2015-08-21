#ifndef V4R_PCL_SEGMENTATION_METHODS__
#define V4R_PCL_SEGMENTATION_METHODS__

#include <pcl/common/common.h>
#include <v4r/core/macros.h>

namespace v4r
{

template <typename PointT> class
V4R_EXPORTS PCLSegmenter
{
public:
    class Parameter
    {
    public:
        int seg_type_, min_cluster_size_, max_vertical_plane_size_, num_plane_inliers_;
        double max_angle_plane_to_ground_,
               sensor_noise_max_,
               table_range_min_,
               table_range_max_,
               chop_at_z_,
               angular_threshold_deg_;
        bool use_transform_to_world_;
        Parameter (int seg_type = 0,
                   int min_cluster_size=500,
                   int max_vertical_plane_size=5000,
                   int num_plane_inliers=1000,
                   double max_angle_plane_to_ground = 15.f,
                   double sensor_noise_max = 0.01f,
                   double table_range_min = 0.6f,
                   double table_range_max = 1.2f,
                   double chop_at_z = std::numeric_limits<double>::max(),
                   double angular_threshold_deg = 10.f,
                   bool use_transform_to_world=false)
            :
              seg_type_ (seg_type),
              min_cluster_size_ (min_cluster_size),
              max_vertical_plane_size_ (max_vertical_plane_size),
              num_plane_inliers_ (num_plane_inliers),
              max_angle_plane_to_ground_ (max_angle_plane_to_ground),
              sensor_noise_max_ (sensor_noise_max),
              table_range_min_ (table_range_min),
              table_range_max_ (table_range_max),
              chop_at_z_ (chop_at_z),
              angular_threshold_deg_ (angular_threshold_deg),
              use_transform_to_world_ (use_transform_to_world)
        {

        }
    };

protected:
    typename pcl::PointCloud<PointT>::Ptr input_cloud_;
    pcl::PointCloud<pcl::Normal>::Ptr input_normal_cloud_;
    Eigen::Vector4f extracted_table_plane_;
    Parameter param_;
    Eigen::Matrix4f transform_to_world_;

public:
    PCLSegmenter(const Parameter &p = Parameter())
    {
        param_ = p;
        input_cloud_.reset( new pcl::PointCloud<PointT>() );
        input_normal_cloud_.reset( new pcl::PointCloud<pcl::Normal>() );
    }

    void do_segmentation(std::vector<pcl::PointIndices> & indices);

    void set_input_cloud(const pcl::PointCloud<PointT> &cloud)
    {
        *input_cloud_ = cloud;
    }

    void set_input_normal_cloud(const pcl::PointCloud<pcl::Normal> &normals)
    {
        *input_normal_cloud_ = normals;
    }

    void set_transform2world(const Eigen::Matrix4f &tf)
    {
        transform_to_world_ = tf;
    }

    Eigen::Vector4f get_table_plane() const
    {
        return extracted_table_plane_;
    }

    void printParams(std::ostream &ostr = std::cout) const;
};

}

#endif
