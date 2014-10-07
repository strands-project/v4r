
#include "ioModule.h"

#include <vector>
#include <string>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace object_modeller
{
namespace registration
{

class GlobalRegistration :
        public InOutModule<std::vector<Eigen::Matrix4f>,
                            boost::tuples::tuple<
                                                 std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>,
                                                 std::vector<pcl::PointCloud<pcl::Normal>::Ptr>,
                                                 std::vector<std::vector<float> > > >
{
private:
    float views_overlap_;
    bool fast_overlap;
    int mv_iterations;
    float min_dot;
    bool mv_use_weights_;
    float max_angle;
    float lateral_sigma;
    bool organized_normals;
    float w_t;
    float canny_low;
    float canny_high;
    bool sobel;
    bool depth_edges;

public:
    GlobalRegistration(std::string config_name="globalRegistration");

    std::vector<Eigen::Matrix4f> process(boost::tuples::tuple<
                                         std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>,
                                         std::vector<pcl::PointCloud<pcl::Normal>::Ptr>,
                                         std::vector<std::vector<float> > > pointClouds);


    std::vector<Eigen::Matrix4f> process(std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> & pointClouds,
                                         std::vector<pcl::PointCloud<pcl::Normal>::Ptr> & normals,
                                         std::vector<std::vector<float> > & weights);


    std::string getName()
    {
        return "Global registration";
    }

    void setMinOverlap(float f)
    {
        views_overlap_ = f;
    }

    void setMinDot(float f)
    {
        min_dot = f;
    }
};

}
}
