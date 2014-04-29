
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
    std::vector<Eigen::Matrix4f> process(boost::tuples::tuple<
                                         std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>,
                                         std::vector<pcl::PointCloud<pcl::Normal>::Ptr>,
                                         std::vector<std::vector<float> > > pointClouds);
    GlobalRegistration();
    virtual void applyConfig(Config &config);

    std::string getName()
    {
        return "Global registration";
    }
};

}
}
