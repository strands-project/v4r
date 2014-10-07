
#include "ioModule.h"

#include <vector>
#include <string>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace object_modeller
{
namespace modelling
{

class NmBasedCloudIntegration :
        public InOutModule<std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr>,
                           boost::tuples::tuple<
                                std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>,
                                std::vector<Eigen::Matrix4f>,
                                std::vector<std::vector<int> >,
                                std::vector<pcl::PointCloud<pcl::Normal>::Ptr>,
                                std::vector<std::vector<float> > > >
{
protected:
    float resolution;
    bool organized_normals;
    int min_points_per_voxel;
    float final_resolution;
    bool depth_edges;
    float max_angle;
    float lateral_sigma;
    float w_t;

public:
    std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> process(boost::tuples::tuple<
                 std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>,
                 std::vector<Eigen::Matrix4f>,
                 std::vector<std::vector<int> >,
                std::vector<pcl::PointCloud<pcl::Normal>::Ptr>,
                std::vector<std::vector<float> > > input);

    std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> processImpl(std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> clouds,
                                                                      std::vector<Eigen::Matrix4f> poses,
                                                                      std::vector<std::vector<int> > indices,
                                                                     std::vector<pcl::PointCloud<pcl::Normal>::Ptr> normals,
                                                                     std::vector<std::vector<float> > weights);

    std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> process(
                 std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> & pointClouds,
                 std::vector<Eigen::Matrix4f> & transformations,
                 std::vector<std::vector<int> > & indices,
                std::vector<pcl::PointCloud<pcl::Normal>::Ptr> & normals,
                std::vector<std::vector<float> > & weights);

    NmBasedCloudIntegration(std::string config_name="nmBasedCloudIntegration");

    std::string getName()
    {
        return "NM Based Cloud Integration";
    }
};


class NmBasedCloudIntegrationMultiSeq :
        public NmBasedCloudIntegration
{
private:
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> pointClouds;
    std::vector<Eigen::Matrix4f> transformations;
    std::vector<std::vector<int> > indices;
    std::vector<pcl::PointCloud<pcl::Normal>::Ptr> normals;
    std::vector<std::vector<float> > weights;

public:
    NmBasedCloudIntegrationMultiSeq(std::string config_name="nmBasedCloudIntegrationMultiSeq") : NmBasedCloudIntegration(config_name) {}

    std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> process(boost::tuples::tuple<
                 std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>,
                 std::vector<Eigen::Matrix4f>,
                 std::vector<std::vector<int> >,
                std::vector<pcl::PointCloud<pcl::Normal>::Ptr>,
                std::vector<std::vector<float> > > input);

    virtual int getNrOutputSequences()
    {
        return 1;
    }
};

}
}
