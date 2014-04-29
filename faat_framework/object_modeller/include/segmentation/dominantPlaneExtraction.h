
#include "ioModule.h"

#include <vector>
#include <string>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace object_modeller
{
namespace segmentation
{

class DominantPlaneExtraction :
        public InOutModule<std::vector<std::vector<int> >,
                           std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> >
{
private:
    int num_plane_inliers;
    int seg_type;
    float plane_threshold;

public:
    DominantPlaneExtraction();
    std::vector<std::vector<int> > process(std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> input);

    virtual void applyConfig(Config &config);

    std::string getName()
    {
        return "Dominant Plane Extraction";
    }
};

}
}
