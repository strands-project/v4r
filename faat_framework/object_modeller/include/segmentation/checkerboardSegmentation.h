
#include "ioModule.h"

#include <vector>
#include <string>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

//#include <cv.h>
#include <opencv2/opencv.hpp>

namespace object_modeller
{
namespace segmentation
{

class CheckerboardSegmentation :
        public InOutModule<std::vector<std::vector<int> >,
                           std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> >
{
private:
    std::vector<cv::Size> boardSizes;

public:
    CheckerboardSegmentation(std::vector<cv::Size> boardSizes, std::string config_name="checkerboardSegmentation");

    std::vector<std::vector<int> > process(std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> input);

    std::string getName()
    {
        return "Checkerboard Segmentation";
    }
};

}
}
