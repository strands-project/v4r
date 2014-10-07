
#include "ioModule.h"

#include <vector>
#include <string>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace object_modeller
{
namespace util
{

class MultiplyMatrix :
        public InOutModule<std::vector<Eigen::Matrix4f>,
                           boost::tuples::tuple<std::vector<Eigen::Matrix4f>, std::vector<Eigen::Matrix4f> > >
{
public:
    MultiplyMatrix(std::string config_name="multiplyMatrix") : InOutModule(config_name)
    {}

    std::vector<Eigen::Matrix4f> process(boost::tuples::tuple<std::vector<Eigen::Matrix4f>, std::vector<Eigen::Matrix4f> > input);

    std::string getName()
    {
        return "Multiply matrix";
    }
};

class MultiplyMatrixSingle :
        public InOutModule<std::vector<Eigen::Matrix4f>,
                           boost::tuples::tuple<std::vector<Eigen::Matrix4f>, Eigen::Matrix4f> >
{
public:
    MultiplyMatrixSingle(std::string config_name="multiplyMatrixSingle") : InOutModule(config_name)
    {}

    std::vector<Eigen::Matrix4f> process(boost::tuples::tuple<std::vector<Eigen::Matrix4f>, Eigen::Matrix4f> input);

    std::string getName()
    {
        return "Multiply matrix Single";
    }
};

}
}
