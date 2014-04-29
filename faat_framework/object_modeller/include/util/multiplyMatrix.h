
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
private:
    std::vector<Eigen::Matrix4f> input;
    std::vector<Eigen::Matrix4f> result;
    std::vector<Eigen::Matrix4f> param;

public:
    virtual void applyConfig(Config &config);

    std::vector<Eigen::Matrix4f> process(boost::tuples::tuple<std::vector<Eigen::Matrix4f>, std::vector<Eigen::Matrix4f> > input);

    std::string getName()
    {
        return "Multiply matrix";
    }
};

}
}
