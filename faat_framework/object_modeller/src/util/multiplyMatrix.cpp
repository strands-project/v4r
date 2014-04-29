
#include "util/multiplyMatrix.h"

#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>

namespace object_modeller
{
namespace util
{

void MultiplyMatrix::applyConfig(Config &config)
{

}

std::vector<Eigen::Matrix4f> MultiplyMatrix::process(boost::tuples::tuple<std::vector<Eigen::Matrix4f>, std::vector<Eigen::Matrix4f> > in)
{
    std::vector<Eigen::Matrix4f> input = in.get<0>();
    std::vector<Eigen::Matrix4f> param = in.get<1>();

    std::vector<Eigen::Matrix4f> result;

    std::cout << "process multiply" << std::endl;
    std::cout << "input size " << input.size() << std::endl;
    std::cout << "param size " << param.size() << std::endl;
    result.clear();

    for (size_t i = 0; i < input.size (); i++)
    {
        std::cout << "process " << i << std::endl;
        result.push_back(input[i] * param[i]);
    }

    std::cout << "process complete " << std::endl;

    return result;
}

}
}
