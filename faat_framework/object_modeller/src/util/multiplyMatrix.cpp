
#include "util/multiplyMatrix.h"

#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>

namespace object_modeller
{
namespace util
{

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

std::vector<Eigen::Matrix4f> MultiplyMatrixSingle::process(boost::tuples::tuple<std::vector<Eigen::Matrix4f>, Eigen::Matrix4f> in)
{
    std::vector<Eigen::Matrix4f> input = in.get<0>();
    Eigen::Matrix4f param = in.get<1>();

    std::vector<Eigen::Matrix4f> result;

    std::cout << "process multiply single" << std::endl;

    std::cout << "multiply matrix for seq " << activeSequence << " with " << std::endl;
    std::cout << param << std::endl;

    for (size_t i = 0; i < input.size (); i++)
    {
        std::cout << "multiply matrix single before: " << activeSequence << "/" <<  i << ": " << std::endl;
        std::cout << input[i] << std::endl;

        //std::cout << "process " << i << std::endl;
        result.push_back(param * input[i]);

        std::cout << "multiply matrix single after: " << activeSequence << "/" <<  i << ": " << std::endl;
        std::cout << result[i] << std::endl;
    }

    //std::cout << "process complete " << std::endl;

    return result;
}

}
}
