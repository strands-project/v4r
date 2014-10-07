
#include "util/vectorMask.h"

#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>

namespace object_modeller
{
namespace util
{

template <class TType>
std::vector<std::vector<TType> > VectorMask<TType>::process(boost::tuples::tuple<
             std::vector<std::vector<TType> >,
             std::vector<std::vector<int> > > input)
{
    std::vector<std::vector<TType> > vec = boost::tuples::get<0>(input);
    std::vector<std::vector<int> > indices = boost::tuples::get<1>(input);

    std::vector<std::vector<TType> > result;

    for (size_t i = 0; i < vec.size (); i++)
    {
        std::vector<TType> filtered;

        for (int j=0;j<indices[i].size();j++)
        {
            filtered.push_back(vec[i][indices[i][j]]);
        }

        result.push_back(filtered);
    }

    return result;
}

template class VectorMask<float>;

}
}
