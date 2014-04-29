
#include "ioModule.h"

#include <vector>
#include <string>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace object_modeller
{
namespace util
{

template<class TPointType>
class Mask :
        public InOutModule<std::vector<typename pcl::PointCloud<TPointType>::Ptr>,
                           boost::tuples::tuple<
                                std::vector<typename pcl::PointCloud<TPointType>::Ptr>,
                                std::vector<std::vector<int> > > >
{
private:

public:
    std::vector<typename pcl::PointCloud<TPointType>::Ptr> process(boost::tuples::tuple<
                 std::vector<typename pcl::PointCloud<TPointType>::Ptr>,
                 std::vector<std::vector<int> > > input);

    virtual void applyConfig(Config &config);

    std::string getName()
    {
        return "Mask point clouds";
    }
};

}
}
