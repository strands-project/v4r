
#include "ioModule.h"

#include <vector>
#include <string>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace object_modeller
{
namespace util
{

template<class TType>
class VectorMask :
        public InOutModule<std::vector<std::vector<TType> >,
                           boost::tuples::tuple<
                                std::vector<std::vector<TType> >,
                                std::vector<std::vector<int> > > >
{
private:

public:
    VectorMask(std::string config_name="vectorMask") : InOutModule<std::vector<std::vector<TType> >,
                                                                    boost::tuples::tuple<
                                                                         std::vector<std::vector<TType> >,
                                                                         std::vector<std::vector<int> > > >(config_name)
    {}

    std::vector<std::vector<TType> > process(boost::tuples::tuple<
                 std::vector<std::vector<TType> >,
                 std::vector<std::vector<int> > > input);

    std::string getName()
    {
        return "Mask vector";
    }
};

}
}
