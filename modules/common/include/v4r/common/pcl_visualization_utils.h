#ifndef PCL_VISUALIZATION_UTILS_H
#define PCL_VISUALIZATION_UTILS_H

#include <v4r/core/macros.h>
#include <pcl/visualization/cloud_viewer.h>
#include <string>

namespace v4r
{
      class V4R_EXPORTS pcl_visualizer
      {
      public:
          static std::vector<int> visualization_framework (pcl::visualization::PCLVisualizer &vis,
                                                size_t number_of_views,
                                                size_t number_of_subwindows_per_view,
                                                const std::vector<std::string> &title_subwindows = std::vector<std::string>());
      };
}
#endif // PCL_VISUALIZATION_UTILS_H

