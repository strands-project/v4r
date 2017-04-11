#pragma once

#include <pcl/common/common.h>

struct IndexPoint
{
  int idx;
};

POINT_CLOUD_REGISTER_POINT_STRUCT (pcl::Histogram<1344>,
    (float[1344], histogram, histogram1344)
)

POINT_CLOUD_REGISTER_POINT_STRUCT (IndexPoint,
    (int, idx, idx)
)

