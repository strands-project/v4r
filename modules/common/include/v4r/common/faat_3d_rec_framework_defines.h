#ifndef FAAT_3D_REC_FRAMEWORK_DEFINES_H
#define FAAT_3D_REC_FRAMEWORK_DEFINES_H

#include <pcl/common/common.h>

struct IndexPoint
{
  int idx;
};

//This stuff is needed to be able to make the SHOT histograms persistent
POINT_CLOUD_REGISTER_POINT_STRUCT (pcl::Histogram<352>,
    (float[352], histogram, histogram352)
)

POINT_CLOUD_REGISTER_POINT_STRUCT (pcl::Histogram<1344>,
    (float[1344], histogram, histogram1344)
)

POINT_CLOUD_REGISTER_POINT_STRUCT (IndexPoint,
    (int, idx, idx)
)

POINT_CLOUD_REGISTER_POINT_STRUCT (pcl::Histogram<128>,
    (float[128], histogram, histogramSIFT)
)

#endif
