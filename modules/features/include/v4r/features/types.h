#ifndef V4R_FEATURE_TYPES__
#define V4R_FEATURE_TYPES__

namespace v4r
{
        enum FeatureType
        {
            SIFT_GPU = 0x01, // 00000001
            SIFT_OPENCV = 0x02, // 00000010
            SHOT  = 0x04, // 00000100
            OURCVFH  = 0x08,  // 00001000
            FPFH = 0x10,  // 00010000
            ESF = 0x20  // 00100000
        };
}

#endif
