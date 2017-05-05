#include <v4r/common/img_utils.h>

namespace v4r
{

cv::Mat
cropImage(const cv::Mat &img, const cv::Rect &roi_original, size_t margin, bool make_square)
{
    int width = img.cols;
    int height = img.rows;

    cv::Rect roi (roi_original);

    if (margin)
    {
        roi.x = std::max<int>(0, roi_original.x - margin);
        roi.y = std::max<int>(0, roi_original.y - margin);
        roi.width = std::min<int>(width-1-roi.x, roi_original.width + 2*margin);
        roi.height = std::min<int>(height-1-roi.y, roi_original.height + 2*margin);
    }

    if (make_square)
    {
        if(roi.width > roi.height)
        {
            int extension_half = (roi.width - roi.height) / 2;
            roi.y = std::max(0, roi.y - extension_half);
            roi.height = std::min<int>(height - roi.y, roi.width);

            if(roi.height < roi.width) // in case the roi reaches outside of image, use some pixel from the other side instead to avoid changing aspect ratio
                roi.y = std::max(0, roi.y - (roi.width - roi.height));
        }
        else
        {
            int extension_half = (roi.height - roi.width) / 2;
            roi.x = std::max(0, roi.x - extension_half);
            roi.width = std::min<int>(width - roi.x, roi.height);

            if(roi.width < roi.height) // in case the roi reaches outside of image, use some pixel from the other side instead to avoid changing aspect ratio
                roi.x = std::max(0, roi.x - (roi.height - roi.width));
        }
    }

    return cv::Mat(img(roi));
}

}
