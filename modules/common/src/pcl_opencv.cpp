#include <glog/logging.h>
#include <omp.h>
#include <pcl/impl/instantiate.hpp>

#include <v4r/common/miscellaneous.h>
#include <v4r/common/pcl_opencv.h>
#include <v4r/common/zbuffering.h>

namespace v4r
{
template<class PointT>
cv::Rect
PCLOpenCVConverter<PointT>::computeROIfromIndices()
{
    CHECK (!indices_.empty() && cloud_->isOrganized());

    int width = cloud_->width, height = cloud_->height;

    int min_u = width-1, min_v = height-1, max_u = 0, max_v = 0;

    for (size_t i = 0; i < indices_.size(); i++)
    {
        const int &idx = indices_[i];

        int col = idx % width;
        int row = idx / width;

        if( row < min_v ) min_v = row;
        if( row > max_v ) max_v = row;
        if( col < min_u ) min_u = col;
        if( col > max_u ) max_u = col;
    }

    min_u = std::max<int>(0, min_u);
    min_v = std::max<int>(0, min_v);
    max_u = std::min<int>(width-1, max_u);
    max_v = std::min<int>(height-1, max_v);

    return cv::Rect( cv::Point(min_u, min_v), cv::Point (max_u, max_v) );
}

template<class PointT>
cv::Mat
PCLOpenCVConverter<PointT>::getRGBImage()
{
    if(cloud_->isOrganized())
        output_matrix_ = cv::Mat_<cv::Vec3b>(cloud_->height, cloud_->width);
    else
    {
        if(!cam_)
            throw std::runtime_error("Point cloud is not organized and camera parameters are not set.");

        output_matrix_ = cv::Mat_<cv::Vec3b>(cam_->getHeight(), cam_->getWidth());
    }

    output_matrix_.setTo( background_color_ );

    return fillMatrix<uchar>(&PCLOpenCVConverter<PointT>::getRGB);
}


template<class PointT>
cv::Mat
PCLOpenCVConverter<PointT>::getNormalizedDepth()
{
    if(cloud_->isOrganized())
        output_matrix_ = cv::Mat_<uchar>(cloud_->height, cloud_->width);
    else
        output_matrix_ = cv::Mat_<uchar>(cam_->getHeight(), cam_->getWidth());

    output_matrix_.setTo( 255 );

    return fillMatrix<uchar>(&PCLOpenCVConverter<PointT>::getZNormalized);
}

template<class PointT>
cv::Mat
PCLOpenCVConverter<PointT>::extractDepth()
{
    if(cloud_->isOrganized())
        output_matrix_ = cv::Mat_<float>(cloud_->height, cloud_->width);
    else
        output_matrix_ = cv::Mat_<float>(cam_->getHeight(), cam_->getWidth());

    output_matrix_.setTo( std::numeric_limits<float>::quiet_NaN());

    return fillMatrix<float>(&PCLOpenCVConverter<PointT>::getZ);
}

template<class PointT>
cv::Mat
PCLOpenCVConverter<PointT>::getOccupiedPixel()
{
    if(cloud_->isOrganized())
        output_matrix_ = cv::Mat_<uchar>(cloud_->height, cloud_->width);
    else
        output_matrix_ = cv::Mat_<uchar>(cam_->getHeight(), cam_->getWidth());

    output_matrix_.setTo(0);

    return fillMatrix<uchar>(&PCLOpenCVConverter<PointT>::getOccupied);
}

template<class PointT>
template<typename MatType>
cv::Mat
PCLOpenCVConverter<PointT>::fillMatrix( std::vector<MatType> (PCLOpenCVConverter<PointT>::*pf)(int v, int u) )
{
    CHECK( !cloud_->points.empty() );

    if(!cloud_->isOrganized())
        computeOrganizedCloud();

    boost::dynamic_bitset<> fg_mask;
    if ( !indices_.empty() )
        fg_mask = createMaskFromIndices(indices_, cloud_->points.size());
    else
    {
        fg_mask = boost::dynamic_bitset<> (cloud_->points.size());
        fg_mask.set();
    }

    MatType* p = (MatType*)output_matrix_.data;
    int nChannels = output_matrix_.channels();

    for (size_t v = 0; v < cloud_->height; v++)
    {
        for (size_t u = 0; u < cloud_->width; u++)
        {
            if( !remove_background_ || fg_mask[v*cloud_->width + u] )
            {
                std::vector<MatType> val = (this->*pf)(v,u);
                for(size_t c=0; c<val.size(); c++)
                    *p++ = val[c];
            }
            else
                p += nChannels;
        }
    }

    indices_.empty() ? roi_ = cv::Rect(cv::Point(0,0), cv::Point(cloud_->width, cloud_->height) ) : roi_ = computeROIfromIndices();
    return output_matrix_;
}



template<class PointT>
void
PCLOpenCVConverter<PointT>::computeOrganizedCloud()
{
    CHECK( !cloud_->isOrganized() && cam_ && cam_->getWidth()>0 && cam_->getHeight()>0 );

    ZBufferingParameter zBufParams;
    zBufParams.do_smoothing_ = true;
    zBufParams.smoothing_radius_ = 2;
    ZBuffering<PointT> zbuf (cam_, zBufParams);
    typename pcl::PointCloud<PointT>::Ptr organized_cloud (new pcl::PointCloud<PointT>);
    zbuf.renderPointCloud( *cloud_,  *organized_cloud);
    index_map_ = zbuf.getIndexMap();

    if(!indices_.empty())
    {
        size_t width = cam_->getWidth();
        size_t height = cam_->getHeight();

        boost::dynamic_bitset<> fg_mask = createMaskFromIndices(indices_, cloud_->points.size());

        std::vector<int> new_indices(indices_.size());

        size_t kept=0;
        for(size_t u=0; u<width; u++)
        {
            for(size_t v=0; v<height; v++)
            {
                int orig_idx = index_map_(v,u);

                if(orig_idx >= 0 && fg_mask[orig_idx] )
                {
                    new_indices[kept++] = v*width + u;
                }
            }
        }
        new_indices.resize(kept);
        indices_.swap(new_indices);
    }

    cloud_ = organized_cloud;
}


#define PCL_INSTANTIATE_PCLOpenCVConverter(T) template class V4R_EXPORTS PCLOpenCVConverter<T>;
PCL_INSTANTIATE(PCLOpenCVConverter, PCL_RGB_POINT_TYPES )

}
