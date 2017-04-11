#include <v4r/segmentation/all_headers.h>
#include <v4r/segmentation/plane_extractor_organized_multiplane.h>
#include <v4r/segmentation/plane_extractor_sac.h>
#include <v4r/segmentation/plane_extractor_sac_normals.h>
#include <v4r/segmentation/plane_extractor_tile.h>
#include <v4r/segmentation/segmenter_2d_connected_components.h>
#include <v4r/segmentation/segmenter_euclidean.h>
#include <v4r/segmentation/segmenter_organized_connected_component.h>
#include <v4r/segmentation/smooth_Euclidean_segmenter.h>

#include <pcl/impl/instantiate.hpp>

namespace v4r
{

template<typename PointT>
typename Segmenter<PointT>::Ptr
initSegmenter(int method, std::vector<std::string> &params )
{
    typename Segmenter<PointT>::Ptr cast_segmenter;

    if(method == SegmentationType::EuclideanSegmentation)
    {
        EuclideanSegmenterParameter param;
        params = param.init(params);
        typename EuclideanSegmenter<PointT>::Ptr seg (new EuclideanSegmenter<PointT> (param));
        cast_segmenter = boost::dynamic_pointer_cast<Segmenter<PointT> > (seg);
    }
//    else if(method == SegmentationType::SmoothEuclideanClustering)
//    {
//        SmoothEuclideanSegmenterParameter param;
//        params = param.init(params);
//        typename SmoothEuclideanSegmenter<PointT>::Ptr seg (new SmoothEuclideanSegmenter<PointT> (param));
//        cast_segmenter = boost::dynamic_pointer_cast<Segmenter<PointT> > (seg);
//    }
    else if(method == SegmentationType::OrganizedConnectedComponents)
    {
        OrganizedConnectedComponentSegmenterParameter param;
        params = param.init(params);
        typename OrganizedConnectedComponentSegmenter<PointT>::Ptr seg (new OrganizedConnectedComponentSegmenter<PointT> (param));
        cast_segmenter = boost::dynamic_pointer_cast<Segmenter<PointT> > (seg);
    }
    else if(method == SegmentationType::ConnectedComponents2D)
    {
        ConnectedComponentsSegmenterParameter param;
        params = param.init(params);
        typename ConnectedComponentsSegmenter<PointT>::Ptr seg (new ConnectedComponentsSegmenter<PointT> (param));
        cast_segmenter = boost::dynamic_pointer_cast<Segmenter<PointT> > (seg);
    }
    else
    {
        std::cerr << "Segmentation method " << method << " not implemented!" << std::endl;
    }

    return cast_segmenter;
}


template<typename PointT>
typename PlaneExtractor<PointT>::Ptr
initPlaneExtractor(int method, std::vector<std::string> &params )
{
    typename PlaneExtractor<PointT>::Ptr cast_plane_extractor;
    if(method == PlaneExtractionType::OrganizedMultiplane)
    {
        PlaneExtractorParameter param;
        params = param.init(params);
        typename OrganizedMultiPlaneExtractor<PointT>::Ptr pe (new OrganizedMultiPlaneExtractor<PointT> (param));
        cast_plane_extractor = boost::dynamic_pointer_cast<PlaneExtractor<PointT> > (pe);
    }
    else if(method == PlaneExtractionType::SAC)
    {
        PlaneExtractorParameter param;
        params = param.init(params);
        typename SACPlaneExtractor<PointT>::Ptr pe (new SACPlaneExtractor<PointT> (param));
        cast_plane_extractor = boost::dynamic_pointer_cast<PlaneExtractor<PointT> > (pe);
    }
    else if(method == PlaneExtractionType::SACNormals)
    {
        PlaneExtractorParameter param;
        params = param.init(params);
        typename SACNormalsPlaneExtractor<PointT>::Ptr pe (new SACNormalsPlaneExtractor<PointT> (param));
        cast_plane_extractor = boost::dynamic_pointer_cast<PlaneExtractor<PointT> > (pe);
    }
    else if(method == PlaneExtractionType::Tile)
    {
        PlaneExtractorTileParameter param;
        params = param.init(params);
        typename PlaneExtractorTile<PointT>::Ptr pe (new PlaneExtractorTile<PointT> (param));
        cast_plane_extractor = boost::dynamic_pointer_cast<PlaneExtractor<PointT> > (pe);
    }
    else
    {
        std::cerr << "Plane extraction method " << method << " not implemented!" << std::endl;
    }

    return cast_plane_extractor;
}

#define PCL_INSTANTIATE_initPlaneExtractor(T) template V4R_EXPORTS typename PlaneExtractor<T>::Ptr initPlaneExtractor<T>(int, std::vector<std::string> &);
PCL_INSTANTIATE(initPlaneExtractor, PCL_XYZ_POINT_TYPES )

#define PCL_INSTANTIATE_initSegmenter(T) template V4R_EXPORTS typename Segmenter<T>::Ptr initSegmenter<T>(int, std::vector<std::string> &);
PCL_INSTANTIATE(initSegmenter, PCL_XYZ_POINT_TYPES )


}
