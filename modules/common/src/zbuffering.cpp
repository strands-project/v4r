#include <v4r/common/zbuffering.h>
#include <v4r/common/miscellaneous.h>
#include <omp.h>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <pcl/impl/instantiate.hpp>

namespace v4r
{

///////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT>
void
ZBuffering<PointT>::filter (const typename pcl::PointCloud<PointT> & model, std::vector<int> & indices_to_keep)
{
    float cx = cam_->getCx();
    float cy = cam_->getCy();
    float f = cam_->getFocalLength();
    size_t width = cam_->getWidth();
    size_t height = cam_->getHeight();

    indices_to_keep.resize (model.points.size ());
    size_t kept = 0;
    for (size_t i = 0; i < model.points.size (); i++)
    {
        float x = model.points[i].x;
        float y = model.points[i].y;
        float z = model.points[i].z;
        int u = static_cast<int> (f * x / z + cx);
        int v = static_cast<int> (f * y / z + cy);

        if (u >= (width - param_.u_margin_) || v >= (height - param_.v_margin_) || u < param_.u_margin_ || v < param_.v_margin_)
            continue;

        //Check if poeint depth (distance to camera) is greater than the (u,v) meaning that the point is not visible
        if ( pcl_isfinite( depth_[v * width + u] ) && (z - param_.inlier_threshold_) > depth_[v * width + u])
            continue;

        indices_to_keep[kept] = static_cast<int> (i);
        kept++;
    }

    indices_to_keep.resize (kept);
}

//template<typename PointT>
//void
//ZBuffering<PointT>::erode(const Eigen::MatrixXf &input,
//                          Eigen::MatrixXf &output,
//                          int erosion_size,
//                          int erosion_elem)
//{
//    cv::Mat_<float> input_cv;
//    cv::eigen2cv(input, input_cv);

//    int erosion_type = cv::MORPH_RECT;
//    if( erosion_elem == 0 ){ erosion_type = cv::MORPH_RECT; }
//    else if( erosion_elem == 1 ){ erosion_type = cv::MORPH_CROSS; }
//    else if( erosion_elem == 2) { erosion_type = cv::MORPH_ELLIPSE; }

//    cv::Mat element = cv::getStructuringElement( erosion_type,
//                                         cv::Size( 2*erosion_size + 1, 2*erosion_size+1 ),
//                                         cv::Point( erosion_size, erosion_size ) );
//    cv::Mat eroded_input;
//    cv::erode( input_cv, eroded_input, element );
//    cv::cv2eigen(eroded_input, output);
//}

template<typename PointT>
void
ZBuffering<PointT>::renderPointCloud(const pcl::PointCloud<PointT> &cloud, pcl::PointCloud<PointT> & rendered_view)
{
    float cx = cam_->getCx();
    float cy = cam_->getCy();
    float f = cam_->getFocalLength();
    size_t width = cam_->getWidth();
    size_t height = cam_->getHeight();

    rendered_view.points.resize( width * height );
    rendered_view.width = width;
    rendered_view.height = height;
    rendered_view.is_dense = false;

#pragma omp parallel for
    for (size_t i=0; i< width * height ; i++)    // initialize points to infinity
        rendered_view.points[i].x = rendered_view.points[i].y = rendered_view.points[i].z = std::numeric_limits<float>::quiet_NaN();

    std::vector<omp_lock_t> pt_locks (width * height);
    for(size_t i=0; i<pt_locks.size(); i++)
        omp_init_lock(&pt_locks[i]);

#pragma omp parallel for schedule (dynamic)
    for (size_t i=0; i<cloud.points.size(); i++)
    {
        const PointT &pt = cloud.points[i];
        int u = static_cast<int> (f * pt.x / pt.z + cx);
        int v = static_cast<int> (f * pt.y / pt.z + cy);

        if (u >= width || v >= height  || u < 0 || v < 0)
            continue;

        int idx = v * width + u;

        omp_set_lock(&pt_locks[idx]);
        PointT &r_pt = rendered_view.points[idx];

        if ( !pcl_isfinite( r_pt.z ) || (pt.z < r_pt.z) )
            r_pt = pt;

        omp_unset_lock(&pt_locks[idx]);
    }

    for(size_t i=0; i<pt_locks.size(); i++)
        omp_destroy_lock(&pt_locks[i]);


    if (param_.do_smoothing_)
    {
        pcl::PointCloud<PointT> rendered_view_unsmooth = rendered_view;

        for (int u = param_.smoothing_radius_; u < (width - param_.smoothing_radius_); u++)
        {
            for (int v = param_.smoothing_radius_; v < (height - param_.smoothing_radius_); v++)
            {
                float min = std::numeric_limits<float>::max();
                int min_uu = u, min_vv = v;
                for (int uu = (u - param_.smoothing_radius_); uu <= (u + param_.smoothing_radius_); uu++)
                {
                    for (int vv = (v - param_.smoothing_radius_); vv <= (v + param_.smoothing_radius_); vv++)
                    {
                        if( uu<0 || vv<0 || uu>=width || vv>=height)    // this shouldn't happen anyway
                            continue;

                        PointT &p = rendered_view_unsmooth.at(uu,vv);
                        if ( !pcl_isfinite(min) || (pcl::isFinite(p) && ( p.z < min)) ) {
                            min = p.z;
                            min_uu = uu;
                            min_vv = vv;
                        }
                    }
                }

                rendered_view.at(u,v) = rendered_view_unsmooth.at(min_uu, min_vv);
            }
        }
    }
}

template<typename PointT>
void
ZBuffering<PointT>::computeDepthMap (const typename pcl::PointCloud<PointT> & cloud, Eigen::MatrixXf &depth_image, std::vector<int> &visible_indices)
{
    float cx = cam_->getCx();
    float cy = cam_->getCy();
    float f = cam_->getFocalLength();
    size_t width = cam_->getWidth();
    size_t height = cam_->getHeight();

    indices_map_.reset (new std::vector<int>( width * height, -1 ));

    std::vector<omp_lock_t> pt_locks (width * height);
    for(size_t i=0; i<pt_locks.size(); i++)
        omp_init_lock(&pt_locks[i]);

    if(!cloud.isOrganized() || param_.force_unorganized_)
    {
        depth_image = std::numeric_limits<float>::quiet_NaN() * Eigen::MatrixXf::Ones(height, width);

#pragma omp parallel for schedule(dynamic)
        for(size_t i=0; i<cloud.points.size(); i++)
        {
            const PointT &pt = cloud.points[i];
            int u = static_cast<int> (f * pt.x / pt.z + cx);
            int v = static_cast<int> (f * pt.y / pt.z + cy);

            if (u >= width || v >= height  || u < 0 || v < 0)
                continue;

            int idx = v * width + u;

            omp_set_lock(&pt_locks[idx]);

            if ( ( pcl_isfinite(pt.z) && !pcl_isfinite( depth_image(v,u) )) ||
                 (pt.z < depth_image(v,u) ) ) {
                depth_image(v,u) = pt.z;
                indices_map_->at(idx) = i;
            }

            omp_unset_lock(&pt_locks[idx]);
        }
    }
    else {
        if ( cloud.points.size() != height * width)
            throw std::runtime_error("Occlusion cloud does not have the same size as provided by the parameters img_height and img_width!");

        depth_image = Eigen::MatrixXf(height, width);

#pragma omp parallel for schedule (dynamic)
        for(size_t u=0; u<cloud.width; u++)
        {
            for(size_t v=0; v<cloud.height; v++)
            {
                const PointT &pt = cloud.at(u,v);
                if (pcl_isfinite(pt.z))
                {
                    depth_image(v,u) = pt.z;
                    indices_map_->at(v * width + u) = v * width + u;
                }
            }
        }
    }

    for(size_t i=0; i<pt_locks.size(); i++)
        omp_destroy_lock(&pt_locks[i]);

    visible_indices.resize(indices_map_->size());
    size_t kept=0;
    for(size_t i=0; i < height * width; i++)
    {
        if(indices_map_->at(i) >= 0)
        {
            visible_indices[kept] = indices_map_->at(i);
            kept++;
        }
    }
    visible_indices.resize(kept);
}

template<typename PointT>
void
ZBuffering<PointT>::computeDepthMap (const typename pcl::PointCloud<PointT> & scene)
{
    float cx = cam_->getCx();
    float cy = cam_->getCy();
    float f = cam_->getFocalLength();
    size_t width = cam_->getWidth();
    size_t height = cam_->getHeight();

    //compute the focal length
    if (param_.compute_focal_length_)
    {
        float max_u, max_v, min_u, min_v;
        max_u = max_v = std::numeric_limits<float>::max () * -1;
        min_u = min_v = std::numeric_limits<float>::max ();

        for (size_t i = 0; i < scene.points.size (); i++)
        {
            float b_x = scene.points[i].x / scene.points[i].z;
            if (b_x > max_u)
                max_u = b_x;
            if (b_x < min_u)
                min_u = b_x;

            float b_y = scene.points[i].y / scene.points[i].z;
            if (b_y > max_v)
                max_v = b_y;
            if (b_y < min_v)
                min_v = b_y;
        }

        float maxC = std::max (std::max (std::abs (max_u), std::abs (max_v)), std::max (std::abs (min_u), std::abs (min_v)));
        f = (cx) / maxC;
    }

    depth_.resize(width * height, std::numeric_limits<float>::quiet_NaN());
    std::vector<omp_lock_t> depth_locks (width * height);
    for(size_t i=0; i<depth_locks.size(); i++)
        omp_init_lock(&depth_locks[i]);

    std::vector<int> indices2input (width * height, -1);

#pragma omp parallel for schedule (dynamic)
    for (size_t i=0; i<scene.points.size(); i++)
    {
        const PointT &pt = scene.points[i];
        int u = static_cast<int> (f * pt.x / pt.z + cx);
        int v = static_cast<int> (f * pt.y / pt.z + cy);

        if (u >= width - param_.u_margin_ || v >= height - param_.v_margin_ || u < param_.u_margin_ || v < param_.v_margin_)
            continue;

        int idx = v * width + u;

        omp_set_lock(&depth_locks[idx]);

        if ( (pt.z < depth_[idx]) || !pcl_isfinite(depth_[idx]) ) {
            depth_[idx] = pt.z;
            indices2input [idx] = i;
        }

        omp_unset_lock(&depth_locks[idx]);
    }

    for(size_t i=0; i<depth_locks.size(); i++)
        omp_destroy_lock(&depth_locks[i]);

    if (param_.do_smoothing_)
    {
        //Dilate and smooth the depth map
        std::vector<float> depth_smooth (width * height, std::numeric_limits<float>::quiet_NaN());
        std::vector<int> indices2input_smooth = indices2input;

        for (int u = param_.smoothing_radius_; u < (width - param_.smoothing_radius_); u++)
        {
            for (int v = param_.smoothing_radius_; v < (height - param_.smoothing_radius_); v++)
            {
                float min = std::numeric_limits<float>::max();
                int min_idx = v * width + u;
                for (int j = (u - param_.smoothing_radius_); j <= (u + param_.smoothing_radius_); j++)
                {
                    for (int i = (v - param_.smoothing_radius_); i <= (v + param_.smoothing_radius_); i++)
                    {
                        if( j<0 || i<0 || j>=height || i>=width)    // this shouldn't happen anyway
                            continue;

                        int idx = i * width + j;
                        if (pcl_isfinite(depth_[idx]) && (depth_[idx] < min)) {
                            min = depth_[idx];
                            min_idx = idx;
                        }
                    }
                }

                if ( min < std::numeric_limits<float>::max() - 0.001 ) {
                    depth_smooth[v * width + u] = min;
                    indices2input_smooth[v * width + u] = indices2input[min_idx];
                }
            }
        }
        depth_ = depth_smooth;
        indices2input = indices2input_smooth;
    }


    boost::dynamic_bitset<> pt_is_visible(scene.points.size(), 0);
    for(size_t i=0; i<indices2input.size(); i++)
    {
        int input_idx = indices2input[i];
        if(input_idx>=0)
            pt_is_visible.set(input_idx);
    }
    kept_indices_ = createIndicesFromMask<int>(pt_is_visible);
}

///////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT>
void
ZBuffering<PointT>::filter (const typename pcl::PointCloud<PointT> & model, typename pcl::PointCloud<PointT> & filtered)
{
    std::vector<int> indices_to_keep;
    filter(model, indices_to_keep);
    pcl::copyPointCloud (model, indices_to_keep, filtered);
}

#define PCL_INSTANTIATE_ZBuffering(T) template class V4R_EXPORTS ZBuffering<T>;
PCL_INSTANTIATE(ZBuffering, PCL_XYZ_POINT_TYPES )

}


