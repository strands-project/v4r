#include <v4r/recognition/recognition_model_hv.h>

namespace v4r
{

template<typename ModelT>
void
HVRecognitionModel<ModelT>::processSilhouette(bool do_smoothing,
                                              size_t smoothing_radius,
                                              bool do_erosion,
                                              size_t erosion_radius,
                                              size_t img_width)
{
    size_t img_height = image_mask_.size() / img_width;
//        std::stringstream fn; fn << "/tmp/rendered_image_" << i << ".txt";
//        std::ofstream f(fn.str().c_str());
//        for(size_t px=0; px<image_mask_.size(); px++)
//            f<< image_mask_[px] << " ";
//        f.close();

    if(do_smoothing)
    {
        std::vector<bool> img_mask_smooth (image_mask_);
        for(int u=0; u<img_width; u++)
        {
            for(int v=0; v <img_height; v++)
            {
                bool found = false;

                for(int uu = u-smoothing_radius; uu < u+smoothing_radius && !found; uu++)
                {
                    for(int vv = v-smoothing_radius; vv < v+smoothing_radius; vv++)
                    {
                        if( uu<0 || vv <0 || uu>(img_width-1) || vv>(img_height-1) )
                            continue;

                        if( image_mask_[vv*img_width+uu] )
                        {
                            img_mask_smooth[v*img_width+u] = true;
                            found = true;
                            break;
                        }
                    }
                }
            }
        }
        image_mask_ = img_mask_smooth;

//            fn.str(""); fn << "/tmp/rendered_image_smooth_" << i << ".txt";
//            f.open(fn.str().c_str());
//            for(size_t px=0; px<image_mask_.size(); px++)
//                f<< image_mask_[px] << " ";
//            f.close();
    }

    if(do_erosion)
    {
        std::vector<bool> img_mask_eroded = image_mask_;
        for(int u=0; u<img_width; u++)
        {
            for(int v=0; v <img_height; v++)
            {
                bool found = false;

                for(int uu = u-erosion_radius; uu < u+erosion_radius && !found; uu++)
                {
                    for(int vv = v-erosion_radius; vv < v+erosion_radius; vv++)
                    {
                        if( uu<0 || vv <0 || uu>(img_width-1) || vv>(img_height-1) )
                            continue;

                        if( !image_mask_[vv*img_width+uu] )
                        {
                            img_mask_eroded[v*img_width+u] = false;
                            found = true;
                            break;
                        }
                    }
                }
            }
        }
        image_mask_ = img_mask_eroded;

//            fn.str(""); fn << "/tmp/rendered_image_eroded_" << i << ".txt";
//            f.open(fn.str().c_str());
//            for(size_t px=0; px<image_mask_.size(); px++)
//                f<< image_mask_[px] << " ";
//            f.close();
    }
}



template class V4R_EXPORTS HVRecognitionModel<pcl::PointXYZRGB>;
//template class V4R_EXPORTS HVRecognitionModel<pcl::PointXYZ>;
}

