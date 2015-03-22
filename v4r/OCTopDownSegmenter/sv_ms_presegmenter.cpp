#include "sv_ms_presegmenter.h"
#include <pcl/segmentation/supervoxel_clustering.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <opencv2/opencv.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/common/time.h>
#include <vtkPolyLine.h>
#include "v4r/on_nurbs/fitting_surface_depth_im.h"
#include "v4r/Segmentation/SlicRGBD.h"
#include <v4r/ORUtils/pcl_opencv.h>
#include <pcl/common/pca.h>

//#define VIS_MERGE
//#define VIS_CC

#ifdef VIS_MERGE
pcl::visualization::PCLVisualizer vis_merge("merging");
#endif

template<typename PointT>
v4rOCTopDownSegmenter::SVMumfordShahPreSegmenter<PointT>::SVMumfordShahPreSegmenter()
{
    nyu_ = 0.001f;
    lambda_ = 0.005f;
    sigma_ = 0.00001f;
    alpha_ = 1.f;

    label_colors_.clear();
    vis_segmentation_.reset(new pcl::visualization::PCLVisualizer ("segmentation"));

#ifdef VIS_CC
    vis_cc.reset(new pcl::visualization::PCLVisualizer ("connected components"));
#endif
    vis_at_each_move_ = false;
    ds_resolution_ = 0.005f;
    MAX_MODEL_TYPE_ = BSPLINE_MODEL_TYPE_5x5;
    pixelwise_refinement_ = false;
    supervoxel_seed_resolution_ = 0.03f;
    supervoxel_resolution_ = 0.004f;
    color_importance_ = 0;
    spatial_importance_ = 1;
    normal_importance_ = 3;

    sRGB_LUT.resize(256, -1);
    sXYZ_LUT.resize(4000, -1);

    boundary_window_ = 1;
    use_SLIC_RGBD_ = false;
}

template<typename PointT>
v4rOCTopDownSegmenter::SVMumfordShahPreSegmenter<PointT>::~SVMumfordShahPreSegmenter()
{

}

/// VISUALIZATION FUNCTIONS

#ifdef VIS_MERGE
template<typename PointT>
void visMerge(v4rOCTopDownSegmenter::MergeCandidate<PointT> & merge,
              typename pcl::PointCloud<PointT>::Ptr & cloud_)
{
    vis_merge.removePointCloud("c1");
    vis_merge.removePointCloud("c2");

    typename pcl::PointCloud<PointT>::Ptr c1, c2;
    c1.reset(new pcl::PointCloud<PointT>);
    c2.reset(new pcl::PointCloud<PointT>);
    pcl::copyPointCloud(*cloud_, merge.r1_->indices_, *c1);
    pcl::copyPointCloud(*cloud_, merge.r2_->indices_, *c2);

    {
        pcl::visualization::PointCloudColorHandlerCustom<PointT> handler(c1, 255, 0, 0);
        vis_merge.addPointCloud(c1, handler, "c1");
    }

    {
        pcl::visualization::PointCloudColorHandlerCustom<PointT> handler(c2, 0, 255, 0);
        vis_merge.addPointCloud(c2, handler, "c2");
    }

    vis_merge.spin();
}
#endif

template<typename PointT>
void v4rOCTopDownSegmenter::SVMumfordShahPreSegmenter<PointT>::visualizeSegmentation
(std::vector<boost::shared_ptr<Region<PointT> > > & regions, int viewport)
{

    int max_label = regions.size();
    if((int)label_colors_.size() != max_label)
    {
        label_colors_.reserve (max_label + 1);
        srand (static_cast<unsigned int> (time (0)));
        while ((int)label_colors_.size () <= max_label )
        {
            uint8_t r = static_cast<uint8_t>( (rand () % 256));
            uint8_t g = static_cast<uint8_t>( (rand () % 256));
            uint8_t b = static_cast<uint8_t>( (rand () % 256));
            label_colors_.push_back (static_cast<uint32_t>(r) << 16 | static_cast<uint32_t>(g) << 8 | static_cast<uint32_t>(b));
        }
    }

    pcl::PointCloud<pcl::PointNormal>::Ptr curvature_cloud(new pcl::PointCloud<pcl::PointNormal>());
    //pcl::copyPointCloud(*cloud_, *curvature_cloud);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cc(new pcl::PointCloud<pcl::PointXYZRGB>(*cloud_));
    Eigen::Vector3f origin = Eigen::Vector3f::Zero();
    std::vector<int> valid(cloud_->points.size(), false);

    double max_curvature = 0.f;

    for(size_t i=0; i < regions.size(); i++)
    {
        if(!regions[i]->valid_)
            continue;

        for(size_t j=0; j < regions[i]->indices_.size(); j++)
        {
            cloud_cc->at(regions[i]->indices_[j]).rgb = label_colors_[i];
            valid[regions[i]->indices_[j]] = true;
        }

        if(regions[i]->current_model_type_ == PLANAR_MODEL_TYPE_ && regions[i]->plane_model_defined_)
        {
            //change the position of the 3D points
            Eigen::Vector3d pp = regions[i]->mean_;
            Eigen::Vector3f point_on_plane = pp.cast<float>();
            for(size_t j=0; j < regions[i]->indices_.size(); j++)
            {

                Eigen::Vector3f p = cloud_cc->at(regions[i]->indices_[j]).getVector3fMap();
                Line l(origin, p);
                Eigen::Vector3f intersect = intersectLineWithPlane(l, regions[i]->planar_model_.first,
                                                                   regions[i]->planar_model_.second, point_on_plane);

                cloud_cc->at(regions[i]->indices_[j]).getVector3fMap() = intersect;
            }
        }
        else if( ( regions[i]->current_model_type_ == BSPLINE_MODEL_TYPE_3x3 || regions[i]->current_model_type_ == BSPLINE_MODEL_TYPE_5x5) && regions[i]->bspline_model_defined_)
        {
            //change the position of the 3D points
            /*double u,v,z;
            for(size_t j=0; j < regions[i]->indices_.size(); j++)
            {
                u = (regions[i]->indices_[j] % cloud_cc->width);
                v = (regions[i]->indices_[j] / cloud_cc->width);

                ON_NurbsSurface& bspline = regions[i]->bspline_model_;
                if(u<bspline.Knot(0,0) || u>bspline.Knot(0,bspline.KnotCount(0)-1))
                {
                    std::cout << "v4rOCTopDownSegmenter::SVMumfordShahPreSegmenter::visualizeSegmentation] Error, index u out of bounds." << std::endl;
                }

                if(v<bspline.Knot(1,0) || v>bspline.Knot(1,bspline.KnotCount(1)-1))
                {
                    std::cout << "[v4rOCTopDownSegmenter::SVMumfordShahPreSegmenter::visualizeSegmentation] Error, index v out of bounds." << std::endl;
                }

                bspline.Evaluate(u, v, 0, 1, &z);

                Eigen::Vector3f p = cloud_cc->at(regions[i]->indices_[j]).getVector3fMap();
                Line l(origin, p);
                cloud_cc->at(regions[i]->indices_[j]).getVector3fMap() = l.eval(z);

                pcl::PointNormal pp;
                pp.getVector3fMap() = cloud_cc->at(regions[i]->indices_[j]).getVector3fMap();

                double ptd[6];
                bspline.Evaluate(u,v,2,1,ptd);
                double c = (ptd[3]+ptd[5])*0.5;
                pp.curvature = c * c;
                curvature_cloud->push_back(pp);
            }*/

            pcl::IndicesPtr ind;
            ind.reset(new std::vector<int>(regions[i]->indices_));

            pcl::PCA<PointT> basis;
            basis.setInputCloud(cloud_);
            basis.setIndices(ind);

            // #################### PLANE PROJECTION #########################
            typename pcl::PointCloud<PointT>::Ptr proj_cloud(new pcl::PointCloud<PointT>);
            basis.project(pcl::PointCloud<PointT>(*cloud_,*ind), *proj_cloud);

            ON_NurbsSurface& bspline = regions[i]->bspline_model_;


            // evaluate error and curvature for each point
            Eigen::VectorXd curvatures(proj_cloud->size(), 1);
            int nder = 2; // number of derivatives
            int nvals = bspline.Dimension()*(nder+1)*(nder+2)/2;
            double P[nvals];
            Eigen::Vector3d n, xu, xv, xuu, xvv, xuv;
            Eigen::Matrix2d II;

            for (std::size_t j = 0; j < proj_cloud->size(); j++)
            {
                const PointT& p0 = proj_cloud->at(j);
                bspline.Evaluate (p0.x, p0.y, nder, bspline.Dimension(), P);

                // positions
                PointT p1, p2;
                p1.x = P[0];    p1.y = P[1];    p1.z = P[2];
                basis.reconstruct(p1,p2);
                cloud_cc->at(regions[i]->indices_[j]).getVector3fMap() = p2.getVector3fMap();

                // 1st derivatives (for normals)
                xu(0) = P[3];    xu(1) = P[4];    xu(2) = P[5];
                xv(0) = P[6];    xv(1) = P[7];    xv(2) = P[8];

                n = xu.cross(xv);
                n.normalize();

                // 2nd derivatives (for curvature)
                xuu(0) = P[9];     xuu(1) = P[10];    xuu(2) = P[11];
                xuv(0) = P[12];    xuv(1) = P[13];    xuv(2) = P[14];
                xvv(0) = P[15];    xvv(1) = P[16];    xvv(2) = P[17];

                // fundamental form
                II(0,0) = n.dot(xuu);   // principal curvature along u
                II(0,1) = n.dot(xuv);
                II(1,0) = II(0,1);
                II(1,1) = n.dot(xvv);   // principal curvature along v

                //      float mean = 0.5*( II(0,0)+II(1,1) ); // mean curvature
                //      float gauss = II(0,0)*II(1,1) - II(0,1)*II(1,0); // gauss curvature

                //double& c = curvatures[j];
                double c = sqrt(II(0,0)*II(0,0) + II(1,1)*II(1,1)); // norm of principal curvatures

                pcl::PointNormal pp;
                pp.getVector3fMap() = cloud_cc->at(regions[i]->indices_[j]).getVector3fMap();
                pp.curvature = c;
                curvature_cloud->push_back(pp);

                if(c > max_curvature)
                {
                    max_curvature = c;
                }
            }
        }
    }

    std::cout << "MAX CURVATURE:" << max_curvature << std::endl;

    for(size_t i=0; i < cloud_cc->points.size(); i++)
    {
        if(!valid[i])
        {
            cloud_cc->points[i].x = cloud_cc->points[i].y = cloud_cc->points[i].z = std::numeric_limits<float>::quiet_NaN();
        }
    }

    //vis_segmentation_->removeAllPointClouds();

    {
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler(cloud_cc);
        vis_segmentation_->addPointCloud(cloud_cc, handler, "cloud", viewport);
    }

    if(save_impath_.compare("") != 0)
    {
        std::stringstream save_to;
        save_to << save_impath_ << "cloud_projected.pcd";

        pcl::io::savePCDFileBinary(save_to.str(), *cloud_cc);
    }

    {
        //visualize curvature
        int v1,v2;
        pcl::visualization::PCLVisualizer vis_curv("curvature");
        vis_curv.createViewPort(0,0,0.5,1,v1);
        vis_curv.createViewPort(0.5,0,1,1,v2);
        pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointNormal> handler(curvature_cloud, "curvature");
        vis_curv.addPointCloud(curvature_cloud, handler, "curvature", v2);
        vis_curv.spin();
        vis_curv.addPointCloud(cloud_, "test", v1);
        vis_curv.spin();
    }
    //vis_segmentation_->spin();
}

template<typename PointT>
void v4rOCTopDownSegmenter::SVMumfordShahPreSegmenter<PointT>::visualizeRegions
(std::vector<boost::shared_ptr<Region<PointT> > > & regions, int viewport)
{

    int max_label = regions.size();
    if((int)label_colors_.size() != max_label)
    {
        label_colors_.reserve (max_label + 1);
        srand (static_cast<unsigned int> (time (0)));
        while ((int)label_colors_.size () <= max_label )
        {
            uint8_t r = static_cast<uint8_t>( (rand () % 256));
            uint8_t g = static_cast<uint8_t>( (rand () % 256));
            uint8_t b = static_cast<uint8_t>( (rand () % 256));
            label_colors_.push_back (static_cast<uint32_t>(r) << 16 | static_cast<uint32_t>(g) << 8 | static_cast<uint32_t>(b));
        }
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cc_planes(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cc_bsplines(new pcl::PointCloud<pcl::PointXYZRGB>());

    for(size_t i=0; i < regions.size(); i++)
    {
        if(!regions[i]->valid_)
            continue;

        for(size_t j=0; j < regions[i]->indices_.size(); j++)
        {

            pcl::PointXYZRGB p;
            p.getVector3fMap() = cloud_->at(regions[i]->indices_[j]).getVector3fMap();
            p.rgb = label_colors_[i];

            if(regions[i]->current_model_type_ == BSPLINE_MODEL_TYPE_3x3)
            {
                cloud_cc_bsplines->push_back(p);
            }
            else
            {
                //cloud_cc->at(regions[i]->indices_[j]).rgb = label_colors_[i];
                cloud_cc_planes->push_back(p);
            }
        }
    }

    //vis_cc->removeAllPointClouds();

    {
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler(cloud_cc_planes);
        vis_segmentation_->addPointCloud(cloud_cc_planes, handler, "cloud_planes", viewport);
    }

    {
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler(cloud_cc_bsplines);
        vis_segmentation_->addPointCloud(cloud_cc_bsplines, handler, "cloud_cc_bsplines", viewport);
        //vis_segmentation_->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "cloud_cc_bsplines");
    }

    if(save_impath_.compare("") != 0)
    {
        std::stringstream save_to;
        save_to << save_impath_ << "cloud_1.pcd";

        pcl::PointCloud<pcl::PointXYZRGB> cloud_to_save;
        pcl::copyPointCloud(*cloud_cc_planes, cloud_to_save);
        cloud_to_save += *cloud_cc_bsplines;

        pcl::io::savePCDFileBinary(save_to.str(), cloud_to_save);
    }

    //vis_cc->spin();
}

template<typename PointT>
void v4rOCTopDownSegmenter::SVMumfordShahPreSegmenter<PointT>::projectRegionsOnImage
(std::vector<boost::shared_ptr<Region<PointT> > > & regions,
 std::string append)
{

    int max_label = regions.size();
    if((int)label_colors_.size() != max_label)
    {
        label_colors_.reserve (max_label + 1);
        srand (static_cast<unsigned int> (time (0)));
        while ((int)label_colors_.size () <= max_label )
        {
            uint8_t r = static_cast<uint8_t>( (rand () % 256));
            uint8_t g = static_cast<uint8_t>( (rand () % 256));
            uint8_t b = static_cast<uint8_t>( (rand () % 256));
            label_colors_.push_back (static_cast<uint32_t>(r) << 16 | static_cast<uint32_t>(g) << 8 | static_cast<uint32_t>(b));
        }
    }

    cv::Mat image(cloud_->height, cloud_->width, CV_8UC3);

    for(int r=0; r < (int)cloud_->height; r++)
    {
        for(int c=0; c < (int)cloud_->width; c++)
        {
            unsigned char rs = cloud_->at(c, r).r;
            unsigned char gs = cloud_->at(c, r).g;
            unsigned char bs = cloud_->at(c, r).b;
            image.at<cv::Vec3b>(r,c) = cv::Vec3b(bs,gs,rs);
        }
    }

    cv::Mat image_clone = image.clone();

    float factor = 0.05f;

    for(size_t i=0; i < regions.size(); i++)
    {
        if(!regions[i]->valid_)
            continue;

        for(size_t j=0; j < regions[i]->indices_.size(); j++)
        {
            int r, c;
            int idx = regions[i]->indices_[j];
            r = idx / cloud_->width;
            c = idx % cloud_->width;

            uint32_t rgb = label_colors_[i];
            unsigned char rs = (rgb >> 16) & 0x0000ff;
            unsigned char gs = (rgb >> 8) & 0x0000ff;
            unsigned char bs = (rgb) & 0x0000ff;

            cv::Vec3b im = image.at<cv::Vec3b>(r,c);
            image.at<cv::Vec3b>(r,c) = cv::Vec3b((unsigned char)(im[0] * factor + bs * (1 - factor)),
                    (unsigned char)(im[1] * factor + gs * (1 - factor)),
                    (unsigned char)(im[2] * factor + rs * (1 - factor)));
        }
    }

    cv::Mat collage = cv::Mat(image.rows, image.cols * 2, CV_8UC3);
    collage.setTo(cv::Vec3b(0,0,0));
    //collage(cv::Range(0, collage.rows), cv::Range(0, collage.cols)) = image_clone + cv::Scalar(cv::Vec3b(0, 0, 0));

    for(int r=0; r < (int)cloud_->height; r++)
    {
        for(int c=0; c < (int)cloud_->width; c++)
        {
            collage.at<cv::Vec3b>(r,c) = image_clone.at<cv::Vec3b>(r,c);
        }
    }

    collage(cv::Range(0, collage.rows), cv::Range(collage.cols/2, collage.cols)) = image + cv::Scalar(cv::Vec3b(0, 0, 0));

    cv::imshow("regions", collage);
    cv::waitKey(0);

    //cv::imshow("image", image_clone);

    std::stringstream save_to;
    save_to << save_impath_ << append << ".png";
    cv::imwrite(save_to.str(), image);
}

/// END VISUALIZATION FUNCTIONS


template<typename PointT>
void v4rOCTopDownSegmenter::SVMumfordShahPreSegmenter<PointT>::createLabelCloud
(std::vector<boost::shared_ptr<Region<PointT> > > & regions)
{

    labeled_cloud_.reset(new pcl::PointCloud<pcl::PointXYZL>());
    pcl::copyPointCloud(*cloud_, *labeled_cloud_);

    for(size_t i=0; i < cloud_->points.size(); i++)
    {
        labeled_cloud_->points[i].label = 0;
    }

    uint32_t label = 1;
    for(size_t i=0; i < regions.size(); i++)
    {
        if(!regions[i]->valid_)
            continue;

        for(size_t j=0; j < regions[i]->indices_.size(); j++)
        {
            labeled_cloud_->at(regions[i]->indices_[j]).label = label;
        }

        label++;
    }
}

template<typename PointT>
int v4rOCTopDownSegmenter::SVMumfordShahPreSegmenter<PointT>::extractSVBoundaries()
{
    int kw = boundary_window_;
    cv::Mat boundaries(cloud_->height, cloud_->width, CV_8UC1);
    boundaries.setTo(0);
    int n_boundaries = 0;

    for(int r=0; r < (int)cloud_->height; r++)
    {
        for(int c=0; c < (int)cloud_->width; c++)
        {

            Eigen::Vector3f p = cloud_->at(c,r).getVector3fMap();
            if(!pcl_isfinite(p[2]))
                continue;

            bool is_boundary = false;
            uint32_t label = supervoxels_labels_cloud_->at(c,r).label;

            std::set<uint32_t> labels;
            for(int u=std::max(0, r - kw); u <= std::min((int)(cloud_->height - 1), r + kw); u++)
            {
                for(int v=std::max(0, c - kw); v <= std::min((int)(cloud_->width - 1), c + kw); v++)
                {

                    //4-neighborhood
                    /*if(u != r && v != c)
                        continue;*/

                    uint32_t label_uv = supervoxels_labels_cloud_->at(v,u).label;
                    Eigen::Vector3f p_uv = cloud_->at(v,u).getVector3fMap();
                    if(!pcl_isfinite(p_uv[2]))
                        continue;

                    //check if label_uv and label are adjacent
                    if(label >= adjacent_.size() || label_uv >= adjacent_.size())
                    {
                        //std::cout << label << " " << label_uv << " " << adjacent_.size() << std::endl;
                        continue;
                    }

                    int ri, rj;
                    ri = std::max(label, label_uv);
                    rj = std::min(label, label_uv);

                    if(adjacent_[ri][rj] < 0)
                        continue;

                    if(!merge_candidates_[adjacent_[ri][rj]].isValid())
                        continue;

                    /*if(label >= merge_candidates_.size() || label_uv >= merge_candidates_.size())
                        continue;

                    if(!(merge_candidates_[std::max(label, label_uv)][std::min(label, label_uv)].isValid()))
                        continue;*/

                    if(label_uv != label)
                    {
                        is_boundary = true;
                    }

                    labels.insert(label_uv);

                }
            }

            if(is_boundary)
            {
                //std::cout << labels.size() << std::endl;
                boundaries.at<unsigned char>(r,c) = 255;
                for(std::set<uint32_t>::iterator it = labels.begin(); it != labels.end(); it++)
                {
                    //std::cout << label << " " << *it << std::endl;

                    uint32_t label_uv = *it;

                    if(label >= adjacent_.size() || label_uv >= adjacent_.size())
                        continue;

                    int ri, rj;
                    ri = std::max(label, label_uv);
                    rj = std::min(label, label_uv);

                    if(adjacent_[ri][rj] < 0)
                        continue;

                    if(!merge_candidates_[adjacent_[ri][rj]].isValid())
                        continue;

                    merge_candidates_[adjacent_[ri][rj]].boundary_length_++;
                    merge_candidates_[adjacent_[ri][rj]].boundary_length_++;

                    /*if(label >= merge_candidates_.size() || *it >= merge_candidates_.size())
                        continue;

                    merge_candidates_[label][*it].boundary_length_++;
                    merge_candidates_[*it][label].boundary_length_++;*/
                }

                n_boundaries++;
            }
        }
    }

    for(size_t i=0; i < adjacent_.size(); i++)
    {
        for(size_t j=0; j < i; j++)
        {
            if(adjacent_[i][j] < 0)
                continue;

            if(!merge_candidates_[adjacent_[i][j]].isValid())
                continue;

            if(merge_candidates_[adjacent_[i][j]].boundary_length_ == 0)
            {
                merge_candidates_[adjacent_[i][j]].valid_ = false;
                adjacent_[i][j] = -1;
            }

            //std::cout << "boundary length:" << merge_candidates_[adjacent_[i][j]].boundary_length_ << std::endl;
        }
    }

    cv::imshow("boundaries", boundaries);
    cv::waitKey(0);

    return n_boundaries;
}

template<typename PointT>
void v4rOCTopDownSegmenter::SVMumfordShahPreSegmenter<PointT>::computeMeanAndCovarianceMatrixSVMS
(Eigen::Matrix3d & covariance, Eigen::Vector3d & mean, Eigen::Matrix<double, 1, 9, Eigen::RowMajor> & accu, std::vector<int> & indices)
{

    typename pcl::PointCloud<PointT> & cloud = *cloud_;
    for (std::vector<int>::const_iterator iIt = indices.begin (); iIt != indices.end (); ++iIt)
    {
        accu [0] += cloud[*iIt].x * cloud[*iIt].x;
        accu [1] += cloud[*iIt].x * cloud[*iIt].y;
        accu [2] += cloud[*iIt].x * cloud[*iIt].z;
        accu [3] += cloud[*iIt].y * cloud[*iIt].y;
        accu [4] += cloud[*iIt].y * cloud[*iIt].z;
        accu [5] += cloud[*iIt].z * cloud[*iIt].z;
        accu [6] += cloud[*iIt].x;
        accu [7] += cloud[*iIt].y;
        accu [8] += cloud[*iIt].z;
    }

    accu /= static_cast<double> (indices.size());
    mean[0] = accu[6]; mean[1] = accu[7]; mean[2] = accu[8];
    covariance.coeffRef (0) = accu [0] - accu [6] * accu [6];
    covariance.coeffRef (1) = accu [1] - accu [6] * accu [7];
    covariance.coeffRef (2) = accu [2] - accu [6] * accu [8];
    covariance.coeffRef (4) = accu [3] - accu [7] * accu [7];
    covariance.coeffRef (5) = accu [4] - accu [7] * accu [8];
    covariance.coeffRef (8) = accu [5] - accu [8] * accu [8];
    covariance.coeffRef (3) = covariance.coeff (1);
    covariance.coeffRef (6) = covariance.coeff (2);
    covariance.coeffRef (7) = covariance.coeff (5);
}

template<typename PointT>
void v4rOCTopDownSegmenter::SVMumfordShahPreSegmenter<PointT>::computeColorMean
(Eigen::Vector3d & mean, std::vector<int> & indices)
{

    typename pcl::PointCloud<PointT> & cloud = *cloud_;
    for (std::vector<int>::const_iterator iIt = indices.begin (); iIt != indices.end (); ++iIt)
    {
        /*unsigned char r, b, g;
        r = cloud[*iIt].r;
        g = cloud[*iIt].g;
        b = cloud[*iIt].b;

        mean[0] += r;
        mean[1] += g;
        mean[2] += b;*/

        /*assert(scene_LAB_values_[*iIt][0] >= -0.1f);
        assert(scene_LAB_values_[*iIt][1] >= -0.1f);
        assert(scene_LAB_values_[*iIt][2] >= -0.1f);*/

        mean += scene_LAB_values_[*iIt];
    }

    mean /= static_cast<double> (indices.size());
}

struct candidate_comp {
    bool operator() (const std::pair<int, float> & lhs, const std::pair<int, float>& rhs) const
    {
        return lhs.second < rhs.second;
    }
};

template<typename PointT>
void v4rOCTopDownSegmenter::SVMumfordShahPreSegmenter<PointT>::fillLineOfSights()
{
    los_.resize(cloud_->points.size());
    Eigen::Vector3f origin = Eigen::Vector3f::Zero();

    for(size_t i=0; i < los_.size(); i++)
    {
        Eigen::Vector3f p = cloud_->points[i].getVector3fMap();
        if(!pcl_isfinite(p[2]))
            continue;

        Line ll(origin, p);
        los_[i] = ll;
    }
}

template<typename PointT>
void v4rOCTopDownSegmenter::SVMumfordShahPreSegmenter<PointT>::addSupervoxelConnectionsToViewer (
        PointT &supervoxel_center,
        typename pcl::PointCloud<PointT> &adjacent_supervoxel_centers,
        std::string supervoxel_name,
        boost::shared_ptr<pcl::visualization::PCLVisualizer> & viewer)
{
    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New ();
    vtkSmartPointer<vtkCellArray> cells = vtkSmartPointer<vtkCellArray>::New ();
    vtkSmartPointer<vtkPolyLine> polyLine = vtkSmartPointer<vtkPolyLine>::New ();

    //Iterate through all adjacent points, and add a center point to adjacent point pair
    typename pcl::PointCloud<PointT>::iterator adjacent_itr = adjacent_supervoxel_centers.begin ();
    for ( ; adjacent_itr != adjacent_supervoxel_centers.end (); ++adjacent_itr)
    {
        points->InsertNextPoint (supervoxel_center.data);
        points->InsertNextPoint (adjacent_itr->data);
    }
    // Create a polydata to store everything in
    vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New ();
    // Add the points to the dataset
    polyData->SetPoints (points);
    polyLine->GetPointIds  ()->SetNumberOfIds(points->GetNumberOfPoints ());
    for(unsigned int i = 0; i < points->GetNumberOfPoints (); i++)
        polyLine->GetPointIds ()->SetId (i,i);
    cells->InsertNextCell (polyLine);
    // Add the lines to the dataset
    polyData->SetLines (cells);
    viewer->addModelFromPolyData (polyData,supervoxel_name);
}

template<typename PointT>
void v4rOCTopDownSegmenter::SVMumfordShahPreSegmenter<PointT>::overSegmentation(std::vector< std::vector<int> > & adjacent,
                                                                                std::vector<int> & label_to_idx)
{

    if(use_SLIC_RGBD_ && cloud_->isOrganized())
    {
        double wc = 10;  //xy compactness factor. use a value ranging from 10 to 40 depending on your needs. Default is 10
        double wz = 1000; //2000
        double wn_cosa = 5000; //1500
        int num_superpixels = 1000;
        v4r::SlicRGBD::Parameter param(100, wc, wz, wn_cosa, 0.02, 15., true);
        v4r::SlicRGBD slic(param);
        slic.setNumberOfSuperpixel(num_superpixels);

        cv::Mat_<cv::Vec3b> im_draw;
        PCLOpenCV::ConvertPCLCloud2Image<PointT>(cloud_, im_draw);

        int numlabels(0);
        cv::Mat_<int> labels;

        slic.setCloud(cloud_, surface_normals_);
        slic.segmentSuperpixel(labels, numlabels);

        //drawLabels(im_draw, labels);
        for (unsigned i=0; i<slic.valid.size(); i++)
            if (!slic.valid[i]) {
                cv::Vec3b &col = im_draw(i);
                col[0] *= 0.2;
                col[1] *= 0.2;
                col[2] *= 0.2;
            }

        slic.drawContours(im_draw, labels, 255,0,0);
        cout<<"number of labels: "<<numlabels << " " << slic.valid.size() <<endl;

        cv::imshow("image",im_draw);
        cv::waitKey(0);

        //fill adjacent and label_to_idx
        adjacent.resize(numlabels);
        label_to_idx.resize(numlabels);

        for(size_t i=0; i < (numlabels); i++)
        {
            adjacent[i].resize(numlabels, -1);
            label_to_idx[i] = i;
        }

        int kw = boundary_window_;

        supervoxels_labels_cloud_.reset(new pcl::PointCloud<pcl::PointXYZL>);
        pcl::copyPointCloud(*cloud_, *supervoxels_labels_cloud_);

        for(int r=0; r < (int)cloud_->height; r++)
        {
            for(int c=0; c < (int)cloud_->width; c++)
            {

                supervoxels_labels_cloud_->at(c,r).label = -1;

                int idx_rc = r * cloud_->width + c;
                if(!slic.valid[idx_rc])
                    continue;

                Eigen::Vector3f p_rc = cloud_->at(c,r).getVector3fMap();

                int label = labels.at<int>(r,c);
                supervoxels_labels_cloud_->at(c,r).label = label;

                for(int u=std::max(0, r - kw); u <= std::min((int)(cloud_->height - 1), r + kw); u++)
                {
                    for(int v=std::max(0, c - kw); v <= std::min((int)(cloud_->width - 1), c + kw); v++)
                    {
                        int idx_uv = u * cloud_->width + v;
                        if(!slic.valid[idx_uv])
                            continue;

                        Eigen::Vector3f p_uv = cloud_->at(v,u).getVector3fMap();
                        if( (p_uv - p_rc).norm() >= 0.2f)
                            continue;

                        int label_uv = labels.at<int>(u,v);
                        if(label_uv != label)
                        {
                            adjacent[std::max(label,label_uv)][std::min(label, label_uv)] = 0;
                        }
                    }
                }
            }
        }
    }
    else
    {
        typename pcl::SupervoxelClustering<PointT> super (supervoxel_resolution_, supervoxel_seed_resolution_, false);
        super.setInputCloud (cloud_);
        super.setColorImportance (color_importance_);
        super.setSpatialImportance (spatial_importance_);
        super.setNormalImportance (normal_importance_);
        super.setNormalCloud(surface_normals_);
        std::map <uint32_t, typename pcl::Supervoxel<PointT>::Ptr > supervoxel_clusters;
        pcl::console::print_highlight ("Extracting supervoxels!\n");
        super.extract (supervoxel_clusters);
        super.refineSupervoxels(5, supervoxel_clusters);
        pcl::console::print_info ("Found %d supervoxels\n", supervoxel_clusters.size ());

        supervoxels_labels_cloud_ = super.getLabeledCloud();
        uint32_t max_label = super.getMaxLabel();

        std::cout << max_label << " " << supervoxels_labels_cloud_->isOrganized() << std::endl;

        supervoxels_rgb_.reset(new pcl::PointCloud<pcl::PointXYZRGBA>);
        supervoxels_rgb_ = super.getColoredCloud();

        label_to_idx.resize(max_label + 1, -1);
        typename std::map <uint32_t, typename pcl::Supervoxel<PointT>::Ptr>::iterator sv_itr,sv_itr_end;
        sv_itr = supervoxel_clusters.begin ();
        sv_itr_end = supervoxel_clusters.end ();
        int i=0;
        for ( ; sv_itr != sv_itr_end; ++sv_itr, i++)
        {
            label_to_idx[sv_itr->first] = i;
            //std::cout << sv_itr->first << " " << i << std::endl;
        }

        adjacent.resize(supervoxel_clusters.size());
        for(size_t i=0; i < (supervoxel_clusters.size()); i++)
            adjacent[i].resize(supervoxel_clusters.size(), -1);

        //#define VIS_SV
#ifdef VIS_SV
        boost::shared_ptr<pcl::visualization::PCLVisualizer> vis;
        vis.reset(new pcl::visualization::PCLVisualizer("super voxels"));
        vis->addPointCloud(supervoxels_rgb_, "labels");
#endif

        std::multimap<uint32_t, uint32_t> supervoxel_adjacency;
        super.getSupervoxelAdjacency (supervoxel_adjacency);
        //To make a graph of the supervoxel adjacency, we need to iterate through the supervoxel adjacency multimap
        std::multimap<uint32_t,uint32_t>::iterator label_itr = supervoxel_adjacency.begin ();
        std::cout << "super voxel adjacency size:" << supervoxel_adjacency.size() << std::endl;
        for ( ; label_itr != supervoxel_adjacency.end (); )
        {
            //First get the label
            uint32_t supervoxel_label = label_itr->first;

#ifdef VIS_SV
            typename pcl::PointCloud<PointT> adjacent_supervoxel_centers;
#endif
            std::multimap<uint32_t,uint32_t>::iterator adjacent_itr = supervoxel_adjacency.equal_range (supervoxel_label).first;
            for ( ; adjacent_itr!=supervoxel_adjacency.equal_range (supervoxel_label).second; ++adjacent_itr)
            {
                adjacent[ std::max (label_to_idx[supervoxel_label], label_to_idx[adjacent_itr->second]) ]
                        [ std::min (label_to_idx[supervoxel_label], label_to_idx[adjacent_itr->second]) ] = 0;

#ifdef VIS_SV
                typename pcl::Supervoxel<PointT>::Ptr neighbor_supervoxel = supervoxel_clusters.at (adjacent_itr->second);
                pcl::PointXYZRGB p;
                p.getVector3fMap() = neighbor_supervoxel->centroid_.getVector3fMap();
                p.rgb = neighbor_supervoxel->centroid_.rgb;
                adjacent_supervoxel_centers.push_back (p);
#endif
            }

#ifdef VIS_SV
            //Now we make a name for this polygon
            typename pcl::Supervoxel<PointT>::Ptr supervoxel = supervoxel_clusters.at (supervoxel_label);
            std::stringstream ss;
            ss << "supervoxel_" << supervoxel_label;

            pcl::PointXYZRGB p;
            p.getVector3fMap() = supervoxel->centroid_.getVector3fMap();
            p.rgb = supervoxel->centroid_.rgb;
            addSupervoxelConnectionsToViewer (p, adjacent_supervoxel_centers, ss.str (), vis);
#endif


            //Move iterator forward to next label
            label_itr = supervoxel_adjacency.upper_bound (supervoxel_label);
        }

#ifdef VIS_SV
        vis->spin();
#endif
    }
}

template<typename PointT>
void v4rOCTopDownSegmenter::SVMumfordShahPreSegmenter<PointT>::assertAdjacencyAndCandidateConsistency()
{
    for(int i=0; i < (int)merge_candidates_.size(); i++)
    {
        if(merge_candidates_[i].isValid())
        {
            int id1 = merge_candidates_[i].r1_->id_;
            int id2 = merge_candidates_[i].r2_->id_;

            int points_to = adjacent_[std::max(id1,id2)][std::min(id1,id2)];
            assert(i == points_to);
        }
    }
}

template<typename PointT>
void v4rOCTopDownSegmenter::SVMumfordShahPreSegmenter<PointT>::prepare(std::vector<int> & label_to_idx,
                                                                       std::vector<std::vector<int> > & sv_indices)
{
    //0. fill line of sights
    fillLineOfSights();
    cloud_z_.resize(cloud_->size());

    for(int r=0; r < (int)cloud_->height; r++)
    {
        for(int c=0; c < (int)cloud_->width; c++)
        {

            Eigen::Vector3f p = cloud_->at(c,r).getVector3fMap();

            cloud_z_(r*cloud_->width + c) = p.norm(); //p(2);

            if(!pcl_isfinite(p[2]))
                continue;

            if(supervoxels_labels_cloud_->at(c,r).label < 0 ||
                    supervoxels_labels_cloud_->at(c,r).label >= label_to_idx.size())
                continue;

            int idx = label_to_idx[supervoxels_labels_cloud_->at(c, r).label];

            if(idx < 0) //some points dont have an appropiate label apparently
                continue;

            assert(idx < sv_indices.size());
            sv_indices[idx].push_back(r * cloud_->width + c);
        }
    }
}

template<typename PointT>
void v4rOCTopDownSegmenter::SVMumfordShahPreSegmenter<PointT>::process()
{
    CURRENT_MODEL_TYPE_ = PLANAR_MODEL_TYPE_;

    scene_LAB_values_.resize(cloud_->points.size(), Eigen::Vector3d(-1,-1,-1));
    for(size_t i=0; i < cloud_->points.size(); i++)
    {
        Eigen::Vector3f p = cloud_->at(i).getVector3fMap();

        if(!pcl_isfinite(p[2]))
            continue;


        float LRefs, aRefs, bRefs;
        unsigned char rs, gs, bs;
        rs = cloud_->points[i].r;
        gs = cloud_->points[i].g;
        bs = cloud_->points[i].b;

        RGB2CIELAB (rs, gs, bs, LRefs, aRefs, bRefs);
        LRefs /= 100.0f; aRefs /= 120.0f; bRefs /= 120.0f;    //normalized LAB components (0<L<1, -1<a<1, -1<b<1)

        scene_LAB_values_[i] = (Eigen::Vector3d(LRefs, aRefs, bRefs));
    }

    if(save_impath_.compare("") != 0 && cloud_->isOrganized())
    {

        cv::Mat image(cloud_->height, cloud_->width, CV_8UC3);

        for(size_t i=0; i < cloud_->points.size(); i++)
        {
            float LRefs, aRefs, bRefs;
            unsigned char rs, gs, bs;
            rs = cloud_->points[i].r;
            gs = cloud_->points[i].g;
            bs = cloud_->points[i].b;

            int r,c;
            r = static_cast<int>(i) / cloud_->width;
            c = static_cast<int>(i) % cloud_->width;
            image.at<cv::Vec3b>(r,c) = cv::Vec3b(bs,gs,rs);
        }

        std::stringstream save_to;
        save_to << save_impath_ << "_color.png";
        cv::imwrite(save_to.str(), image);
    }

    //1. compute supervoxels
    std::vector<int> label_to_idx;
    adjacent_.clear ();
    overSegmentation(adjacent_, label_to_idx);

    //2. minimize MS functional

    //2.1 precompute information useful to merge regions
    //e.g. for plane fit, we will need the mean for each region

    //create regions
    std::vector<std::vector<int> > sv_indices;
    sv_indices.resize(adjacent_.size());

    prepare(label_to_idx, sv_indices);

    std::vector<boost::shared_ptr<Region<PointT> > > regions;
    regions.resize(adjacent_.size());

#pragma omp parallel for
    for(size_t i=0; i < adjacent_.size(); i++)
    {
        regions[i].reset(new Region<PointT>);
        regions[i]->id_ = (int)i;
        regions[i]->indices_ = sv_indices[i];

        //compute mean and covariance matrix for this regions
        computeMeanAndCovarianceMatrixSVMS(regions[i]->covariance_, regions[i]->mean_,
                                           regions[i]->accu_, regions[i]->indices_);

        //compute color mean
        computeColorMean(regions[i]->color_mean_, regions[i]->indices_);

        //add planar and color error for regions
        if(regions[i]->indices_.size() >= 3)
        {
            computePlaneError(regions[i]);
            //regions[i]->planar_error_ = regions[i]->computePlaneError(cloud_, los_);
        }
        else
        {
            regions[i]->planar_error_ = 0;
        }

        regions[i]->color_error_ = regions[i]->computeColorError(cloud_, scene_LAB_values_);

        //regions[i]->planar_error_ = 0; //individual regions have no error?
        std::cout << regions[i]->planar_error_ << " " << regions[i]->color_error_ << std::endl;
    }

    //initialize neighbours
    std::vector< std::vector< boost::shared_ptr<Region<PointT> > > > neighbours;
    neighbours.resize(sv_indices.size());

    for(size_t i=0; i < adjacent_.size(); i++)
    {
        for(size_t j=0; j < adjacent_[i].size(); j++)
        {
            if(adjacent_[i][j] >= 0)
            {
                neighbours[i].push_back(regions[j]);
            }
        }
    }

    //create merge candidates
    for(size_t i=0; i < sv_indices.size(); i++)
    {
        for(size_t j=0; j < neighbours[i].size(); j++)
        {
            int j_id = neighbours[i][j]->id_;
            if(j_id >= (int)i)
                continue;

            MergeCandidate<PointT> m;
            m.valid_ = true;
            m.r1_ = regions[i];
            m.r2_ = neighbours[i][j];

            if( !(m.r1_->indices_.size() >= 3 && m.r2_->indices_.size() >= 3))
                continue;

            //compute cost of merging these two regions
            //plane model (cost will be the distance from the points to the model)
            //m.planar_error_ = m.computePlaneError(cloud_, los_);
            computePlaneError(m);
            m.boundary_length_ = 0;

            //compute color error of merging these two regions
            m.color_error_ = m.computeColorError(cloud_, scene_LAB_values_);
            m.min_nyu_ = -1;

            adjacent_[i][j_id] = merge_candidates_.size();
            merge_candidates_.push_back(m);
        }
    }

    assertAdjacencyAndCandidateConsistency();

    /*for(size_t i=0; i < adjacent_.size(); i++)
    {
        for(size_t j=0; j < adjacent_.size(); j++)
            std::cout << adjacent_[i][j] << " ";

        std::cout << std::endl;
    }*/

    //2.2 extract super pixels boundaries and adjacency
    //done by going through the image and counting label changes
    for(size_t i=0; i < supervoxels_labels_cloud_->size(); i++)
    {
        if(supervoxels_labels_cloud_->at(i).label < 0 ||
                supervoxels_labels_cloud_->at(i).label >= label_to_idx.size())
            continue;

        supervoxels_labels_cloud_->at(i).label = label_to_idx[supervoxels_labels_cloud_->at(i).label];
    }

    projectRegionsOnImage(regions, "_1");

    //this also defines with merge_candidates will be considered
    int boundary_length = extractSVBoundaries();

    //initialize min_nyu with boundaries and model error
    float nyu = nyu_;

    /*for(size_t i=0; i < regions.size(); i++)
    {
        for(size_t j=0; j < regions.size(); j++)
        {
            merge_candidates_[i][j].min_nyu_ = -1;
        }
    }*/

    std::cout << regions.size() << " " << sv_indices.size() << std::endl;

    float total_cost = nyu * boundary_length;
    for(size_t i=0; i < regions.size(); i++)
    {
        if(regions[i]->valid_)
        {
            total_cost += regions[i]->getModelTypeError();
        }
    }

    int n_valid = 0;
    for(size_t i=0; i < merge_candidates_.size(); i++)
    {
        if(merge_candidates_[i].isValid())
        {
            int reg_i, reg_j;
            reg_i = merge_candidates_[i].r1_->id_;
            reg_j = merge_candidates_[i].r2_->id_;
            float p_err = regions[reg_i]->planar_error_ + regions[reg_j]->planar_error_ - merge_candidates_[i].planar_error_;
            float c_err = regions[reg_i]->color_error_ + regions[reg_j]->color_error_ - merge_candidates_[i].color_error_;

            merge_candidates_[i].min_nyu_ = -(alpha_ * p_err + sigma_ * c_err) / merge_candidates_[i].boundary_length_;
            n_valid++;
        }
        else
        {
            std::cout << "merge is not valid, why? extractSVBoundaries sets candidates to false if boundary length between regions is zero!" << std::endl;
        }
    }

    //    for(size_t i=0; i < sv_indices.size(); i++)
    //    {
    //        for(size_t j=0; j < sv_indices.size(); j++)
    //        {
    //            if(merge_candidates_[i][j].isValid())
    //            {
    //                float p_err = regions[i]->planar_error_ + regions[j]->planar_error_ - merge_candidates_[i][j].planar_error_;
    //                float c_err = regions[i]->color_error_ + regions[j]->color_error_ - merge_candidates_[i][j].color_error_;

    //                merge_candidates_[i][j].min_nyu_ = -(alpha_ * p_err + sigma_ * c_err) / merge_candidates_[i][j].boundary_length_;
    //                n_valid++;
    //            }
    //        }
    //    }

    std::cout << "Number of valid merges:" << n_valid << " " << merge_candidates_.size() << " " << sv_indices.size() * sv_indices.size() << std::endl;
    std::cout << "Cost:" << total_cost << std::endl;

    //regiong merging
    int iter=0;

#ifdef VIS_MERGE
    vis_merge.addPointCloud(cloud_);
#endif

    double tt_merge, tt_recompute;
    tt_merge = tt_recompute = 0.0;
    double tt_minimum = 0.0;

    pcl::StopWatch t;

    v4rOCTopDownSegmenter::StopWatch t_minimum, t_merge, t_recompute;
    t.reset();

    std::multiset< std::pair<int, float>, candidate_comp > sorted_merge_candidates;
    for(size_t i=0; i < adjacent_.size(); i++)
    {
        for(size_t j=0; j < adjacent_.size(); j++)
        {
            if(adjacent_[i][j] < 0)
                continue;

            if(merge_candidates_[adjacent_[i][j]].isValid())
            {
                sorted_merge_candidates.insert(
                            std::make_pair( (int)i * (int)adjacent_.size() + (int)j,
                                            merge_candidates_[adjacent_[i][j]].min_nyu_));

                //std::cout << "color error:" << merge_candidates_[adjacent_[i][j]].color_error_  << " " << merge_candidates_[adjacent_[i][j]].min_nyu_ << std::endl;
            }
        }
    }

    std::cout << "Finished insterting in multiset... " << adjacent_.size() << " " << sv_indices.size() << std::endl;

    typedef std::multiset< std::pair<int, float>, candidate_comp >::iterator ItMS;

    //ItMS it = sorted_merge_candidates.begin();
    //std::cout << it->first / (int)sv_indices.size() << " " << it->first % (int)sv_indices.size()  << " " << it->second << std::endl;

    int max_id = 0;
    for(size_t i=0; i < regions.size(); i++)
    {
        if(regions[i]->id_ > max_id)
        {
            max_id = regions[i]->id_;
        }
    }
    std::cout << sv_indices.size() << " " << adjacent_.size() << " " << regions.size() << " " << max_id << std::endl;

    while(true)
    {
        //what is the smallest nyu for all valid merge candidates
        int i_,j_;

        t_minimum.reset();

        //TODO: This will fail if two or more candidates have the same key... => FIX THIS!
        ItMS it = sorted_merge_candidates.begin();

        std::pair<ItMS,ItMS> range = sorted_merge_candidates.equal_range(std::make_pair(0, it->second));

        int elems_range=0;
        ItMS it_range;
        for (it_range = range.first; it_range != range.second; ++it_range)
            elems_range++;

        i_ = it->first / (int)adjacent_.size();
        j_ = it->first % (int)adjacent_.size();
        float min_nyu = it->second;

        if(elems_range > 1)
        {
            std::cout << "elements in range START:" << elems_range << std::endl;
            std::cout << min_nyu << std::endl;

            std::cout << sorted_merge_candidates.size() << std::endl;
            for (it_range = range.first; it_range != range.second; ++it_range)
            {
                int ii_ = it_range->first / (int)adjacent_.size();
                int jj_ = it_range->first % (int)adjacent_.size();
                std::cout << ii_ << " " << jj_ << " " << it_range->second << std::endl;
            }
        }

        /// remove this pair from candidate set
        sorted_merge_candidates.erase(sorted_merge_candidates.begin());

        tt_minimum += t_minimum.getTimeMicroSeconds();

        assert(i_ > j_);

        /// if not valid, ignore and move to next
        if(adjacent_[i_][j_] < 0 || !merge_candidates_[adjacent_[i_][j_]].isValid())
            continue;

        //std::cout << min_nyu << " " << i_ << " - " << j_ << " " << merge_candidates_[i_][j_].isValid() << std::endl;
        //std::cout << merge_candidates_[i_][j_].smoothness_ << " " << lambda_ << " " << min_nyu << std::endl;

        if(nyu < min_nyu)
        {
            if(CURRENT_MODEL_TYPE_ >= MAX_MODEL_TYPE_)
                break;
            else
            {
                CURRENT_MODEL_TYPE_++;
                std::cout << "CURRENT_MODEL_TYPE increased to: " << CURRENT_MODEL_TYPE_ << " " << sorted_merge_candidates.size() << std::endl;
                /// clear sorted candidates
                sorted_merge_candidates.clear();

                //do i need to try to replace single regions with current model type fits?
                /// compute merge costs for current model type and for all valid regions
                int valid_merges = 0;
                for(size_t i=0; i < sv_indices.size(); i++)
                {
                    for(size_t j=0; j < sv_indices.size(); j++)
                    {
                        if(adjacent_[i][j] < 0)
                            continue;

                        if(merge_candidates_[adjacent_[i][j]].isValid())
                        {
                            //float previous_error = merge_candidates_[adjacent_[i][j]].getModelTypeError();

                            if(CURRENT_MODEL_TYPE_ == BSPLINE_MODEL_TYPE_3x3)
                            {
                                //merge_candidates_[adjacent_[i][j]].bspline_error_ = merge_candidates_[adjacent_[i][j]].computeBsplineError(cloud_z_, cloud_->width);
                                computeBSplineError(merge_candidates_[adjacent_[i][j]], 3, 3, 3);
                            }
                            else if(CURRENT_MODEL_TYPE_ == BSPLINE_MODEL_TYPE_5x5)
                            {
                                computeBSplineError(merge_candidates_[adjacent_[i][j]], 3, 5, 5);
                                //merge_candidates_[adjacent_[i][j]].bspline_error_ = merge_candidates_[adjacent_[i][j]].computeBsplineError(cloud_z_, cloud_->width, 3, 5, 5);
                            }
                            //std::cout << merge_candidates_[i][j].bspline_error_ << " " << merge_candidates_[i][j].planar_error_ << std::endl;

                            merge_candidates_[adjacent_[i][j]].increaseModelType();

                            /// re-insert candidates after evaluating cost with new model
                            /// at this point, i am trying to replace two planar regions with one bspline, so:

                            float incr_regional_fit = (regions[i]->getModelTypeError() + regions[j]->getModelTypeError() - merge_candidates_[adjacent_[i][j]].getModelTypeError());
                            float incr_smoothness = (regions[i]->mComp() + regions[j]->mComp() - merge_candidates_[adjacent_[i][j]].mComp());
                            float c_err = regions[i]->color_error_ + regions[j]->color_error_ - merge_candidates_[adjacent_[i][j]].color_error_;

                            merge_candidates_[adjacent_[i][j]].min_nyu_ = (-alpha_ * incr_regional_fit - lambda_ * incr_smoothness - sigma_ * c_err) / merge_candidates_[adjacent_[i][j]].boundary_length_;

                            sorted_merge_candidates.insert(
                                        std::make_pair( (int)i * (int)sv_indices.size() + (int)j,
                                                        merge_candidates_[adjacent_[i][j]].min_nyu_));

                            valid_merges++;

                            //float current_error = merge_candidates_[adjacent_[i][j]].getModelTypeError();
                            //std::cout << current_error - previous_error << std::endl;
                        }
                    }
                }

                ItMS it = sorted_merge_candidates.begin();
                std::cout << "valid merges:" << valid_merges << " " << sorted_merge_candidates.size() << " " << CURRENT_MODEL_TYPE_ << std::endl;
                std::cout << "min nyu:" << it->second << " " << nyu_ << std::endl;

                projectRegionsOnImage(regions, "_2");
                //visualizeSegmentation(regions);

                continue;
            }
        }

        //std::cout << "iteration:" << iter << std::endl;
        //assertAdjacencyAndCandidateConsistency();

        /// apply merges with cost smaller or equal than min_nyu
        t_merge.reset();
        std::vector< std::pair<int, int> > recompute_candidates;
        merge_candidates_[adjacent_[i_][j_]].valid_ = false;
        int adj = adjacent_[i_][j_];
        adjacent_[i_][j_] = -1;

        float region_i_error = regions[i_]->getModelTypeError();
        float region_j_error = regions[j_]->getModelTypeError();

        //after merging, the regional information is not valid anymore
        if(merge_candidates_[adj].r1_->id_ != i_)
        {
            std::cout << merge_candidates_[adj].r1_->id_ << " " << i_ << std::endl;
            std::cout << merge_candidates_[adj].r2_->id_ << " " << j_ << std::endl;
        }

        /*assert(merge_candidates_[adj].r1_->id_ == i_);
        assert(merge_candidates_[adj].r2_->id_ == j_);*/

        merge_candidates_[adj].merge(adjacent_, merge_candidates_, recompute_candidates);

#ifdef VIS_MERGE
        std::cout << merge_candidates_[adj].boundary_length_ << std::endl;
        visMerge(merge_candidates_[adj], cloud_);
#endif

        //std::cout << merge_candidates_[i_][j_].planar_error_ << " " << merge_candidates_[i_][j_].boundary_length_ << " " << nyu << std::endl;
        //total cost gets increased by planar error
        float data_error_increase = (merge_candidates_[adj].getModelTypeError() - region_i_error - region_j_error);
        total_cost = total_cost + data_error_increase - nyu * merge_candidates_[adj].boundary_length_;

        tt_merge += t_merge.getTimeMicroSeconds();


        /// recompute costs for recompute_candidates pairs and set them to valid
        /// compute model error in parallel
        t_recompute.reset();

        //#pragma omp parallel for schedule(dynamic)
        for(size_t i = 0; i < recompute_candidates.size(); i++)
        {
            int ii, jj;
            ii = recompute_candidates[i].first;
            jj = recompute_candidates[i].second;

            assert(ii > jj);
            if(!(ii == i_ || jj == i_))
            {
                std::cout << ii << " " << jj << " " << i_ << " " << j_ << std::endl;
                //assert(ii == i_ || jj == i_);
            }

            MergeCandidate<PointT> m;
            m.valid_ = true;
            m.r1_ = regions[ii];
            m.r2_ = regions[jj];

            if(CURRENT_MODEL_TYPE_ == PLANAR_MODEL_TYPE_)
            {
                //m.planar_error_ = m.computePlaneError(cloud_, los_);
                computePlaneError(m);
            }
            else if(CURRENT_MODEL_TYPE_ == BSPLINE_MODEL_TYPE_3x3)
            {

                //m.bspline_error_ = m.computeBsplineError(cloud_z_, cloud_->width);
                computeBSplineError(m, 3, 3, 3);
                if(m.getModelType() < CURRENT_MODEL_TYPE_)
                {
                    m.increaseModelType();
                }
            }
            else if(CURRENT_MODEL_TYPE_ == BSPLINE_MODEL_TYPE_5x5)
            {
                //m.bspline_error_ = m.computeBsplineError(cloud_z_, cloud_->width, 3, 5, 5);
                computeBSplineError(m, 3, 5, 5);
                if(m.getModelType() < CURRENT_MODEL_TYPE_)
                {
                    m.increaseModelType();
                }
            }


            m.color_error_ = m.computeColorError(cloud_, scene_LAB_values_);

            //add additional boundary length to the merge candidate (coming from j_ and the other regions)
            int new_region = jj;
            if(ii != i_)
                new_region = ii;

            int bj_new, bi_new;
            bj_new = bi_new = 0;

            if(adjacent_[std::max(new_region, j_)][std::min(new_region, j_)] >= 0)
            {
                bj_new = merge_candidates_[adjacent_[std::max(new_region, j_)][std::min(new_region, j_)]].boundary_length_;
            }

            if(adjacent_[std::max(new_region, i_)][std::min(new_region, i_)] >= 0)
            {
                bi_new = merge_candidates_[adjacent_[std::max(new_region, i_)][std::min(new_region, i_)]].boundary_length_;
            }

            assert(bj_new != 0 || bi_new != 0);

            m.boundary_length_
                    =  bj_new + bi_new;

            float p_err = (regions[ii]->getModelTypeError() + regions[jj]->getModelTypeError() - m.getModelTypeError());
            float c_err = (regions[ii]->color_error_ + regions[jj]->color_error_ - m.color_error_);
            float smoothness_err = regions[ii]->mComp() + regions[jj]->mComp() - m.mComp();

            m.min_nyu_ = (-(alpha_ * p_err + lambda_ * smoothness_err + sigma_ * c_err)) / m.boundary_length_;

            if(adjacent_[ii][jj] >= 0)
            {
                //the move existed already

                float previous_nyu = merge_candidates_[adjacent_[ii][jj]].min_nyu_;

                //remove and insert to candidate set
                std::pair<ItMS,ItMS> range = sorted_merge_candidates.equal_range(std::make_pair(0, previous_nyu));
                ItMS it;
                ItMS it_found;

                int idx = ii * adjacent_.size() + jj;
                bool found = false;
                int elems_range=0;
                for (it = range.first; it != range.second; ++it)
                {
                    elems_range++;
                    if(it->first == idx)
                    {
                        found = true;
                        it_found = it;
                    }
                }

                if(found)
                    sorted_merge_candidates.erase(it_found);
            }

            sorted_merge_candidates.insert(std::make_pair(ii * (int)adjacent_.size() + jj,
                                                          m.min_nyu_));

            adjacent_[ii][jj] = merge_candidates_.size();
            merge_candidates_.push_back(m);
        }

        tt_recompute +=t_recompute.getTimeMicroSeconds();

        //invalidate adjacents for r2 (j_), j_^th row and j_^th column
        /*for(int j=0; j < adjacent_.size(); j++)
        {
            adjacent_[j_][j] = -1;
        }

        for(int j=0; j < adjacent_.size(); j++)
        {
            adjacent_[j][j_] = -1;
        }*/

        /// update sorted list
        t_minimum.reset();
        tt_minimum += t_minimum.getTimeMicroSeconds();

        if(iter % 100 == 0)
            std::cout << "MS cost:" << total_cost << std::endl;

        //if(vis_at_each_move_)
        //visualizeSegmentation(regions);

        //projectRegionsOnImage(regions, "_1");
        iter++;
    }

    double ms = t.getTime();
    std::cout << "Converged after" << iter << " iterations, in " << ms << " ms" << std::endl;
    std::cout << " tmerge: " << tt_merge / 1000 << " ms trecompute: " << tt_recompute / 1000 << " ms tminimum:" << tt_minimum / 1000.0 << " ms" << std::endl;

    //visualizeSegmentation(regions);

#ifdef VIS_CC
    visualizeRegions(regions);
#endif

    projectRegionsOnImage(regions, "_3");

    for(size_t r = 0; r < regions.size(); r++)
    {
        if(!regions[r]->valid_)
            continue;

        for(size_t i=0; i < regions[r]->indices_.size(); i++)
        {
            supervoxels_labels_cloud_->at(regions[r]->indices_[i]).label = (uint32_t)regions[r]->id_;
        }
    }

    //extractSVBoundaries();

    /// pixelwise refinement (fix erroneous oversegmented boundaries)
    if(pixelwise_refinement_)
    {
        if(!cloud_->isOrganized())
        {
            std::cout << "!cloud_->isOrganized()" << std::endl;
            vis_segmentation_->removeAllPointClouds();
            visualizeSegmentation(regions, 0);
            vis_segmentation_->spin();
        }

        refinement(regions);

        if(!cloud_->isOrganized())
        {
            vis_segmentation_->removeAllPointClouds();
            visualizeSegmentation(regions, 0);
            vis_segmentation_->spin();
        }
    }

#ifdef VIS_CC
    visualizeRegions(regions);
#endif
    projectRegionsOnImage(regions, "_4");

    //int v1,v2;
    //vis_segmentation_->createViewPort(0,0,1,0.5,v1);
    //vis_segmentation_->createViewPort(0,0.5,1,1,v2);
    vis_segmentation_->removeAllPointClouds();
    vis_segmentation_->setBackgroundColor(1,1,1);
    //visualizeRegions(regions, v1);
    visualizeSegmentation(regions, 0);
    vis_segmentation_->spin();

    vis_segmentation_->removeAllPointClouds();
    visualizeRegions(regions, 0);
    vis_segmentation_->spin();

    createLabelCloud(regions);
}

template<typename PointT>
void v4rOCTopDownSegmenter::SVMumfordShahPreSegmenter<PointT>::refinement(std::vector<boost::shared_ptr<Region<PointT> > > & regions)
{
    //for pixels at the boundary, we will do moves that swap the region membership that minimize the MS cost
    //this aims at reducing wiggles at the boundary between regions as well as data error
    //once a move is accepted, then update edges closeby

    //for each pixel, we can have multiple PixelMoves
    std::vector<std::vector<PixelMove> > pixel_moves;
    pixel_moves.resize(cloud_->points.size());
    fillPixelMoves(pixel_moves, 0, 0, cloud_->height, cloud_->width);

    for(int i=0; i < regions.size(); i++)
    {
        if(regions[i]->valid_)
        {
            std::set<int> ii(regions[i]->indices_.begin(), regions[i]->indices_.end());
            regions[i]->indices_set_ = ii;
        }
    }

    v4rOCTopDownSegmenter::StopWatch t1;
    double t1_ = 0;

    int iter_refinement = 0;

    typedef std::pair<int, int> pixel_move_with_id;

    //std::multiset< std::pair<pixel_move_with_id, float>, candidate_comp > improving_moves_multiset;

    t1.reset();

    std::vector<pixel_move_with_id> moves_improving;
    moves_improving.resize(cloud_->height * cloud_->width);

    while(true)
    {

        //pcl::ScopeTime ttt("one refinement iteration");

        int improving_moves = 0;
        int total_pms = 0;
        int recomputed_moves = 0;

        int kw = 1;

        for(int r=0; r < (int)cloud_->height; ++r)
        {
            for(int c=0; c < (int)cloud_->width; ++c)
            {
                int idx = r * cloud_->width + c;
                if(pixel_moves[idx].size() == 0)
                    continue;

                if(!pixel_moves[idx][0].recompute_)
                {
                    for(size_t i=0; i < pixel_moves[idx].size(); i++)
                    {
                        if(pixel_moves[idx][i].improvement_ < 0)
                        {
                            moves_improving[improving_moves].first = idx;
                            moves_improving[improving_moves].second = i;
                            improving_moves++;

                        }

                        total_pms++;
                    }

                    continue;
                }

                for(size_t i=0; i < pixel_moves[idx].size(); i++)
                {
                    PixelMove & pm = pixel_moves[idx][i];

                    assert(regions[pm.r1_]->valid_ && regions[pm.r2_]->valid_);

                    int cur_region = pm.current_region_;

                    assert(cur_region == pm.r1_ || cur_region == pm.r2_);

                    int swap_region = pm.r2_;
                    if(pm.r1_ != cur_region)
                    {
                        swap_region = pm.r1_;
                        cur_region = pm.r2_;
                    }

                    ///compute the data error of this pixel if it belongs to either region
                    float err_curr = regions[cur_region]->getModelErrorForPoint(cloud_, idx, los_);
                    float err_if_swap = regions[swap_region]->getModelErrorForPoint(cloud_, idx, los_);

                    /*float cerr_curr = regions[cur_region]->getColorErrorForPoint(scene_LAB_values_, idx);
                    float cerr_if_swap = regions[swap_region]->getColorErrorForPoint(scene_LAB_values_, idx);*/

                    //if error delta is negative, swaping is better
                    pm.data_error_delta_ = 0;
                    pm.data_error_delta_ = err_if_swap - err_curr;
                    //pm.color_error_delta_ = cerr_if_swap - cerr_curr;

                    ///compute the boundary length in the NN if pixel belongs to either region
                    uint32_t label = cur_region; //no swap
                    int boundary_cur = 0;
                    int boundary_swap = 0;

                    for(int u=std::max(0, r - kw); u <= std::min((int)(cloud_->height - 1), r + kw); u++)
                    {
                        for(int v=std::max(0, c - kw); v <= std::min((int)(cloud_->width - 1), c + kw); v++)
                        {
                            uint32_t label_uv = supervoxels_labels_cloud_->at(v,u).label;
                            Eigen::Vector3f p_uv = cloud_->at(v,u).getVector3fMap();
                            if(!pcl_isfinite(p_uv[2]))
                                continue;

                            //check if label_uv and label are adjacent
                            if(label >= merge_candidates_.size() || label_uv >= merge_candidates_.size())
                                continue;

                            if(adjacent_[std::max(label, label_uv)][std::min(label, label_uv)] < 0)
                                continue;

                            if(!(merge_candidates_[adjacent_[std::max(label, label_uv)][std::min(label, label_uv)]].isValid()))
                                continue;

                            if(label_uv != label)
                            {
                                boundary_cur++;
                            }
                        }
                    }

                    label = swap_region; //no swap
                    for(int u=std::max(0, r - kw); u <= std::min((int)(cloud_->height - 1), r + kw); u++)
                    {
                        for(int v=std::max(0, c - kw); v <= std::min((int)(cloud_->width - 1), c + kw); v++)
                        {
                            if(v == c && r == u) //do not count the same pixel
                                continue;

                            uint32_t label_uv = supervoxels_labels_cloud_->at(v,u).label;
                            Eigen::Vector3f p_uv = cloud_->at(v,u).getVector3fMap();
                            if(!pcl_isfinite(p_uv[2]))
                                continue;

                            //check if label_uv and label are adjacent

                            if(label >= adjacent_.size() || label_uv >= adjacent_.size())
                                continue;

                            int ri, rj;
                            ri = std::max(label, label_uv);
                            rj = std::min(label, label_uv);

                            if(adjacent_[ri][rj] < 0)
                                continue;

                            if(!merge_candidates_[adjacent_[ri][rj]].isValid())
                                continue;

                            /*if(label >= merge_candidates_.size() || label_uv >= merge_candidates_.size())
                                continue;

                            if(!(merge_candidates_[std::max(label, label_uv)][std::min(label, label_uv)].isValid()))
                                continue;*/

                            if(label_uv != label)
                            {
                                boundary_swap++;
                            }
                        }
                    }

                    pm.boundary_length_delta_ = (boundary_swap - boundary_cur);

                    ///improvement by swapping pixel
                    float improvement = alpha_ * pm.data_error_delta_ + sigma_ * pm.color_error_delta_ + nyu_ * pm.boundary_length_delta_;
                    pm.improvement_ = improvement;
                    pm.recompute_ = false;

                    if(improvement < 0)
                    {
                        moves_improving[improving_moves].first = idx;
                        moves_improving[improving_moves].second = i;
                        improving_moves++;

                    }

                    total_pms++;
                    recomputed_moves++;
                }

                //t4_ += t4.getTimeMicroSeconds();
            }
        }

        ///apply best improving move
        //update supervoxels_label_cloud
        //update indices for r1 and r2 of the moves

        //            if( (iter_refinement % 50) == 0)
        //            {
        //                std::cout << "Number of improving moves:" << improving_moves << " recomputed:" << recomputed_moves << " total:" << total_pms << std::endl;
        //            }


        if(improving_moves > 0)
        {


            int best_cost = 0;
            int best_i = 0;

            for(int i=0; i < improving_moves; i++)
            {
                PixelMove & pm = pixel_moves[moves_improving[i].first][moves_improving[i].second];
                if(pm.improvement_ < best_cost)
                {
                    best_i = i;
                    best_cost = pm.improvement_;
                }
            }

            pixel_move_with_id & best_move = moves_improving[best_i];
            PixelMove & pm = pixel_moves[best_move.first][best_move.second];

            //std::cout << "improvement:" << pm.improvement_ << std::endl;

            int cur_region = pm.current_region_;
            int swap_region = pm.r2_;
            if(pm.r1_ != cur_region)
            {
                swap_region = pm.r1_;
                cur_region = pm.r2_;
            }

            int idx_being_swapped = pm.idx_;
            regions[cur_region]->indices_set_.erase(idx_being_swapped);
            regions[swap_region]->indices_set_.insert(idx_being_swapped);
            supervoxels_labels_cloud_->at(idx_being_swapped).label = swap_region;

            int kw_recompute=2;
            int r, c;
            r = idx_being_swapped / cloud_->width;
            c = idx_being_swapped % cloud_->width;

            int rs,cs,rend,cend;
            rs = std::max(0, r - kw_recompute);
            cs = std::max(0, c - kw_recompute);

            rend = std::min((int)cloud_->height, r + kw_recompute);
            cend = std::min((int)cloud_->width, c + kw_recompute);

            for(int rr=rs; rr < rend; rr++)
            {
                for(int cc=cs; cc < cend; cc++)
                {
                    int idx = rr * cloud_->width + cc;
                    pixel_moves[idx].clear();
                }
            }

            fillPixelMoves(pixel_moves, rs, cs, rend, cend);
        }
        else
        {
            break;
        }

        iter_refinement++;
    }

    t1_ += t1.getTimeMicroSeconds();

    std::cout << t1_ / 1000.0 << " miliseconds" << std::endl;

    for(size_t i=0; i < regions.size(); i++)
    {
        if(regions[i]->valid_)
        {
            regions[i]->indices_.clear();
            std::copy(regions[i]->indices_set_.begin(), regions[i]->indices_set_.end(), std::back_inserter(regions[i]->indices_));
        }
    }
}

template<typename PointT>
void v4rOCTopDownSegmenter::SVMumfordShahPreSegmenter<PointT>::fillPixelMoves(std::vector<std::vector<PixelMove> > & pixel_moves,
                                                                              int rs, int cs, int rend, int cend)
{
    int kw = 1;

    for(int r=rs; r < rend; r++)
    {
        for(int c=cs; c < cend; c++)
        {

            Eigen::Vector3f p = cloud_->at(c,r).getVector3fMap();
            if(!pcl_isfinite(p[2]))
                continue;

            bool is_boundary = false;
            uint32_t label = supervoxels_labels_cloud_->at(c,r).label;

            std::set<uint32_t> labels;
            for(int u=std::max(0, r - kw); u <= std::min((int)(cloud_->height - 1), r + kw); u++)
            {
                for(int v=std::max(0, c - kw); v <= std::min((int)(cloud_->width - 1), c + kw); v++)
                {

                    //4-neighborhood
                    /*if(u != r && v != c)
                        continue;*/

                    uint32_t label_uv = supervoxels_labels_cloud_->at(v,u).label;
                    Eigen::Vector3f p_uv = cloud_->at(v,u).getVector3fMap();
                    if(!pcl_isfinite(p_uv[2]))
                        continue;

                    //check if label_uv and label are adjacent
                    /*if(label >= merge_candidates_.size() || label_uv >= merge_candidates_.size())
                        continue;

                    if(!(merge_candidates_[std::max(label, label_uv)][std::min(label, label_uv)].isValid()))
                        continue;*/

                    //check if label_uv and label are adjacent
                    if(label >= adjacent_.size() || label_uv >= adjacent_.size())
                        continue;

                    int ri, rj;
                    ri = std::max(label, label_uv);
                    rj = std::min(label, label_uv);

                    if(adjacent_[ri][rj] < 0)
                        continue;

                    if(!merge_candidates_[adjacent_[ri][rj]].isValid())
                        continue;

                    if(label_uv != label)
                    {
                        is_boundary = true;
                    }

                    labels.insert(label_uv);

                }
            }

            if(is_boundary)
            {
                for(std::set<uint32_t>::iterator it = labels.begin(); it != labels.end(); it++)
                {
                    if(label >= adjacent_.size() || *it >= adjacent_.size())
                        continue;

                    int idx = r * cloud_->width + c;
                    PixelMove pm;
                    pm.valid_ = true;
                    pm.r1_ = std::max(label, *it);
                    pm.r2_ = std::min(label, *it);
                    pm.current_region_ = label;
                    pm.idx_ = idx;
                    pm.recompute_ = true;
                    pixel_moves[idx].push_back(pm);
                }
            }
        }
    }
}


//template<typename PointT> float v4rOCTopDownSegmenter::SVMumfordShahPreSegmenter<PointT>::sRGB_LUT[256] = {- 1};
//template<typename PointT> float v4rOCTopDownSegmenter::SVMumfordShahPreSegmenter<PointT>::sXYZ_LUT[4000] = {- 1};

template class v4rOCTopDownSegmenter::SVMumfordShahPreSegmenter<pcl::PointXYZRGB>;
//template class v4rOCTopDownSegmenter::SVMumfordShahPreSegmenter<pcl::PointXYZRGBA>;
