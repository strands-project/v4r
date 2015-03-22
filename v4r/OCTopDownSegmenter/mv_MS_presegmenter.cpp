#include "mv_MS_presegmenter.h"
#include <pcl/segmentation/supervoxel_clustering.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <opencv2/opencv.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/common/time.h>
#include <vtkPolyLine.h>
#include "v4r/on_nurbs/fitting_surface_depth_im.h"
#include <numeric>
#include <pcl/common/pca.h>

template<typename PointT>
v4rOCTopDownSegmenter::MVMumfordShahPreSegmenter<PointT>::MVMumfordShahPreSegmenter()
{
    boundary_radius_ = 0.015;
}

template<typename PointT>
v4rOCTopDownSegmenter::MVMumfordShahPreSegmenter<PointT>::~MVMumfordShahPreSegmenter()
{

}

template<typename PointT>
void v4rOCTopDownSegmenter::MVMumfordShahPreSegmenter<PointT>::visualizeSegmentation
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

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cc(new pcl::PointCloud<pcl::PointXYZRGB>(*cloud_));
    Eigen::Vector3f origin = Eigen::Vector3f::Zero();
    std::vector<int> valid(cloud_->points.size(), false);

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
            Eigen::Vector3f nn = regions[i]->planar_model_.first;

            for(size_t j=0; j < regions[i]->indices_.size(); j++)
            {

                //project point to plane
                Eigen::Vector3f p = cloud_cc->at(regions[i]->indices_[j]).getVector3fMap();
                Eigen::Vector3f v = p - point_on_plane;
                float dist = v.dot(nn);
                Eigen::Vector3f intersect = p - dist * nn;

                cloud_cc->at(regions[i]->indices_[j]).getVector3fMap() = intersect;
            }
        }
        else if(regions[i]->current_model_type_ == BSPLINE_MODEL_TYPE_3x3 && regions[i]->bspline_model_defined_)
        {
            //change the position of the 3D points

            pcl::IndicesPtr ind;
            ind.reset(new std::vector<int>(regions[i]->indices_));

            pcl::PCA<PointT> basis;
            basis.setInputCloud(cloud_);
            basis.setIndices(ind);

            // #################### PLANE PROJECTION #########################
            typename pcl::PointCloud<PointT>::Ptr proj_cloud(new pcl::PointCloud<PointT>);
            basis.project(pcl::PointCloud<PointT>(*cloud_,*ind), *proj_cloud);

            ON_NurbsSurface& bspline = regions[i]->bspline_model_;

            std::cout << "Bspline: order: " << bspline.Order(0) << " x " << bspline.Order(1);
            std::cout << " CP: " << bspline.CVCount(0) << " x " << bspline.CVCount(1) << std::endl;


            // evaluate error and curvature for each point
            Eigen::VectorXd curvatures(proj_cloud->size(), 1);
            int nder = 2; // number of derivatives
            int nvals = bspline.Dimension()*(nder+1)*(nder+2)/2;
            double P[nvals];
            Eigen::Vector3d n, xu, xv, xuu, xvv, xuv;
            Eigen::Matrix2d II;
            double c_min(DBL_MAX), c_max(DBL_MIN);
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

              double& c = curvatures[j];
              c = sqrt(II(0,0)*II(0,0) + II(1,1)*II(1,1)); // norm of principal curvatures

              if(c<c_min)
                c_min = c;
              if(c>c_max)
                c_max = c;

              pcl::PointNormal pp;
              pp.getVector3fMap() = cloud_cc->at(regions[i]->indices_[j]).getVector3fMap();
              pp.curvature = c;
              curvature_cloud->push_back(pp);

            }

            /*for (std::size_t j = 0; j < proj_cloud->size(); j++)
            {
              pcl::PointXYZRGB& p_cc = cloud_cc->at(regions[i]->indices_[j]);
              p_cc.r = static_cast<uint8_t>( 255*(curvatures[j] - c_min) / (c_max - c_min) );
              p_cc.g = 0;
              p_cc.b = 255 - p_cc.r;
            }*/


//            double z;
//            for (std::size_t j = 0; j < proj_cloud->size(); j++)
//            {
//                regions[i]->bspline_model_.Evaluate (proj_cloud->at(j).x, proj_cloud->at(j).y, 0, 1, &z);

//                PointT p1, p2;
//                p1.x = proj_cloud->at(j).x;
//                p1.y = proj_cloud->at(j).y;
//                p1.z = z;

//                basis.reconstruct(p1,p2);
//                cloud_cc->at(regions[i]->indices_[j]).getVector3fMap() = p2.getVector3fMap();
//            }

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
            }*/
        }
    }

    int non_valid = 0;
    for(size_t i=0; i < cloud_cc->points.size(); i++)
    {
        if(!valid[i])
        {
            cloud_cc->points[i].x = cloud_cc->points[i].y = cloud_cc->points[i].z = std::numeric_limits<float>::quiet_NaN();
            non_valid++;
        }
    }

    if(non_valid > 0)
    {
        cloud_cc->is_dense = false;
    }

    {
        //visualize curvature
        int v1,v2;
        pcl::visualization::PCLVisualizer vis_curv("curvature");
        vis_curv.createViewPort(0,0,0.5,1,v1);
        vis_curv.createViewPort(0.5,0,1,1,v2);
        pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointNormal> handler(curvature_cloud, "curvature");
        vis_curv.addPointCloud(curvature_cloud, handler, "curvature", v2);
        vis_curv.addPointCloud(cloud_, "test", v1);
        vis_curv.spin();
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

    //vis_segmentation_->spin();
}

template<typename PointT>
void v4rOCTopDownSegmenter::MVMumfordShahPreSegmenter<PointT>::visualizeRegions
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

/// END VISUALIZATION FUNCTIONS

template<typename PointT>
void v4rOCTopDownSegmenter::MVMumfordShahPreSegmenter<PointT>::prepare(std::vector<int> & label_to_idx,
                                                                       std::vector<std::vector<int> > & sv_indices)
{
    for(int r=0; r < (int)cloud_->points.size(); r++)
    {
        int idx = label_to_idx[supervoxels_labels_cloud_->at(r).label];

        if(idx < 0) //WHY is this happening?
            continue;

        assert(idx < sv_indices.size());
        sv_indices[idx].push_back(r);
    }

    //create octree required for extractSVBoundaries...
    octree_.reset(new pcl::octree::OctreePointCloudSearch<PointT> (supervoxel_resolution_));
    octree_->setInputCloud(cloud_);
    octree_->addPointsFromInputCloud();
}

template<typename PointT>
int v4rOCTopDownSegmenter::MVMumfordShahPreSegmenter<PointT>::extractSVBoundaries()
{
    int n_boundaries = 0;
    pcl::PointCloud<pcl::PointXYZ>::Ptr boundary_points(new pcl::PointCloud<pcl::PointXYZ>);

    //for each point in the cloud, detect those that are at the boundary by looking at the labels of the point
    //in a neighbourd
    std::vector<int> pointIdxNKNSearch;
    std::vector<float> pointNKNSquaredDistance;
    float radius_search = boundary_radius_;

    for(int r=0; r < (int)cloud_->points.size(); r++)
    {
        PointT p = cloud_->at(r);

        bool is_boundary = false;
        uint32_t label = supervoxels_labels_cloud_->at(r).label;

        std::set<uint32_t> labels;
        if(octree_->radiusSearch (p, radius_search, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
        {
            for(size_t i=0; i < pointIdxNKNSearch.size(); i++)
            {
                //compare labels
                uint32_t label_uv = supervoxels_labels_cloud_->at(pointIdxNKNSearch[i]).label;

                //check if label_uv and label are adjacent
                if(label >= adjacent_.size() || label_uv >= adjacent_.size())
                {
                    continue;
                }

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
            //std::cout << labels.size() << std::endl;
            for(std::set<uint32_t>::iterator it = labels.begin(); it != labels.end(); it++)
            {
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
            }

            n_boundaries++;
            pcl::PointXYZ p_bound;
            p_bound.getVector3fMap() = p.getVector3fMap();
            boundary_points->push_back(p_bound);
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

    pcl::visualization::PCLVisualizer vis_boundary("boundaries");
    vis_boundary.addPointCloud(cloud_);

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler(boundary_points, 255, 0, 0);
    vis_boundary.addPointCloud<pcl::PointXYZ>(boundary_points, handler, "BOUNDARIES");
    vis_boundary.addCoordinateSystem(0.3f);
    vis_boundary.spin();

    return n_boundaries;
}

struct Generator {
    Generator() : m_value( 0 ) { }
    int operator()() { return m_value++; }
    int m_value;
};

template<typename PointT>
void v4rOCTopDownSegmenter::MVMumfordShahPreSegmenter<PointT>::refinement(std::vector<boost::shared_ptr<Region<PointT> > > & regions)
{
    //for pixels at the boundary, we will do moves that swap the region membership that minimize the MS cost
    //this aims at reducing wiggles at the boundary between regions as well as data error
    //once a move is accepted, then update edges closeby

    //for each pixel, we can have multiple PixelMoves
    std::vector<std::vector<PixelMove> > pixel_moves;
    pixel_moves.resize(cloud_->points.size());
    std::vector<int> indices_to_compute(cloud_->size());

    std::generate( indices_to_compute.begin(), indices_to_compute.end(), Generator() );
    //std::iota(indices_to_compute.begin(), indices_to_compute.end(), 0); //c11

    fillPixelMoves(pixel_moves, indices_to_compute, false);

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

    t1.reset();

    std::vector<pixel_move_with_id> moves_improving;
    moves_improving.resize(cloud_->size());

    while(true)
    {

        //pcl::ScopeTime ttt("one refinement iteration");

        int improving_moves = 0;
        int total_pms = 0;
        int recomputed_moves = 0;

        for(int idx=0; idx < (int)cloud_->size(); ++idx)
        {
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
                float err_curr = regions[cur_region]->getModelErrorForPointUnorganized(cloud_, idx);
                float err_if_swap = regions[swap_region]->getModelErrorForPointUnorganized(cloud_, idx);

                /*float cerr_curr = regions[cur_region]->getColorErrorForPoint(scene_LAB_values_, idx);
                float cerr_if_swap = regions[swap_region]->getColorErrorForPoint(scene_LAB_values_, idx);*/

                //if error delta is negative, swaping is better
                pm.data_error_delta_ = 0;
                pm.data_error_delta_ = err_if_swap - err_curr;
                //pm.color_error_delta_ = cerr_if_swap - cerr_curr;
                pm.color_error_delta_ = 0;

                ///compute the boundary length in the neighbourhood if pixel belongs to either region
                int boundary_cur = 0;
                int boundary_swap = 0;

                std::vector<int> pointIdxNKNSearch;
                std::vector<float> pointNKNSquaredDistance;
                float radius_search = boundary_radius_;
                PointT p = cloud_->at(idx);

                uint32_t label = cur_region; //no swap
                if(octree_->radiusSearch (p, radius_search, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
                {
                    for(size_t uu=1; uu < pointIdxNKNSearch.size(); uu++)
                    {
                        uint32_t label_uv = supervoxels_labels_cloud_->at(pointIdxNKNSearch[uu]).label;

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

                    label = swap_region; //swap

                    for(size_t uu=1; uu < pointIdxNKNSearch.size(); uu++)
                    {
                        uint32_t label_uv = supervoxels_labels_cloud_->at(pointIdxNKNSearch[uu]).label;

                        //check if label_uv and label are adjacent
                        if(label >= merge_candidates_.size() || label_uv >= merge_candidates_.size())
                            continue;

                        if(adjacent_[std::max(label, label_uv)][std::min(label, label_uv)] < 0)
                            continue;

                        if(!(merge_candidates_[adjacent_[std::max(label, label_uv)][std::min(label, label_uv)]].isValid()))
                            continue;

                        if(label_uv != label)
                        {
                            boundary_swap++;
                        }
                    }
                }

                pm.boundary_length_delta_ = (boundary_swap - boundary_cur);

                ///improvement by swapping pixel
                float improvement = alpha_ * pm.data_error_delta_ + nyu_ * pm.boundary_length_delta_;
                //float improvement = alpha_ * pm.data_error_delta_;
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

            std::cout << "improvement:" << pm.improvement_ << " " << pm.idx_ << std::endl;

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

            //recompute for points around the point being swapped
            std::vector<int> pointIdxNKNSearch;
            std::vector<float> pointNKNSquaredDistance;
            float radius_search = boundary_radius_ * 2.f;
            PointT p = cloud_->at(idx_being_swapped);

            if(octree_->radiusSearch (p, radius_search, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
            {
                for(size_t uu=0; uu < pointIdxNKNSearch.size(); uu++)
                {
                    pixel_moves[pointIdxNKNSearch[uu]].clear();
                }

                fillPixelMoves(pixel_moves, pointIdxNKNSearch);
            }
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

    /*{
        pixel_moves.resize(cloud_->points.size());
        std::vector<int> indices_to_compute(cloud_->size());

        std::generate( indices_to_compute.begin(), indices_to_compute.end(), Generator() );
        //std::iota(indices_to_compute.begin(), indices_to_compute.end(), 0); //c11

        fillPixelMoves(pixel_moves, indices_to_compute, true);
    }*/
}

template<typename PointT>
void v4rOCTopDownSegmenter::MVMumfordShahPreSegmenter<PointT>::fillPixelMoves(std::vector<std::vector<PixelMove> > & pixel_moves,
                                                                              std::vector<int> & to_compute,
                                                                              bool vis)
{

    std::cout << to_compute.size() << " " << cloud_->size() << std::endl;

    int n_boundaries = 0;
    pcl::PointCloud<pcl::PointXYZ>::Ptr boundary_points(new pcl::PointCloud<pcl::PointXYZ>);

    //for each point in the cloud, detect those that are at the boundary by looking at the labels of the point
    //in a neighbourd
    std::vector<int> pointIdxNKNSearch;
    std::vector<float> pointNKNSquaredDistance;
    float radius_search = boundary_radius_;

    std::vector<int>::iterator it;
    for(it = to_compute.begin(); it != to_compute.end(); it++)
    {
        PointT p = cloud_->at(*it);

        bool is_boundary = false;
        uint32_t label = supervoxels_labels_cloud_->at(*it).label;

        std::set<uint32_t> labels;
        if(octree_->radiusSearch (p, radius_search, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
        {
            for(size_t i=0; i < pointIdxNKNSearch.size(); i++)
            {
                //compare labels
                uint32_t label_uv = supervoxels_labels_cloud_->at(pointIdxNKNSearch[i]).label;

                //check if label_uv and label are adjacent
                if(label >= adjacent_.size() || label_uv >= adjacent_.size())
                {
                    continue;
                }

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
            //std::cout << labels.size() << std::endl;
            for(std::set<uint32_t>::iterator itt = labels.begin(); itt != labels.end(); itt++)
            {
                uint32_t label_uv = *itt;
                if(label >= adjacent_.size() || label_uv >= adjacent_.size())
                    continue;

                int ri, rj;
                ri = std::max(label, label_uv);
                rj = std::min(label, label_uv);

                if(adjacent_[ri][rj] < 0)
                    continue;

                if(!merge_candidates_[adjacent_[ri][rj]].isValid())
                    continue;

                int idx = *it;
                PixelMove pm;
                pm.valid_ = true;
                pm.r1_ = std::max(label, label_uv);
                pm.r2_ = std::min(label, label_uv);
                pm.current_region_ = label;
                pm.idx_ = idx;
                pm.recompute_ = true;
                pixel_moves[idx].push_back(pm);

            }

            n_boundaries++;
            pcl::PointXYZ p_bound;
            p_bound.getVector3fMap() = p.getVector3fMap();
            boundary_points->push_back(p_bound);
        }

    }

    if(vis)
    {
        pcl::visualization::PCLVisualizer vis_boundary("boundaries");
        vis_boundary.addPointCloud(cloud_);

        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler(boundary_points, 255, 0, 0);
        vis_boundary.addPointCloud<pcl::PointXYZ>(boundary_points, handler, "BOUNDARIES");
        vis_boundary.addCoordinateSystem(0.3f);
        vis_boundary.spin();
    }
}

template class v4rOCTopDownSegmenter::MVMumfordShahPreSegmenter<pcl::PointXYZRGB>;
//template class v4rOCTopDownSegmenter::SVMumfordShahPreSegmenter<pcl::PointXYZRGBA>;
