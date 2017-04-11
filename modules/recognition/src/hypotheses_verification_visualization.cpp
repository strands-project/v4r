#include <v4r/recognition/hypotheses_verification_visualization.h>
#include <pcl/impl/instantiate.hpp>

namespace v4r
{

template<typename ModelT, typename SceneT>
void
HV_CuesVisualizer<ModelT, SceneT>::visualize(const HypothesisVerification<ModelT, SceneT> *hv, const boost::dynamic_bitset<> & active_solution, float cost, int times_evaluated)
{
    if(!vis_go_cues_)
    {
        vis_go_cues_.reset(new pcl::visualization::PCLVisualizer("visualizeGOCues"));
        vis_go_cues_->createViewPort(0, 0, 0.33, 0.5, vp_scene_scene_);
        vis_go_cues_->createViewPort(0.33, 0, 0.66, 0.5, vp_scene_active_hypotheses_);
        vis_go_cues_->createViewPort(0.66, 0, 1, 0.5, vp_model_fitness_);
        vis_go_cues_->createViewPort(0, 0.5, 0.33, 1, vp_scene_fitness_);
        vis_go_cues_->createViewPort(0.33, 0.5, 0.66, 1, vp_scene_duplicity_);
        vis_go_cues_->createViewPort(0.66, 0.5, 1, 1, vp_scene_smooth_regions_);
        vis_go_cues_->setBackgroundColor(vis_param_->bg_color_[0], vis_param_->bg_color_[1], vis_param_->bg_color_[2]);
    }

    vis_go_cues_->removeAllPointClouds();
    vis_go_cues_->removeAllShapes();

    size_t model_outliers = 0;
    float pairwise_cost = 0.f;

    // model uni term
    size_t num_active_hypotheses = 0;
    for(size_t i=0; i<active_solution.size(); i++)
    {
        if(active_solution[i])
        {
            model_outliers += hv->global_hypotheses_[i]->visible_pt_is_outlier_.count();
            num_active_hypotheses++;
        }
    }


    // pairwise term
    for(size_t i=0; i<active_solution.size(); i++)
    {
        for(size_t j=0; j<i; j++)
        {
            if(active_solution[i] && active_solution[j])
                pairwise_cost += hv->intersection_cost_(i,j);
        }
    }


    std::stringstream out, model_fitness_txt;
    out << "Active Hypotheses: " << active_solution << std::endl
        << "Cost: " << std::setprecision(5) << cost << " , #Evaluations: " << times_evaluated
        << std::endl << "; pairwise cost: " << pairwise_cost << "; total cost: " << cost << std::endl;
    model_fitness_txt << "model fitness. Outliers: " << model_outliers;


    vis_go_cues_->addText ("Scene", 1, 30, 16, vis_param_->text_color_[0], vis_param_->text_color_[1], vis_param_->text_color_[2], "inliers_outliers", vp_scene_scene_);
    vis_go_cues_->addText (out.str(), 1, 30, 16, vis_param_->text_color_[0], vis_param_->text_color_[1], vis_param_->text_color_[2], "active_hypotheses", vp_scene_active_hypotheses_);
    vis_go_cues_->addText (model_fitness_txt.str(), 1, 30, 16, vis_param_->text_color_[0], vis_param_->text_color_[1], vis_param_->text_color_[2], "model fitness", vp_model_fitness_);
    vis_go_cues_->addPointCloud (hv->scene_cloud_downsampled_, "scene_cloud", vp_scene_scene_);

    pcl::visualization::PointCloudColorHandlerCustom<SceneT> gray (hv->scene_cloud_downsampled_, 128, 128, 128);
    vis_go_cues_->addPointCloud(hv->scene_cloud_downsampled_, gray, "input_active_hypotheses", vp_scene_active_hypotheses_);
    vis_go_cues_->setPointCloudRenderingProperties( pcl::visualization::PCL_VISUALIZER_OPACITY, 0.2, "input_active_hypotheses");


    // ==== VISUALIZE ACTIVE HYPOTHESES =======
    {
        for(size_t i=0; i < active_solution.size(); i++)
        {
            if(active_solution[i])
            {
                HVRecognitionModel<ModelT> &rm = *(hv->global_hypotheses_[i]);
                std::stringstream model_name; model_name << "model_" << i;
                vis_go_cues_->addPointCloud(rm.visible_cloud_, model_name.str(), vp_scene_active_hypotheses_);
                vis_go_cues_->addPointCloud(rm.visible_cloud_, model_name.str()+"_smooth", vp_scene_smooth_regions_);

                typename pcl::PointCloud<ModelT>::Ptr model_fit_cloud (new pcl::PointCloud<ModelT> (*rm.visible_cloud_));
                for( ModelT &mp : model_fit_cloud->points)
                    mp.r = mp.g = mp.b = 0.f;

                for(size_t cidx=0; cidx < rm.model_scene_c_.size(); cidx++)
                {
                    const ModelSceneCorrespondence &c = rm.model_scene_c_[cidx];
                    int sidx = c.scene_id_;
                    int midx = c.model_id_;

                    if(sidx<0)
                        continue;

                    CHECK ( hv->getFitness(c) <= 1 );

                    ModelT &mp = model_fit_cloud->points[midx];

                    // scale green color channels with fitness terms
                    mp.g   = 255.f * hv->getFitness(c);
                }

                for(size_t midx=0; midx < model_fit_cloud->points.size(); midx++)
                {
                    if ( rm.visible_pt_is_outlier_[ midx ] )
                        model_fit_cloud->points[ midx ].r = 255.f;
                }

                model_name << "_fitness";
                vis_go_cues_->addPointCloud(model_fit_cloud, model_name.str(), vp_model_fitness_);
            }
        }
    }

    Eigen::Array<bool, Eigen::Dynamic, 1> scene_pt_is_explained( hv->scene_cloud_downsampled_->points.size() ); // needed for smooth region visualization
    scene_pt_is_explained.setConstant(hv->scene_cloud_downsampled_->points.size(), false);

    // ==== VISUALIZE SCENE FITNESS =======
    {
        typename pcl::PointCloud<SceneT>::Ptr scene_fit_cloud (new pcl::PointCloud<SceneT> (*hv->scene_cloud_downsampled_));
        for( SceneT &p : scene_fit_cloud->points)
            p.r = p.g = p.b = 0.f;


        double scene_fit =0.;
        for(size_t s_id=0; s_id < hv->scene_pts_explained_solution_.size(); s_id++)
        {
            const std::vector<PtFitness> &s_pt = hv->scene_pts_explained_solution_[s_id];
            if(  !s_pt.empty() )
            {
                SceneT &p = scene_fit_cloud->points[s_id];
                double s_fit_tmp = static_cast<double>( s_pt.back().fit_ );
                scene_fit += s_fit_tmp;
                p.g = 255.f * s_fit_tmp; // uses the maximum value for scene explanation
                scene_pt_is_explained(s_id) = true;
            }
        }
        vis_go_cues_->addPointCloud(scene_fit_cloud, "scene fitness cloud", vp_scene_fitness_);
        if(!vis_param_->no_text_)
        {
            std::stringstream scene_fitness_txt; scene_fitness_txt << "scene fitness: " << scene_fit;
            vis_go_cues_->addText (scene_fitness_txt.str(), 1, 30, 16, vis_param_->text_color_[0], vis_param_->text_color_[1], vis_param_->text_color_[2], "scene fitness", vp_scene_fitness_);
        }
    }

    // ==== VISUALIZE DUPLICATED POINTS FITNESS =======
    {
        double duplicity =0.;
        typename pcl::PointCloud<SceneT>::Ptr duplicity_cloud (new pcl::PointCloud<SceneT> (*hv->scene_cloud_downsampled_));
        for( SceneT &p : duplicity_cloud->points)
            p.r = p.g = p.b = 0.f;

        for(size_t s_id=0; s_id < hv->scene_pts_explained_solution_.size(); s_id++)
        {
            const std::vector<PtFitness> &s_pt = hv->scene_pts_explained_solution_[s_id];

            SceneT &p = duplicity_cloud->points[s_id];
            if ( s_pt.size() > 1 ) // two or more hypotheses explain the same scene point
            {
                double duplicity_tmp = static_cast<double>( s_pt[ s_pt.size() - 2 ].fit_ );
                p.r = 255 * duplicity_tmp; // uses the second best explanation
                duplicity += duplicity_tmp;
            }
        }
        vis_go_cues_->addPointCloud(duplicity_cloud, "duplicity cloud", vp_scene_duplicity_);

        if(!vis_param_->no_text_)
        {
            std::stringstream duplicity_txt; duplicity_txt << "duplicity: " << duplicity;
            vis_go_cues_->addText (duplicity_txt.str(), 1, 30, vis_param_->fontsize_, vis_param_->text_color_[0], vis_param_->text_color_[1], vis_param_->text_color_[2], "duplicity txt", vp_scene_duplicity_);
        }
    }


    // ---- VISUALIZE SMOOTH SEGMENTATION -------
    {
        int max_label = hv->scene_pt_smooth_label_id_.maxCoeff() + 1;
        if( max_label >= 1 )
        {
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene_smooth_labels_rgb (new pcl::PointCloud<pcl::PointXYZRGB>(*hv->scene_cloud_downsampled_));

            if(!vis_param_->no_text_)
                vis_go_cues_->addText( "smooth regions", 1, 30, vis_param_->fontsize_, vis_param_->text_color_[0], vis_param_->text_color_[1], vis_param_->text_color_[2], "smooth seg text", vp_scene_smooth_regions_);


            Eigen::Matrix3Xf label_colors (3, max_label);
            size_t num_smooth_regions_of_interest = 0;
            for(int i=0; i<max_label; i++)
            {
                float r,g,b;
                if( i==0 )
                    r = g = b = 255; // label 0 will be white
                else
                {
                    r = rand () % 255;
                    g = rand () % 255;
                    b = rand () % 255;
                }
                label_colors(0,i) = r;
                label_colors(1,i) = g;
                label_colors(2,i) = b;


                auto s_pt_in_region = (hv->scene_pt_smooth_label_id_.array() == i );
                auto explained_pt_in_region = (s_pt_in_region.array() && scene_pt_is_explained.array());
                size_t num_explained_pts_in_region = explained_pt_in_region.count();
                size_t num_pts_in_smooth_regions = s_pt_in_region.count();

                if(!vis_param_->no_text_ && num_explained_pts_in_region)
                {
                    std::stringstream lbl_txt; lbl_txt << std::fixed << std::setprecision(2) << num_explained_pts_in_region << " /" << " " << num_pts_in_smooth_regions;

                    if ( num_explained_pts_in_region > hv->param_.min_pts_smooth_cluster_to_be_epxlained_ &&
                         (float)(num_explained_pts_in_region) / num_pts_in_smooth_regions < hv->param_.min_ratio_cluster_explained_ )
                        lbl_txt << " !!";   // violates smooth region check

                    std::stringstream txt_id; txt_id << "smooth_cluster_txt " << i;
                    vis_go_cues_->addText( lbl_txt.str(), 10, 40+12*num_smooth_regions_of_interest++, vis_param_->fontsize_, r/255, g/255, b/255, txt_id.str(), vp_scene_smooth_regions_);
                }
            }

            for(int i=0; i < hv->scene_pt_smooth_label_id_.rows(); i++)
            {
                int l = hv->scene_pt_smooth_label_id_(i);
                pcl::PointXYZRGB &p = scene_smooth_labels_rgb->points[i];
                p.r = label_colors(0,l);
                p.g = label_colors(1,l);
                p.b = label_colors(2,l);
            }
            vis_go_cues_->addPointCloud(scene_smooth_labels_rgb, "smooth labels", vp_scene_smooth_regions_);
        }
    }

    vis_go_cues_->resetCamera();
    vis_go_cues_->spin();
}


template<typename ModelT, typename SceneT>
void
HV_ModelVisualizer<ModelT, SceneT>::visualize(const HypothesisVerification<ModelT, SceneT> *hv, const HVRecognitionModel<ModelT> &rm)
{
    if(!vis_model_)
    {
        vis_model_.reset (new pcl::visualization::PCLVisualizer ("model cues"));
        vis_model_->createViewPort(0   , 0  , 0.2, 0.33, vp_model_scene_);
        vis_model_->createViewPort(0.2 , 0  , 0.4, 0.33, vp_model_);
        vis_model_->createViewPort(0.4 , 0  , 0.6, 0.33, vp_model_scene_overlay_);
        vis_model_->createViewPort(0.6 , 0  , 0.8, 0.33, vp_model_outliers_);
        vis_model_->createViewPort(0.8 , 0  , 1  , 0.33, vp_model_scene_fit_);

        vis_model_->createViewPort(0. , 0.33 , 0.2 ,0.66 , vp_model_visible_);
        vis_model_->createViewPort(0.2, 0.33 , 0.4 ,0.66 , vp_model_total_fit_);
        vis_model_->createViewPort(0.4, 0.33 , 0.6 ,0.66 , vp_model_3d_fit_);
        vis_model_->createViewPort(0.6, 0.33 , 0.8 ,0.66 , vp_model_color_fit_);
        vis_model_->createViewPort(0.8, 0.33 , 1   ,0.66 , vp_model_normals_fit_);

        vis_model_->setBackgroundColor(vis_param_->bg_color_[0], vis_param_->bg_color_[1], vis_param_->bg_color_[2]);
    }

    vis_model_->removeAllPointClouds();
    vis_model_->removeAllShapes();

    vis_model_->addPointCloud(hv->scene_cloud_downsampled_, "scene1", vp_model_scene_);
    vis_model_->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, vis_param_->vis_pt_size_, "scene1", vp_model_scene_);

    pcl::visualization::PointCloudColorHandlerCustom<SceneT> gray (hv->scene_cloud_downsampled_, 128, 128, 128);
    vis_model_->addPointCloud(rm.visible_cloud_, "model", vp_model_);
    vis_model_->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, vis_param_->vis_pt_size_, "model", vp_model_);
    vis_model_->addPointCloud(hv->scene_cloud_downsampled_, gray, "input_rm_vp_model_", vp_model_);
    vis_model_->setPointCloudRenderingProperties( pcl::visualization::PCL_VISUALIZER_OPACITY, 0.2, "input_rm_vp_model_");


    if(!vis_param_->no_text_)
    {
        vis_model_->addText("scene",10,10, vis_param_->fontsize_, vis_param_->text_color_[0], vis_param_->text_color_[1] ,vis_param_->text_color_[2], "scene",vp_model_scene_);
        vis_model_->addText("model",10,10, vis_param_->fontsize_, vis_param_->text_color_[0], vis_param_->text_color_[1] ,vis_param_->text_color_[2], "model",vp_model_);
    }

    if(!vis_param_->no_text_)
    {
        std::stringstream txt; txt << "visible ratio: " << std::fixed << std::setprecision(2) << rm.visible_indices_by_octree_.size() / (float)rm.complete_cloud_->points.size();
        vis_model_->addText(txt.str(), 10, 10, vis_param_->fontsize_, vis_param_->text_color_[0], vis_param_->text_color_[1] ,vis_param_->text_color_[2], "visible model cloud", vp_model_visible_);
    }

    // ===== VISUALIZE VISIBLE PART =================
    {
        typename pcl::PointCloud<ModelT>::Ptr visible_cloud_colored (new pcl::PointCloud<ModelT> (*rm.complete_cloud_));
        for(ModelT &mp : visible_cloud_colored->points)
            mp.r = mp.g = mp.b = 0.f;

        for(int idx : rm.visible_indices_by_octree_)
            visible_cloud_colored->points[idx].r = 255;

        vis_model_->addPointCloud(visible_cloud_colored, "model2", vp_model_visible_);
        vis_model_->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, vis_param_->vis_pt_size_, "model2", vp_model_);
    }

    // ===== VISUALIZE FITNESS SCORES =================
    {
        typename pcl::PointCloud<ModelT>::Ptr model_3D_fit_cloud (new pcl::PointCloud<ModelT> (*rm.visible_cloud_));
        typename pcl::PointCloud<ModelT>::Ptr model_color_fit_cloud (new pcl::PointCloud<ModelT> (*rm.visible_cloud_));
        typename pcl::PointCloud<ModelT>::Ptr model_normals_fit_cloud (new pcl::PointCloud<ModelT> (*rm.visible_cloud_));
        typename pcl::PointCloud<ModelT>::Ptr model_fit_cloud (new pcl::PointCloud<ModelT> (*rm.visible_cloud_));
        for(size_t p=0; p < model_3D_fit_cloud->points.size(); p++)
        {
            ModelT &mp3d = model_3D_fit_cloud->points[p];
            ModelT &mpC = model_color_fit_cloud->points[p];
            ModelT &mpN = model_normals_fit_cloud->points[p];
            ModelT &mp = model_fit_cloud->points[p];
            mp3d.r = mp3d.b = mpC.r = mpC.b = mpN.r = mpN.b = mp.r = mp.b = 0.f;
            mp3d.g = mpC.g = mpN.g = mp.g = 0.f;
        }

        Eigen::VectorXf normals_fitness = Eigen::VectorXf::Zero (rm.visible_cloud_->points.size());
        Eigen::VectorXf color_fitness = Eigen::VectorXf::Zero (rm.visible_cloud_->points.size());
        Eigen::VectorXf fitness_3d = Eigen::VectorXf::Zero (rm.visible_cloud_->points.size());

        boost::dynamic_bitset<> model_explained_pts ( rm.visible_cloud_->points.size(), 0);

        for(size_t cidx=0; cidx < rm.model_scene_c_.size(); cidx++)
        {
            const ModelSceneCorrespondence &c = rm.model_scene_c_[cidx];
            int sidx = c.scene_id_;
            int midx = c.model_id_;

            if(sidx<0)
                continue;

            if( !model_explained_pts[midx] )
            {
                model_explained_pts.set(midx);

                normals_fitness(midx) = hv->modelSceneNormalsCostTerm(c);
                color_fitness(midx) = hv->modelSceneColorCostTerm(c);
                fitness_3d(midx) =  hv->modelScene3DDistCostTerm(c);

                CHECK ( normals_fitness(midx) <= 1 );
                CHECK ( color_fitness(midx) <= 1 );
                CHECK ( fitness_3d(midx) <= 1 );
                CHECK ( hv->getFitness(c) <= 1 );

                ModelT &mp3d = model_3D_fit_cloud->points[midx];
                ModelT &mpC = model_color_fit_cloud->points[midx];
                ModelT &mpN = model_normals_fit_cloud->points[midx];
                ModelT &mp = model_fit_cloud->points[midx];

                // scale green color channels with fitness terms
                mp3d.g = 255.f * fitness_3d(midx);
                mpC.g  = 255.f * color_fitness(midx);
                mpN.g  = 255.f * normals_fitness(midx);
                mp.g   = 255.f * hv->getFitness(c);
            }
        }

        if(!vis_param_->no_text_)
        {
            std::stringstream txt;
            txt.str(""); txt << std::fixed << std::setprecision(2)  << "3D fitness (" << (int)(hv->param_.w_3D_*100) << "\%): " << (float)fitness_3d.sum() / rm.visible_indices_.size();
            vis_model_->addText(txt.str(),10,10, vis_param_->fontsize_, vis_param_->text_color_[0], vis_param_->text_color_[1], vis_param_->text_color_[2], "3D distance",vp_model_3d_fit_);
            txt.str(""); txt << "color fitness(" << (int)(hv->param_.w_color_ *100) << "\%): " << std::fixed << std::setprecision(2) << (float)color_fitness.sum() / rm.visible_indices_.size();
            vis_model_->addText(txt.str(),10,10, vis_param_->fontsize_, vis_param_->text_color_[0], vis_param_->text_color_[1], vis_param_->text_color_[2], "color distance",vp_model_color_fit_);
            txt.str(""); txt << "normals fitness(" << (int)(hv->param_.w_normals_*100) << "\%): " << std::fixed << std::setprecision(2) << (float)normals_fitness.sum() / rm.visible_indices_.size();
            vis_model_->addText(txt.str(), 10, 10, vis_param_->fontsize_, vis_param_->text_color_[0], vis_param_->text_color_[1], vis_param_->text_color_[2],  "normals distance",vp_model_normals_fit_);
            txt.str(""); txt << "model fitness: " << std::fixed << std::setprecision(2) << rm.model_fit_ << "; normalized: " << rm.model_fit_ / rm.visible_cloud_->points.size();
            vis_model_->addText(txt.str(),10,10, vis_param_->fontsize_,vis_param_->text_color_[0], vis_param_->text_color_[1] ,vis_param_->text_color_[2], "model fitness",vp_model_total_fit_);
        }

        vis_model_->addPointCloud(model_3D_fit_cloud, "3D_distance", vp_model_3d_fit_);
        vis_model_->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, vis_param_->vis_pt_size_, "3D_distance", vp_model_3d_fit_);
        vis_model_->addPointCloud(hv->scene_cloud_downsampled_, gray, "input_rm_vp_model_scene_3d_dist_", vp_model_3d_fit_);
        vis_model_->setPointCloudRenderingProperties( pcl::visualization::PCL_VISUALIZER_OPACITY, 0.2, "input_rm_vp_model_scene_3d_dist_");

        vis_model_->addPointCloud(model_color_fit_cloud, "color_distance", vp_model_color_fit_);
        vis_model_->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, vis_param_->vis_pt_size_, "color_distance", vp_model_color_fit_);
        vis_model_->addPointCloud(hv->scene_cloud_downsampled_, gray, "input_rm_vp_model_scene_color_dist_", vp_model_color_fit_);
        vis_model_->setPointCloudRenderingProperties( pcl::visualization::PCL_VISUALIZER_OPACITY, 0.2, "input_rm_vp_model_scene_color_dist_");

        vis_model_->addPointCloud(model_normals_fit_cloud, "normals_distance", vp_model_normals_fit_);
        vis_model_->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, vis_param_->vis_pt_size_, "normals_distance", vp_model_normals_fit_);
        vis_model_->addPointCloud(hv->scene_cloud_downsampled_, gray, "input_rm_vp_model_scene_normals_dist_", vp_model_normals_fit_);
        vis_model_->setPointCloudRenderingProperties( pcl::visualization::PCL_VISUALIZER_OPACITY, 0.2, "input_rm_vp_model_scene_normals_dist_");

        vis_model_->addPointCloud(model_fit_cloud, "model_fitness", vp_model_total_fit_);
        vis_model_->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, vis_param_->vis_pt_size_, "model_fitness", vp_model_total_fit_);
        vis_model_->addPointCloud(hv->scene_cloud_downsampled_, gray, "input_rm_vp_model_scene_model_fit_", vp_model_total_fit_);
        vis_model_->setPointCloudRenderingProperties( pcl::visualization::PCL_VISUALIZER_OPACITY, 0.2, "input_rm_vp_model_scene_model_fit_");
    }


    // ===== VISUALIZE MODEL OUTLIERS =================
    {
        typename pcl::PointCloud<ModelT>::Ptr outlier_cloud (new pcl::PointCloud<ModelT> (*rm.visible_cloud_));
        for(size_t p=0; p < outlier_cloud->points.size(); p++)
        {
            ModelT &mp = outlier_cloud->points[p];
            mp.r = mp.g = mp.b = 0.f;

            if( rm.visible_pt_is_outlier_[p] )
                mp.r = 255.f;
        }

        vis_model_->addPointCloud(outlier_cloud, "outlier_cloud", vp_model_outliers_);
        vis_model_->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, vis_param_->vis_pt_size_, "outlier_cloud", vp_model_outliers_);
        vis_model_->addPointCloud(hv->scene_cloud_downsampled_, gray, "input_rm_vp_model_outliers_", vp_model_outliers_);
        vis_model_->setPointCloudRenderingProperties( pcl::visualization::PCL_VISUALIZER_OPACITY, 0.2, "input_rm_vp_model_outliers_");
        if(!vis_param_->no_text_)
        {
            std::stringstream txt;
            txt.str(""); txt << "model outliers (" << rm.visible_pt_is_outlier_.count() << " / " << rm.visible_cloud_->points.size() << " ( " << std::fixed << std::setprecision(2)  << 100.f * (float)rm.visible_pt_is_outlier_.count() / rm.visible_cloud_->points.size() << " % )";
            vis_model_->addText(txt.str(),10,10, vis_param_->fontsize_, vis_param_->text_color_[0], vis_param_->text_color_[1], vis_param_->text_color_[2], "model outliers txt", vp_model_outliers_);
        }
    }

    // ==== VISUALIZE SCENE FITNESS CLOUD =====
    {
        typename pcl::PointCloud<SceneT>::Ptr scene_fit_cloud (new pcl::PointCloud<SceneT> (*hv->scene_cloud_downsampled_));
        for(int p=0; p < rm.scene_explained_weight_.rows(); p++)
        {
            SceneT &sp = scene_fit_cloud->points[p];
            sp.r = sp.b = 0.f;
            sp.g = 255.f * rm.scene_explained_weight_.coeff(p);
        }

        if(!vis_param_->no_text_)
        {
            std::stringstream txt;
            txt.str(""); txt << "scene pts explained (fitness: " << rm.scene_explained_weight_.sum() << "; normalized: " << rm.scene_explained_weight_.sum()/hv->scene_cloud_downsampled_->points.size() << ")";
            vis_model_->addText(txt.str(),10,10, vis_param_->fontsize_,0,0,0,"scene fitness",vp_model_scene_fit_);
            vis_model_->addText("scene and visible model",10,10, vis_param_->fontsize_, vis_param_->text_color_[0], vis_param_->text_color_[1], vis_param_->text_color_[2], "scene_and_model",vp_model_scene_overlay_);
            vis_model_->addPointCloud(hv->scene_cloud_downsampled_, "scene_model_1", vp_model_scene_overlay_);
        }

        vis_model_->addPointCloud(scene_fit_cloud, "scene_fitness", vp_model_scene_fit_);
        vis_model_->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, vis_param_->vis_pt_size_, "scene_fitness", vp_model_scene_fit_);
    }

    vis_model_->addPointCloud(rm.visible_cloud_, "scene_model_2", vp_model_scene_overlay_);

    vis_model_->resetCamera();
    vis_model_->spin();
}


template<typename ModelT, typename SceneT>
void
HV_PairwiseVisualizer<ModelT, SceneT>::visualize( const HypothesisVerification<ModelT, SceneT> *hv )
{
    if(!vis_pairwise_)
    {
        vis_pairwise_.reset( new pcl::visualization::PCLVisualizer("intersection") );
        vis_pairwise_->createViewPort(0    , 0   , 0.25 , 1   , vp_pair_1_);
        vis_pairwise_->createViewPort(0.25 , 0   , 0.5  , 1   , vp_pair_2_);
        vis_pairwise_->createViewPort(0.5  , 0   , 1    , 1   , vp_pair_3_);
        vis_pairwise_->setBackgroundColor(vis_param_->bg_color_[0], vis_param_->bg_color_[1], vis_param_->bg_color_[2]);
    }

    for(size_t i=1; i< hv->global_hypotheses_.size(); i++)
    {
        const HVRecognitionModel<ModelT> &rm_a = *(hv->global_hypotheses_[i]);

        for(size_t j=0; j<i; j++)
        {
            const HVRecognitionModel<ModelT> &rm_b = *(hv->global_hypotheses_[j]);

            std::stringstream txt; txt <<  "intersection cost (" << i << ", " << j << "): " << hv->intersection_cost_(j,i);

            vis_pairwise_->removeAllPointClouds();
            vis_pairwise_->removeAllShapes();
            vis_pairwise_->addText(txt.str(), 10, 10, vis_param_->fontsize_, vis_param_->text_color_[0], vis_param_->text_color_[1], vis_param_->text_color_[2],  "intersection_text", vp_pair_3_ );
            vis_pairwise_->addPointCloud(rm_a.visible_cloud_, "cloud_a", vp_pair_1_);
            vis_pairwise_->addPointCloud(rm_b.visible_cloud_, "cloud_b", vp_pair_2_);
            vis_pairwise_->addPointCloud(rm_a.visible_cloud_, "cloud_aa", vp_pair_3_);
            vis_pairwise_->addPointCloud(rm_b.visible_cloud_, "cloud_bb", vp_pair_3_);
            vis_pairwise_->resetCamera();
            vis_pairwise_->spin();
        }
    }
}

#define PCL_INSTANTIATE_HV_CuesVisualizer(MT, ST) template class V4R_EXPORTS HV_CuesVisualizer<MT, ST>;
PCL_INSTANTIATE_PRODUCT(HV_CuesVisualizer, ((pcl::PointXYZRGB))((pcl::PointXYZRGB)) )

#define PCL_INSTANTIATE_HV_ModelVisualizer(MT, ST) template class V4R_EXPORTS HV_ModelVisualizer<MT, ST>;
PCL_INSTANTIATE_PRODUCT(HV_ModelVisualizer, ((pcl::PointXYZRGB))((pcl::PointXYZRGB)) )

#define PCL_INSTANTIATE_HV_PairwiseVisualizer(MT, ST) template class V4R_EXPORTS HV_PairwiseVisualizer<MT, ST>;
PCL_INSTANTIATE_PRODUCT(HV_PairwiseVisualizer, ((pcl::PointXYZRGB))((pcl::PointXYZRGB)) )

}
