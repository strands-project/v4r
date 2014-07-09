#include <pcl/point_types.h>
#include <faat_pcl/recognition/hv/ghv_cuda_wrapper.h>
#include <pcl/common/transforms.h>
#include <pcl/common/time.h>
#include <boost/thread.hpp>

//#define VIS

template<typename PointT>
faat_pcl::recognition::GHVCudaWrapper<PointT>::GHVCudaWrapper ()
{
#ifdef VIS
    vis_.reset(new pcl::visualization::PCLVisualizer("GHV gpu"));
#endif
}

template<typename PointT>
void
faat_pcl::recognition::GHVCudaWrapper<PointT>::extractEuclideanClustersSmooth (const typename pcl::PointCloud<PointT> &cloud,
                                                                               const pcl::PointCloud<pcl::Normal> &normals, float tolerance,
                                                                               const typename pcl::search::Search<PointT>::Ptr &tree,
                                                                               std::vector<pcl::PointIndices> &clusters, double eps_angle,
                                                                               float curvature_threshold, unsigned int min_pts_per_cluster,
                                                                               unsigned int max_pts_per_cluster)
  {

    if (tree->getInputCloud ()->points.size () != cloud.points.size ())
    {
      PCL_ERROR("[pcl::extractEuclideanClusters] Tree built for a different point cloud dataset\n");
      return;
    }
    if (cloud.points.size () != normals.points.size ())
    {
      PCL_ERROR("[pcl::extractEuclideanClusters] Number of points in the input point cloud different than normals!\n");
      return;
    }

    // Create a bool vector of processed point indices, and initialize it to false
    std::vector<bool> processed (cloud.points.size (), false);

    std::vector<int> nn_indices;
    std::vector<float> nn_distances;
    // Process all points in the indices vector
    int size = static_cast<int> (cloud.points.size ());
    for (int i = 0; i < size; ++i)
    {
      if (processed[i])
        continue;

      std::vector<unsigned int> seed_queue;
      int sq_idx = 0;
      seed_queue.push_back (i);

      processed[i] = true;

      while (sq_idx < static_cast<int> (seed_queue.size ()))
      {

        if (normals.points[seed_queue[sq_idx]].curvature > curvature_threshold)
        {
          sq_idx++;
          continue;
        }

        // Search for sq_idx
        if (!tree->radiusSearch (seed_queue[sq_idx], tolerance, nn_indices, nn_distances))
        {
          sq_idx++;
          continue;
        }

        for (size_t j = 1; j < nn_indices.size (); ++j) // nn_indices[0] should be sq_idx
        {
          if (processed[nn_indices[j]]) // Has this point been processed before ?
            continue;

          if (normals.points[nn_indices[j]].curvature > curvature_threshold)
          {
            continue;
          }

          //processed[nn_indices[j]] = true;
          // [-1;1]

          double dot_p = normals.points[seed_queue[sq_idx]].normal[0] * normals.points[nn_indices[j]].normal[0]
              + normals.points[seed_queue[sq_idx]].normal[1] * normals.points[nn_indices[j]].normal[1] + normals.points[seed_queue[sq_idx]].normal[2]
              * normals.points[nn_indices[j]].normal[2];

          if (fabs (acos (dot_p)) < eps_angle)
          {
            processed[nn_indices[j]] = true;
            seed_queue.push_back (nn_indices[j]);
          }
        }

        sq_idx++;
      }

      // If this queue is satisfactory, add to the clusters
      if (seed_queue.size () >= min_pts_per_cluster && seed_queue.size () <= max_pts_per_cluster)
      {
        pcl::PointIndices r;
        r.indices.resize (seed_queue.size ());
        for (size_t j = 0; j < seed_queue.size (); ++j)
          r.indices[j] = seed_queue[j];

        std::sort (r.indices.begin (), r.indices.end ());
        r.indices.erase (std::unique (r.indices.begin (), r.indices.end ()), r.indices.end ());
        clusters.push_back (r); // We could avoid a copy by working directly in the vector
      }
    }
  }

template<typename PointT>
void
faat_pcl::recognition::GHVCudaWrapper<PointT>::smoothSceneSegmentation()
{

    pcl::ScopeTime t("smoothSceneSegmentation() in single thread");
    scene_downsampled_tree_.reset (new pcl::search::KdTree<PointT>);
    scene_downsampled_tree_->setInputCloud (scene_cloud_);

    eps_angle_threshold_ = 0.25;
    min_points_ = 20;
    curvature_threshold_ = 0.04f;
    cluster_tolerance_ = 0.015f;

    std::vector<pcl::PointIndices> clusters;
    extractEuclideanClustersSmooth (*scene_cloud_, *scene_normals_, cluster_tolerance_,
                                     scene_downsampled_tree_, clusters, eps_angle_threshold_,
                                     curvature_threshold_, min_points_);

    clusters_cloud_.reset (new pcl::PointCloud<pcl::PointXYZL>);
    clusters_cloud_rgb_.reset(new pcl::PointCloud<pcl::PointXYZRGBA>);
    clusters_cloud_->points.resize (scene_cloud_->points.size ());
    clusters_cloud_->width = scene_cloud_->width;
    clusters_cloud_->height = 1;

    clusters_cloud_rgb_->points.resize (scene_cloud_->points.size ());
    clusters_cloud_rgb_->width = scene_cloud_->width;
    clusters_cloud_rgb_->height = 1;

    for (size_t i = 0; i < scene_cloud_->points.size (); i++)
    {
        pcl::PointXYZL p;
        p.getVector3fMap () = scene_cloud_->points[i].getVector3fMap ();
        p.label = 0;
        clusters_cloud_->points[i] = p;
        clusters_cloud_rgb_->points[i].getVector3fMap() = p.getVector3fMap();
        clusters_cloud_rgb_->points[i].r = clusters_cloud_rgb_->points[i].g = clusters_cloud_rgb_->points[i].b = 100;
    }

    uint32_t label = 1;
    for (size_t i = 0; i < clusters.size (); i++)
    {
        for (size_t j = 0; j < clusters[i].indices.size (); j++)
            clusters_cloud_->points[clusters[i].indices[j]].label = label;
        label++;
    }

    std::vector<uint32_t> label_colors_;
    int max_label = label;
    label_colors_.reserve (max_label + 1);
    srand (static_cast<unsigned int> (time (0)));
    while (label_colors_.size () <= max_label )
    {
        uint8_t r = static_cast<uint8_t>( (rand () % 256));
        uint8_t g = static_cast<uint8_t>( (rand () % 256));
        uint8_t b = static_cast<uint8_t>( (rand () % 256));
        label_colors_.push_back (static_cast<uint32_t>(r) << 16 | static_cast<uint32_t>(g) << 8 | static_cast<uint32_t>(b));
    }

    {
        for(size_t i=0; i < clusters_cloud_->points.size(); i++)
        {
            if (clusters_cloud_->points[i].label != 0)
            {
                clusters_cloud_rgb_->points[i].rgb = label_colors_[clusters_cloud_->points[i].label];
            }
        }
    }
}

template<typename PointT>
void
faat_pcl::recognition::GHVCudaWrapper<PointT>::uploadToGPU (faat_pcl::recognition_cuda::GHV & ghv_)
{
    //upload scene and occlusion clouds
    faat_pcl::recognition_cuda::XYZPointCloud * occ_cloud = new faat_pcl::recognition_cuda::XYZPointCloud;
    faat_pcl::recognition_cuda::XYZPointCloud * scene = new faat_pcl::recognition_cuda::XYZPointCloud;

    {
        occ_cloud->height_ = occlusion_cloud_->height;
        occ_cloud->width_ = occlusion_cloud_->width;
        occ_cloud->on_device_ = false;
        occ_cloud->points = new faat_pcl::recognition_cuda::xyz_p[occlusion_cloud_->points.size()];
        for(size_t i=0; i < occlusion_cloud_->points.size(); i++)
        {
            occ_cloud->points[i].x = occlusion_cloud_->points[i].x;
            occ_cloud->points[i].y = occlusion_cloud_->points[i].y;
            occ_cloud->points[i].z = occlusion_cloud_->points[i].z;
        }

        ghv_.setSceneCloud(occ_cloud);
    }

    typedef pcl::PointCloud<PointT> CloudS;
    typedef typename pcl::traits::fieldList<typename CloudS::PointType>::type FieldListS;

    bool exists_s;
    float rgb_s;

    {
        float3 * scene_RGB_values = new float3[scene_cloud_->points.size()];

        pcl::for_each_type<FieldListS> (
                        pcl::CopyIfFieldExists<typename CloudS::PointType, float> (scene_cloud_->points[0],
                                                                                   "rgb", exists_s, rgb_s));

        scene->height_ = scene_cloud_->height;
        scene->width_ = scene_cloud_->width;
        scene->on_device_ = false;
        scene->points = new faat_pcl::recognition_cuda::xyz_p[scene_cloud_->points.size()];
        for(size_t i=0; i < scene_cloud_->points.size(); i++)
        {
            scene->points[i].x = scene_cloud_->points[i].x;
            scene->points[i].y = scene_cloud_->points[i].y;
            scene->points[i].z = scene_cloud_->points[i].z;

            if(exists_s)
            {
                pcl::for_each_type<FieldListS> (
                                pcl::CopyIfFieldExists<typename CloudS::PointType, float> (scene_cloud_->points[i],
                                                                                           "rgb", exists_s, rgb_s));

                uint32_t rgb = *reinterpret_cast<int*> (&rgb_s);
                unsigned char rs = (rgb >> 16) & 0x0000ff;
                unsigned char gs = (rgb >> 8) & 0x0000ff;
                unsigned char bs = (rgb) & 0x0000ff;

                float LRefs, aRefs, bRefs;
                RGB2CIELAB (rs, gs, bs, LRefs, aRefs, bRefs);
                LRefs /= 100.0f; aRefs /= 120.0f; bRefs /= 120.0f;    //normalized LAB components (0<L<1, -1<a<1, -1<b<1)

                scene_RGB_values[i].x = LRefs;
                scene_RGB_values[i].y = aRefs;
                scene_RGB_values[i].z = bRefs;

            }
        }

        ghv_.setScenePointCloud(scene);

        if(exists_s)
        {
            std::cout << "going to upload color..." << std::endl;
            ghv_.setSceneRGBValues(scene_RGB_values, scene_cloud_->points.size());
        }

        delete[] scene_RGB_values;
    }

    //upload model clouds, transforms and mapping to GPU
    std::vector<faat_pcl::recognition_cuda::XYZPointCloud *> models_gpu;
    std::vector<faat_pcl::recognition_cuda::XYZPointCloud *> models_normals_gpu;
    std::vector<float3 *> models_color_gpu;
    std::vector<int> models_sizes_gpu;

    std::cout << models_.size() << " " << models_normals_.size() << " " << transforms_.size() << " " << transforms_to_models_.size() << std::endl;

    for(size_t i=0; i < models_.size(); i++)
    {

        float3 * color = new float3[models_[i]->points.size()];
        pcl::for_each_type<FieldListS> (
                        pcl::CopyIfFieldExists<typename CloudS::PointType, float> (models_[i]->points[0],
                                                                                   "rgb", exists_s, rgb_s));

        faat_pcl::recognition_cuda::XYZPointCloud * model = new faat_pcl::recognition_cuda::XYZPointCloud;
        model->height_ = models_[i]->height;
        model->width_ = models_[i]->width;
        model->on_device_ = false;
        model->points = new faat_pcl::recognition_cuda::xyz_p[models_[i]->points.size()];
        for(size_t k=0; k < models_[i]->points.size(); k++)
        {
            model->points[k].x = models_[i]->points[k].x;
            model->points[k].y = models_[i]->points[k].y;
            model->points[k].z = models_[i]->points[k].z;

            if(exists_s)
            {
                pcl::for_each_type<FieldListS> (
                                pcl::CopyIfFieldExists<typename CloudS::PointType, float> (models_[i]->points[k],
                                                                                           "rgb", exists_s, rgb_s));

                uint32_t rgb = *reinterpret_cast<int*> (&rgb_s);
                unsigned char rs = (rgb >> 16) & 0x0000ff;
                unsigned char gs = (rgb >> 8) & 0x0000ff;
                unsigned char bs = (rgb) & 0x0000ff;

                float LRefs, aRefs, bRefs;
                RGB2CIELAB (rs, gs, bs, LRefs, aRefs, bRefs);
                LRefs /= 100.0f; aRefs /= 120.0f; bRefs /= 120.0f;    //normalized LAB components (0<L<1, -1<a<1, -1<b<1)

                color[k].x = LRefs;
                color[k].y = aRefs;
                color[k].z = bRefs;

            }
        }

        models_color_gpu.push_back(color);
        models_gpu.push_back(model);

        faat_pcl::recognition_cuda::XYZPointCloud * normals = new faat_pcl::recognition_cuda::XYZPointCloud;
        normals->height_ = models_[i]->height;
        normals->width_ = models_[i]->width;
        normals->on_device_ = false;
        normals->points = new faat_pcl::recognition_cuda::xyz_p[models_[i]->points.size()];
        for(size_t k=0; k < models_[i]->points.size(); k++)
        {
            Eigen::Vector3f n = models_normals_[i]->points[k].getNormalVector3fMap();
            normals->points[k].x = n[0];
            normals->points[k].y = n[1];
            normals->points[k].z = n[2];
        }

        models_normals_gpu.push_back(normals);
        models_sizes_gpu.push_back(models_[i]->points.size());
    }

    ghv_.setModelColors(models_color_gpu, models_.size(), models_sizes_gpu);

    {
        //upload scene normals
        faat_pcl::recognition_cuda::XYZPointCloud * normals = new faat_pcl::recognition_cuda::XYZPointCloud;
        normals->height_ = scene_normals_->height;
        normals->width_ = scene_normals_->width;
        normals->on_device_ = false;
        normals->points = new faat_pcl::recognition_cuda::xyz_p[scene_normals_->points.size()];
        for(size_t k=0; k < scene_normals_->points.size(); k++)
        {
            Eigen::Vector3f n = scene_normals_->points[k].getNormalVector3fMap();
            normals->points[k].x = n[0];
            normals->points[k].y = n[1];
            normals->points[k].z = n[2];
        }

        ghv_.setSceneNormals(normals);
    }

    std::vector<faat_pcl::recognition_cuda::HypothesisGPU> hypotheses;
    for(size_t i=0; i < transforms_.size(); i++)
    {
        faat_pcl::recognition_cuda::HypothesisGPU hyp;
        hyp.model_idx_ = transforms_to_models_[i];

        for(size_t ii=0; ii < 4; ii++)
          for(size_t jj=0; jj < 4; jj++)
            hyp.transform_.mat[ii][jj] = transforms_[i](ii,jj);

        hypotheses.push_back(hyp);
    }

    ghv_.setModelClouds(models_gpu, models_normals_gpu);
    ghv_.setHypotheses(hypotheses);

    delete scene;
    delete occ_cloud;
}

template<typename PointT>
void faat_pcl::recognition::GHVCudaWrapper<PointT>::addPlanarModels(std::vector<faat_pcl::PlaneModel<PointT> > & models)
{
    planar_model_hypotheses_ = models;

    //simply add to transforms_, models_ and model_normals_
    //... and it should be fine

    //also modify transforms_to_models_
    int size_models = models_.size();

    for(size_t i=0; i < planar_model_hypotheses_.size(); i++)
    {
        models_.push_back(planar_model_hypotheses_[i].plane_cloud_);
        transforms_.push_back(Eigen::Matrix4f::Identity());

        pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal> ());
        normals->points.resize(planar_model_hypotheses_[i].plane_cloud_->points.size());
        Eigen::Vector3f plane_normal;
        plane_normal[0] = planar_model_hypotheses_[i].coefficients_.values[0];
        plane_normal[1] = planar_model_hypotheses_[i].coefficients_.values[1];
        plane_normal[2] = planar_model_hypotheses_[i].coefficients_.values[2];

        for(size_t k=0; k < planar_model_hypotheses_[i].plane_cloud_->points.size(); k++)
        {
            normals->points[k].getNormalVector3fMap() = plane_normal;
        }

        models_normals_.push_back(normals);
        transforms_to_models_.push_back(size_models + i);
    }
}

template<typename PointT>
void
faat_pcl::recognition::GHVCudaWrapper<PointT>::verify ()
{

    sol.clear();

#ifdef VIS
    vis_->removeAllPointClouds();
    int v1,v2, v3, v4;
    vis_->createViewPort(0,0,0.5,0.5,v1);
    vis_->createViewPort(0.5,0,1,0.5,v2);
    vis_->createViewPort(0,0.5,0.5,1,v3);
    vis_->createViewPort(0.5,0.5,1,1,v4);

    vis_->addPointCloud<PointT>(scene_cloud_, "cloud", v1);

    /*for(size_t i=0; i < transforms_.size(); i++)
    {
        std::stringstream model_name;

        PointInTPtr model_aligned(new pcl::PointCloud<PointT>);
        pcl::transformPointCloud(*models_[transforms_to_models_[i]],*model_aligned, transforms_[i]);

        model_name << "model_" << i;

        pcl::visualization::PointCloudColorHandlerRandom<PointT> scene_handler(model_aligned);
        vis.addPointCloud<PointT> (model_aligned, scene_handler, model_name.str(), v4);
    }

    vis.spin();*/

#endif

    faat_pcl::recognition_cuda::GHV ghv_;

    {
        pcl::ScopeTime t("total time with smooth segmentation (CPU), upload, cues and optimization");
        pcl::StopWatch t_cues;
        t_cues.reset();

        boost::thread thread_1 (&faat_pcl::recognition::GHVCudaWrapper<PointT>::smoothSceneSegmentation, this);

        {
            pcl::ScopeTime t("upload");
            uploadToGPU(ghv_);
        }

        {
            pcl::ScopeTime t("compute visibility");
            ghv_.computeVisibility();
        }

        {
            pcl::ScopeTime t("model cues");
            ghv_.computeExplainedAndModelCues();
        }

        //after joining, upload labels in main thread
        thread_1.join();
        std::vector<int> labels(clusters_cloud_->points.size(), 0);
        for(size_t i=0; i < clusters_cloud_->points.size(); i++)
            labels[i] = clusters_cloud_->points[i].label;

        ghv_.setSmoothLabels(labels);

        {
            pcl::ScopeTime t("clutter cue");
            ghv_.computeClutterCue();
        }

        cues_time_ = static_cast<float>(t_cues.getTimeSeconds());

        {
            pcl::StopWatch t_cues;
            t_cues.reset();
            pcl::ScopeTime t("optimize GPU");
            ghv_.optimize();
            t_opt_ = static_cast<float>(t_cues.getTimeSeconds());
        }

        std::cout << "Number of hypotheses:" << transforms_.size() << std::endl;
    }

    std::vector< std::vector<int> > visible_points;
    ghv_.getVisible(visible_points);

    visible_points_ = 0;
    for(size_t i=0; i < transforms_.size(); i++)
        visible_points_ += visible_points[i].size();

    std::vector< std::pair<int, int> > visible_and_outlier_sizes;
    ghv_.getVisibleAndOutlierSizes(visible_and_outlier_sizes);

    std::vector< std::vector<int> > explained;
    std::vector< std::vector<float> > explained_weights;
    ghv_.getExplainedPointsAndWeights(explained, explained_weights);

    std::vector< std::vector<int> > unexplained;
    std::vector< std::vector<float> > unexplained_weights;
    ghv_.getUnexplainedPointsAndWeights(unexplained, unexplained_weights);

    ghv_.getSolution(sol);
    ghv_.freeMemory();

#ifdef VIS

    {
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBA> scene_handler(clusters_cloud_rgb_);
        vis_->addPointCloud<pcl::PointXYZRGBA> (clusters_cloud_rgb_, scene_handler, "smooth_segmentation", v3);
    }

    vis_->removeAllPointClouds(v4);

    for(size_t i=0; i < transforms_.size(); i++)
    {
        //std::cout << "visible:" << visible_points[i].size() << " total:" << models_[transforms_to_models_[i]]->points.size() << std::endl;

        if(!sol[i])
            continue;

        PointInTPtr model_aligned(new pcl::PointCloud<PointT>);
        pcl::transformPointCloud(*models_[transforms_to_models_[i]],*model_aligned, transforms_[i]);

        std::stringstream model_name_visible;
        PointInTPtr model_aligned_visible(new pcl::PointCloud<PointT>(*model_aligned));
        pcl::copyPointCloud(*model_aligned, visible_points[i], *model_aligned_visible);
        model_name_visible << "model_visible" << i;

        pcl::visualization::PointCloudColorHandlerRandom<PointT> scene_handler(model_aligned_visible);
        vis_->addPointCloud<PointT> (model_aligned_visible, scene_handler, model_name_visible.str(), v4);

        std::pair<int, int> vis_outliers = visible_and_outlier_sizes[i];
        //float ratio = vis_outliers.second / static_cast<float>(vis_outliers.first);

        std::cout << explained[i].size() << " " << vis_outliers.second << std::endl;

        std::stringstream model_name;
        {
            //add full models
            //PointInTPtr model_aligned_visible(new pcl::PointCloud<PointT>(*model_aligned));
            model_name << "model_" << i;

            pcl::PointCloud<pcl::PointXYZRGB>::Ptr model_color(new pcl::PointCloud<pcl::PointXYZRGB>(*model_aligned));

            pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> scene_handler(model_color);
            vis_->addPointCloud<pcl::PointXYZRGB> (model_color, scene_handler, model_name.str(), v2);
        }

        pcl::PointCloud<pcl::PointXYZI>::Ptr explained_cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::copyPointCloud(*scene_cloud_, explained[i], *explained_cloud);
        for(size_t kk=0; kk < explained_cloud->points.size(); kk++)
        {
            explained_cloud->points[kk].intensity = explained_weights[i][kk] * 255.f;
        }

        model_name << "_explained";
        pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> random_handler (explained_cloud, "intensity");
        vis_->addPointCloud<pcl::PointXYZI>(explained_cloud, random_handler, model_name.str(), v1);

        /*{
            model_name << "unexplained";

            pcl::PointCloud<pcl::PointXYZRGB>::Ptr clutter (new pcl::PointCloud<pcl::PointXYZRGB> ());
            pcl::copyPointCloud(*scene_cloud_, unexplained[i], *clutter);
            for(size_t k=0; k < clutter->points.size(); k++)
            {
                if(unexplained_weights[i][k] > 1)
                {
                    clutter->points[k].r = 255;
                    clutter->points[k].g = 255;
                    clutter->points[k].b = 0;
                }
                else
                {
                    clutter->points[k].r = round(255.0 * unexplained_weights[i][k]);
                    clutter->points[k].b = round(255.0 * unexplained_weights[i][k]);
                    clutter->points[k].g = 40.f;
                }
            }

            pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> scene_handler(clutter);
            vis.addPointCloud<pcl::PointXYZRGB> (clutter, scene_handler, model_name.str(), v4);
        }*/

        /*vis.spin();

        vis.removeAllPointClouds();
        vis.addPointCloud<PointT>(scene_cloud_, "cloud");

        {
            pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBA> scene_handler(clusters_cloud_rgb_);
            vis.addPointCloud<pcl::PointXYZRGBA> (clusters_cloud_rgb_, scene_handler, "smooth_segmentation", v3);
        }*/

        /*vis.spin();
        vis.removePointCloud("explained", v1);
        vis.removePointCloud(model_name.str(), v2);*/
    }

    vis_->spin();
#endif

}

template<typename PointT> float faat_pcl::recognition::GHVCudaWrapper<PointT>::sRGB_LUT[256] = {- 1};
template<typename PointT> float faat_pcl::recognition::GHVCudaWrapper<PointT>::sXYZ_LUT[4000] = {- 1};

//template class faat_pcl::recognition::GHVCudaWrapper<pcl::PointXYZ>;
template class faat_pcl::recognition::GHVCudaWrapper<pcl::PointXYZRGB>;
