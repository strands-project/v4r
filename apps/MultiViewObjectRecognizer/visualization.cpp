#include "visualization.h"
#include <v4r/recognition/model.h>
#include <pcl/common/transforms.h>
#include <time.h>

namespace v4r
{

template<typename PointT>
void
ObjectRecognitionVisualizer<PointT>::flipOpacity(const std::string& cloud_name, double max_opacity) const
{
    double opacity;
    vis_->getPointCloudRenderingProperties( pcl::visualization::PCL_VISUALIZER_OPACITY, opacity, cloud_name );
    opacity > 0 ? opacity = 0 :opacity = max_opacity;
    vis_->setPointCloudRenderingProperties( pcl::visualization::PCL_VISUALIZER_OPACITY, opacity, cloud_name );
}

template<typename PointT>
void
ObjectRecognitionVisualizer<PointT>::pointPickingEventOccured (const pcl::visualization::PointPickingEvent &event) const
{
    float x,y,z;
    event.getPoint(x,y,z);
//    std::cout << "Point ID: " << event.getPointIndex() << " Clicked Point: " << x << "/" << y << "/" << z << std::endl;

    pcl::PointXYZ searchPoint;
    searchPoint.x = x;
    searchPoint.y = y;
    searchPoint.z = z;

    int K = 1;
    std::vector<int> pointIdxNKNSearch(K);
    std::vector<float> pointNKNSquaredDistance(K);
    if ( kdtree_ && kdtree_->nearestKSearch (searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0 )
    {
      for (size_t i = 0; i < pointIdxNKNSearch.size (); i++)
      {
//          std::cout << "NN: " << pointIdxNKNSearch[i] << ", Distance: " << sqrt(pointNKNSquaredDistance[i]) << std::endl;
          int s_kp_idx = pointIdxNKNSearch[i];

          int counter=0;
          bool found = false;
          for(size_t ohg_id=0; ohg_id<generated_object_hypotheses_.size(); ohg_id++)
          {
              const ObjectHypothesesGroup<PointT> &ohg = generated_object_hypotheses_[ohg_id];

              for(size_t oh_id=0; oh_id<ohg.ohs_.size(); oh_id++)
              {
                  const ObjectHypothesis<PointT> &oh = *ohg.ohs_[oh_id];

                  for( const pcl::Correspondence &c : oh.corr_ )
                  {

    //                  pcl::PointXYZRGB p = cloud_->points [c.index_match];
    //                  std::cout << "Point " << counter << ": " << p.x << "/" << p.y << "/" << p.z << " (distance: " << (p.getVector3fMap()-searchPoint.getVector3fMap()).norm() << ")" << std::endl;
                      if(counter == s_kp_idx)
                      {
                          std::cout << "Feature Distance: " << c.distance << std::endl;
    //                      pcl::PointXYZRGB p = cloud_->points [c.index_match];
    //                      std::cout << "****Found Point: " << p.x << "/" << p.y << "/" << p.z << " (distance: " << (p.getVector3fMap()-searchPoint.getVector3fMap()).norm() << ")" << std::endl;

                          found = true;
                          break;
                      }
                      counter++;
                  }
    //              std::cout << std::endl;
                  if(found)
                      break;
              }
           }
      }
    }
}

template<typename PointT>
void
ObjectRecognitionVisualizer<PointT>::keyboardEventOccurred (const pcl::visualization::KeyboardEvent &event) const
{
    if ( event.getKeySym() == "k" && event.keyDown() )
    {
        double opacity;
        vis_->getPointCloudRenderingProperties( pcl::visualization::PCL_VISUALIZER_OPACITY, opacity, "kp_cloud_scene" );

        if(opacity==1)
        {
            flipOpacity("kp_cloud_scene");
            flipOpacity("kp_cloud_model");
            flipOpacity("kp_cloud_scene2");
            flipOpacity("kp_cloud_model2");
            for(Line &l : corrs_)
                l();
            corrs_.swap(corrs2_);
            for(Line &l : corrs_)
                l();
        }
        else
        {
            vis_->getPointCloudRenderingProperties( pcl::visualization::PCL_VISUALIZER_OPACITY, opacity, "kp_cloud_scene2" );
            if(opacity==1)
            {
                flipOpacity("kp_cloud_scene2");
                flipOpacity("kp_cloud_model2");
            }
            else
            {
                flipOpacity("kp_cloud_scene");
                flipOpacity("kp_cloud_model");
                for(Line &l : corrs_)
                    l();
                corrs_.swap(corrs2_);
                for(Line &l : corrs_)
                    l();
            }
        }
    }

    else if ( event.getKeySym() == "d" && event.keyDown() )
    {
        flipOpacity("input_vp2", 0.2);
        flipOpacity("input_vp3", 0.2);
    }

    else if ( event.getKeySym() == "l" && event.keyDown() )
    {
        for(Line &l : corrs_)
            l();
    }
}


template<typename PointT>
void
ObjectRecognitionVisualizer<PointT>::visualize() const
{
    corrs_.clear();
    corrs2_.clear();

    if(!vis_) {
        vis_.reset(new pcl::visualization::PCLVisualizer("single-view recognition results"));
        vis_->createViewPort(0,0,1,0.33,vp1_);
        vis_->createViewPort(0,0.33,1,0.66,vp2_);
        vis_->createViewPort(0,0.66,1,1,vp3_);
    }

    size_t gen_hyps = 0;
    for(const auto &ohg : generated_object_hypotheses_)
        gen_hyps += ohg.ohs_.size();

    vis_->removeAllShapes();
    std::stringstream generated_hyp_ss; generated_hyp_ss << "genereated hypotheses (" << gen_hyps << ")";
    std::stringstream verified_hyp_ss; verified_hyp_ss << "verified hypotheses (" << verified_object_hypotheses_.size() << ")";
    vis_->addText("input cloud", 10, 10, 20, 1, 1, 1, "input_test", vp1_);
    vis_->addText(generated_hyp_ss.str(), 10, 10, 20, 0, 0, 0, "generated hypotheses_text", vp2_);
    vis_->addText("l...toggle correspondence lines", 10, 50, 12, 0, 0, 0, "toggle_lines", vp2_);
    vis_->addText("k...toggle keypoints", 10, 70, 12, 0, 0, 0, "toggle_keypoints", vp2_);
    vis_->addText("d...toggle input cloud", 10, 90, 12, 0, 0, 0, "toggle_input", vp2_);
    vis_->addText(verified_hyp_ss.str(), 10, 10, 20, 0, 0, 0, "verified hypotheses_text", vp3_);


    vis_->removeAllPointClouds();
    vis_->removeAllPointClouds(vp1_);
    vis_->removeAllPointClouds(vp2_);
    vis_->removeAllPointClouds(vp3_);

    typename pcl::PointCloud<PointT>::Ptr vis_cloud (new pcl::PointCloud<PointT>);
    pcl::copyPointCloud(*cloud_, *vis_cloud);
    vis_cloud->sensor_origin_ = Eigen::Vector4f::Zero();
    vis_cloud->sensor_orientation_ = Eigen::Quaternionf::Identity();

    if(normals_)
        vis_->addPointCloudNormals<PointT,pcl::Normal>( cloud_, normals_, 300, 0.02f, "normals", vp1_);

#if PCL_VERSION >= 100800
    vis_->removeAllCoordinateSystems(vp2_);
    vis_->removeAllCoordinateSystems(vp3_);
        for(size_t co_id=0; co_id<coordinate_axis_ids_.size(); co_id++)
            vis_->removeCoordinateSystem( coordinate_axis_ids_[co_id] );
#endif
        coordinate_axis_ids_.clear();

    if(vis_param_->no_text_)
        vis_->setBackgroundColor(1,1,1,vp1_);
    else
        vis_->setBackgroundColor(.0f, .0f, .0f, vp1_);


    pcl::PointCloud<pcl::PointXYZRGB>::Ptr kp_cloud_scene (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr kp_cloud_scene2 (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr kp_cloud_model, kp_cloud_model2;
    if(lomdb_)
    {
        kp_cloud_model.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
        kp_cloud_model2.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
    }
    srand (time(NULL));

    for(size_t ohg_id=0; ohg_id<generated_object_hypotheses_.size(); ohg_id++)
    {
        for(size_t i=0; i<generated_object_hypotheses_[ohg_id].ohs_.size(); i++)
        {
            const ObjectHypothesis<PointT> &oh = *(generated_object_hypotheses_[ohg_id].ohs_[i]);
            bool found;
            typename Model<PointT>::ConstPtr m = m_db_->getModelById(oh.class_id_, oh.model_id_, found);
            const std::string model_id = oh.model_id_.substr(0, oh.model_id_.length() - 4);
            std::stringstream model_label;
            model_label << model_id << "_" << ohg_id << "_" << i;
            typename pcl::PointCloud<PointT>::Ptr model_aligned ( new pcl::PointCloud<PointT>() );
            typename pcl::PointCloud<PointT>::ConstPtr model_cloud = m->getAssembled(5);
            pcl::transformPointCloud( *model_cloud, *model_aligned, oh.transform_);
            vis_->addPointCloud(model_aligned, model_label.str(), vp2_);


            // assign unique color for each object hypothesis
            const uint8_t r = rand()%255;
            const uint8_t g = rand()%255;
            const uint8_t b = rand()%255;

            pcl::PointCloud<pcl::PointXYZRGB>::Ptr kp_cloud_scene_tmp (new pcl::PointCloud<pcl::PointXYZRGB>);
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr kp_cloud_scene_tmp2 (new pcl::PointCloud<pcl::PointXYZRGB>);

            pcl::PointCloud<pcl::PointXYZRGB>::Ptr kp_cloud_model_tmp, kp_cloud_model_tmp2;
            if(lomdb_)
            {
                kp_cloud_model_tmp.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
                kp_cloud_model_tmp2.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
            }

            for( const pcl::Correspondence &c : oh.corr_ )
            {
                pcl::PointXYZRGB p = vis_cloud->points [c.index_match];
                p.r = r;
                p.g = g;
                p.b = b;
                kp_cloud_scene_tmp->points.push_back(p);

                if(lomdb_)
                {
                    pcl::PointXYZ m_kp = lomdb_->l_obj_models_.at(m->id_)->keypoints_->points[c.index_query];
                    pcl::PointXYZRGB m_kp_color;
                    m_kp_color.getVector3fMap() = m_kp.getVector3fMap();
                    m_kp_color.r = r;
                    m_kp_color.g = g;
                    m_kp_color.b = b;
                    kp_cloud_model_tmp->points.push_back(m_kp_color);
                }


                // also show unique color for each correspondence
                const uint8_t rr = rand()%255;
                const uint8_t gg = rand()%255;
                const uint8_t bb = rand()%255;

                p = vis_cloud->points [c.index_match];
                p.r = rr;
                p.g = gg;
                p.b = bb;
                kp_cloud_scene_tmp2->points.push_back(p);

                if(lomdb_)
                {
                    pcl::PointXYZ m_kp = lomdb_->l_obj_models_.at(m->id_)->keypoints_->points[c.index_query];
                    pcl::PointXYZRGB m_kp_color;
                    m_kp_color.getVector3fMap() = m_kp.getVector3fMap();
                    m_kp_color.r = rr;
                    m_kp_color.g = gg;
                    m_kp_color.b = bb;
                    kp_cloud_model_tmp2->points.push_back(m_kp_color);
                }
            }

            if(lomdb_)
            {
                pcl::transformPointCloud(*kp_cloud_model_tmp, *kp_cloud_model_tmp, oh.transform_);
                pcl::transformPointCloud(*kp_cloud_model_tmp2, *kp_cloud_model_tmp2, oh.transform_);
                *kp_cloud_model += *kp_cloud_model_tmp;
                *kp_cloud_model2 += *kp_cloud_model_tmp2;
            }
            *kp_cloud_scene += *kp_cloud_scene_tmp;
            *kp_cloud_scene2 += *kp_cloud_scene_tmp2;


    #if PCL_VERSION >= 100800
            Eigen::Matrix4f tf_tmp = oh.transform_;
            Eigen::Matrix3f rot_tmp  = tf_tmp.block<3,3>(0,0);
            Eigen::Vector3f trans_tmp = tf_tmp.block<3,1>(0,3);
            Eigen::Affine3f affine_trans;
            affine_trans.fromPositionOrientationScale(trans_tmp, rot_tmp, Eigen::Vector3f::Ones());
            std::stringstream co_id; co_id << ohg_id << i << "vp2";
            vis_->addCoordinateSystem(0.2f, affine_trans, co_id.str(), vp2_);
            coordinate_axis_ids_.push_back(co_id.str());
    #endif
        }
    }

    for(size_t i=0; i<verified_object_hypotheses_.size(); i++)
    {
        const ObjectHypothesis<PointT> &oh = *verified_object_hypotheses_[i];
        bool found_model;
        typename Model<PointT>::ConstPtr m = m_db_->getModelById(oh.class_id_, oh.model_id_, found_model);
        const std::string model_id = m->id_.substr(0, m->id_.length() - 4);
        std::stringstream model_label;
        model_label << model_id << "_verified_" << i;
        typename pcl::PointCloud<PointT>::Ptr model_aligned ( new pcl::PointCloud<PointT>() );
        typename pcl::PointCloud<PointT>::ConstPtr model_cloud = m->getAssembled(5);
        pcl::transformPointCloud( *model_cloud, *model_aligned, oh.transform_);
        vis_->addPointCloud(model_aligned, model_label.str(), vp3_);

#if PCL_VERSION >= 100800
        Eigen::Matrix4f tf_tmp = oh.transform_;
        Eigen::Matrix3f rot_tmp  = tf_tmp.block<3,3>(0,0);
        Eigen::Vector3f trans_tmp = tf_tmp.block<3,1>(0,3);
        Eigen::Affine3f affine_trans;
        affine_trans.fromPositionOrientationScale(trans_tmp, rot_tmp, Eigen::Vector3f::Ones());
        std::stringstream co_id; co_id << i << "vp3";
        vis_->addCoordinateSystem(0.2f, affine_trans, co_id.str(), vp3_);
        coordinate_axis_ids_.push_back(co_id.str());
#endif
    }

    if(vis_param_->no_text_)
        vis_->setBackgroundColor(1,1,1,vp2_);
    else
        vis_->setBackgroundColor(.5f, .5f, .5f, vp2_);


    vis_->addPointCloud(vis_cloud, "input", vp1_);
    pcl::visualization::PointCloudColorHandlerCustom<PointT> gray2 (vis_cloud, 255, 255, 255);
    vis_->addPointCloud(vis_cloud, gray2, "input_vp2", vp2_);
    vis_->setPointCloudRenderingProperties( pcl::visualization::PCL_VISUALIZER_OPACITY, 0.2, "input_vp2");
    pcl::visualization::PointCloudColorHandlerCustom<PointT> gray3 (vis_cloud, 128, 128, 128);
    vis_->addPointCloud(vis_cloud, gray3, "input_vp3", vp3_);
    vis_->setPointCloudRenderingProperties( pcl::visualization::PCL_VISUALIZER_OPACITY, 0.2, "input_vp3");
    vis_->addPointCloud(kp_cloud_scene, "kp_cloud_scene", vp1_);
    vis_->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 15, "kp_cloud_scene");
    vis_->addPointCloud(kp_cloud_scene2, "kp_cloud_scene2", vp1_);
    vis_->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 15, "kp_cloud_scene2");
    vis_->setPointCloudRenderingProperties( pcl::visualization::PCL_VISUALIZER_OPACITY, 0, "kp_cloud_scene2");
    vis_->setBackgroundColor(1.f, 1.f, 1.f, vp3_);

    if(lomdb_)
    {
        vis_->addPointCloud(kp_cloud_model, "kp_cloud_model", vp2_);
        vis_->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 15, "kp_cloud_model");
        vis_->addPointCloud(kp_cloud_model2, "kp_cloud_model2", vp2_);
        vis_->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 15, "kp_cloud_model2");
        vis_->setPointCloudRenderingProperties( pcl::visualization::PCL_VISUALIZER_OPACITY, 0, "kp_cloud_model2");

        for(size_t c_id=0; c_id<kp_cloud_model2->points.size(); c_id++)
        {
            const pcl::PointXYZRGB &m = kp_cloud_model->points[c_id];
            const pcl::PointXYZRGB &s = kp_cloud_scene->points[c_id];
            const pcl::PointXYZRGB &m2 = kp_cloud_model2->points[c_id];
            const pcl::PointXYZRGB &s2 = kp_cloud_scene2->points[c_id];
            std::stringstream line_ss; line_ss << "line_" << c_id;
            Line l(vis_, s, m, s.r/255., s.g/255., s.b/255., line_ss.str(), vp2_);
            Line l2(vis_, s2, m2, s2.r/255., s2.g/255., s2.b/255., line_ss.str(), vp2_);
            corrs_.push_back( l );
            corrs2_.push_back( l2 );
            corrs_.back();
        }
    }

    boost::function<void (const pcl::visualization::KeyboardEvent &)> f =
            boost::bind (&ObjectRecognitionVisualizer<PointT>::keyboardEventOccurred, this, _1);
    vis_->registerKeyboardCallback(f);


    boost::function<void (const pcl::visualization::PointPickingEvent &)> f2 =
            boost::bind (&ObjectRecognitionVisualizer<PointT>::pointPickingEventOccured, this, _1);
    vis_->registerPointPickingCallback(f2);


    if( kp_cloud_scene && !kp_cloud_scene->points.empty() )
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr kp_cloud_scene_xyz (new pcl::PointCloud<pcl::PointXYZ>);
        pcl::copyPointCloud( *kp_cloud_scene, *kp_cloud_scene_xyz );
        kdtree_.reset( new pcl::KdTreeFLANN<pcl::PointXYZ>);
        kdtree_->setInputCloud( kp_cloud_scene_xyz );
    }

    vis_->resetCamera();
    vis_->spin();

    //    normals_.reset();
}

template class V4R_EXPORTS ObjectRecognitionVisualizer<pcl::PointXYZRGB>;

}
