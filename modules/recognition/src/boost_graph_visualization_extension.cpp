#include "v4r/recognition/boost_graph_visualization_extension.h"

#include <v4r/recognition/model_only_source.h>
#include <v4r/common/pcl_visualization_utils.h>
#include <v4r/common/pcl_opencv.h>

#include <pcl/common/transforms.h>

typedef v4r::rec_3d_framework::Model<PointT> ModelT;
typedef boost::shared_ptr<ModelT> ModelTPtr;

void BoostGraphVisualizer::visualizeGraph(const Graph &grph, pcl::visualization::PCLVisualizer::Ptr &vis)
{
    //--(bottom: Scene; 2nd from bottom: Single-view-results; 2nd from top: transformed hypotheses; top: verified hypotheses coming from all views)--
    //...

    std::vector<int> viewportNr;
    size_t vis_rows = 5;

    if ( !vis ) //-------Visualize Scene Cloud--------------------
    {
        vis.reset ( new pcl::visualization::PCLVisualizer ( "vis1" ) );
        vis->setWindowName ( "Recognition from Multiple Views" );
    }
    vis->removeAllPointClouds();
<<<<<<< HEAD
    std::vector<std::string> subwindow_title;
    subwindow_title.push_back("original scene");
    subwindow_title.push_back("generated hypotheses from single-view only");
    subwindow_title.push_back("verified hypotheses from single-view only");
    subwindow_title.push_back("generated multi-view hypotheses (previous + current observations)");
    subwindow_title.push_back("verified multi-view hypotheses");
=======
>>>>>>> v4r_root/master
    viewportNr = v4r::common::pcl_visualizer::visualization_framework ( vis, num_vertices(grph), vis_rows);

    std::pair<vertex_iter, vertex_iter> vp;
    size_t view_id = 0;
    for ( vp = vertices ( grph ); vp.first != vp.second; ++vp.first )
    {
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler_rgb ( grph[*vp.first].pScenePCl_f );
        std::stringstream cloud_name;
<<<<<<< HEAD
        cloud_name << "view_cloud_" << grph[*vp.first].pScenePCl->header.frame_id <<  "_" << view_id;
=======
        cloud_name << "view_cloud_" << grph[*vp.first].pScenePCl->header.frame_id;
>>>>>>> v4r_root/master
        vis->addPointCloud<pcl::PointXYZRGB> ( grph[*vp.first].pScenePCl_f, handler_rgb, cloud_name.str (), viewportNr[view_id * vis_rows + 0] );

        for ( size_t hyp_id = 0; hyp_id < grph[*vp.first].hypothesis_sv_.size(); hyp_id++ )
        {
            //visualize models
            //            std::string model_id = grph[*vp.first].hypothesis_sv_[hyp_id].model_id_;
            Eigen::Matrix4f trans = grph[*vp.first].hypothesis_sv_[hyp_id].transform_;
            ModelTPtr model = grph[*vp.first].hypothesis_sv_[hyp_id].model_;

            std::stringstream name;
<<<<<<< HEAD
            name << cloud_name.str() << "_sv__hypothesis_" << hyp_id << view_id;
=======
            name << cloud_name.str() << "_sv__hypothesis_" << hyp_id;
>>>>>>> v4r_root/master

            typename pcl::PointCloud<PointT>::Ptr model_aligned ( new pcl::PointCloud<PointT> );
            ConstPointInTPtr model_cloud = model->getAssembled (0.005f);
            pcl::transformPointCloud (*model_cloud, *model_aligned, trans);

            pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb_handler1 ( model_aligned );
            vis->addPointCloud<PointT> ( model_aligned, rgb_handler1, name.str (), viewportNr[view_id * vis_rows + 1] );

            if ( grph[*vp.first].hypothesis_sv_[hyp_id].verified_ )	//--show-verified-extended-hypotheses
            {
                name << "__verified";
                pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb_handler2 ( model_aligned );
                vis->addPointCloud<PointT> ( model_aligned, rgb_handler2, name.str (), viewportNr[view_id * vis_rows + 2] );
            }
        }
        for(size_t plane_id=0; plane_id < grph[*vp.first].verified_planes_.size(); plane_id++)
        {
            std::stringstream plane_name;
<<<<<<< HEAD
            plane_name << "plane_sv_" << plane_id << "_vrtx" << grph[*vp.first].pScenePCl->header.frame_id <<  "_" << view_id;
=======
            plane_name << "plane_sv_" << plane_id << "_vrtx" << grph[*vp.first].pScenePCl->header.frame_id;
>>>>>>> v4r_root/master
            pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb_handler ( grph[*vp.first].verified_planes_[plane_id] );
            vis->addPointCloud<PointT> ( grph[*vp.first].verified_planes_[plane_id], rgb_handler, plane_name.str (), viewportNr[view_id * vis_rows + 2] );
        }

        for ( size_t hyp_id = 0; hyp_id < grph[*vp.first].hypothesis_mv_.size(); hyp_id++ )
        {
            Eigen::Matrix4f trans = grph[*vp.first].hypothesis_mv_[hyp_id].transform_;
            ModelTPtr model = grph[*vp.first].hypothesis_mv_[hyp_id].model_;

            std::stringstream name;
<<<<<<< HEAD
            name << cloud_name.str() << "_mv__hypothesis_" << hyp_id << view_id;
=======
            name << cloud_name.str() << "_mv__hypothesis_" << hyp_id;
>>>>>>> v4r_root/master

            typename pcl::PointCloud<PointT>::Ptr model_aligned ( new pcl::PointCloud<PointT> );
            ConstPointInTPtr model_cloud = model->getAssembled (0.005f);
            pcl::transformPointCloud (*model_cloud, *model_aligned, trans);

            pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb_handler3 ( model_aligned );
            vis->addPointCloud<PointT> ( model_aligned, rgb_handler3, name.str (), viewportNr[view_id * vis_rows + 3] );

            if ( grph[*vp.first].hypothesis_mv_[hyp_id].verified_ )	//--show-verified-extended-hypotheses
            {
                name << "__verified";
                pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb_handler4 ( model_aligned );
                vis->addPointCloud<PointT> ( model_aligned, rgb_handler4, name.str (), viewportNr[view_id * vis_rows + 4] );
            }
        }

        for(size_t plane_id=0; plane_id < grph[*vp.first].verified_planes_.size(); plane_id++)
        {
            std::stringstream plane_name;
<<<<<<< HEAD
            plane_name << "plane_mv_" << plane_id << "_vrtx" << grph[*vp.first].pScenePCl->header.frame_id << "_" << view_id;
=======
            plane_name << "plane_mv_" << plane_id << "_vrtx" << grph[*vp.first].pScenePCl->header.frame_id;
>>>>>>> v4r_root/master
            pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb_handler ( grph[*vp.first].verified_planes_[plane_id] );
            vis->addPointCloud<PointT> ( grph[*vp.first].verified_planes_[plane_id], rgb_handler, plane_name.str (), viewportNr[view_id * vis_rows + 4] );
        }
        view_id++;
    }
    vis->spin ();
    //vis->getInteractorStyle()->saveScreenshot ( "singleview.png" );
}

void BoostGraphVisualizer::visualizeEdge (const Edge &edge, const Graph &grph)
{
    Vertex src = source ( edge, grph );
    Vertex trgt = target ( edge, grph );

    Eigen::Matrix4f transform;

    if ( grph[edge].source_id.compare( grph[src].pScenePCl->header.frame_id ) == 0)
    {
        transform = grph[edge].transformation;
    }
    else if (grph[edge].target_id.compare( grph[src].pScenePCl->header.frame_id ) == 0)
    {
        transform = grph[edge].transformation.inverse();
    }
    else
    {
        PCL_ERROR("Something is messed up with the transformation!");
    }

    if(!edge_vis_)
        edge_vis_.reset (new pcl::visualization::PCLVisualizer());

    edge_vis_->removeAllPointClouds();
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler_rgb_verified (grph[trgt].pScenePCl_f);
    edge_vis_->addPointCloud<pcl::PointXYZRGB> (grph[trgt].pScenePCl_f, handler_rgb_verified, "Hypothesis_1");
    PointInTPtr transformed_PCl (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::transformPointCloud (*grph[src].pScenePCl_f, *transformed_PCl, transform);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler_rgb_verified2 (transformed_PCl);
    edge_vis_->addPointCloud<pcl::PointXYZRGB> (transformed_PCl, handler_rgb_verified2, "Hypothesis_2");
    std::stringstream window_title;
    window_title << "transform of source view_id " << grph[src].pScenePCl->header.frame_id << " to target view_id " << grph[trgt].pScenePCl->header.frame_id << " with edge " << grph[edge].model_name;
    edge_vis_->setWindowName(window_title.str());
    edge_vis_->spin ();
}

void BoostGraphVisualizer::visualizeWorkflow ( const Vertex &vrtx, const Graph &grph, boost::shared_ptr< pcl::PointCloud<PointT> > pAccumulatedKeypoints)
{

    if(!keypoints_vis_)
    {
        keypoints_vis_.reset (new pcl::visualization::PCLVisualizer());
        keypoints_vis_->createViewPort(0,0,0.5,1,v1);
        keypoints_vis_->createViewPort(0.5,0,1,1,v2);
        keypoints_vis_->addText ("Single-view keypoints", 10, 10, "sv", v1);
        keypoints_vis_->addText ("Single-view keypoints (red) + extended keypoints (green)", 10, 10, "mv", v2);
    }
    else
    {
        keypoints_vis_->removeAllPointClouds(v1);
        keypoints_vis_->removeAllPointClouds(v2);
        keypoints_vis_->removeAllShapes(v1);
        keypoints_vis_->removeAllShapes(v2);
        keypoints_vis_->updateText("Single-view keypoints", 10, 10, "sv");
        keypoints_vis_->updateText("Single-view keypoints (red) + extended keypoints (green)", 10, 10, "mv");
    }


    //----display-keypoints--------------------

     pcl::PointCloud<PointT>::Ptr keypointsVis (new pcl::PointCloud<PointT>(*grph[vrtx].pKeypointsMultipipe_));
     pcl::PointCloud<PointT>::Ptr extendedKeypointsVis (new pcl::PointCloud<PointT>());
     extendedKeypointsVis->points.insert(extendedKeypointsVis->points.end(),
                                         pAccumulatedKeypoints->points.begin() + keypointsVis->size (),
                                         pAccumulatedKeypoints->points.end());

     for (size_t keyId = 0; keyId < keypointsVis->size (); keyId++)
     {
//         keypointsVis->points[keyId].r = 255;
//         keypointsVis->points[keyId].g = 0;
//         keypointsVis->points[keyId].b = 0;
         PointT ptTemp;
         ptTemp.getVector3fMap() = keypointsVis->points[keyId].getVector3fMap();
         std::stringstream keySs;
         keySs << "sphere_" << keyId;
         keypoints_vis_->addSphere(ptTemp, 0.003, 255, 0, 0, keySs.str(), v1);
         keySs << "_v2";
         keypoints_vis_->addSphere(ptTemp, 0.003, 255, 0, 0, keySs.str(), v2);
     }

     for (size_t keyId = 0; keyId < extendedKeypointsVis->size (); keyId++)
     {
//         extendedKeypointsVis->points[keyId].r = 0;
//         extendedKeypointsVis->points[keyId].g = 255;
//         extendedKeypointsVis->points[keyId].b = 0;
         PointT ptTemp;
         ptTemp.getVector3fMap() = extendedKeypointsVis->points[keyId].getVector3fMap();
         std::stringstream keySs;
         keySs << "sphere_extended_" << keyId;
         keypoints_vis_->addSphere(ptTemp, 0.003, 0, 255, 0, keySs.str(), v2);
     }

     pcl::PointCloud<PointT>::Ptr pTotalCloud  (new pcl::PointCloud<PointT>());
     *pTotalCloud = *(grph[vrtx].pScenePCl);// + *keypointsVis;
     //*pTotalCloud += *extendedKeypointsVis;

     pcl::visualization::PointCloudColorHandlerRGBField<PointT> handler_rgb (grph[vrtx].pScenePCl_f);
     keypoints_vis_->addPointCloud<PointT> (grph[vrtx].pScenePCl_f, handler_rgb, "total_v1", v1);

     pcl::visualization::PointCloudColorHandlerRGBField<PointT> handler_rgb2 (grph[vrtx].pScenePCl_f);
     keypoints_vis_->addPointCloud<PointT> (grph[vrtx].pScenePCl_f, handler_rgb2, "total_v2", v2);
     keypoints_vis_->spin ();

     keypoints_vis_->removeAllPointClouds(v1);
     keypoints_vis_->removeAllPointClouds(v2);
     keypoints_vis_->updateText("Single-view: Correspondence Grouping and Hypotheses Generation", 10, 10, "sv");
     keypoints_vis_->updateText("Multi-view: Correspondence Grouping and Hypotheses Generation", 10, 10, "mv");

     for ( size_t hyp_id = 0; hyp_id < grph[vrtx].hypothesis_sv_.size(); hyp_id++ )
     {
         //visualize models
         //            std::string model_id = grph[*vp.first].hypothesis_sv_[hyp_id].model_id_;
         Eigen::Matrix4f trans = grph[vrtx].hypothesis_sv_[hyp_id].transform_;
         ModelTPtr model = grph[vrtx].hypothesis_sv_[hyp_id].model_;

         std::stringstream name;
         name << "_sv__hypothesis_" << hyp_id;

         typename pcl::PointCloud<PointT>::Ptr model_aligned ( new pcl::PointCloud<PointT> );
         ConstPointInTPtr model_cloud = model->getAssembled (0.005f);
         pcl::transformPointCloud (*model_cloud, *model_aligned, trans);

         pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb_handler1 ( model_aligned );
         keypoints_vis_->addPointCloud<PointT> ( model_aligned, rgb_handler1, name.str (), v1 );
     }

     for ( size_t hyp_id = 0; hyp_id < grph[vrtx].hypothesis_mv_.size(); hyp_id++ )
     {
         //visualize models
         //            std::string model_id = grph[*vp.first].hypothesis_sv_[hyp_id].model_id_;
         Eigen::Matrix4f trans = grph[vrtx].hypothesis_mv_[hyp_id].transform_;
         ModelTPtr model = grph[vrtx].hypothesis_mv_[hyp_id].model_;

         std::stringstream name;
         name << "_mv__hypothesis_" << hyp_id;

         typename pcl::PointCloud<PointT>::Ptr model_aligned ( new pcl::PointCloud<PointT> );
         ConstPointInTPtr model_cloud = model->getAssembled (0.005f);
         pcl::transformPointCloud (*model_cloud, *model_aligned, trans);

         pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb_handler1 ( model_aligned );
         keypoints_vis_->addPointCloud<PointT> ( model_aligned, rgb_handler1, name.str (), v2 );
     }

    keypoints_vis_->spin();
    keypoints_vis_->removeAllPointClouds(v1);
    keypoints_vis_->removeAllPointClouds(v2);
    keypoints_vis_->updateText("Single-view: Verified Hypotheses", 10, 10, "sv");
    keypoints_vis_->updateText("Multi-view: Verified Hypotheses", 10, 10, "mv");

    for ( size_t hyp_id = 0; hyp_id < grph[vrtx].hypothesis_sv_.size(); hyp_id++ )
    {
        if ( grph[vrtx].hypothesis_sv_[hyp_id].verified_ )	//--show-verified-extended-hypotheses
        {
            Eigen::Matrix4f trans = grph[vrtx].hypothesis_sv_[hyp_id].transform_;
            ModelTPtr model = grph[vrtx].hypothesis_sv_[hyp_id].model_;
            typename pcl::PointCloud<PointT>::Ptr model_aligned ( new pcl::PointCloud<PointT> );
            ConstPointInTPtr model_cloud = model->getAssembled (0.005f);
            pcl::transformPointCloud (*model_cloud, *model_aligned, trans);

            std::stringstream name;
            name << "_sv__hypothesis_" << hyp_id << "__verified";
            pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb_handler2 ( model_aligned );
            keypoints_vis_->addPointCloud<PointT> ( model_aligned, rgb_handler2, name.str (), v1 );
        }
    }

    for ( size_t hyp_id = 0; hyp_id < grph[vrtx].hypothesis_mv_.size(); hyp_id++ )
    {
        if (  grph[vrtx].hypothesis_mv_[hyp_id].verified_ )	//--show-verified-extended-hypotheses
        {
            Eigen::Matrix4f trans = grph[vrtx].hypothesis_mv_[hyp_id].transform_;
            ModelTPtr model = grph[vrtx].hypothesis_mv_[hyp_id].model_;
            typename pcl::PointCloud<PointT>::Ptr model_aligned ( new pcl::PointCloud<PointT> );
            ConstPointInTPtr model_cloud = model->getAssembled (0.005f);
            pcl::transformPointCloud (*model_cloud, *model_aligned, trans);

            std::stringstream name;
            name << "_mv__hypothesis_" << hyp_id << "__verified";
            pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb_handler2 ( model_aligned );
            keypoints_vis_->addPointCloud<PointT> ( model_aligned, rgb_handler2, name.str (), v2 );
        }
    }
   keypoints_vis_->spin();
}


void BoostGraphVisualizer::createImage(const Vertex &src, const Graph &grph, std::string outputfile)
{
    cv::Mat_ < cv::Vec3b > colorImage;
    PCLOpenCV::ConvertPCLCloud2Image<PointT> (grph[src].pScenePCl_f, colorImage);
    cv::imwrite(outputfile, colorImage);

//    //transform gt_cloud to organized point cloud and then to image
//    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_organized(new pcl::PointCloud<pcl::PointXYZRGB>);
//    cloud_organized->width = scene->width;
//    cloud_organized->height = scene->height;
//    cloud_organized->is_dense = scene->is_dense;
//    cloud_organized->points.resize(scene->points.size());
//    for(size_t kk=0; kk < cloud_organized->points.size(); kk++)
//    {
//        cloud_organized->points[kk].x = cloud_organized->points[kk].y = cloud_organized->points[kk].z =
//                std::numeric_limits<float>::quiet_NaN();

//        cloud_organized->points[kk].r = cloud_organized->points[kk].g = cloud_organized->points[kk].b = 255;
//    }

//    float f = 525.f;
//    float cx = (static_cast<float> (scene->width) / 2.f - 0.5f);
//    float cy = (static_cast<float> (scene->height) / 2.f - 0.5f);

//    int ws2 = 1;
//    for (size_t kk = 0; kk < gt_cloud->points.size (); kk++)
//    {
//      float x = gt_cloud->points[kk].x;
//      float y = gt_cloud->points[kk].y;
//      float z = gt_cloud->points[kk].z;
//      int u = static_cast<int> (f * x / z + cx);
//      int v = static_cast<int> (f * y / z + cy);

//      for(int uu = (u-ws2); uu < (u+ws2); uu++)
//      {
//          for(int vv = (v-ws2); vv < (v+ws2); vv++)
//          {
//              //Not out of bounds
//                if ((uu >= static_cast<int> (scene->width)) ||
//                    (vv >= static_cast<int> (scene->height)) || (uu < 0) || (vv < 0))
//                  continue;

//                float z_oc = cloud_organized->at (uu, vv).z;

//                if(pcl_isnan(z_oc))
//                {
//                    cloud_organized->at (uu, vv) = gt_cloud->points[kk];
//                }
//                else
//                {
//                    if(z < z_oc)
//                    {
//                        cloud_organized->at (uu, vv) = gt_cloud->points[kk];
//                    }
//                }
//          }
//      }

//      /*//Not out of bounds
//      if ((u >= static_cast<int> (scene->width)) ||
//          (v >= static_cast<int> (scene->height)) || (u < 0) || (v < 0))
//        continue;

//      float z_oc = gt_cloud_organized->at (u, v).z;

//      if(pcl_isnan(z_oc))
//      {
//          gt_cloud_organized->at (u, v) = gt_cloud->points[kk];
//      }
//      else
//      {
//          if(z < z_oc)
//          {
//              gt_cloud_organized->at (u, v) = gt_cloud->points[kk];
//          }
//      }*/
//    }

//    {
//        std::stringstream rel_path;
//        rel_path << scene_images_gt_path << "/";
//        for(size_t k=0; k < (strs.size() - 1); k++)
//        {
//            rel_path << strs[k] << "/";
//            bf::path p = rel_path.str();
//            if(!bf::exists(p))
//            {
//                bf::create_directory(p);
//            }
//        }

//        std::cout << rel_path.str() << "/" << scene_name << ".jpg" << std::endl;

//        std::stringstream image_path;
//        image_path << rel_path.str() << "/" << scene_name << ".jpg";


//        cv::Mat_ < cv::Vec3b > colorImage;
//        PCLOpenCV::ConvertPCLCloud2Image<PointT> (cloud_organized, colorImage);
//        /*cv::namedWindow("image");
//        cv::imshow("image", colorImage);
//        cv::waitKey(0);*/
//        cv::imwrite(image_path.str(), colorImage);
//    }
}
