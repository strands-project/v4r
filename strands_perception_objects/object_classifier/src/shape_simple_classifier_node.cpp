/*
 * shape_simple_classifier_node.cpp
 *
 *  Created on: Sep 7, 2013
 *      Author: aitor
 */

#include "ros/ros.h"
#include "sensor_msgs/PointCloud2.h"
#include <pcl/common/common.h>
#include <pcl/console/parse.h>
#include <pcl_conversions.h>
#include <pcl/filters/passthrough.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <faat_pcl/3d_rec_framework/pc_source/mesh_source.h>
#include <faat_pcl/3d_rec_framework/feature_wrapper/global/vfh_estimator.h>
#include <faat_pcl/3d_rec_framework/feature_wrapper/global/esf_estimator.h>
#include <faat_pcl/3d_rec_framework/feature_wrapper/global/cvfh_estimator.h>
#include <faat_pcl/3d_rec_framework/utils/metrics.h>
#include <faat_pcl/3d_rec_framework/pipeline/global_nn_classifier.h>
#include "classifier_srv_definitions/segment_and_classify.h"
#include "classifier_srv_definitions/classify.h"
#include "object_perception_msgs/classification.h"
#include <pcl/apps/dominant_plane_segmentation.h>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/lexical_cast.hpp>
#include <Eigen/Eigenvalues>
#include "segmentation_srv_definitions/segment.h"
#include "std_msgs/String.h"
#include "std_msgs/Int32MultiArray.h"
#include "geometry_msgs/Point32.h"

class ShapeClassifier
{
  private:
    typedef pcl::PointXYZ PointT;
    std::string models_dir_;
    std::string training_dir_;
    std::string desc_name_;
    int NN_;
    double chop_at_z_;
    pcl::PointCloud<PointT>::Ptr frame_;
    std::vector<pcl::PointIndices> cluster_indices_;
    std::vector < std::string > categories_;
    std::vector<float> conf_;

    boost::shared_ptr<faat_pcl::rec_3d_framework::GlobalNNPipeline<flann::L1, PointT, pcl::ESFSignature640> > classifier_;
    ros::ServiceServer segment_and_classify_service_;
    ros::ServiceServer classify_service_;
    ros::NodeHandle *n_;

    bool classify(classifier_srv_definitions::classify::Request & req,
                  classifier_srv_definitions::classify::Response & response)
    {
        pcl::fromROSMsg(req.cloud, *frame_);
        classifier_->setInputCloud(frame_);

        for(size_t i=0; i < req.clusters_indices.size(); i++)
        {
          std::vector<int> cluster_indices_int;
          Eigen::Vector4f centroid;
          Eigen::Matrix3f covariance_matrix;

          for(size_t kk=0; kk < req.clusters_indices[i].data.size(); kk++)
          {
              cluster_indices_int.push_back(static_cast<int>(req.clusters_indices[i].data[kk]));
          }

          classifier_->setIndices(cluster_indices_int);
          classifier_->classify ();
          classifier_->getCategory (categories_);
          classifier_->getConfidence (conf_);

          std::cout << "for cluster " << i << " with size " << cluster_indices_int.size() << ", I have following hypotheses: " << std::endl;

          object_perception_msgs::classification class_tmp;
          for(size_t kk=0; kk < categories_.size(); kk++)
          {
            std::cout << categories_[kk] << " with confidence " << conf_[kk] << std::endl;
            std_msgs::String str_tmp;
            str_tmp.data = categories_[kk];
            class_tmp.class_type.push_back(str_tmp);
            class_tmp.confidence.push_back(conf_[kk]);
          }
          response.class_results.push_back(class_tmp);

          pcl::PointCloud<PointT>::Ptr pClusterPCl_transformed (new pcl::PointCloud<PointT>());
          pcl::computeMeanAndCovarianceMatrix(*frame_, cluster_indices_int, covariance_matrix, centroid);
          //Eigen::EigenSolver<Eigen::Matrix3f> es(covariance_matrix);
          //std::cout << "eigenvector and eigenvalues for cluster " << i << std::endl;
          //Eigen::Matrix3f eigenvec_matrix = es.eigenvectors().real();
          //std::cout << es.eigenvalues().real() << std::endl << " vec: " << es.eigenvectors().real() << std::endl;
          Eigen::Matrix3f eigvects;
          Eigen::Vector3f eigvals;
          pcl::eigen33(covariance_matrix, eigvects,  eigvals);
          //std:cout << "PCL Eigen: " << std::endl << eigvals << std::endl << eigvects << std::endl;

          Eigen::Vector3f centroid_transformed = eigvects.transpose() * centroid.topRows(3);

          Eigen::Matrix4f transformation_matrix = Eigen::Matrix4f::Zero(4,4);
          transformation_matrix.block<3,3>(0,0) = eigvects.transpose();
          transformation_matrix.block<3,1>(0,3) = -centroid_transformed;
          transformation_matrix(3,3) = 1;

          pcl::transformPointCloud(*frame_, cluster_indices_int, *pClusterPCl_transformed, transformation_matrix);

          //pcl::transformPointCloud(*frame_, cluster_indices_int, *frame_eigencoordinates_, eigvects);
          PointT min_pt, max_pt;
          pcl::getMinMax3D(*pClusterPCl_transformed, min_pt, max_pt);
          std::cout << "Elongations along eigenvectors: " << max_pt.x - min_pt.x << ", " << max_pt.y - min_pt.y
                    << ", " << max_pt.z - min_pt.z << std::endl;
          geometry_msgs::Point32 centroid_ros;
          centroid_ros.x = centroid[0];
          centroid_ros.y = centroid[1];
          centroid_ros.z = centroid[2];
          response.centroid.push_back(centroid_ros);

          /*boost::shared_ptr<pcl::visualization::PCLVisualizer> vis;
          vis.reset(new pcl::visualization::PCLVisualizer("cluster visualization"));
          vis->addCoordinateSystem(0.2f);
          vis->addPointCloud(pClusterPCl_transformed, "scene_cloud_eigen_transformed");
          vis->spin();*/

//          if(categories_.size() == 1)
//          {
//            std_msgs::String ss;
//            ss.data = categories_[0];
//            response.categories.push_back(ss);
//            response.confidence.push_back(conf_[0]);
//          }
//          else if(categories_.size() == 0)
//          {
//            //weird case, do nothing...
//          }
//          else
//          {
//            //at least 2 categories
//            std::vector< std::pair<float, std::string> > conf_categories_map_;
//            for (size_t kk = 0; kk < categories_.size (); kk++)
//            {
//              conf_categories_map_.push_back(std::make_pair(conf_[kk], categories_[kk]));
//            }

//            std::sort (conf_categories_map_.begin (), conf_categories_map_.end (),
//                       boost::bind (&std::pair<float, std::string>::first, _1) > boost::bind (&std::pair<float, std::string>::first, _2));

//            /*for (size_t kk = 0; kk < categories.size (); kk++)
//            {
//              std::cout << conf_categories_map_[kk].first << std::endl;
//            }*/

//            if( (conf_categories_map_[1].first / conf_categories_map_[0].first) < 0.85f)
//            {


//              if (!boost::starts_with(conf_categories_map_[0].second, "unknown"))
//              {
//                std_msgs::String ss;
//                ss.data = conf_categories_map_[0].second;
//                response.categories.push_back(ss);
//                response.confidence.push_back(conf_categories_map_[0].first);
//              }
//            }
//          }
        }

        response.clusters_indices = req.clusters_indices;
        return true;
    }

    bool segmentAndClassify(classifier_srv_definitions::segment_and_classify::Request & req,
                               classifier_srv_definitions::segment_and_classify::Response & response)
    {
      //------- Segmentation------------------
      /*Eigen::Vector4f table_plane;
      doSegmentation<PointT>(frame_, cluster_indices_, table_plane, chop_at_z_);*/
      //std::vector<pcl::PointCloud<PointT>::Ptr> clusters;
      /*
      float tolerance = 0.005f;
      pcl::apps::DominantPlaneSegmentation<pcl::PointXYZ> dps;
      dps.setInputCloud (frame_);
      dps.setMaxZBounds (chop_at_z_);
      dps.setObjectMinHeight (tolerance);
      dps.setMinClusterSize (1000);
      dps.setWSize (9);
      dps.setDistanceBetweenClusters (0.1f);

      dps.setDownsamplingSize (0.01f);
      dps.compute_fast (clusters);
      dps.getIndicesClusters (cluster_indices_);
      Eigen::Vector4f table_plane;
      dps.getTableCoefficients (table_plane);*/
      //-------------------------------------

      ros::ServiceClient segmentation_client = n_->serviceClient<segmentation_srv_definitions::segment>("/object_segmenter_service/object_segmenter");
      segmentation_srv_definitions::segment seg_srv;
      seg_srv.request.cloud = req.cloud;
      if (segmentation_client.call(seg_srv))
      {
          std::cout << "Number of clusters:" << static_cast<int>(seg_srv.response.clusters_indices.size()) << std::endl;
          classifier_srv_definitions::classify srv;
          srv.request.cloud = req.cloud;
          srv.request.clusters_indices = seg_srv.response.clusters_indices;
          classify(srv.request, srv.response);
          response.class_results = srv.response.class_results;
          response.clusters_indices = srv.response.clusters_indices;
          response.centroid = srv.response.centroid;
      }
      else
      {
          ROS_ERROR("Failed to call segmentation service.");
          return false;
      }
      return true;
    }

  public:
    ShapeClassifier()
    {
      //default values
      desc_name_ = "esf";
      NN_ = 50;
      chop_at_z_ = 1.f;
      frame_.reset(new pcl::PointCloud<PointT>());
    }

    void initialize(int argc, char ** argv)
    {
        ros::init (argc, argv, "classifier_service");
        n_ = new ros::NodeHandle ( "~" );
        n_->getParam ( "models_dir", models_dir_ );
        n_->getParam ( "training_dir", training_dir_ );
        n_->getParam ( "descriptor_name", desc_name_ );
        n_->getParam ( "nn", NN_ );
        n_->getParam ( "chop_z", chop_at_z_ );

        ROS_INFO("models_dir, training dir, desc:  %s, %s, %s",  models_dir_.c_str(), training_dir_.c_str(), desc_name_.c_str());

      if(models_dir_.compare("") == 0)
      {
        PCL_ERROR("Set -models_dir option in the command line, ABORTING");
        return;
      }

      if(training_dir_.compare("") == 0)
      {
        PCL_ERROR("Set -training_dir option in the command line, ABORTING");
        return;
      }

      boost::shared_ptr<faat_pcl::rec_3d_framework::MeshSource<PointT> > mesh_source (new faat_pcl::rec_3d_framework::MeshSource<PointT>);
      mesh_source->setPath (models_dir_);
      mesh_source->setResolution (150);
      mesh_source->setTesselationLevel (0);
      mesh_source->setViewAngle (57.f);
      mesh_source->setRadiusSphere (1.f);
      mesh_source->setModelScale (1.f);
      mesh_source->generate (training_dir_);

      boost::shared_ptr<faat_pcl::rec_3d_framework::Source<PointT> > cast_source;
      cast_source = boost::static_pointer_cast<faat_pcl::rec_3d_framework::MeshSource<PointT> > (mesh_source);

      boost::shared_ptr<faat_pcl::rec_3d_framework::ESFEstimation<PointT, pcl::ESFSignature640> > estimator;
      estimator.reset (new faat_pcl::rec_3d_framework::ESFEstimation<PointT, pcl::ESFSignature640>);

      boost::shared_ptr<faat_pcl::rec_3d_framework::GlobalEstimator<PointT, pcl::ESFSignature640> > cast_estimator;
      cast_estimator = boost::dynamic_pointer_cast<faat_pcl::rec_3d_framework::ESFEstimation<PointT, pcl::ESFSignature640> > (estimator);

      classifier_.reset(new faat_pcl::rec_3d_framework::GlobalNNPipeline<flann::L1, PointT, pcl::ESFSignature640>);
      classifier_->setDataSource (cast_source);
      classifier_->setTrainingDir (training_dir_);
      classifier_->setDescriptorName (desc_name_);
      classifier_->setFeatureEstimator (cast_estimator);
      classifier_->setNN (NN_);
      classifier_->initialize (false);

      segment_and_classify_service_ = n_->advertiseService("segment_and_classify", &ShapeClassifier::segmentAndClassify, this);
      classify_service_ = n_->advertiseService("classify", &ShapeClassifier::classify, this);
      ros::spin();
    }
};

int
main (int argc, char ** argv)
{
  ShapeClassifier m;
  m.initialize (argc, argv);

  return 0;
}
