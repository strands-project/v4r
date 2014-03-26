#ifndef MY_CLASSIFIER_H
#define MY_CLASSIFIER_H

#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET

#include <vector>
#include <cv.h>
#include <ros/ros.h>
#include <pcl/common/common.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/ml/kmeans.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/io.h>

#include <faat_pcl/3d_rec_framework/pc_source/unregistered_views_source.h>
#include <faat_pcl/3d_rec_framework/img_source/source2d.h>
#include <faat_pcl/3d_rec_framework/feature_wrapper/local/image/sift_local_estimator.h>
#include <faat_pcl/3d_rec_framework/feature_wrapper/global/esf_estimator.h>
#include <faat_pcl/3d_rec_framework/pipeline/local_recognizer.h>

typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT>::Ptr PointInTPtr;
typedef pcl::Histogram<128> SIFTFeatureT;
typedef pcl::PointCloud<SIFTFeatureT>::Ptr SIFTFeatureTPtr;
typedef pcl::ESFSignature640 ESFFeatureT;
typedef pcl::PointCloud<ESFFeatureT>::Ptr ESFFeatureTPtr;
typedef faat_pcl::rec_3d_framework::Model<PointT> ModelT;
typedef faat_pcl::rec_3d_framework::Model2D Model2DT;
typedef boost::shared_ptr<ModelT> ModelTPtr;
typedef boost::shared_ptr<Model2DT> Model2DTPtr;

class MyClassifier
{
private:
    pcl::PointCloud<PointT>::Ptr pInputCloud_;
    std::string models_dir_, training_dir_;
    pcl::KdTreeFLANN<ESFFeatureT> kdtree_;
    ros::NodeHandle *n_;
    std::string test_filename_;
    boost::shared_ptr < faat_pcl::rec_3d_framework::UnregisteredViewsSource  <PointT>
            > pPCSource_;
    boost::shared_ptr < faat_pcl::rec_3d_framework::Source2D> pImgSource_;
    boost::shared_ptr < faat_pcl::rec_3d_framework::Source<PointT> > cast_source_;
    boost::shared_ptr<std::vector<ModelTPtr> > models3D_;
    boost::shared_ptr<std::vector<Model2DTPtr> > models2D_;
    boost::shared_ptr < faat_pcl::rec_3d_framework::SIFTLocalEstimation<PointT, SIFTFeatureT > > sift_estimator_;
    boost::shared_ptr < faat_pcl::rec_3d_framework::ESFEstimation<PointT, ESFFeatureT > > esf_estimator_;
    pcl::PointCloud<ESFFeatureT>::Ptr esf_signatures_ ;

public:
    MyClassifier()
    {
        pInputCloud_.reset(new pcl::PointCloud<PointT>());
        models_dir_ = "/home/thomas/data/Cat50_TestDB_small/pcd_binary";
    }

    void init(int argc, char ** argv);
    void setInputCloud(pcl::PointCloud<PointT> &cloud);
    void trainClassifier();
    void classify();
};

#endif //MY_CLASSIFIER_H
