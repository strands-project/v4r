#include "my_classifier.h"
#include <iostream>
#include <fstream>

void MyClassifier::init(int argc, char **argv)
{
    ros::init(argc, argv, "my_classifier");
    n_ = new ros::NodeHandle ( "~" );
    n_->getParam ( "test_filename", test_filename_ );
    std::string indices_prefix = "object_indices_";


    // load a .pcd file, extract indices of object and classify
    if(bf::exists(test_filename_))
    {
        std::string directory, filename;
        char sep = '/';
         #ifdef _WIN32
            sep = '\\';
         #endif

        size_t position = test_filename_.rfind(sep);
           if (position != std::string::npos)
           {
              directory = test_filename_.substr(0, position);
              filename = test_filename_.substr(position+1, test_filename_.length()-1);
           }

       std::stringstream path_oi;
       path_oi << directory << "/" << indices_prefix << filename ;

        if(bf::exists(path_oi.str()))
        {
            pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);
            pcl::io::loadPCDFile (test_filename_, *cloud);

            pcl::PointCloud<IndexPoint> obj_indices_cloud;
            pcl::io::loadPCDFile (path_oi.str(), obj_indices_cloud);
            pcl::PointCloud<PointT>::Ptr pFilteredCloud;
            pFilteredCloud.reset(new pcl::PointCloud<PointT>());
            pcl::PointIndices indices;
            indices.indices.resize(obj_indices_cloud.points.size());
            for(size_t kk=0; kk < obj_indices_cloud.points.size(); kk++)
              indices.indices[kk] = obj_indices_cloud.points[kk].idx;
            pcl::copyPointCloud(*cloud, indices, *pFilteredCloud);

            this->setInputCloud(*pFilteredCloud);
            std::cout << "Test point cloud and indices loaded for file: " << test_filename_ << std::endl;
        }
    }
}

void MyClassifier::setInputCloud(pcl::PointCloud<PointT> &cloud)
{
    *pInputCloud_ = cloud;
}

void MyClassifier::trainClassifier()
{
    int num_clusters = 80;

    pImgSource_.reset(new faat_pcl::rec_3d_framework::Source2D ());
    pImgSource_->setPath(models_dir_);
    pImgSource_->generate();
    models2D_ = pImgSource_->getModels();

    pPCSource_.reset(new faat_pcl::rec_3d_framework::UnregisteredViewsSource  <PointT>());
    pPCSource_->setPath(models_dir_);
    std::string dummy = "";
    pPCSource_->generate(dummy);
    cast_source_ = boost::static_pointer_cast<faat_pcl::rec_3d_framework::UnregisteredViewsSource<PointT> > (pPCSource_);
    models3D_ = cast_source_->getModels ();

    sift_estimator_.reset (new faat_pcl::rec_3d_framework::SIFTLocalEstimation<PointT, SIFTFeatureT >());
    esf_estimator_.reset (new faat_pcl::rec_3d_framework::ESFEstimation<PointT, ESFFeatureT >());

    esf_signatures_.reset(new pcl::PointCloud<ESFFeatureT>());
    esf_signatures_->width = models3D_->size();
    esf_signatures_->height = 1;
    esf_signatures_->resize(models3D_->size());
    //std::vector<std::string> model_v;


    // ---Assign class names to unique discrete identifiers ------
    std::map<std::string, size_t> class_map;
    for(size_t i=0; i < models3D_->size(); i++)
    {
        if(class_map.find(models3D_->at(i)->class_) == class_map.end())
        {
            size_t class_label = class_map.size();
            std::cout << "Class " << models3D_->at(i)->class_
                      << " corresponds to id " << class_label << std::endl;
            class_map[models3D_->at(i)->class_] = class_label;
        }
    }
    for(size_t i=0; i < models2D_->size(); i++)
    {
        if(class_map.find(models2D_->at(i)->class_) == class_map.end())
        {
            size_t class_label = class_map.size();
            std::cout << "Class " << models2D_->at(i)->class_
                      << " corresponds to id " << class_label << std::endl;
            class_map[models2D_->at(i)->class_] = class_label;
        }
    }
    size_t num_classes = class_map.size();
    // -------------------


    // ---Assign view names to unique discrete identifiers ------
    std::vector< std::map <std::string, size_t> > view_map_per_class_v;
    view_map_per_class_v.resize(num_classes);
    for(size_t i=0; i < models3D_->size(); i++)
    {
        if(view_map_per_class_v[class_map[models3D_->at(i)->class_]].find(
                    models3D_->at(i)->id_) == view_map_per_class_v[class_map[models3D_->at(i)->class_]].end())
        {
            size_t model_label = view_map_per_class_v[class_map[models3D_->at(i)->class_]].size();
            std::cout << "model id " << models3D_->at(i)->id_ << " for class "
                      << models3D_->at(i)->class_ << " corresponds to id "
                      << model_label << std::endl;
            view_map_per_class_v[class_map[models3D_->at(i)->class_]][models3D_->at(i)->id_] = model_label;

        }
    }
    for(size_t i=0; i < models2D_->size(); i++)
    {
        if(view_map_per_class_v[class_map[models2D_->at(i)->class_]].find(
                    models2D_->at(i)->id_) == view_map_per_class_v[class_map[models2D_->at(i)->class_]].end())
        {
            size_t model_label = view_map_per_class_v[class_map[models2D_->at(i)->class_]].size();
            std::cout << "model id " << models2D_->at(i)->id_ << " for class "
                      << models2D_->at(i)->class_ << " corresponds to id "
                      << model_label << std::endl;
            view_map_per_class_v[class_map[models2D_->at(i)->class_]][models2D_->at(i)->id_] = model_label;

        }
    }
    // -------------------

    size_t num_signatures=0;
    std::vector<std::vector<std::vector<SIFTFeatureT> > >
            sift_signatures_per_class_per_view_v;
    sift_signatures_per_class_per_view_v.resize(num_classes);

    for(size_t i=0; i<sift_signatures_per_class_per_view_v.size(); i++)
    {
        //check if this is correct
        sift_signatures_per_class_per_view_v[i].resize(view_map_per_class_v[i].size());
    }

    for(size_t i=0; i < models3D_->size(); i++)
    {
        PointInTPtr sift_keypoints;
        sift_keypoints.reset(new pcl::PointCloud<PointT>());
        SIFTFeatureTPtr sift_signatures;
        sift_signatures.reset(new pcl::PointCloud<SIFTFeatureT>());
        std::vector<float> sift_scale;
        sift_estimator_->setIndices(models3D_->at(i)->indices_->at(0));
        sift_estimator_->estimate(models3D_->at(i)->views_->at(0), sift_keypoints, sift_signatures, sift_scale);
        //num_signatures_per_view_per_class_v[class_map[models_->at(i)->class_].
          //      push_back(sift_signatures->points.size());

        for(size_t kk=0; kk<sift_signatures->points.size(); kk++)
        {
            sift_signatures_per_class_per_view_v[ class_map[models3D_->at(i)->class_] ]
                    [view_map_per_class_v[class_map[models3D_->at(i)->class_]][models3D_->at(i)->id_]]
                    .push_back(sift_signatures->points[kk]);
            num_signatures++;
        }

        /*pcl::PointCloud<ESFFeatureT>::CloudVectorType esf_signature;
        std::vector<Eigen::Vector3f> centroids;
        pcl::PointCloud<PointT>::Ptr pFilteredCloud, pProcessedCloud;
        pFilteredCloud.reset(new pcl::PointCloud<PointT>());
        pProcessedCloud.reset(new pcl::PointCloud<PointT>());
        pcl::copyPointCloud(*(models3D_->at(i)->views_->at(0)), models3D_->at(i)->indices_->at(0), *pFilteredCloud);
        //std::vector<PointT> test = pFilteredCloud->points();
        esf_estimator_->estimate(pFilteredCloud, pProcessedCloud, esf_signature, centroids);
        for(int kk=0; kk<640; kk++)
        {
            esf_signatures_->points[i].histogram[kk] = esf_signature[0].points[0].histogram[kk];
        }*/
    }

    for(size_t i=0; i < models2D_->size(); i++)
    {
        //cv::Mat image = cv::imread("/home/thomas/data/Cat50_TestDB_small/image_color/1.png", CV_LOAD_IMAGE_COLOR);
        std::vector<SiftGPU::SiftKeypoint> keypoints;
        SIFTFeatureTPtr sift_signatures;
        sift_signatures.reset(new pcl::PointCloud<SIFTFeatureT>());
        std::vector<float> sift_scale;
        sift_estimator_->estimate(*(models2D_->at(i)->view_), keypoints, sift_signatures, sift_scale);

//        std::vector<SiftGPU::SiftKeypoint> keypoints;
//        SIFTFeatureTPtr sift_signatures;
//        sift_signatures.reset(new pcl::PointCloud<SIFTFeatureT>());
//        std::vector<float> sift_scale;
//        sift_estimator_->estimate(models2D_->at(i)->view_, keypoints, sift_signatures, sift_scale);

        //num_signatures_per_view_per_class_v[class_map[models_->at(i)->class_].
          //      push_back(sift_signatures->points.size());

        for(size_t kk=0; kk<sift_signatures->points.size(); kk++)
        {
            sift_signatures_per_class_per_view_v[ class_map[models2D_->at(i)->class_] ]
                    [view_map_per_class_v[class_map[models2D_->at(i)->class_]][models2D_->at(i)->id_]]
                    .push_back(sift_signatures->points[kk]);
            num_signatures++;
        }
    }
    //kdtree_.setInputCloud(esf_signatures_);

    cv::Mat labels;
    std::vector<cv::Mat> labels_per_class_v;
    labels_per_class_v.resize(num_classes);

    std::cout << "Generating Bag of Words..." << std::endl;
    //{
     //   pcl::ScopeTime ttt ("Codebook Generation");
        cv::Mat centers(num_clusters, 128, CV_32F);
        cv::Mat samples(num_signatures, 128, CV_32F);
        size_t row_id=0;
        for(size_t class_id=0; class_id < sift_signatures_per_class_per_view_v.size(); class_id++)
        {
            for (size_t  view_id=0; view_id < sift_signatures_per_class_per_view_v[class_id].size(); view_id++)
            {
                for (size_t signature_id=0; signature_id < sift_signatures_per_class_per_view_v[class_id][view_id].size(); signature_id++)
                {
                    for(size_t jjj=0; jjj < 128; jjj++)
                    {
                        samples.at<float>(row_id,jjj) =
                                static_cast<float>(sift_signatures_per_class_per_view_v[class_id][view_id][signature_id].histogram[jjj]);
                    }
                row_id++;
                }
           }
        }
        cv::kmeans(samples, num_clusters, labels, cv::TermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 50, 1.0), 1, cv::KMEANS_RANDOM_CENTERS, centers);
        std::cout << "labels = "<< std::endl << " "  << labels << std::endl << std::endl;
        std::cout << "centers = "<< std::endl << " "  << centers << std::endl << std::endl;

        std::vector<std::vector<std::vector<size_t> > > hist_per_class_per_view_v;
        hist_per_class_per_view_v.resize(num_classes);
        row_id=0;
        for(size_t class_id=0; class_id < sift_signatures_per_class_per_view_v.size(); class_id++)
        {
            hist_per_class_per_view_v[class_id].resize(view_map_per_class_v[class_id].size());

            for (size_t  view_id=0; view_id < sift_signatures_per_class_per_view_v[class_id].size(); view_id++)
            {
                hist_per_class_per_view_v[class_id][view_id].resize(num_clusters);
                for(size_t hist_bin_id=0; hist_bin_id < num_clusters; hist_bin_id++)
                {
                    hist_per_class_per_view_v[class_id][view_id][hist_bin_id]=0;
                }

                for (size_t signature_id=0; signature_id < sift_signatures_per_class_per_view_v[class_id][view_id].size(); signature_id++)
                {
                    int hist_bin_id = labels.at<int>(row_id, 0);
                    hist_per_class_per_view_v[class_id][view_id][hist_bin_id] ++;
                    row_id++;
                }
           }
        }


        for(size_t class_id=0; class_id < hist_per_class_per_view_v.size(); class_id++)
        {
            ofstream myfile;
            std::stringstream filename;
            filename << "hist_class" << class_id << ".txt";
            myfile.open(filename.str().c_str());

            for(size_t view_id=0; view_id < hist_per_class_per_view_v[class_id].size(); view_id++)
            {
                for(size_t hist_bin_id=0; hist_bin_id < num_clusters; hist_bin_id++)
                {
                    myfile << hist_per_class_per_view_v[class_id][view_id][hist_bin_id] << " " ;
                }
                myfile << "\n";
            }
           myfile.close();
         }
    // pcl::Kmeans::Centroids centroids_sift_clusters = mycluster.get_centroids();

      /*boost::shared_ptr < faat_pcl::rec_3d_framework::LocalEstimator<PointT, pcl::Histogram<128> > > cast_estimator;
      cast_estimator = boost::dynamic_pointer_cast<faat_pcl::rec_3d_framework::SIFTLocalEstimation<PointT, pcl::Histogram<128> > > (estimator);

      std::string desc_name = "sift";
      std::string idx_flann_fn = "sift_flann.idx";
      boost::shared_ptr<faat_pcl::rec_3d_framework::LocalRecognitionPipeline<flann::L1, PointT, pcl::Histogram<128> > > new_sift_local_;
      new_sift_local_.reset (new faat_pcl::rec_3d_framework::LocalRecognitionPipeline<flann::L1, PointT, pcl::Histogram<128> > (idx_flann_fn));
      new_sift_local_->setDataSource (cast_source);
      new_sift_local_->setTrainingDir (training_dir_);
      new_sift_local_->setDescriptorName (desc_name);
      new_sift_local_->setICPIterations (0);
      new_sift_local_->setFeatureEstimator (cast_estimator);
      new_sift_local_->setUseCache (true);
      new_sift_local_->setKnn (5);
      new_sift_local_->setUseCache (true);
      new_sift_local_->initialize (false);*/
}

void MyClassifier::classify()
{
    if(pInputCloud_->size() <= 0)
    {
        PCL_ERROR("No input cloud defined.");
    }
    else
    {
        pcl::PointCloud<ESFFeatureT>::CloudVectorType esf_signature;
        std::vector<Eigen::Vector3f> centroids;
        pcl::PointCloud<PointT>::Ptr pProcessedCloud;
        pProcessedCloud.reset(new pcl::PointCloud<PointT>());

        esf_estimator_->estimate(pInputCloud_, pProcessedCloud, esf_signature, centroids);
        int K = 10;

        std::vector<int> pointIdxNKNSearch(K);
        std::vector<float> pointNKNSquaredDistance(K);

        if ( kdtree_.nearestKSearch (esf_signature[0].points[0], K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0 )
        {
          for (size_t i = 0; i < pointIdxNKNSearch.size (); ++i)
           std::cout << "  Class:  "  <<   models3D_->at(pointIdxNKNSearch[i])->class_
                        << "  Id:  "  <<   models3D_->at(pointIdxNKNSearch[i])->id_
                      << " (squared distance: " << pointNKNSquaredDistance[i] << ")" << std::endl;
        }
    }
}
