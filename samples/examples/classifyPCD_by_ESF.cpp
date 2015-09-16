/*
 * Author: Thomas Faeulhammer
 * Date: Sept., 2015
 *
 * Segments PCD files and classifies them using the ESF descriptor
 *
 */

#include <v4r/recognition/source.h>
#include <v4r/recognition/mesh_source.h>
#include <v4r/features/global_estimator.h>
#include <v4r/features/esf_estimator.h>
#include <v4r/recognition/global_nn_classifier.h>
#include <v4r/segmentation/pcl_segmentation_methods.h>

#include <pcl/console/parse.h>
#include <pcl/common/centroid.h>
#include <pcl/visualization/pcl_visualizer.h>

class ShapeClassifier
{
private:
    typedef pcl::PointXYZ PointT;
    std::string models_dir_, training_dir_, test_dir_;
    int knn_;
    typename boost::shared_ptr<v4r::Source<PointT> > source_;
    v4r::GlobalNNPipeline<flann::L1, PointT, pcl::ESFSignature640> classifier_;
    pcl::visualization::PCLVisualizer::Ptr vis_;
    v4r::PCLSegmenter<pcl::PointXYZRGB>::Parameter seg_param_;

    int vp1_, vp2_;
    std::vector<pcl::PointIndices> found_clusters_;
    std::vector< std::vector < std::string > > categories_;
    std::vector< std::vector < float > >confidences_;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_;
    bool visualize_;

public:
    ShapeClassifier()
    {
        training_dir_ = "/tmp/trained_models/";
        knn_ = 10;
        cloud_.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
        visualize_ = true;
        seg_param_.seg_type_ = 1;
    }

    void printUsage(int argc, char ** argv)
    {
        (void) argc;
        std::cerr << "Usage: "
                  << argv[0] << " "
                  << "  -test_dir "  << "dir_with_pcd_files " << std::endl
                  << "  -models_dir " << "directory containing model files (.ply)  for training" << std::endl
                  << "  [-training_dir "   << "directory where trained features will be stored or loaded if they already exist (default: " << training_dir_ << ")] " << std::endl
                  << "  [-visualize " << "if true, visualizes results (default: " << visualize_ << ")] " << std::endl
                  << "  [-NN " << "number of nearest neighbor considered for classification (default: " << knn_ << ")] "
                  << "  [-seg_type " << "segmentation_method_used (default: " << seg_param_.seg_type_ << ")] "
                  << "  [-chop_z " << "distance in meters for points considered inside the field of view (default: " << seg_param_.seg_type_ << ")] "<< std::endl;
    }

    void init(int argc, char** argv)
    {
        if(!pcl::console::parse_argument (argc, argv,  "-test_dir", test_dir_))
            printUsage(argc, argv);

        if(!pcl::console::parse_argument (argc, argv,  "-models_dir", models_dir_))
            printUsage(argc, argv);

        pcl::console::parse_argument (argc, argv, "-training_dir", training_dir_);
        pcl::console::parse_argument (argc, argv, "-NN", knn_);
        pcl::console::parse_argument (argc, argv,  "-chop_z", seg_param_.chop_at_z_ );
        pcl::console::parse_argument (argc, argv,  "-visualize", visualize_);

        boost::shared_ptr<v4r::MeshSource<PointT> > mesh_source (new v4r::MeshSource<PointT>);
        mesh_source->setPath (models_dir_);
        mesh_source->setResolution (150);
        mesh_source->setTesselationLevel (0);
        mesh_source->setViewAngle (57.f);
        mesh_source->setRadiusSphere (3.f);
        mesh_source->setModelScale (1.f);
        mesh_source->setTesselationLevel(1);
        mesh_source->generate (training_dir_);

        source_ = boost::static_pointer_cast<v4r::MeshSource<PointT> > (mesh_source);

        boost::shared_ptr<v4r::ESFEstimation<PointT, pcl::ESFSignature640> > estimator;
        estimator.reset (new v4r::ESFEstimation<PointT, pcl::ESFSignature640>);

        boost::shared_ptr<v4r::GlobalEstimator<PointT, pcl::ESFSignature640> > cast_estimator;
        cast_estimator = boost::dynamic_pointer_cast<v4r::ESFEstimation<PointT, pcl::ESFSignature640> > (estimator);

        classifier_.setDataSource(source_);
        classifier_.setTrainingDir(training_dir_);
        classifier_.setDescriptorName("esf");
        classifier_.setFeatureEstimator (cast_estimator);
        classifier_.setNN(knn_);
        classifier_.initialize (false);
    }


    void eval()
    {
        v4r::PCLSegmenter<pcl::PointXYZRGB> seg_(seg_param_);
        std::vector< std::string> sub_folder_names;
        if(!v4r::io::getFoldersInDirectory( test_dir_, "", sub_folder_names) )
        {
            std::cerr << "No subfolders in directory " << test_dir_ << ". " << std::endl;
            sub_folder_names.push_back("");
        }

        std::sort(sub_folder_names.begin(), sub_folder_names.end());
        for (size_t sub_folder_id=0; sub_folder_id < sub_folder_names.size(); sub_folder_id++)
        {
            const std::string sequence_path = test_dir_ + "/" + sub_folder_names[ sub_folder_id ];

            std::vector< std::string > views;
            v4r::io::getFilesInDirectory(sequence_path, views, "", ".*.pcd", false);
            std::sort(views.begin(), views.end());
            for (size_t v_id=0; v_id<views.size(); v_id++)
            {
                const std::string fn = test_dir_ + "/" + sub_folder_names[sub_folder_id] + "/" + views[ v_id ];

                std::cout << "Segmenting file " << fn << std::endl;
                pcl::io::loadPCDFile(fn, *cloud_);

                seg_.set_input_cloud(*cloud_);
                seg_.do_segmentation(found_clusters_);

                categories_. resize( found_clusters_.size() );
                confidences_.resize( found_clusters_.size() );

                for(size_t i=0; i < found_clusters_.size(); i++)
                {
                    typename pcl::PointCloud<PointT>::Ptr clusterXYZ (new pcl::PointCloud<PointT>());
                    pcl::copyPointCloud(*cloud_, found_clusters_[i], *clusterXYZ);

                    categories_[i].clear();
                    confidences_[i].clear();

                    classifier_.setInputCloud(clusterXYZ);
                    classifier_.classify();
                    classifier_.getCategory(categories_[i]);
                    classifier_.getConfidence(confidences_[i]);
                }

                if(visualize_)
                    visualize();

            }
        }
    }
    void visualize()
    {
        if(!vis_)
        {
            vis_.reset ( new pcl::visualization::PCLVisualizer("Classification Results") );
            vis_->createViewPort(0,0,0.5,1,vp1_);
            vis_->createViewPort(0.5,0,1,1,vp2_);
        }
        vis_->removeAllPointClouds();
        vis_->removeAllShapes();
        vis_->addPointCloud(cloud_, "cloud", vp1_);

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud (new pcl::PointCloud<pcl::PointXYZRGB>());
        for(size_t i=0; i < found_clusters_.size(); i++)
        {
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster (new pcl::PointCloud<pcl::PointXYZRGB>());
            pcl::copyPointCloud(*cloud_, found_clusters_[i], *cluster);

            for(size_t t=0; t<categories_[i].size(); t++)
            {
                std::cout << categories_[i][t];
                if ( t < confidences_[i].size() )
                    std::cout << " " << std::setprecision(2) << confidences_[i][t];
                std::cout << std::endl;
            }

            const uint8_t r = rand()%255;
            const uint8_t g = rand()%255;
            const uint8_t b = rand()%255;
            for(size_t pt_id=0; pt_id<cluster->points.size(); pt_id++)
            {
                cluster->points[pt_id].r = r;
                cluster->points[pt_id].g = g;
                cluster->points[pt_id].b = b;
            }
            *colored_cloud += *cluster;

            std::stringstream text;
            for(size_t t=0; t< std::min(categories_[i].size(), (size_t)3) ; t++)
            {
                if( t != 0 )
                    text << "; ";
                text << categories_[i][t];
                if ( t < confidences_[i].size() )
                    text << " " << std::setprecision(3) << confidences_[i][t];
            }

            std::stringstream text_id; text_id << i;
            vis_->addText(text.str(), 0, 20*i, 15, r/255.f, g/255.f, b/255.f, text_id.str(), vp2_);
        }
        vis_->addPointCloud(colored_cloud, "segments", vp2_);
        vis_->spin();
    }
};

int main(int argc, char** argv)
{
    ShapeClassifier c;
    c.init(argc, argv);
    c.eval();

    return 0;
}
