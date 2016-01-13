/*
 * Author: Thomas Faeulhammer
 * Date: Sept., 2015
 *
 * Segments PCD files and classifies them using the ESF descriptor
 *
 */

#include <v4r/features/global_estimator.h>
#include <v4r/features/esf_estimator.h>
#include <v4r/io/filesystem.h>
#include <v4r/ml/svmWrapper.h>
#include <v4r/recognition/global_nn_classifier.h>
#include <v4r/recognition/source.h>
#include <v4r/recognition/mesh_source.h>
#include <v4r/segmentation/pcl_segmentation_methods.h>

#include <opencv2/opencv.hpp>

#include <pcl/common/centroid.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>
#include <boost/program_options.hpp>
#include <glog/logging.h>

#include <fstream>
#include <sstream>

namespace po = boost::program_options;

typedef pcl::PointXYZ PointT;

std::string MESH_DIR_, MODELS_DIR_;
int KNN_;

void
init(v4r::GlobalNNClassifier<flann::L1, PointT> &esf_classifier) {

    boost::shared_ptr<v4r::MeshSource<PointT> > mesh_source (new v4r::MeshSource<PointT>);
    mesh_source->setMeshDir(MESH_DIR_);
    mesh_source->setPath (MODELS_DIR_);
    mesh_source->setResolution (150);
    mesh_source->setTesselationLevel (0);
//    mesh_source->setViewAngle (57.f);
    mesh_source->setRadiusSphere (3.f);
    mesh_source->setModelScale (1.f);
    mesh_source->setTesselationLevel(1);
    mesh_source->generate ();

    boost::shared_ptr<v4r::Source<PointT> > source;
    source = boost::static_pointer_cast<v4r::MeshSource<PointT> > (mesh_source);

    boost::shared_ptr<v4r::ESFEstimation<PointT> > estimator;
    estimator.reset (new v4r::ESFEstimation<PointT>);

    boost::shared_ptr<v4r::GlobalEstimator<PointT> > cast_estimator;
    cast_estimator = boost::dynamic_pointer_cast<v4r::ESFEstimation<PointT> > (estimator);

    esf_classifier.setDataSource(source);
    esf_classifier.setTrainingDir(MODELS_DIR_);
    esf_classifier.setDescriptorName("esf");
    esf_classifier.setFeatureEstimator (cast_estimator);
    esf_classifier.setNN(KNN_);
    esf_classifier.initialize (false);
}

int main(int argc, char** argv)
{
    bool do_esf = true;
    bool eval_only_closest_cluster = true;
    std::map<std::string, size_t> class2label;

    std::string test_dir, svm_path, class2label_file;
    std::string out_dir = "/tmp/class_results/";
    v4r::PCLSegmenter<pcl::PointXYZRGB>::Parameter seg_param;

    v4r::GlobalNNClassifier<flann::L1, PointT> esf_classifier;

    KNN_ = 5;

    google::InitGoogleLogging(argv[0]);

    po::options_description desc("Depth-map and point cloud Rendering from mesh file\n======================================\n**Allowed options");
    desc.add_options()
            ("help,h", "produce help message")
            ("mesh_dir,i", po::value<std::string>(&MESH_DIR_)->required(), "root directory containing mesh files (.ply) for each class. Each class is represented by a sub folder with the folder name indicating the class name. Inside these folders, there are .ply files with object models of this class. Each file represents an object identity.")
            ("models_dir,m", po::value<std::string>(&MODELS_DIR_)->required(), "directory containing the object models (will be generated if not exists)")
            ("test_dir,t", po::value<std::string>(&test_dir)->required(), "directory containing *.pcd files for testing")
            ("out_dir,o", po::value<std::string>(&out_dir)->default_value(out_dir), "output directory")
            ("do_esf,e", po::value<bool>(&do_esf)->default_value(do_esf), "if true, includes esf classification (shape based).")
            ("eval_only_closest_cluster", po::value<bool>(&eval_only_closest_cluster)->default_value(eval_only_closest_cluster), "if true, evaluates only the closest segmented cluster with respect to the camera.")
            ("kNN,k", po::value<int>(&KNN_)->default_value(KNN_), "defines the number k of nearest neighbor for classification")
            ("chop_z,z", po::value<double>(&seg_param.chop_at_z_ )->default_value(seg_param.chop_at_z_, boost::str(boost::format("%.2e") % seg_param.chop_at_z_)), "")
            ("seg_type", po::value<int>(&seg_param.seg_type_ )->default_value(seg_param.seg_type_), "")
            ("min_cluster_size", po::value<int>(&seg_param.min_cluster_size_ )->default_value(seg_param.min_cluster_size_), "")
            ("max_vertical_plane_size", po::value<int>(&seg_param.max_vertical_plane_size_ )->default_value(seg_param.max_vertical_plane_size_), "")
            ("num_plane_inliers", po::value<int>(&seg_param.num_plane_inliers_ )->default_value(seg_param.num_plane_inliers_), "")
            ("max_angle_plane_to_ground", po::value<double>(&seg_param.max_angle_plane_to_ground_ )->default_value(seg_param.max_angle_plane_to_ground_), "")
            ("sensor_noise_max", po::value<double>(&seg_param.sensor_noise_max_ )->default_value(seg_param.sensor_noise_max_), "")
            ("table_range_min", po::value<double>(&seg_param.table_range_min_ )->default_value(seg_param.table_range_min_), "")
            ("table_range_max", po::value<double>(&seg_param.table_range_max_ )->default_value(seg_param.table_range_max_), "")
            ("angular_threshold_deg", po::value<double>(&seg_param.angular_threshold_deg_ )->default_value(seg_param.angular_threshold_deg_), "")
            ("trained_svm_model", po::value<std::string>(&svm_path), "")
            ("class2label_file", po::value<std::string>(&class2label_file), "")
   ;
     po::variables_map vm;
     po::store(po::parse_command_line(argc, argv, desc), vm);
     if (vm.count("help"))
     {
         std::cout << desc << std::endl;
         return false;
     }

     try
     {
         po::notify(vm);
     }
     catch(std::exception& e)
     {
         std::cerr << "Error: " << e.what() << std::endl << std::endl << desc << std::endl;
         return false;
     }

    ::svm_model *svm_mod_ = ::svm_load_model(svm_path.c_str());


    std::ifstream f(class2label_file.c_str());
    std::vector<std::string> words;
    std::string line;
    while (std::getline(f, line)) {
        boost::trim_right(line);
        split( words, line, boost::is_any_of("\t "));
        class2label[words[0]] = static_cast<size_t>(atoi(words[1].c_str()));
    }
    f.close();

    if (do_esf)
        init(esf_classifier);

    v4r::PCLSegmenter<pcl::PointXYZRGB> seg(seg_param);

    std::vector< std::string> sub_folders = v4r::io::getFoldersInDirectory( test_dir );
    if( sub_folders.empty() )
        sub_folders.push_back("");


    for (const std::string &sub_folder : sub_folders){
        const std::string sequence_path = test_dir + "/" + sub_folder;
        const std::string out_dir_full = out_dir + "/" +  sub_folder;
        v4r::io::createDirIfNotExist(out_dir_full);

        std::vector< std::string > views = v4r::io::getFilesInDirectory(sequence_path, ".*.pcd", false);
        for ( const std::string &view : views ) {
            const std::string fn = sequence_path + "/" + view;

            std::cout << "Segmenting file " << fn << std::endl;
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>());
            pcl::io::loadPCDFile(fn, *cloud);
            seg.set_input_cloud(*cloud);

            std::vector<pcl::PointIndices> found_clusters;
            seg.do_segmentation(found_clusters);

            std::vector< std::vector < std::string > > categories (found_clusters.size());
            std::vector< std::vector < float > > confidences (found_clusters.size());

            if (eval_only_closest_cluster) { // only save classification result for cluster which is closest to the camera (w.r.t. to centroid)
                int min_id=-1;
                double min_centroid = std::numeric_limits<double>::max();

                for(size_t i=0; i < found_clusters.size(); i++)
                {
                    typename pcl::PointCloud<PointT>::Ptr clusterXYZ (new pcl::PointCloud<PointT>());
                    pcl::copyPointCloud(*cloud, found_clusters[i], *clusterXYZ);
                    Eigen::Vector4f centroid;
                    pcl::compute3DCentroid (*clusterXYZ, centroid);

        //            double dist = centroid[0]*centroid[0] + centroid[1]*centroid[1] + centroid[2]*centroid[2];

                    if (centroid[2] < min_centroid) {
                        min_centroid = centroid[2];
                        min_id = i;
                    }
                }

                if (min_id >= 0) {
                    std::vector<pcl::PointIndices> closest_cluster;
                    closest_cluster.push_back( found_clusters[min_id] );
                    found_clusters = closest_cluster;
                }
            }

            std::string out_fn = out_dir_full + "/" + view;
            boost::replace_all(out_fn, ".pcd", ".anno_test");
            std::ofstream of (out_fn.c_str());
            for(size_t i=0; i < found_clusters.size(); i++) {
                typename pcl::PointCloud<PointT>::Ptr clusterXYZ (new pcl::PointCloud<PointT>());
                pcl::copyPointCloud(*cloud, found_clusters[i], *clusterXYZ);

                Eigen::Vector4f centroid;
                pcl::compute3DCentroid (*clusterXYZ, centroid);
                std::cout << centroid[2] << " " << centroid[0]*centroid[0] + centroid[1]*centroid[1] + centroid[2]*centroid[2] << std::endl;

                if(do_esf) {
                    esf_classifier.setInputCloud(clusterXYZ);
                    esf_classifier.classify();
                    esf_classifier.getCategory(categories[i]);
                    esf_classifier.getConfidence(confidences[i]);

                    std::cout << "Predicted Label (ESF): " << categories[i][0] << std::endl;
                    of << categories[i][0] << " " << confidences[i][0] << " ";
                    for(size_t c_id=0; c_id<categories[i].size(); c_id++) {
                        std::cout << categories[i][c_id] << "(" << confidences[i][c_id] << ")" << std::endl;
                    }
                }

                of << std::endl;
//                feature_extraction_pipeline<float>(argc, argv);
            }
            of.close();
        }
    }
    return 0;
}

