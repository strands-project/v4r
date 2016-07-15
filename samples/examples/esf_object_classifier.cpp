/*
 * Author: Thomas Faeulhammer
 * Date: Sept., 2015
 *
 * Segments PCD files and classifies them using the ESF descriptor
 *
 */

#include <v4r/features/esf_estimator.h>
#include <v4r/io/filesystem.h>
#include <v4r/ml/svmWrapper.h>
#include <v4r/recognition/global_nn_classifier.h>
#include <v4r/recognition/mesh_source.h>
#include <v4r/segmentation/all_headers.h>

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

int main(int argc, char** argv)
{
    typedef pcl::PointXYZ PointT;
    bool eval_only_closest_cluster = true;
    std::string mesh_dir, models_dir;
    int knn;
    std::string test_dir;
    std::string out_dir = "/tmp/class_results/";
    v4r::GlobalNNClassifier<flann::L1, PointT> esf_classifier;

    knn = 5;
    int segmentation_method = v4r::SegmentationType::DominantPlane;

    google::InitGoogleLogging(argv[0]);

    po::options_description desc("Depth-map and point cloud Rendering from mesh file\n======================================\n**Allowed options");
    desc.add_options()
            ("help,h", "produce help message")
            ("mesh_dir,i", po::value<std::string>(&mesh_dir)->required(), "root directory containing mesh files (.ply) for each class. Each class is represented by a sub folder with the folder name indicating the class name. Inside these folders, there are .ply files with object models of this class. Each file represents an object identity.")
            ("models_dir,m", po::value<std::string>(&models_dir)->required(), "directory containing the object models (will be generated if not exists)")
            ("test_dir,t", po::value<std::string>(&test_dir)->required(), "directory containing *.pcd files for testing")
            ("out_dir,o", po::value<std::string>(&out_dir)->default_value(out_dir), "output directory")
            ("eval_only_closest_cluster", po::value<bool>(&eval_only_closest_cluster)->default_value(eval_only_closest_cluster), "if true, evaluates only the closest segmented cluster with respect to the camera.")
            ("kNN,k", po::value<int>(&knn)->default_value(knn), "defines the number k of nearest neighbor for classification")
            ("seg_method", po::value<int>(&segmentation_method)->default_value(segmentation_method), "segmentation method used")
            ;
    po::variables_map vm;
    po::parsed_options parsed = po::command_line_parser(argc, argv).options(desc).allow_unregistered().run();
    std::vector<std::string> to_pass_further = po::collect_unrecognized(parsed.options, po::include_positional);
    po::store(parsed, vm);
    if (vm.count("help"))
    { std::cout << desc << std::endl; return false; }

    try {po::notify(vm);}
    catch(std::exception& e) { std::cerr << "Error: " << e.what() << std::endl << std::endl << desc << std::endl; return false; }

    std::vector< std::string> class_labels = v4r::io::getFoldersInDirectory( mesh_dir );
    CHECK( !class_labels.empty() ) << "No subfolders in directory " << mesh_dir << ". " << std::endl;


    v4r::MeshSource<PointT>::Ptr mesh_source (new v4r::MeshSource<PointT>);
    mesh_source->setPath ( models_dir );
    mesh_source->setMeshDir( mesh_dir );
    mesh_source->setResolution (150);
    mesh_source->setTesselationLevel (0);
    mesh_source->setRadiusSphere (3.f);
    mesh_source->setModelScale (1.f);
    mesh_source->setTesselationLevel(1);
    mesh_source->generate ();

    v4r::Source<PointT>::Ptr source = boost::static_pointer_cast<v4r::MeshSource<PointT> > (mesh_source);
    v4r::ESFEstimation<PointT>::Ptr estimator (new v4r::ESFEstimation<PointT>);
    v4r::GlobalEstimator<PointT>::Ptr cast_estimator = boost::dynamic_pointer_cast<v4r::ESFEstimation<PointT> > (estimator);

    esf_classifier.setDataSource(source);
    esf_classifier.setTrainingDir(models_dir);
    esf_classifier.setDescriptorName("esf");
    esf_classifier.setFeatureEstimator (cast_estimator);
    esf_classifier.setNN(knn);
    esf_classifier.initialize (false);


    // Set-up segmenter
    typename v4r::Segmenter<pcl::PointXYZRGB>::Ptr cast_segmenter;
    if(segmentation_method == v4r::SegmentationType::DominantPlane)
    {
        typename v4r::DominantPlaneSegmenter<pcl::PointXYZRGB>::Parameter param;
        to_pass_further = param.init(to_pass_further);
        typename v4r::DominantPlaneSegmenter<pcl::PointXYZRGB>::Ptr seg (new v4r::DominantPlaneSegmenter<pcl::PointXYZRGB> (param));
        cast_segmenter = boost::dynamic_pointer_cast<v4r::Segmenter<pcl::PointXYZRGB> > (seg);
    }
    else if(segmentation_method == v4r::SegmentationType::MultiPlane)
    {
        typename v4r::MultiplaneSegmenter<pcl::PointXYZRGB>::Parameter param;
        to_pass_further = param.init(to_pass_further);
        typename v4r::MultiplaneSegmenter<pcl::PointXYZRGB>::Ptr seg (new v4r::MultiplaneSegmenter<pcl::PointXYZRGB> (param));
        cast_segmenter = boost::dynamic_pointer_cast<v4r::Segmenter<pcl::PointXYZRGB> > (seg);
    }
    else if(segmentation_method == v4r::SegmentationType::EuclideanSegmentation)
    {
        typename v4r::EuclideanSegmenter<pcl::PointXYZRGB>::Parameter param;
        to_pass_further = param.init(to_pass_further);
        typename v4r::EuclideanSegmenter<pcl::PointXYZRGB>::Ptr seg (new v4r::EuclideanSegmenter<pcl::PointXYZRGB> (param));
        cast_segmenter = boost::dynamic_pointer_cast<v4r::Segmenter<pcl::PointXYZRGB> > (seg);
    }
    else if(segmentation_method == v4r::SegmentationType::SmoothEuclideanClustering)
    {
        typename v4r::SmoothEuclideanSegmenter<pcl::PointXYZRGB>::Parameter param;
        to_pass_further = param.init(to_pass_further);
        typename v4r::SmoothEuclideanSegmenter<pcl::PointXYZRGB>::Ptr seg (new v4r::SmoothEuclideanSegmenter<pcl::PointXYZRGB> (param));
        cast_segmenter = boost::dynamic_pointer_cast<v4r::Segmenter<pcl::PointXYZRGB> > (seg);
    }

    std::vector< std::string> sub_folders = v4r::io::getFoldersInDirectory( test_dir );
    if( sub_folders.empty() ) {
        std::cerr << "No subfolders in directory " << test_dir << ". " << std::endl;
        sub_folders.push_back("");
    }


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
            cast_segmenter->setInputCloud(cloud);
            cast_segmenter->segment();
            std::vector<pcl::PointIndices> found_clusters;
            cast_segmenter->getSegmentIndices(found_clusters);

            if (found_clusters.empty())
            {
                std::cerr << "Segmentation failed! Classifying whole image..." << std::endl;
                pcl::PointIndices all_idx;
                all_idx.indices.resize(cloud->points.size());
                for(size_t i=0; i<cloud->points.size(); i++)
                    all_idx.indices[i] = i;
            }

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
                //                std::cout << centroid[2] << " " << centroid[0]*centroid[0] + centroid[1]*centroid[1] + centroid[2]*centroid[2] << std::endl;

                esf_classifier.setInputCloud(clusterXYZ);
                esf_classifier.classify();
                esf_classifier.getCategory(categories[i]);
                esf_classifier.getConfidence(confidences[i]);

                std::cout << "Predicted Label (ESF): " << categories[i][0] << std::endl;
                of << categories[i][0] << " " << confidences[i][0] << " ";
                for(size_t c_id=0; c_id<categories[i].size(); c_id++) {
                    std::cout << categories[i][c_id] << "(" << confidences[i][c_id] << ")" << std::endl;
                }

                of << std::endl;
            }
            of.close();
        }
    }
    return 0;
}
