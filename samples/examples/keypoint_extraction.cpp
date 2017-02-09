#include <iostream>

#include <opencv2/opencv.hpp>

#include <pcl/common/time.h>
#include <pcl/features/boundary.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl_1_8/features/organized_edge_detection.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/search/kdtree.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <v4r/common/pcl_utils.h>
#include <v4r/common/pcl_opencv.h>
#include <v4r/io/filesystem.h>
#include <v4r/keypoints/all_headers.h>
#include <v4r/features/sift_local_estimator.h>

#include <boost/any.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <glog/logging.h>

namespace po = boost::program_options;
namespace bf = boost::filesystem;

using namespace v4r;

/**
 * @brief main this example demonstrates keypoint extraction from a point cloud with subsequent filtering
 * @param argc
 * @param argv
 * @return
 */

int
main (int argc, char ** argv)
{
    typedef pcl::PointXYZRGB PointT;
    std::string test_dir;
    bool visualize = false;
    bool filter_planar = true;
    bool filter_boundary_pts = true;
    float chop_z = 3.5f;
    int boundary_width = 5;
    float threshold_planar = 0.02f;
    double vis_pt_size = 7;
    int kp_extraction_type = KeypointType::UniformSampling;

    google::InitGoogleLogging(argv[0]);

    po::options_description desc("Point Cloud Segmentation Example\n======================================\n**Allowed options");
    desc.add_options()
        ("help,h", "produce help message")
        ("test_dir,t", po::value<std::string>(&test_dir)->required(), "Directory with test scenes stored as point clouds (.pcd).")
        ("keypoint_extraction_type", po::value<int>(&kp_extraction_type)->default_value(kp_extraction_type), "keypoint extraction type")
        ("chop_z,z", po::value<float>(&chop_z)->default_value(chop_z), "cut-off distance in meter")
        ("filter_planar", po::value<bool>(&filter_planar)->default_value(filter_planar), "if true, filters planar keypoints")
        ("filter_boundary_pts", po::value<bool>(&filter_boundary_pts)->default_value(filter_boundary_pts), "if true, filters keypoints on depth discontinuities")
        ("threshold_planar", po::value<float>(&threshold_planar)->default_value(threshold_planar), "curvature threshold (only used if \"filter_planar\" is enabled)")
        ("boundary_width", po::value<int>(&boundary_width)->default_value(boundary_width), "boundary width in pixel (only used if \"filter_boundary_pts\" is enabled)")
        ("visualize,v", po::bool_switch(&visualize), "If set, visualizes segmented clusters.")
    ;
    po::variables_map vm;
    po::parsed_options parsed = po::command_line_parser(argc, argv).options(desc).allow_unregistered().run();
    std::vector<std::string> to_pass_further = po::collect_unrecognized(parsed.options, po::include_positional);
    po::store(parsed, vm);
    if (vm.count("help")) { std::cout << desc << std::endl; }
    try { po::notify(vm); }
    catch(std::exception& e) { std::cerr << "Error: " << e.what() << std::endl << std::endl << desc << std::endl;  }

    typename v4r::KeypointExtractor<PointT>::Ptr kp_extractor = v4r::initKeypointExtractor<PointT> ( kp_extraction_type, to_pass_further );
    typename v4r::NormalEstimator<PointT>::Ptr normal_estimator = v4r::initNormalEstimator<PointT> ( normal_estimation_type, to_pass_further );

    if( !to_pass_further.empty() )
    {
        std::cerr << "Unused command line arguments: ";
        for(size_t c=0; c<to_pass_further.size(); c++)
            std::cerr << to_pass_further[c] << " ";

        std::cerr << "!" << std::endl;
    }


    std::vector< std::string> sub_folder_names = v4r::io::getFoldersInDirectory( test_dir );
    if(sub_folder_names.empty())
        sub_folder_names.push_back("");

    std::sort(sub_folder_names.begin(), sub_folder_names.end());
    for (size_t sub_folder_id=0; sub_folder_id < sub_folder_names.size(); sub_folder_id++)
    {
        bf::path sequence_path = test_dir;
        sequence_path /= sub_folder_names[ sub_folder_id ];

        std::vector< std::string > views = v4r::io::getFilesInDirectory(sequence_path.string(), ".*.pcd", false);

        for (size_t v_id=0; v_id<views.size(); v_id++)
        {
            bf::path in_path = sequence_path;
            in_path /= views[ v_id ];

            std::cout << "Extracting keypoints for file " << in_path.string() << std::endl;

            typename pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);
            std::vector<int> kp_indices;

            pcl::io::loadPCDFile(in_path.string(), *cloud);
            {
                pcl::ScopeTime t("Keypoint extraction");
                kp_extractor->setInputCloud( cloud );
                pcl::PointCloud<PointT>::Ptr kp_cloud (new pcl::PointCloud<PointT>);
                kp_extractor->compute( *kp_cloud );
                kp_indices = kp_extractor->getKeypointIndices();
            }

            boost::dynamic_bitset<> kp_is_kept( kp_indices.size() );
            kp_is_kept.set();

            if(filter_planar)
            {
                pcl::ScopeTime tt("Filtering planar keypoints");
                typename pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
                pcl::NormalEstimationOMP<PointT, pcl::Normal> normalEstimation;
                normalEstimation.setInputCloud(cloud);
                boost::shared_ptr< std::vector<int> > IndicesPtr (new std::vector<int>);
                *IndicesPtr = kp_indices;
                normalEstimation.setIndices(IndicesPtr);
                normalEstimation.setRadiusSearch( planar_support_radius );
                normalEstimation.setSearchMethod(tree);
                pcl::PointCloud<pcl::Normal>::Ptr normals_for_planarity_check ( new pcl::PointCloud<pcl::Normal> );
                {
                    pcl::ScopeTime tn("Computing normals");
                    normal_estimator->setInputCloud( cloud );
                    normals_for_planarity_check = normal_estimator->compute();
                    pcl::copyPointCloud(*normals_for_planarity_check, kp_indices, *normals_for_planarity_check);
                }

                CHECK(kp_indices.size() == normals_for_planarity_check->points.size());

                for(size_t i=0; i<kp_indices.size(); i++)
                {
                    if(normals_for_planarity_check->points[i].curvature < threshold_planar )
                        kp_is_kept.reset(i);
                }
                std::cout << "Kept " << kp_is_kept.count() << " points from " << kp_indices.size() << " after planar filtering." << std::endl;
            }

            if(filter_boundary_pts)
            {
                pcl::ScopeTime t("Filtering boundary points");
                CHECK( cloud->isOrganized() );
                //compute depth discontinuity edges
                pcl_1_8::OrganizedEdgeBase<PointT, pcl::Label> oed;
                oed.setDepthDisconThreshold (0.05f); //at 1m, adapted linearly with depth
                oed.setMaxSearchNeighbors(100);
                oed.setEdgeType (  pcl_1_8::OrganizedEdgeBase<PointT, pcl::Label>::EDGELABEL_OCCLUDING
                                 | pcl_1_8::OrganizedEdgeBase<PointT, pcl::Label>::EDGELABEL_OCCLUDED
                                 | pcl_1_8::OrganizedEdgeBase<PointT, pcl::Label>::EDGELABEL_NAN_BOUNDARY
                                 );
                oed.setInputCloud ( cloud );

                pcl::PointCloud<pcl::Label> labels;
                std::vector<pcl::PointIndices> edge_indices;
                oed.compute (labels, edge_indices);

                // count indices to allocate memory beforehand
                size_t kept=0;
                for (size_t j = 0; j < edge_indices.size (); j++)
                    kept += edge_indices[j].indices.size ();

                std::vector<int> discontinuity_edges (kept);

                kept=0;
                for (size_t j = 0; j < edge_indices.size (); j++)
                {
                    for (size_t i = 0; i < edge_indices[j].indices.size (); i++)
                        discontinuity_edges[kept++] = edge_indices[j].indices[i];
                }

                cv::Mat boundary_mask = cv::Mat_<unsigned char>::zeros( cloud->height, cloud->width);
                for(size_t i=0; i<discontinuity_edges.size(); i++)
                {
                    int idx = discontinuity_edges[i];
                    int u = idx%cloud->width;
                    int v = idx/cloud->width;

                    boundary_mask.at<unsigned char>(v,u) = 255;
                }


                cv::Mat element = cv::getStructuringElement( cv::MORPH_ELLIPSE,
                                                             cv::Size( 2*boundary_width + 1, 2*boundary_width+1 ),
                                                             cv::Point( boundary_width, boundary_width ) );
                cv::Mat boundary_mask_dilated;
                cv::dilate( boundary_mask, boundary_mask_dilated, element );

                for(size_t i=0; i<kp_indices.size(); i++)
                {
                    int idx = kp_indices[i];
                    int u = idx%cloud->width;
                    int v = idx/cloud->width;

                    if ( boundary_mask_dilated.at<unsigned char>(v,u) )
                        kp_is_kept.reset(i);
                }
                std::cout << "Kept " << kp_is_kept.count() << " points from " << kp_indices.size() << " after filtering boundary pts." << std::endl;
            }

//            {
//                pcl::ScopeTime ("SIFT extraction");
////                v4r::PCLOpenCVConverter<PointT> conv;
////                conv.setInputCloud( cloud );
////                cv::Mat img = conv.getRGBImage();

//                sift_estimator->setInputCloud( cloud );
//                std::vector<std::vector<float> > signatures;
//                sift_estimator->compute( signatures );
//                kp_indices2 = sift_estimator->getKeypointIndices();
//                std::cout << "Size: " << kp_indices2.size() << std::endl;

//                dog_kp_indices_filtered = kp_indices2;
//            }

            if(visualize)
            {
                //reset view point - otherwise this messes up PCL's visualization (this does not affect recognition results)
                cloud->sensor_orientation_ = Eigen::Quaternionf::Identity();
                cloud->sensor_origin_ = Eigen::Vector4f::Zero(4);
                pcl::visualization::PCLVisualizer::Ptr vis (new pcl::visualization::PCLVisualizer());
                int vp1, vp2, vp3;
                vis->createViewPort(0,0,0.33,1,vp1);
                vis->createViewPort(0.33,0,0.66,1,vp2);
                vis->createViewPort(0.66,0,1,1,vp3);
                vis->addPointCloud<PointT>(cloud, "input_vp1", vp1);
                vis->addPointCloud<PointT>(cloud, "input_vp2", vp2);
                vis->addPointCloud<PointT>(cloud, "input_vp3", vp3);

                pcl::PointCloud<PointT>::Ptr kp_cloud (new pcl::PointCloud<PointT> );
                pcl::PointCloud<PointT>::Ptr kp_cloud_accepted (new pcl::PointCloud<PointT> );
                pcl::PointCloud<PointT>::Ptr kp_cloud_rejected (new pcl::PointCloud<PointT> );

                pcl::copyPointCloud(*cloud, kp_indices, *kp_cloud);
                pcl::copyPointCloud(*kp_cloud, kp_is_kept, *kp_cloud_accepted);
                boost::dynamic_bitset<> kp_is_rejected = ~kp_is_kept;
                pcl::copyPointCloud(*kp_cloud, kp_is_rejected, *kp_cloud_rejected);

                pcl::visualization::PointCloudColorHandlerCustom<PointT> red (kp_cloud, 0, 255, 0);
                vis->addPointCloud<PointT>( kp_cloud, red, "keypoints_cloud", vp2);
                vis->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, vis_pt_size, "keypoints_cloud", vp2);

                pcl::visualization::PointCloudColorHandlerCustom<PointT> accepted_color (kp_cloud_accepted, 0, 255, 0);
                vis->addPointCloud<PointT>( kp_cloud_accepted, accepted_color, "keypoints_accepted", vp3);
                vis->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, vis_pt_size, "keypoints_accepted", vp3);

                pcl::visualization::PointCloudColorHandlerCustom<PointT> rejected_color (kp_cloud_rejected, 255, 0, 0);
                vis->addPointCloud<PointT>( kp_cloud_rejected, rejected_color, "keypoints_rejected", vp3);
                vis->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, vis_pt_size, "keypoints_rejected", vp3);

                vis->resetCameraViewpoint();
                vis->spin();
            }
        }
    }

    return 0;
}
