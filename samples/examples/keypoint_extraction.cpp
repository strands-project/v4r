#include <iostream>

#include <opencv/highgui.h>

#include <pcl/common/time.h>
#include <pcl/features/boundary.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/passthrough.h>
#include <pcl_1_8/features/organized_edge_detection.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/search/kdtree.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <v4r/common/miscellaneous.h>
#include <v4r/common/pcl_utils.h>
#include <v4r/common/pcl_opencv.h>
#include <v4r/common/normal_estimator_z_adpative.h>
#include <v4r/common/normals.h>
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
    int filter_boundary_pts = 7; //  according to the edge types defined in pcl::OrganizedEdgeBase (EDGELABEL_OCCLUDING  | EDGELABEL_OCCLUDED | EDGELABEL_NAN_BOUNDARY)
    float chop_z = 3.5f;
    float planar_support_radius = 0.04f; ///< Radius used to check keypoints for planarity.
    float max_depth_change_factor =  0.02f; //10.0f;
    float normal_smoothing_size = 20.f; // 10.0f;
    bool use_depth_dependent_smoothing = false; // false
    int boundary_width = 5;
    float threshold_planar = 0.02f;
    double vis_pt_size = 7;
    int kp_extraction_type = KeypointType::UniformSampling;
    bool use_sift = false;
    int normal_estimation_type = NormalEstimatorType::PCL_INTEGRAL_NORMAL;

    google::InitGoogleLogging(argv[0]);

    po::options_description desc("Point Cloud Segmentation Example\n======================================\n**Allowed options");
    desc.add_options()
        ("help,h", "produce help message")
        ("test_dir,t", po::value<std::string>(&test_dir)->required(), "Directory with test scenes stored as point clouds (.pcd).")
        ("keypoint_extraction_type", po::value<int>(&kp_extraction_type)->default_value(kp_extraction_type), "keypoint extraction type")
        ("normal_estimation_type", po::value<int>(&normal_estimation_type)->default_value(normal_estimation_type), "surface normal estimation type")
        ("chop_z,z", po::value<float>(&chop_z)->default_value(chop_z), "cut-off distance in meter")
        ("use_sift", po::value<bool>(&use_sift)->default_value(use_sift), "if true, uses DoG as keypoint extraction method (which is implemented in SIFT-GPU). Ignores keypoint extraction type.")
        ("filter_planar", po::value<bool>(&filter_planar)->default_value(filter_planar), "if true, filters planar keypoints")
        ("filter_boundary_pts", po::value<int>(&filter_boundary_pts)->default_value(filter_boundary_pts), "if true, filters keypoints on depth discontinuities")
        ("planar_support_radius", po::value<float>(&planar_support_radius)->default_value(planar_support_radius), "planar support radius in meter  (only used if \"filter_planar\" is enabled)")
        ("max_depth_change_factor", po::value<float>(&max_depth_change_factor)->default_value(max_depth_change_factor), "max_depth_change_factor  (only used if \"filter_planar\" is enabled)")
        ("normal_smoothing_size", po::value<float>(&normal_smoothing_size)->default_value(normal_smoothing_size), "normal_smoothing_size  (only used if \"filter_planar\" is enabled)")
        ("use_depth_dependent_smoothing", po::value<bool>(&use_depth_dependent_smoothing)->default_value(use_depth_dependent_smoothing), "use_depth_dependent_smoothing  (only used if \"filter_planar\" is enabled)")
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

    std::vector<typename v4r::KeypointExtractor<PointT>::Ptr > kp_extractors = v4r::initKeypointExtractors<PointT> ( kp_extraction_type, to_pass_further );
    typename v4r::NormalEstimator<PointT>::Ptr normal_estimator = v4r::initNormalEstimator<PointT> ( normal_estimation_type, to_pass_further );
    //    typename v4r::KeypointExtractor<PointT>::Ptr shot = v4r::initKeypointExtractor<PointT>( segmentation_method, to_pass_further);

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
            cloud->sensor_origin_ = Eigen::Vector4f::Zero();
            cloud->sensor_orientation_ = Eigen::Quaternionf::Identity();

            // ==== FILTER POINTS BASED ON DISTANCE =====
            for(PointT &p : cloud->points)
            {
                if (pcl::isFinite(p) && p.getVector3fMap().norm() > chop_z)
                    p.x = p.y = p.z = std::numeric_limits<float>::quiet_NaN();
            }

            pcl::PointCloud<pcl::Normal>::Ptr normals ( new pcl::PointCloud<pcl::Normal> );

            bool need_normals = false;
            for(const auto kp : kp_extractors)
            {
                if( kp->needNormals())
                {
                    need_normals = true;
                    break;
                }
            }

            if( need_normals || filter_planar )
            {
                pcl::ScopeTime tn("Computing normals");
                normal_estimator->setInputCloud( cloud );
                normals = normal_estimator->compute();

//                    pcl::NormalEstimationOMP<PointT, pcl::Normal> ne;
//                    typename pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
//                    ne.setSearchMethod(tree);
//                    ne.setRadiusSearch( planar_support_radius );
//                    ne.setIndices(IndicesPtr);

//                    ZAdaptiveNormalsParameter n_param;
//                    n_param.adaptive_ = true;
//                    std::vector<int> radius = {5,5,6,6,7,9,11,12};
//                    n_param.kernel_radius_ = radius;

//                    ZAdaptiveNormalsPCL<PointT> ne(n_param);
//                    ne.setInputCloud( cloud );
//                    normals_for_planarity_check = ne.compute();


//                    pcl::IntegralImageNormalEstimation<PointT, pcl::Normal> ne;
//                    ne.setNormalEstimationMethod (ne.COVARIANCE_MATRIX);
//                    ne.setMaxDepthChangeFactor( max_depth_change_factor );//(10.0f);
//                    ne.setNormalSmoothingSize( normal_smoothing_size );//(10.0f);
//                    ne.setDepthDependentSmoothing( use_depth_dependent_smoothing );
//////                    ne.setBorderPolicy();

//                    ne.setInputCloud(cloud);
//                    ne.compute(*normals);
            }

            {
                if(use_sift)
                {
                    pcl::ScopeTime t("SIFT Keypoint extraction");
                    typename v4r::SIFTLocalEstimation<PointT>::Ptr sift_estimator (new v4r::SIFTLocalEstimation<PointT>);
                    sift_estimator->setInputCloud( cloud );
                    std::vector<std::vector<float> > signatures_foo;
                    sift_estimator->compute( signatures_foo );
                    kp_indices = sift_estimator->getKeypointIndices();
                }
                else
                {
                    boost::dynamic_bitset<> scene_pt_is_keypoint ( cloud->points.size(), 0);

                    for ( typename v4r::KeypointExtractor<PointT>::Ptr ke : kp_extractors)
                    {
                        std::stringstream info_txt; info_txt << ke->getKeypointExtractorName() << " keypoint extraction";
                        pcl::ScopeTime t(info_txt.str().c_str());

                        ke->setInputCloud( cloud );
                        ke->setNormals(normals);
                        ke->compute( );
                        const std::vector<int> kp_indices_tmp = ke->getKeypointIndices();

                        for (int idx : kp_indices_tmp)
                            scene_pt_is_keypoint.set(idx);
                    }
                    kp_indices = v4r::createIndicesFromMask<int>( scene_pt_is_keypoint );
                }
            }


//            kp_indices.resize( cloud->points.size() ) ; // vector with 100 ints.
//            std::iota (std::begin(kp_indices), std::end(kp_indices), 0); // Fill with 0, 1, ..., 99.

            boost::dynamic_bitset<> kp_is_kept( kp_indices.size() );
            kp_is_kept.set();


            if(filter_planar)
            {
                pcl::ScopeTime t("Filtering planar keypoints");
                boost::shared_ptr< std::vector<int> > IndicesPtr (new std::vector<int>);
                *IndicesPtr = kp_indices;
                pcl::copyPointCloud(*normals, kp_indices, *normals);


                float min_curv = std::numeric_limits<float>::max();
                float max_curv = std::numeric_limits<float>::min();
                for(size_t i=0; i<kp_indices.size(); i++)
                {
                    float curv = normals->points[i].curvature;

                    if( curv < threshold_planar )
                        kp_is_kept.reset(i);

                    if(curv < min_curv )
                        min_curv = curv;

                    if(curv > max_curv )
                        max_curv = curv;
                }
                std::cout << "min curv: " << min_curv << ", max curv: " << max_curv << std::endl;
                std::cout << "Kept " << kp_is_kept.count() << " points from " << kp_indices.size() << " after planar filtering." << std::endl;

                v4r::PCLOpenCVConverter<PointT> conv;
                conv.setInputCloud( cloud );
                cv::Mat img = conv.getRGBImage();

                for(size_t i=0; i<cloud->points.size(); i++)
                {
                    int u = i%cloud->width;
                    int v = i/cloud->width;

                    float curv = normals->points[i].curvature;

                    if( pcl_isfinite(curv) )
                        img.at<cv::Vec3b>(v,u) = cv::Vec3b(0, 255 - 255* (curv-min_curv)/(max_curv-min_curv) , 0);
                    else
                        img.at<cv::Vec3b>(v,u) = cv::Vec3b(0, 0, 0);
                }
                cv::imshow("curvature", img);
                cv::waitKey();
            }

            if(filter_boundary_pts)
            {
                pcl::ScopeTime t("Filtering boundary points");
                CHECK( cloud->isOrganized() );
                //compute depth discontinuity edges
                pcl_1_8::OrganizedEdgeBase<PointT, pcl::Label> oed;
                oed.setDepthDisconThreshold (0.05f); //at 1m, adapted linearly with depth
                oed.setMaxSearchNeighbors(100);
                oed.setEdgeType ( filter_boundary_pts  );
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
//                pcl::copyPointCloud(*cloud, uniform_sampling_kp_indices_filtered, *kp_cloud_filtered);

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


//            v4r::PCLOpenCVConverter<PointT> conv;
//            conv.setInputCloud( cloud );
//            cv::Mat img = conv.getRGBImage();

//            for(size_t i=0; i<cloud->points.size(); i++)
//            {
//                int u = i%cloud->width;
//                int v = i/cloud->width;

//                if( !pcl::isFinite( cloud->points[i] ) )
//                    img.at<cv::Vec3b>(v,u) = cv::Vec3b(255,255,255);
//            }

//            for(size_t i=0; i<kp_indices.size(); i++)
//            {
//                int idx = kp_indices[i];
//                int u = idx%cloud->width;
//                int v = idx/cloud->width;

//                cv::Scalar color;
//                if( kp_is_kept[i] )
//                    color = cv::Scalar( 0, 255, 0 );
//                else
//                {
//                    color = cv::Scalar( 0, 0, 255 );
//                    img.at<cv::Vec3b>(v,u) = cv::Vec3b(0,0,255);
//                }

//                if(cloud->points[idx].z > chop_z)
//                    img.at<cv::Vec3b>(v,u) = cv::Vec3b(0,0,255);

//                cv::circle( img, cv::Point(u,v), 1, color, -1, cv::LINE_AA, 0 );
//            }
//            cv::imshow("keypoints", img);
//            cv::waitKey();
        }
    }

    return 0;
}
