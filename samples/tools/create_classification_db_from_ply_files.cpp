
/**
  * @author Thomas Faeulhammer on 19.01.17
  * @brief This file opens mesh files stored as ".ply" and renders PCL point clouds from various views on a specified sphere
  * It creates a model database that can be used for recognition by assuming each the input directory has folders
  * for each object and inside these folders are .ply files showing specific instances of this object.
  * The program outputs a model folder with the object folders on top, in each object folder there is a folder for each instance
  * and within this instance folder there is a "/view" folder with the rendered point clouds and their poses of this particular instance.
  */

#include <iomanip>
#include <iostream>
#include <string>
#include <sstream>
#include <limits>
#include <stdio.h>
#include <opencv2/opencv.hpp>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <v4r/common/miscellaneous.h>
#include <v4r/io/eigen.h>
#include <v4r/io/filesystem.h>
#include <v4r/rendering/depthmapRenderer.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/program_options.hpp>
#include <glog/logging.h>

namespace po = boost::program_options;
namespace bf = boost::filesystem;
using namespace v4r;

std::string cloud_prefix_, pose_prefix_;

template <typename PointT>
void
removeNaNPoints(pcl::PointCloud<PointT> &cloud)
{
    size_t kept=0;
    for(size_t i=0; i<cloud.points.size(); i++)
    {
        const PointT &pt = cloud.points[i];
        if ( pcl::isFinite(pt) )
            cloud.points[kept++] = pt;
    }
    cloud.points.resize(kept);
    cloud.width = kept;
    cloud.height = 1;
}

template <typename PointT>
void
save(const pcl::PointCloud<PointT> &cloud, const std::string &out_dir)
{
    bf::path out_path_bf = out_dir;
    out_path_bf /= "views/" + cloud_prefix_;

    std::string out_cloud_fn;
    size_t counter=0;
    do
    {
        std::stringstream fn; fn << out_path_bf.string() << std::setfill('0') << std::setw(5) << counter++ << ".pcd";
        out_cloud_fn = fn.str();
    }while(io::existsFile( out_cloud_fn ));

    io::createDirForFileIfNotExist ( out_cloud_fn );
    pcl::io::savePCDFileBinaryCompressed ( out_cloud_fn, cloud );

    const Eigen::Matrix4f tf = RotTrans2Mat4f ( cloud.sensor_orientation_, cloud.sensor_origin_ );
    std::string out_pose_fn ( out_cloud_fn );
    boost::replace_last ( out_pose_fn, cloud_prefix_, pose_prefix_);
    boost::replace_last ( out_pose_fn, ".pcd", ".txt");
    io::writeMatrixToFile ( out_pose_fn, tf );
    LOG(INFO) << "Saved rendered cloud to " << out_cloud_fn << ".";
}

int main(int argc, const char * argv[])
{
    std::string input_dir, out_dir;
    bool visualize = false;
    size_t width = 640, height = 480;
    float fx = 535.4, fy = 539.2, cx = 320.1, cy = 247.6;

    float radius_sphere = 3.f;
    size_t subdivisions = 0;
    bool gen_organized = false;

    cloud_prefix_ = "cloud_";
    pose_prefix_ = "pose_";

    google::InitGoogleLogging(argv[0]);

    po::options_description desc("Depth-map and point cloud Rendering from mesh file\n======================================\n**Allowed options");
    desc.add_options()
            ("help,h", "produce help message")
            ("input,i", po::value<std::string>(&input_dir)->required(), "input file path (.ply) or folder containing files")
            ("out_dir,o", po::value<std::string>(&out_dir)->default_value("/tmp/model_database/"), "output directory to store the rendered point clouds in the recognition model structure")
            ("subdivisions,s", po::value<size_t>(&subdivisions)->default_value(subdivisions), "defines the number of subdivsions used for rendering")
            ("radius_sphere,r", po::value<float>(&radius_sphere)->default_value(radius_sphere, boost::str(boost::format("%.2e") % radius_sphere)), "defines the radius of the sphere used for rendering")
            ("width", po::value<size_t>(&width)->default_value(width), "defines the image width")
            ("height", po::value<size_t>(&height)->default_value(height), "defines the image height")
            ("fx", po::value<float>(&fx)->default_value(fx, boost::str(boost::format("%.2e") % fx)), "defines the focal length in x direction used for rendering")
            ("fy", po::value<float>(&fy)->default_value(fy, boost::str(boost::format("%.2e") % fy)), "defines the focal length in y direction used for rendering")
            ("cx", po::value<float>(&cx)->default_value(cx, boost::str(boost::format("%.2e") % cx)), "defines the central point of projection in x direction used for rendering")
            ("cy", po::value<float>(&cy)->default_value(cy, boost::str(boost::format("%.2e") % cy)), "defines the central point of projection in y direction used for rendering")
            ("visualize,v", po::bool_switch(&visualize), "visualize the rendered depth and color map")
            ("generate_organized", po::value<bool>(&gen_organized)->default_value(gen_organized), "if false, removes NaN points from the rendered point cloud")
            ;
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) { std::cout << desc << std::endl; return false; }
    try { po::notify(vm); }
    catch(std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl << std::endl << desc << std::endl;
        return false;
    }

    CHECK( (cx < width) && (cy < height) && (cx > 0) && (cy > 0)) << "Parameters not valid!";

    DepthmapRenderer renderer ( width, height );
    renderer.setIntrinsics ( fx, fy, cx, cy);

    std::vector<std::string> class_names = io::getFoldersInDirectory( input_dir );
    if(class_names.empty())
        class_names.push_back("");

    for(const std::string class_name : class_names )
    {
        bf::path class_path = input_dir;
        class_path /= class_name;

        std::vector<std::string> instance_names = io::getFilesInDirectory( class_path.string(), ".*.ply", false);

        for( const std::string &instance_name : instance_names)
        {
            bf::path input_path (class_path);
            input_path /= instance_name;
            const std::string filename = input_path.string();

            DepthmapRendererModel model( filename );
            renderer.setModel(&model);
            std::vector<Eigen::Vector3f> sphere = renderer.createSphere(radius_sphere, subdivisions);

            LOG(INFO) << "Rendering file " << filename << " ( with color? " << model.hasColor() << ").";    //test if the model has colored elements(note! no textures supported yet.... only colored polygons)z

            for(const Eigen::Vector3f &pt : sphere )
            {
                //get a camera pose looking at the center:
                Eigen::Matrix4f orientation = renderer.getPoseLookingToCenterFrom(pt);

                renderer.setCamPose(orientation);
                float visible;
                cv::Mat color;
                cv::Mat depthmap = renderer.renderDepthmap(visible, color);

                bf::path out_path_bf = out_dir;
                out_path_bf /= class_name;
                out_path_bf /= instance_name;

                std::string out_path = out_path_bf.string();
                size_t lastindex = out_path.find_last_of(".");
                out_path = out_path.substr(0, lastindex);

                if(model.hasColor())
                {
                    pcl::PointCloud<pcl::PointXYZRGB> cloud = renderer.renderPointcloudColor(visible);
                    if( !gen_organized )
                        removeNaNPoints(cloud);

                    if( cloud.points.empty())
                    {
                        std::cerr << "Rendered cloud does not contain any point! " << std::endl;
                        continue;
                    }

                    save (cloud, out_path );

                    if (visualize)
                        cv::imshow("color", color);
                }
                else
                {
                    pcl::PointCloud<pcl::PointXYZ> cloud = renderer.renderPointcloud(visible);
                    if( !gen_organized )
                        removeNaNPoints(cloud);

                    if( cloud.points.empty())
                    {
                        std::cerr << "Rendered cloud does not contain any point! " << std::endl;
                        continue;
                    }

                    save ( cloud, out_path );
                }

                if(visualize)
                {
                    LOG(INFO) << visible << "% visible.";
                    cv::imshow("depthmap", depthmap*0.25);
                    cv::waitKey();
                }
            }
        }
    }

    return 0;
}

