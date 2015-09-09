#include <pcl/common/common.h>
#include <pcl/console/parse.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <opencv/cv.h>
#include <opencv/highgui.h>

#include <algorithm>
#include <random>

#include <boost/filesystem.hpp>

#include <v4r/io/filesystem.h>

#include <v4r/tomgine/tgTomGineThread.h>
#include <v4r/tomgine/tgShapeCreator.h>

#include <v4r/tomgine/tgTextureModelAI.h>
#include <v4r/tomgine/PointCloudRendering.h>

#define USE_GPU

#ifdef USE_GPU
    #include <opencv2/gpu/gpu.hpp>
#endif

using namespace TomGine;

int main(int argc, char *argv[])
{
    std::string dir;

    int img_width = 50;
    int img_height = 50;
    unsigned int subdivisions = 2;
    int polyhedron = TomGine::tgShapeCreator::ICOSAHEDRON;
    const float max_dist = 2.f;
    const float min_dist = 0.1f;
    float bg_noise_level = 0.1;
    float obj_noise_level = 0.01;
    bool render_upper_half_only = true;

    const bool scale = true;
    const bool center = true;

    if ( pcl::console::parse_argument (argc, argv, "-dir", dir) == -1)
    {
        std::cout << "Usage: mesh2pointcloud -dir model-file " <<
                     "[-bg_noise_level background-noise-level (default: " << bg_noise_level << ")] " <<
                     "[-fg_noise_level object-noise-level (default: " << obj_noise_level << ")]" <<
                     "[-img_width img_width (default: "   << img_width  << ")]" <<
                     "[-img_height img_height (default: " << img_height << ")]" <<
                     std::endl << std::endl;
        return 0;
    }
    pcl::console::parse_argument (argc, argv, "-fg_noise_level", obj_noise_level);
    pcl::console::parse_argument (argc, argv, "-bg_noise_level", bg_noise_level);
    pcl::console::parse_argument (argc, argv, "-img_width", img_width);
    pcl::console::parse_argument (argc, argv, "-img_height", img_height);
    pcl::console::parse_argument (argc, argv, "-render_upper_half_only", render_upper_half_only);
    pcl::console::parse_argument (argc, argv, "-subdivisions", subdivisions);
    pcl::console::parse_argument (argc, argv, "-polyhedron", polyhedron);

    std::cout << "Generating " << img_width << "x" << img_height <<
                 " depth images with a background noise level of " << bg_noise_level <<
                 " and a foreground/object noise level of " << obj_noise_level << std::endl;


    std::default_random_engine generator;
    std::normal_distribution<float> cam_pose_distribution(0.0, 1.0);
    std::normal_distribution<float>  bg_distribution (0.0, bg_noise_level);
    std::normal_distribution<float> obj_distribution (0.0, obj_noise_level);



    std::vector< std::string> folder_names;   // equal to class names
    if(!v4r::io::getFoldersInDirectory(dir, "", folder_names))
    {
        std::cerr << "No subfolders in directory " << dir << ". " << std::endl;
        folder_names.push_back("");
    }

    std::vector< std::string > full_filenames;
    std::vector< size_t > class_labels;

    for(size_t class_id=0; class_id < folder_names.size(); class_id++)
    {
        std::stringstream class_images_path_ss;
#ifdef _WIN32
        class_images_path_ss << dir << "\\" <<  folder_names[class_id];
#else
        class_images_path_ss << dir << "/" << folder_names[class_id];
#endif
        std::vector < std::string > files_intern;
        if( v4r::io::getFilesInDirectory(class_images_path_ss.str(), files_intern, "", ".*\\.(ply|PLY|obj|OBJ)", true) == -1)
        {
            std::cerr << "No files in directory " << dir << ". " << std::endl;
            return -1;
        }

        for(size_t view_id=0; view_id < files_intern.size(); view_id++)
        {
            std::stringstream image_path_ss;
#ifdef _WIN32
            image_path_ss << dir << "\\" << folder_names[class_id] << "\\" << files_intern[view_id];
#else
            image_path_ss << dir << "/" << folder_names[class_id] << "/" << files_intern[view_id];
#endif
            full_filenames.push_back(image_path_ss.str());
            class_labels.push_back(class_id);
        }
    }

    const size_t display=5;
    for(size_t file_id=0; file_id < full_filenames.size(); file_id++)
    {
        if ( file_id%display == 0 )
            std::cout << "Rendering file " << file_id << " of " << full_filenames.size()
                      << " (showing this message every " << display << " file(s)). " << std::endl;

//         init viewer


        size_t sphere_id = 0;
        float sphere_radius = 2.8f;
        //      for (float sphere_radius = 1.5f; sphere_radius < 5.1f; sphere_radius = sphere_radius + 1.0f, sphere_id++)
        {
            PointCloudRendering pcr( full_filenames[file_id], scale, center );
//            const std::vector<tgTextureModelAI*>& models = pcr.GetModels();
            TomGine::tgModel sphere;

//            TomGine::tgTomGineThread viewer(img_width, img_height, "mesh2pointcloud");
//            viewer.SetClearColor(0.5f);
//            for(size_t i=0; i<models.size(); i++)
//                viewer.AddModel3D(*models[i]);


            TomGine::tgShapeCreator::CreateSphere(sphere, sphere_radius, subdivisions, polyhedron);
            sphere.m_line_width = 1.0f;
//            viewer.AddModel3D(sphere);
//            viewer.Update();

            // camera
            TomGine::tgCamera cam;
            //        cam.SetViewport(640,480);
            //        cam.SetIntrinsicCV(525,525,320,240,0.01f,10.0f);
            cam.SetViewport(img_width, img_height);

            // To preserve Kinect camera parameters (640x480 / f=525)
            const float fx = img_width * static_cast<float>(525 / 640.0f);
            const float fy = img_height * static_cast<float>(525 / 480.0f);
            const float cx = img_width * static_cast<float>(320 / 640.0f);
            const float cy = img_height * static_cast<float>(240 / 480.0f);
            cam.SetIntrinsicCV(fx, fy, cx , cy, 0.01f, 10.0f);

//             init GL context with fbo
                    int pc_id(-1);

            //        for(size_t i=0; i<sphere.m_vertices.size() && !viewer.Stopped(); i++)
            for(size_t i=0; i<sphere.m_vertices.size(); i++)
            {
                TomGine::vec3& p = sphere.m_vertices[i].pos;


//                const float noise_x = cam_pose_distribution(generator);
//                const float noise_y = cam_pose_distribution(generator);
//                const float noise_z = cam_pose_distribution(generator);

//                p.x = p.x + noise_x / 3.0f;
//                p.y = p.y + noise_y / 3.0f;
//                p.z = p.z + noise_z / 3.0f;

                // set camera
                if(cross(p,TomGine::vec3(0,1,0)).length()<TomGine::epsilon)
                    cam.LookAt(p, TomGine::vec3(0,0,0), TomGine::vec3(0,0,1));
                else
                    cam.LookAt(p, TomGine::vec3(0,0,0), TomGine::vec3(0,1,0));
                cam.ApplyTransform();

                const bool use_world_coordinate_system = false;
                pcr.Generate(cam, use_world_coordinate_system);
                const cv::Mat4f& pointcloud = pcr.GetPointCloud(i);

                if ( p.z < 0 && render_upper_half_only) // only render view on the upper hemisphere (we don't train the bottom part of the models)
                    continue;

#ifdef USE_GPU
                cv::gpu::GpuMat pointcloud_dev, depth_image_dev, mask_dev;
                pointcloud_dev.upload(pointcloud);
#endif
                cv::Mat depth_image (pointcloud.rows, pointcloud.cols, CV_32FC1);
                cv::Mat depth_image_norm (pointcloud.rows, pointcloud.cols, CV_32FC1);
                cv::Mat noise_image (pointcloud.rows, pointcloud.cols, CV_32FC1);
                cv::randn(noise_image, 0, bg_noise_level);
                cv::Mat mask (pointcloud.rows, pointcloud.cols, CV_8UC1);
                for (int row_id=0; row_id < pointcloud.rows; row_id++)
                {
                    for (int col_id=0; col_id < pointcloud.cols; col_id++)
                    {
                       const double obj_noise = obj_distribution(generator);

                        const cv::Vec4f pt = pointcloud.at<cv::Vec4f>(row_id, col_id);
                        const float x = pt[0];
                        const float y = pt[1];
                        const float z = pt[2];
                        const float rgb = pt[3];
                        const float depth = z; //std::sqrt( x*x + y*y + z*z );

                        if (std::isnan(depth))
                        {
                            depth_image.at<float>(row_id, col_id) = std::numeric_limits<float>::max();
                            mask.at<unsigned char>(row_id, col_id) = 0;
                        }
                        else
                        {
                            depth_image.at<float>(row_id, col_id) = depth + obj_noise;
                            mask.at<unsigned char>(row_id, col_id) = 1;
                        }
                    }
                }

                cv::Scalar mean = cv::mean(depth_image, mask);
                depth_image -= mean;
                depth_image_norm = 128 - 128 * depth_image / 1.0f;
                depth_image_norm = cv::max(depth_image_norm, 255*cv::abs(noise_image));
                depth_image_norm = cv::min(depth_image_norm, 255);

//                        float depth_norm = 255 * (1.0f - static_cast<float>((depth - min_dist) / (max_dist - min_dist) ) + obj_noise) ;

//                        mask.at<uchar>(row_id, col_id) = 1; // pixel belongs to object

//                        // clippping
//                        if (depth_norm > 255)
//                            depth_norm = 255;

//                        if (depth_norm < 0.0f || std::isnan(depth_norm))
//                        {
//                            depth_norm = 255 * std::abs( static_cast<float>( bg_noise ) );
//                            mask.at<uchar>(row_id, col_id) = 0; // pixel belongs to background
//                        }

//                        depth_image_norm.at<float>(row_id, col_id) = depth_norm;
//                    }
//                }

                //            cv::Scalar mean, stddev;
                //            cv::meanStdDev(depth_image, mean, stddev, mask);
                //            cv::Scalar mean2, stddev2;
                //            cv::meanStdDev(depth_image, mean2, stddev2);
                //            std::cout << "Mean and dev: " << mean << ", " << stddev << "; " << mean2 << ", " << stddev2 << std::endl;
                //            std::cout << "num channels: " << depth_image.channels() << std::endl;
                boost::filesystem::path full_filename_bf(full_filenames[file_id]);
                cv::Mat normalized_depth_img;
                //           depth_image.convertTo(normalized_depth_img, CV_32FC1, 255, -255*mean.val[0]);
                depth_image_norm.convertTo(normalized_depth_img, CV_8UC1);

                std::stringstream out_img_file_ss;
                std::string filename_only, filename_with_ext;
                filename_with_ext = full_filename_bf.filename().string(); // file.ext
                filename_only = full_filename_bf.stem().string();
                int ext_length = filename_with_ext.length() - filename_only.length();
                out_img_file_ss << full_filenames[file_id].substr(0, full_filenames[file_id].length() - ext_length) << "_view_" << std::setw(4) << std::setfill('0') << i << "__rad_" << sphere_id << ".png";
                //           std::cout << "Writing depth image file to " << out_img_file_ss.str();out_img_file_ss << std::endl;
                cv::imwrite(out_img_file_ss.str(), normalized_depth_img);
                //           cv::namedWindow("rendered img", 1);
                //           cv::imshow("rendered img", depth_image);
                //           cv::waitKey(0);

                // visualize
//                            if(pc_id<0)
//                                pc_id = viewer.AddPointCloud(pointcloud);
//                            else
//                                viewer.SetPointCloud(pc_id, pointcloud);

//                            viewer.AddPoint3D(p.x,p.y,p.z, 255,0,0, 10.0f);

//                            viewer.Update();
//                            viewer.WaitForEvent(TomGine::TMGL_Press, TomGine::TMGL_Space);
//                            viewer.ClearPoints3D();
            }
        }
        //        viewer.Clear();
        //        viewer.WaitForEvent(TomGine::TMGL_Press, TomGine::TMGL_Escape);
    }
    return 0;
}


