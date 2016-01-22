#include <iostream>

#include <opencv2/opencv.hpp>
#include <pcl/console/parse.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/PointIndices.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <v4r/io/filesystem.h>
#include <v4r/segmentation/pcl_segmentation_methods.h>
#include <time.h>

template <typename PointT>
class Segmenter
{
private:
    bool save_bounding_boxes_;
    bool visualize_;
    std::string out_dir_, test_dir_;
    int margin_;
    typename v4r::PCLSegmenter<PointT>::Parameter seg_param_;
//    typename boost::shared_ptr<v4r::PCLSegmenter<PointT>> seg_;

    pcl::visualization::PCLVisualizer::Ptr vis_;

    typename pcl::PointCloud<PointT>::Ptr cloud_;
    std::vector<pcl::PointIndices> found_clusters_;
    int vp1_, vp2_;

public:
    Segmenter()
    {
        out_dir_ = "/tmp/segmentation/";
        save_bounding_boxes_ = true;
        visualize_ = true;
        margin_ = 0;
        cloud_.reset(new pcl::PointCloud<PointT>());
        seg_param_.seg_type_ = 1;
    }

    void printUsage(int argc, char ** argv)
    {
        (void) argc;
        std::cerr << "Usage: "
                  << argv[0] << " "
                  << "-test_dir "  << "dir_with_pcd_files "
                  << "[-out_dir "  << "dir_where_output_images_will_be_written_to (default: " << out_dir_ << ")] "
                  << "[-margin "   << "margin_for_output_image_in_pixel (default: " << margin_ << ")] "
                  << "[-visualize " << "visualize_segmented_clusters (default: " << visualize_ << ")] "
                  << "[-seg_type " << "segmentation_method_used (default: " << seg_param_.seg_type_ << ")] "
                  << std::endl;
    }

    void initialize(int argc, char ** argv)
    {
        if(!pcl::console::parse_argument (argc, argv,  "-test_dir", test_dir_))
            printUsage(argc, argv);

        pcl::console::parse_argument (argc, argv,  "-visualize", visualize_);
        pcl::console::parse_argument (argc, argv,  "-out_dir", out_dir_);
        pcl::console::parse_argument (argc, argv,  "-margin", margin_);

        pcl::console::parse_argument (argc, argv,  "-seg_type", seg_param_.seg_type_ );
        pcl::console::parse_argument (argc, argv,  "-min_cluster_size", seg_param_.min_cluster_size_ );
        pcl::console::parse_argument (argc, argv,  "-max_vertical_plane_size", seg_param_.max_vertical_plane_size_ );
        pcl::console::parse_argument (argc, argv,  "-num_plane_inliers", seg_param_.num_plane_inliers_ );
        pcl::console::parse_argument (argc, argv,  "-max_angle_plane_to_ground", seg_param_.max_angle_plane_to_ground_ );
        pcl::console::parse_argument (argc, argv,  "-sensor_noise_max", seg_param_.sensor_noise_max_ );
        pcl::console::parse_argument (argc, argv,  "-table_range_min", seg_param_.table_range_min_ );
        pcl::console::parse_argument (argc, argv,  "-table_range_max", seg_param_.table_range_max_ );
        pcl::console::parse_argument (argc, argv,  "-chop_z", seg_param_.chop_at_z_ );
        pcl::console::parse_argument (argc, argv,  "-angular_threshold_deg", seg_param_.angular_threshold_deg_ );

//        seg_.reset(  );
    }

    bool eval()
    {
        v4r::PCLSegmenter<PointT> seg_(seg_param_);
        std::vector< std::string> sub_folder_names = v4r::io::getFoldersInDirectory( test_dir_ );
        if(sub_folder_names.empty())
        {
            std::cerr << "No subfolders in directory " << test_dir_ << ". " << std::endl;
            sub_folder_names.push_back("");
        }

        std::sort(sub_folder_names.begin(), sub_folder_names.end());
        for (size_t sub_folder_id=0; sub_folder_id < sub_folder_names.size(); sub_folder_id++)
        {
            const std::string sequence_path = test_dir_ + "/" + sub_folder_names[ sub_folder_id ];
            const std::string out_path = out_dir_ + "/" + sub_folder_names[ sub_folder_id ];
            v4r::io::createDirIfNotExist(out_path);

            std::vector< std::string > views = v4r::io::getFilesInDirectory(sequence_path, ".*.pcd", false);

            for (size_t v_id=0; v_id<views.size(); v_id++)
            {
                const std::string fn = test_dir_ + "/" + sub_folder_names[sub_folder_id] + "/" + views[ v_id ];
                std::string out_fn_prefix = out_path + "/" + views[ v_id ];
                boost::replace_last(out_fn_prefix, ".pcd", "");

                std::cout << "Segmenting file " << fn << std::endl;
                pcl::io::loadPCDFile(fn, *cloud_);

                seg_.set_input_cloud(*cloud_);
                seg_.do_segmentation(found_clusters_);

                if(visualize_)
                    visualize();

                if(save_bounding_boxes_)
                {
                    save_bb_image(out_fn_prefix);
                }
            }
        }

        return 0;
    }

    void visualize()
    {
        if(!vis_)
        {
            vis_.reset ( new pcl::visualization::PCLVisualizer("Segmentation Results") );
            vis_->createViewPort(0,0,0.5,1,vp1_);
            vis_->createViewPort(0.5,0,1,1,vp2_);
        }
        vis_->removeAllPointClouds();
        vis_->removeAllShapes();
        vis_->addPointCloud(cloud_, "cloud", vp1_);


        typename pcl::PointCloud<PointT>::Ptr colored_cloud (new pcl::PointCloud<PointT>());
        for(size_t i=0; i < found_clusters_.size(); i++)
        {
            typename pcl::PointCloud<PointT>::Ptr cluster (new pcl::PointCloud<PointT>());
            pcl::copyPointCloud(*cloud_, found_clusters_[i], *cluster);

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
        }
        vis_->addPointCloud(colored_cloud,"segments", vp2_);
        vis_->spin();
    }

    void save_bb_image(const std::string &filename_prefix)
    {
        for(size_t i=0; i < found_clusters_.size(); i++)
        {
            std::stringstream filename; filename << filename_prefix << "_" << std::setfill('0') << std::setw(5) << i << ".jpg";
            int min_u, min_v, max_u, max_v;
            max_u = max_v = 0;
            min_u = cloud_->width;
            min_v = cloud_->height;

            const std::vector<int> &c_tmp = found_clusters_[i].indices;

            for(size_t idx=0; idx<c_tmp.size(); idx++)
            {
                int u = c_tmp[idx] % cloud_->width;
                int v = (int) (c_tmp[idx] / cloud_->width);

                if (u>max_u)
                    max_u = u;

                if (v>max_v)
                    max_v = v;

                if (u<min_u)
                    min_u = u;

                if (v<min_v)
                    min_v = v;
            }

            min_u = std::max (0, min_u - margin_);
            min_v = std::max (0, min_v - margin_);
            max_u = std::min ((int)cloud_->width, max_u + margin_);
            max_v = std::min ((int)cloud_->height, max_v + margin_);

            int img_width  = max_u - min_u;
            int img_height = max_v - min_v;
            cv::Mat_<cv::Vec3b> image(img_height, img_width);

            for (int row = 0; row < img_height; row++)
            {
              for (int col = 0; col < img_width; col++)
              {
                cv::Vec3b & cvp = image.at<cv::Vec3b> (row, col);
                int position = (row + min_v) * cloud_->width + (col + min_u);
                const PointT &pt = cloud_->points[position];

                cvp[0] = pt.b;
                cvp[1] = pt.g;
                cvp[2] = pt.r;
              }
            }
            cv::imwrite(filename.str(), image);
        }
    }
};


int
main (int argc, char ** argv)
{
    srand (time(NULL));
    Segmenter<pcl::PointXYZRGB> s;
    s.initialize(argc,argv);
    s.eval();
    return 0;
}
