/*
 * local_recognition_mian_dataset.cpp
 *
 *  Created on: Mar 24, 2012
 *      Author: aitor
 */

#undef NDEBUG

#include <pcl/console/parse.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/passthrough.h>
#include <pcl/io/pcd_io.h>

#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/filters/voxel_grid.h>

#include <v4r/OCTopDownSegmenter/edge_based_presegmenter.h>
#include <v4r/OCTopDownSegmenter/sv_ms_presegmenter.h>
#include <v4r/OCTopDownSegmenter/mv_MS_presegmenter.h>
#include <v4r/utils/filesystem_utils.h>

#include <pcl/filters/statistical_outlier_removal.h>
#include <v4r/ORUtils/pcl_opencv.h>

void denoisePoint(Eigen::Vector3f p, Eigen::Vector3f n,
                    float sigma_c, float sigma_s,
                    pcl::PointCloud<pcl::PointXYZRGB>::Ptr & cloud,
                    int r, int c,
                    pcl::PointCloud<pcl::PointXYZRGB>::Ptr & copy,
                    int kernel_width)
{
    if(!pcl_isfinite(p[2]))
        return;

    //use neighborhood to compute weights...
    float sum, normalizer;
    sum = normalizer = 0;
    for(int u=std::max(0, r - kernel_width); u <= std::min((int)(cloud->height - 1), r + kernel_width); u++)
    {
        for(int v=std::max(0, c - kernel_width); v <= std::min((int)(cloud->width - 1), c + kernel_width); v++)
        {
            Eigen::Vector3f p_uv = cloud->at(v,u).getVector3fMap();
            if(!pcl_isfinite(p_uv[2]))
                return;

            float t = (p_uv - p).norm();
            float h = n.dot(p - p_uv);
            float wc, ws;
            wc = std::exp(-t*t / (2*sigma_c*sigma_c));
            ws = std::exp(-h*h / (2*sigma_s*sigma_s));
            sum += wc * ws * h;
            normalizer += wc * ws;
        }
    }

    //std::cout << sum << " " << normalizer << std::endl;

    //copy->at(c,r).getVector3fMap() = copy->at(c,r).getVector3fMap() + n * (sum / normalizer);
    copy->at(c,r).z = copy->at(c,r).z + (sum / normalizer);

}

void bilateral_filter(pcl::PointCloud<pcl::PointXYZRGB>::Ptr & cloud,
                      pcl::PointCloud<pcl::Normal>::Ptr & normals,
                      float sigma_c, float sigma_s,
                      int kernel_width = 5)
{

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr copy(new pcl::PointCloud<pcl::PointXYZRGB>(*cloud));

    for(int r=0; r < (int)copy->height; r++)
    {
        for(int c=0; c < (int)copy->width; c++)
        {
            denoisePoint(cloud->at(c,r).getVector3fMap(),
                         normals->at(c,r).getNormalVector3fMap(),
                         sigma_c, sigma_s,
                         copy, r, c, cloud, kernel_width);

        }
    }

    /*pcl::visualization::PCLVisualizer vis("bilateral filter");
    int v1, v2;
    vis.createViewPort(0,0,0.5,1,v1);
    vis.createViewPort(0.5,0,1,1,v2);

    {
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler(cloud);
        vis.addPointCloud(cloud, handler, "cloud", v1);
    }

    {
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler(copy);
        vis.addPointCloud(copy, handler, "copy", v2);
    }

    vis.spin();*/
}

//./app_test -pcd_file /media/DATA/OSD-0.2/pcd/test51.pcd -z_dist 1.3 -sigma_s 0.005 -sigma_c 0.001 -kernel_width 3 -seg_type 1 -bf 1 -nyu 0.035 -vis_each_move 0
//./app_test -pcd_file /media/aitor14/DATA/OSD-0.2/pcd/test59.pcd -z_dist 1.3 -sigma_s 0.005 -sigma_c 0.001 -kernel_width 3 -seg_type 1 -bf 1 -nyu 0.00015 -vis_each_move 0 -rgbd_plus_labels_file /media/aitor14/DATA/OSD-0.2/results_rgbd_plus_labels/test59.pcd -refinement 1 -lambda 0 -sv_res 0.004 -sv_seed 0.04 -sigma 0

void visualizeOutput(pcl::PointCloud<pcl::PointXYZRGB>::Ptr & scene,
                     std::vector< std::vector<int> > & clusters)
{
    std::vector<uint32_t> label_colors_;

    int max_label = clusters.size();
    if((int)label_colors_.size() != max_label)
    {
        label_colors_.reserve (max_label + 1);
        srand (static_cast<unsigned int> (time (0)));
        while ((int)label_colors_.size () <= max_label )
        {
            uint8_t r = static_cast<uint8_t>( (rand () % 256));
            uint8_t g = static_cast<uint8_t>( (rand () % 256));
            uint8_t b = static_cast<uint8_t>( (rand () % 256));
            label_colors_.push_back (static_cast<uint32_t>(r) << 16 | static_cast<uint32_t>(g) << 8 | static_cast<uint32_t>(b));
        }
    }

    if(scene->isOrganized())
    {
        cv::Mat_<cv::Vec3b> image;
        PCLOpenCV::ConvertPCLCloud2Image<pcl::PointXYZRGB>(scene, image);

        cv::Mat image_clone = image.clone();

        float factor = 0.05f;

        for(size_t i=0; i < clusters.size(); i++)
        {
            for(size_t j=0; j < clusters[i].size(); j++)
            {
                int r, c;
                int idx = clusters[i][j];
                r = idx / scene->width;
                c = idx % scene->width;

                uint32_t rgb = label_colors_[i];
                unsigned char rs = (rgb >> 16) & 0x0000ff;
                unsigned char gs = (rgb >> 8) & 0x0000ff;
                unsigned char bs = (rgb) & 0x0000ff;

                cv::Vec3b im = image.at<cv::Vec3b>(r,c);
                image.at<cv::Vec3b>(r,c) = cv::Vec3b((unsigned char)(im[0] * factor + bs * (1 - factor)),
                        (unsigned char)(im[1] * factor + gs * (1 - factor)),
                        (unsigned char)(im[2] * factor + rs * (1 - factor)));
            }
        }

        cv::Mat collage = cv::Mat(image.rows, image.cols * 2, CV_8UC3);
        collage.setTo(cv::Vec3b(0,0,0));

        for(unsigned int r=0; r < scene->height; r++)
        {
            for(unsigned int c=0; c < scene->width; c++)
            {
                collage.at<cv::Vec3b>(r,c) = image_clone.at<cv::Vec3b>(r,c);
            }
        }

        collage(cv::Range(0, collage.rows), cv::Range(collage.cols/2, collage.cols)) = image + cv::Scalar(cv::Vec3b(0, 0, 0));

        cv::imshow("regions", collage);
        cv::waitKey(0);
    }
    else
    {

        //same thing with point cloud
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cc(new pcl::PointCloud<pcl::PointXYZRGB>(*scene));
        for(size_t i=0; i < clusters.size(); i++)
        {
            for(size_t j=0; j < clusters[i].size(); j++)
            {
                cloud_cc->at(clusters[i][j]).rgb = label_colors_[i];
            }
        }

        pcl::visualization::PCLVisualizer vis("regions");
        vis.addPointCloud(cloud_cc);
        vis.spin();
    }
}

/// Command line
/// Unorganized cloud
/// ./app_test -pcd_file /home/aitor14/Downloads/room1/models/modelname.pcd -z_dist 1.3 -sigma_s 0.005 -sigma_c 0.005 -kernel_width 3 -seg_type 1 -bf 1 -nyu 0.015 -vis_each_move 0 -max_mt 1 -refinement 0 -lambda 0.00001 -sv_res 0.02 -sv_seed 0.2 -filter_cloud 1
/// Organized
/// ./app_test -pcd_file /media/aitor14/DATA/OSD-0.2/pcd/test58.pcd -z_dist 1.3 -sigma_s 0.005 -sigma_c 0.005 -kernel_width 3 -seg_type 1 -bf 1 -nyu 0.01 -vis_each_move 0 -max_mt 1 -refinement 1 -lambda 0.000025
int
main (int argc, char ** argv)
{
  std::string pcd_file = "";
  float z_dist = 3.f;
  float sigma_s, sigma_c;
  sigma_s = 0.1f;
  sigma_c = 0.05f;
  int kernel_width = 5;
  int seg_type = 0;
  bool bf = false;
  bool vis_each_move = false;
  int max_mt = 1;
  bool pixwise_refinement = false;
  float lambda = 0.005f;
  float sigma = 0.00001f;
  float alpha = 1.f;
  float nyu = 0.0001f;

  std::string save_to_;


  float sv_res = 0.004f;
  float sv_seed = 0.03f;
  float sv_color = 0.f;
  float sv_spatial = 1.f;
  float sv_normal = 3.f;

  int boundary_window = 1;
  float boundary_radius = 0.015;

  bool NYU_dataset = false;
  bool filter_cloud = false;
  bool use_SLIC = false;

  std::string output_labels = "";
  std::string rgbd_plus_labels_file = "";

  pcl::console::parse_argument (argc, argv, "-output_labels", output_labels);
  pcl::console::parse_argument (argc, argv, "-pcd_file", pcd_file);
  pcl::console::parse_argument (argc, argv, "-z_dist", z_dist);
  pcl::console::parse_argument (argc, argv, "-sigma_s", sigma_s);
  pcl::console::parse_argument (argc, argv, "-sigma_c", sigma_c);
  pcl::console::parse_argument (argc, argv, "-kernel_width", kernel_width);
  pcl::console::parse_argument (argc, argv, "-seg_type", seg_type);
  pcl::console::parse_argument (argc, argv, "-bf", bf);
  pcl::console::parse_argument (argc, argv, "-vis_each_move", vis_each_move);
  pcl::console::parse_argument (argc, argv, "-max_mt", max_mt);
  pcl::console::parse_argument (argc, argv, "-refinement", pixwise_refinement);
  pcl::console::parse_argument (argc, argv, "-rgbd_plus_labels_file", rgbd_plus_labels_file);
  pcl::console::parse_argument (argc, argv, "-boundary_window", boundary_window);
  pcl::console::parse_argument (argc, argv, "-boundary_radius", boundary_radius);
  pcl::console::parse_argument (argc, argv, "-filter_cloud", filter_cloud);
  pcl::console::parse_argument (argc, argv, "-use_SLIC", use_SLIC);

  //MS regularizers
  pcl::console::parse_argument (argc, argv, "-nyu", nyu);
  pcl::console::parse_argument (argc, argv, "-lambda", lambda);
  pcl::console::parse_argument (argc, argv, "-sigma", sigma);
  pcl::console::parse_argument (argc, argv, "-alpha", alpha);

  pcl::console::parse_argument (argc, argv, "-save_to", save_to_);
  pcl::console::parse_argument (argc, argv, "-NYU_dataset", NYU_dataset);
  pcl::console::parse_argument (argc, argv, "-sv_res", sv_res);
  pcl::console::parse_argument (argc, argv, "-sv_seed", sv_seed);
  pcl::console::parse_argument (argc, argv, "-sv_color", sv_color);
  pcl::console::parse_argument (argc, argv, "-sv_spatial", sv_spatial);
  pcl::console::parse_argument (argc, argv, "-sv_normal", sv_normal);

  std::vector<std::string> to_process;
  /*if(v4r::utils::getFilesInDirectory(pcd_file, to_process, so_far, pattern, true) != -1)
  {


//      void
//      getFilesInDirectory (   const bf::path & dir,
//                              std::vector<std::string> & relative_paths,
//                              const std::string & rel_path_so_far = std::string(""),
//                              const std::string & regex_pattern = std::string(""),
//                              bool recursive = true);
  }
  else
  {*/
      to_process.push_back(pcd_file);
  //}

  std::cout << to_process.size() << std::endl;

  for(size_t k=0; k < to_process.size(); k++)
  {

      std::cout << to_process[k] << std::endl;
      std::string file;
      std::string save_labels_to;

      bf::path dir = pcd_file;
      if(bf::is_directory(dir))
      {
          std::stringstream file_ss;
          file_ss << pcd_file << "/" << to_process[k];
          file = file_ss.str();

          std::stringstream save_toss;
          save_toss << output_labels << "/" << to_process[k];
          save_labels_to = save_toss.str();
      }
      else
      {
          file = pcd_file;
          save_labels_to = output_labels;
      }

      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
      pcl::io::loadPCDFile(file, *cloud);

      bool is_organized = cloud->isOrganized();

      if(NYU_dataset)
      {
          for(size_t i=0; i < cloud->points.size(); i++)
          {
              Eigen::Vector3f p = cloud->points[i].getVector3fMap();
              if(p[2] < 0.3)
              {
                  cloud->points[i].x = cloud->points[i].y = cloud->points[i].z = std::numeric_limits<float>::quiet_NaN();
              }
          }
      }

      float radius_normals = 0.01f;

      if(is_organized)
      {
          pcl::PassThrough<pcl::PointXYZRGB> pass_;
          pass_.setFilterLimits (0.f, z_dist);
          pass_.setFilterFieldName ("z");
          pass_.setInputCloud (cloud);
          pass_.setKeepOrganized (true);
          pass_.filter (*cloud);
      }
      else
      {
          //should we do down sampling here?

          pcl::VoxelGrid<pcl::PointXYZRGB> filter;
          filter.setInputCloud(cloud);
          filter.setDownsampleAllData(true);
          filter.setLeafSize(0.01f,0.01f,0.01f);

          pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_voxel(new pcl::PointCloud<pcl::PointXYZRGB>);
          filter.filter(*cloud_voxel);

          if(filter_cloud)
          {
              pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
              sor.setInputCloud (cloud_voxel);
              sor.setMeanK (50);
              sor.setStddevMulThresh (1.0);
              sor.filter (*cloud);
          }
          else
          {
              cloud = cloud_voxel;
          }

          radius_normals = 0.02f;
          seg_type = 2;
      }

      pcl::PointCloud<pcl::Normal>::Ptr normal_cloud (new pcl::PointCloud<pcl::Normal>);

      pcl::NormalEstimationOMP<pcl::PointXYZRGB, pcl::Normal> ne;
      ne.setRadiusSearch(radius_normals);
      ne.setInputCloud (cloud);
      ne.compute (*normal_cloud);

      if(bf && is_organized)
        bilateral_filter(cloud, normal_cloud, sigma_s, sigma_c, kernel_width);

      if(seg_type == 0)
      {
          pcl::IntegralImageNormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
          ne.setNormalEstimationMethod (ne.AVERAGE_3D_GRADIENT);
          ne.setMaxDepthChangeFactor(0.02f);
          ne.setDepthDependentSmoothing(true);
          ne.setNormalSmoothingSize(5.0f);
          ne.setInputCloud(cloud);
          ne.compute(*normal_cloud);

          v4rOCTopDownSegmenter::EdgeBasedPreSegmenter<pcl::PointXYZRGB> pre_segmenter;
          pre_segmenter.setInputCloud(cloud);
          pre_segmenter.setSurfaceNormals(normal_cloud);
          pre_segmenter.process();
      }
      else if(seg_type == 1)
      {
          v4rOCTopDownSegmenter::SVMumfordShahPreSegmenter<pcl::PointXYZRGB> pre_segmenter;
          pre_segmenter.setInputCloud(cloud);
          pre_segmenter.setSurfaceNormals(normal_cloud);
          pre_segmenter.setNyu(nyu);
          pre_segmenter.setLambda(lambda);
          pre_segmenter.setSigma(sigma);
          pre_segmenter.setAlpha(alpha);
          pre_segmenter.setVisEachMove(vis_each_move);
          pre_segmenter.setMaxModelType(max_mt);
          pre_segmenter.setPixelWiseRefinement(pixwise_refinement);
          pre_segmenter.setSaveImPath(save_to_);
          pre_segmenter.setSVParams(sv_seed, sv_res);
          pre_segmenter.setSVImportanceValues(sv_color, sv_spatial, sv_normal);
          pre_segmenter.setBoundaryWindow(boundary_window);
          pre_segmenter.setUseSLIC(use_SLIC);
          pre_segmenter.process();

          if(pixwise_refinement)
          {
              pcl::PointCloud<pcl::PointXYZL>::Ptr labels;
              pre_segmenter.getLabelCloud(labels);

              if(save_labels_to.compare("") != 0)
                pcl::io::savePCDFileBinary(save_labels_to, *labels);

              if(rgbd_plus_labels_file.compare("") != 0)
              {
                  pcl::PointCloud<pcl::PointXYZRGBL>::Ptr cloud_and_labels(new pcl::PointCloud<pcl::PointXYZRGBL>);
                  pcl::copyPointCloud(*cloud, *cloud_and_labels);
                  for(size_t k=0; k < labels->points.size(); k++)
                  {
                      cloud_and_labels->points[k].label = labels->points[k].label;
                  }

                  pcl::io::savePCDFileBinary(rgbd_plus_labels_file, *cloud_and_labels);
              }
          }

          std::vector<std::vector<int> > segmentation_indices;
          pre_segmenter.getSegmentationIndices(segmentation_indices);
          visualizeOutput(cloud, segmentation_indices);
      }
      else
      {
          v4rOCTopDownSegmenter::MVMumfordShahPreSegmenter<pcl::PointXYZRGB> pre_segmenter;
          pre_segmenter.setInputCloud(cloud);
          pre_segmenter.setSurfaceNormals(normal_cloud);
          pre_segmenter.setNyu(nyu);
          pre_segmenter.setLambda(lambda);
          pre_segmenter.setSigma(sigma);
          pre_segmenter.setAlpha(alpha);
          pre_segmenter.setVisEachMove(vis_each_move);
          pre_segmenter.setMaxModelType(max_mt);
          pre_segmenter.setPixelWiseRefinement(pixwise_refinement);
          pre_segmenter.setSaveImPath(save_to_);
          pre_segmenter.setSVParams(sv_seed, sv_res);
          pre_segmenter.setSVImportanceValues(sv_color, sv_spatial, sv_normal);
          pre_segmenter.setBoundaryRadius(boundary_radius);
          pre_segmenter.process();

          pcl::PointCloud<pcl::PointXYZL>::Ptr labels;
          pre_segmenter.getLabelCloud(labels);

          if(save_labels_to.compare("") != 0)
            pcl::io::savePCDFileBinary(save_labels_to, *labels);

          if(rgbd_plus_labels_file.compare("") != 0)
          {
              pcl::PointCloud<pcl::PointXYZRGBL>::Ptr cloud_and_labels(new pcl::PointCloud<pcl::PointXYZRGBL>);
              pcl::copyPointCloud(*cloud, *cloud_and_labels);
              for(size_t k=0; k < labels->points.size(); k++)
              {
                  cloud_and_labels->points[k].label = labels->points[k].label;
              }

              pcl::io::savePCDFileBinary(rgbd_plus_labels_file, *cloud_and_labels);
          }

          std::vector<std::vector<int> > segmentation_indices;
          pre_segmenter.getSegmentationIndices(segmentation_indices);
          visualizeOutput(cloud, segmentation_indices);
      }
  }

}
