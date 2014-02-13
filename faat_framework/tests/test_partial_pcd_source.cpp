#include <pcl/console/parse.h>
#include <faat_pcl/3d_rec_framework/pc_source/partial_pcd_source.h>
#include <faat_pcl/utils/filesystem_utils.h>

#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <fstream>
#include <faat_pcl/utils/pcl_opencv.h>

namespace bf = boost::filesystem;

struct camPosConstraints
{
  bool
  operator() (const Eigen::Vector3f & pos) const
  {
    if (pos[2] > 0)
      return true;

    return false;
  }
  ;
};

int
main (int argc, char ** argv)
{

  boost::function<bool
  (const Eigen::Vector3f &)> campos_constraints;
  campos_constraints = camPosConstraints ();
  int tes_level_ = 1;
  std::string input;
  float dot_normal = 0.f;
  bool gen_organized_ = false;
  float distance = 1.f;

  pcl::console::parse_argument (argc, argv, "-tes_level", tes_level_);
  pcl::console::parse_argument (argc, argv, "-input", input);
  pcl::console::parse_argument (argc, argv, "-dot_normal", dot_normal);
  pcl::console::parse_argument (argc, argv, "-gen_organized", gen_organized_);
  pcl::console::parse_argument (argc, argv, "-distance", distance);

  std::string training_dir = "test_partial";

  boost::shared_ptr<faat_pcl::rec_3d_framework::PartialPCDSource<pcl::PointXYZRGBNormal, pcl::PointXYZRGB> >
                                                                                                             source (
                                                                                                                     new faat_pcl::rec_3d_framework::PartialPCDSource<
                                                                                                                         pcl::PointXYZRGBNormal,
                                                                                                                         pcl::PointXYZRGB>);
  source->setPath (input);
  source->setModelScale (1.f);
  source->setRadiusSphere (distance);
  source->setTesselationLevel (tes_level_);
  source->setDotNormal (dot_normal);
  source->setGenOrganized(gen_organized_);
  source->setLoadViews (true);
  source->setCamPosConstraints (campos_constraints);
  source->setWindowSizeAndFocalLength(640, 480, 575.f);
  source->genInPlaneRotations(true, 45.f);
  source->setLoadIntoMemory (false);
  source->generate (training_dir);

  bf::path input_path = training_dir;
  std::vector<std::string> model_files;
  std::string pattern_models = ".*view_.*.pcd";
  std::string relative_path = "";
  faat_pcl::utils::getFilesInDirectoryRecursive(input_path, relative_path, model_files, pattern_models);

  for(size_t i=0; i < model_files.size(); i++)
  {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    std::stringstream file;
    file << training_dir << "/" << model_files[i];
    pcl::io::loadPCDFile(file.str(), *cloud);

    {
      cv::Mat_ < cv::Vec3b > colorImage;
      PCLOpenCV::ConvertPCLCloud2Image<pcl::PointXYZRGB> (cloud, colorImage);
      cv::namedWindow("original");
      cv::imshow("original", colorImage);
    }
    cv::waitKey(0);
  }
}
