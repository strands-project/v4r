/*
 * red_green_blue_to_xyzrgb.cpp
 *
 *  Created on: Mar 15, 2013
 *      Author: aitor
 */

/*
 * do_modelling.cpp
 *
 *  Created on: Mar 15, 2013
 *      Author: aitor
 */

#include <pcl/console/parse.h>
#include <faat_pcl/utils/filesystem_utils.h>
#include <pcl/common/common.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/passthrough.h>
#include <pcl/point_types.h>

//MIT dependant stuff...
/*struct PointXYZRedGreenBlue
{
  PCL_ADD_POINT4D;                  // preferred way of adding a XYZ+padding
  float red;
  float green;
  float blue;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW   // make sure our new allocators are aligned
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZRedGreenBlue,
                                   (float, x, x)
                                   (float, y, y)
                                   (float, z, z)
                                   (float, red, red)
                                   (float, green, green)
                                   (float, blue, blue)
                                   )*/

struct PointXYZRedGreenBlue
 {
   PCL_ADD_POINT4D;
   int red;
   int green;
   int blue;
   EIGEN_MAKE_ALIGNED_OPERATOR_NEW   // make sure our new allocators are aligned
 } EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment

 POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZRedGreenBlue,           // here we assume a XYZ + "test" (as fields)
                                    (float, x, x)
                                    (float, y, y)
                                    (float, z, z)
                                    (int, red, red)
                                    (int, green, green)
                                    (int, blue, blue)
)


pcl::PointCloud<pcl::PointXYZRGB> RedGreenBlue_to_RGB(const pcl::PointCloud<PointXYZRedGreenBlue> &cloud)
{
  pcl::PointCloud<pcl::PointXYZRGB> cloud2;
  cloud2.width = cloud.width;
  cloud2.height = cloud.height;
  cloud2.is_dense = false;
  cloud2.points.resize(cloud.width * cloud.height);

  for (uint i = 0; i < cloud.points.size(); i++) {
    cloud2.points[i].x = cloud.points[i].x;
    cloud2.points[i].y = cloud.points[i].y;
    cloud2.points[i].z = cloud.points[i].z;

    int r = cloud.points[i].red;
    int g = cloud.points[i].green;
    int b = cloud.points[i].blue;
    int rgbi = b;


    rgbi += (g << 8);
    rgbi += (r << 16);
    float rgbf; // = *(float*)(&rgbi);
    //memset(&rgbf, 0, sizeof(float));
    memcpy(&rgbf, (float*)(&rgbi), 3);
    cloud2.points[i].rgb = rgbf;
  }

  std::cout << "finished loading cloud..." << std::endl;

  return cloud2;
}

int
main (int argc, char ** argv)
{
  std::string pcd_files_dir_, out_dir_;
  pcl::console::parse_argument (argc, argv, "-pcd_files_dir", pcd_files_dir_);

  std::vector<std::string> files;
  std::string start = "";
  std::string ext = std::string ("pcd");
  bf::path dir = pcd_files_dir_;
  faat_pcl::utils::getFilesInDirectory (dir, start, files, ext);
  std::cout << "Number of scenes in directory is:" << files.size () << std::endl;

  typedef pcl::PointXYZRGB PointType;
  for(size_t i=0; i < files.size(); i++)
  {
    pcl::PointCloud<PointXYZRedGreenBlue> RedGreenBlue;
    pcl::PointCloud<PointType> RGB;
    std::stringstream file_to_read;
    file_to_read << pcd_files_dir_ << "/" << files[i];
    pcl::io::loadPCDFile(file_to_read.str(), RedGreenBlue);
    RGB = RedGreenBlue_to_RGB(RedGreenBlue);

    pcl::io::savePCDFileBinary(file_to_read.str(), RGB);
  }
}
