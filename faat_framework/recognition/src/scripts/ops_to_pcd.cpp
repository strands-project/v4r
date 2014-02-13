#include <pcl/io/pcd_io.h>
#include <pcl/console/parse.h>
#include <fstream>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/common/transforms.h>
int
main (int argc, char ** argv)
{
  std::string ops_file = "";
  std::string pcd_out_file = "";
  bool save_normals = true;
  bool normals_present = true;
  bool flip_z = false;

  pcl::console::parse_argument (argc, argv, "-ops_file", ops_file);
  pcl::console::parse_argument (argc, argv, "-pcd_out_file", pcd_out_file);
  pcl::console::parse_argument (argc, argv, "-save_normals", save_normals);
  pcl::console::parse_argument (argc, argv, "-normals_present", normals_present);
  pcl::console::parse_argument (argc, argv, "-flip_z", flip_z);

  std::ifstream infile(ops_file.c_str());
  std::string line;
  int ln=0;
  int npoints=-1;
  std::vector<Eigen::Vector3f> ps;
  std::vector<Eigen::Vector3f> ns;

  while (std::getline(infile, line))
  {
      std::istringstream iss(line);
      if(ln == 0) {
        iss >> npoints;
      } else {

        Eigen::Vector3f p;
        for(size_t k=0; k < 3; k++)
          iss >> p[k];

        ps.push_back(p);

        if(normals_present) {
          Eigen::Vector3f n;
          for(size_t k=0; k < 3; k++)
            iss >> n[k];

          ns.push_back(n);
        }
      }

      ln++;
  }

  std::cout << ps.size() << " " << ns.size() << " " << npoints <<std::endl;
  if(save_normals && normals_present) {
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud(new pcl::PointCloud<pcl::PointNormal>);
    cloud->points.resize(npoints);
    cloud->width = npoints;
    cloud->height = 1;

    for(size_t i=0; i < ps.size(); i++) {
      cloud->points[i].getVector3fMap() = ps[i];
      cloud->points[i].getNormalVector3fMap() = ns[i];
    }

    pcl::io::savePCDFileBinary(pcd_out_file, *cloud);
  } else {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    cloud->points.resize(npoints);
    cloud->width = npoints;
    cloud->height = 1;

    for(size_t i=0; i < ps.size(); i++) {
      cloud->points[i].getVector3fMap() = ps[i];
    }

    if(flip_z) {
      float rot_angle = pcl::deg2rad (static_cast<float> (180));
      Eigen::Affine3f rot_trans = Eigen::Affine3f (Eigen::AngleAxisf (static_cast<float> (rot_angle), Eigen::Vector3f::UnitY ()));
      pcl::transformPointCloud(*cloud, *cloud, rot_trans);
    }

    pcl::io::savePCDFileBinary(pcd_out_file, *cloud);
  }
}
