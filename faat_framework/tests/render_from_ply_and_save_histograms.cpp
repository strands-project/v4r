/*
 * render_from_ply.cpp
 *
 *  Created on: Mar 18, 2013
 *      Author: aitor
 */

#include <pcl/console/parse.h>
#include <faat_pcl/utils/filesystem_utils.h>
#include <pcl/common/common.h>
#include <pcl/io/pcd_io.h>
#include <pcl/apps/render_views_tesselated_sphere.h>
#include <pcl/filters/voxel_grid.h>
#include <faat_pcl/3d_rec_framework/feature_wrapper/global/esf_estimator.h>
#include <vtkPolyData.h>
#include <vtkTriangle.h>
#include <vtkSmartPointer.h>
#include <vtkCellArray.h>
#include <vtkPLYReader.h>
#include <vtkPolyDataMapper.h>
#include <vtkTransform.h>
#include <vtkTransformFilter.h>
#include <vtkTransformPolyDataFilter.h>

void
writeHistogramToFile(std::string file, float * histogram, int size)
{
  std::ofstream out (file.c_str ());
  if (!out)
  {
    std::cout << "Cannot open file.\n";
    return;
  }

  for(size_t i=0; i < size; i++)
  {
    out << histogram[i];
    if(i < (size - 1))
      out << " ";
  }

  out << std::endl;

  out.close ();
}

using namespace pcl;
int
main (int argc, char ** argv)
{
  std::string ply_file, out_dir;
  float model_scale_ = 0.001f;
  int tes_level_ = 0;
  bool use_vertices = false;
  int res = 250;

  pcl::console::parse_argument (argc, argv, "-ply_file", ply_file);
  pcl::console::parse_argument (argc, argv, "-out_dir", out_dir);
  pcl::console::parse_argument (argc, argv, "-model_scale", model_scale_);
  pcl::console::parse_argument (argc, argv, "-tes_level", tes_level_);
  pcl::console::parse_argument (argc, argv, "-use_vertices", use_vertices);
  pcl::console::parse_argument (argc, argv, "-res", res);

  std::string ply_name;
  std::vector<std::string> strs;
  boost::split (strs, ply_file, boost::is_any_of ("/"));
  ply_name = strs[strs.size() - 1];
  boost::replace_all (ply_name, ".ply", "");

  //check if destination files exist
  bool all_exists = true;
  for(size_t i=0; i < 20; i++)
  {
      std::stringstream out_file;
      out_file << out_dir << "/" << ply_name << "_view_" << i << ".txt";
      bf::path file_path = out_file.str();
      if(!bf::exists(file_path))
      {
          all_exists = false;
      }
  }

  vtkSmartPointer < vtkPLYReader > reader = vtkSmartPointer<vtkPLYReader>::New ();
  reader->SetFileName (ply_file.c_str ());
  reader->Update();

  vtkSmartPointer < vtkTransform > trans = vtkSmartPointer<vtkTransform>::New ();
  trans->Scale (model_scale_, model_scale_, model_scale_);
  trans->Modified ();
  trans->Update ();

  vtkSmartPointer < vtkTransformPolyDataFilter > filter_scale = vtkSmartPointer<vtkTransformPolyDataFilter>::New ();
  filter_scale->SetTransform (trans);
  filter_scale->SetInputConnection(reader->GetOutputPort());
  filter_scale->Update();

  vtkSmartPointer < vtkPolyData > mapper = filter_scale->GetOutput();
  mapper->Update ();

  if(all_exists)
  {
      PCL_WARN("All files exist, doing nothing...\n");
      return 0;
  }

  std::vector<PointCloud<PointXYZ>::Ptr> views_xyz;
  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > poses;
  std::vector<float> enthropies;
  pcl::apps::RenderViewsTesselatedSphere rend;
  rend.setRadiusSphere(1.f);
  rend.setTesselationLevel(tes_level_);
  rend.setGenOrganized(false);
  rend.setUseVertices(use_vertices);
  rend.setResolution(res);
  rend.addModelFromPolyData(mapper);
  rend.generateViews();
  rend.getViews(views_xyz);

  boost::shared_ptr<faat_pcl::rec_3d_framework::ESFEstimation<pcl::PointXYZ, pcl::ESFSignature640> > estimator;
  estimator.reset (new faat_pcl::rec_3d_framework::ESFEstimation<pcl::PointXYZ, pcl::ESFSignature640>);

  bf::path out = out_dir;
  if(!bf::exists(out_dir))
    bf::create_directory(out);

  float voxel_grid_size = 0.001;
  for(size_t i=0; i < views_xyz.size(); i++)
  {

    /*PointCloud<PointXYZ>::Ptr vxgrided(new PointCloud<PointXYZ>);
    pcl::VoxelGrid<PointXYZ> grid_;
    grid_.setInputCloud (views_xyz[i]);
    grid_.setLeafSize (voxel_grid_size, voxel_grid_size, voxel_grid_size);
    grid_.setDownsampleAllData (true);
    grid_.filter (*vxgrided);*/

    PointCloud<PointXYZ>::Ptr processed(new PointCloud<PointXYZ>);
    pcl::PointCloud<pcl::ESFSignature640>::CloudVectorType signatures;
    std::vector<Eigen::Vector3f> centroids;
    estimator->estimate (views_xyz[i], processed, signatures, centroids);

    //std::cout << signatures.size() << std::endl;
    std::stringstream out_file;
    out_file << out_dir << "/" << ply_name << "_view_" << i << ".txt";

    for (size_t idx = 0; idx < signatures.size (); idx++)
    {
      float* hist = signatures[idx].points[0].histogram;
      int size_feat = sizeof(signatures[idx].points[0].histogram) / sizeof(float);
      //std::cout << size_feat << " " << out_file.str() << std::endl;
      writeHistogramToFile(out_file.str(), hist, size_feat);
    }
    /*std::stringstream out_file;
    out_file << out_dir << "/cloud_" << i << ".pcd";
    pcl::io::savePCDFileBinary(out_file.str(), *views_xyz[i]);*/
  }
}

