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
#include <vtkPolyData.h>
#include <vtkTriangle.h>
#include <vtkSmartPointer.h>
#include <vtkCellArray.h>
#include <vtkPLYReader.h>
#include <vtkPolyDataMapper.h>
#include <vtkTransform.h>
#include <vtkTransformFilter.h>
#include <vtkTransformPolyDataFilter.h>

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

  std::vector<PointCloud<PointXYZ>::Ptr> views_xyz;
  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > poses;
  std::vector<float> enthropies;
  pcl::apps::RenderViewsTesselatedSphere rend;
  rend.setRadiusSphere(1.f);
  rend.setTesselationLevel(tes_level_);
  rend.setGenOrganized(true);
  rend.setUseVertices(use_vertices);
  rend.setResolution(res);
  rend.addModelFromPolyData(mapper);
  rend.generateViews();
  rend.getViews(views_xyz);

  for(size_t i=0; i < views_xyz.size(); i++)
  {
    std::stringstream out_file;
    out_file << out_dir << "/cloud_" << i << ".pcd";
    pcl::io::savePCDFileBinary(out_file.str(), *views_xyz[i]);
  }
}

