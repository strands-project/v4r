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
#include <faat_pcl/3d_rec_framework/utils/vtk_model_sampling.h>
#include <faat_pcl/utils/filesystem_utils.h>

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
    std::string dir, out_dir;

    pcl::console::parse_argument (argc, argv, "-out_dir", out_dir);
    pcl::console::parse_argument (argc, argv, "-dir", dir);


    bf::path input = dir;
    std::vector<std::string> model_files;
    std::string pattern_models = ".*.ply";
    std::string rel = "";
    faat_pcl::utils::getFilesInDirectoryRecursive(input, rel, model_files, pattern_models);

    std::cout << model_files.size() << std::endl;
    std::map<std::string, bool> banned_categories;
    //banned_categories.insert(std::make_pair("calculator", true));

    std::map<std::string, bool> banned_models;
    /*banned_models.insert(std::make_pair("axe/dd077df63be453da894a5aa0efb9dd19", true));
    banned_models.insert(std::make_pair("banjo/f94fe93a63234c0c54e499d8b00afd63", true));
    banned_models.insert(std::make_pair("book/b5a3e3cbdb735913e4edbf3acec8591c", true));
    banned_models.insert(std::make_pair("book/bf519a4ea7931f1cdca5ddc967bf6ffa", true));

    banned_models.insert(std::make_pair("bottle/8309e710832c07f91082f2ea630bf69e", true));
    banned_models.insert(std::make_pair("bottle/ketchup_bottle", true));
    banned_models.insert(std::make_pair("bowl/dbc35fcbbb90b5b4a7eee628cf5fc3f7", true));
    banned_models.insert(std::make_pair("clothes_hanger/bc5f3ab1f943a45a3c918bf63e516751", true));

    banned_models.insert(std::make_pair("fighter_jet/946f4fa804f3535a4d12d7a6e7b71cd8", true));
    banned_models.insert(std::make_pair("fighter_jet/959044f10e27b89ee664ce1de3ddc8b4", true));
    banned_models.insert(std::make_pair("flashlight/70cfc22ffe0e9835fc3223d52c1f21a9", true));
    banned_models.insert(std::make_pair("formula_car/9f5ca35eb8c78b134ecfe701ed997099", true));

    banned_models.insert(std::make_pair("guitar/16916a50a064304bf6ed0b697979412e", true));
    banned_models.insert(std::make_pair("guitar/8f4b1f242bc014b88fdda65f2c9bf85", true));
    banned_models.insert(std::make_pair("hammer/rubber_mallet", true));
    banned_models.insert(std::make_pair("keyboard/56794ac8b1257d0799fcd1563ba74ccd", true));

    banned_models.insert(std::make_pair("monster_truck/9c46c8eedc05da8f4de5de03a5e8f584", true));
    banned_models.insert(std::make_pair("mug/6a9b31e1298ca1109c515ccf0f61e75f", true));
    banned_models.insert(std::make_pair("padlock/77491c6e16cbe9a8ea9b15e3702bc79", true));
    banned_models.insert(std::make_pair("screwdriver/ff88df1f5ef7fdc43b9b2061fd503df7", true));

    banned_models.insert(std::make_pair("shoe/3", true));
    banned_models.insert(std::make_pair("shoe/N120311", true));
    banned_models.insert(std::make_pair("shoe/par", true));
    banned_models.insert(std::make_pair("shoe/Snickers", true));

    banned_models.insert(std::make_pair("stapler/586f90ea05f62bd36379618f01f885e3", true));
    banned_models.insert(std::make_pair("stapler/a5cafa3a913187dd8f9dd7647048a0c", true));
    banned_models.insert(std::make_pair("stapler/f39912a4f0516fb897371d1e7cc637f3", true));
    banned_models.insert(std::make_pair("toilet_paper/da6e2afe4689d66170acfa4e32a0a86", true));*/

    bf::path out = out_dir;
    if(!bf::exists(out_dir))
      bf::create_directory(out);

    float voxel_grid_size = 0.001f;
    int count = 0;
    for(size_t i=0; i < model_files.size(); i++)
    {
        std::vector<std::string> strs;
        boost::split (strs, model_files[i], boost::is_any_of ("/"));
        std::string category = strs[0];
        std::map<std::string, bool>::iterator it;
        it = banned_categories.find(category);
        if(it != banned_categories.end())
            continue;

        std::string ply_name = model_files[i];
        boost::replace_all (ply_name, ".ply", "");

        it = banned_models.find(ply_name);
        if(it != banned_models.end())
            continue;

        std::stringstream pathmodel;
        pathmodel << dir << "/" << model_files[i];
        std::string model_path = pathmodel.str ();

        pcl::PointCloud<pcl::PointXYZ>::Ptr model_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        faat_pcl::rec_3d_framework::uniform_sampling (model_path, 100000, *model_cloud, 1);

        boost::shared_ptr<faat_pcl::rec_3d_framework::ESFEstimation<pcl::PointXYZ, pcl::ESFSignature640> > estimator;
        estimator.reset (new faat_pcl::rec_3d_framework::ESFEstimation<pcl::PointXYZ, pcl::ESFSignature640>);


        PointCloud<pcl::PointXYZ>::Ptr vxgrided(new PointCloud<pcl::PointXYZ>);
        pcl::VoxelGrid<pcl::PointXYZ> grid_;
        grid_.setInputCloud (model_cloud);
        grid_.setLeafSize (voxel_grid_size, voxel_grid_size, voxel_grid_size);
        grid_.setDownsampleAllData (true);
        grid_.filter (*vxgrided);

        PointCloud<PointXYZ>::Ptr processed(new PointCloud<PointXYZ>);
        pcl::PointCloud<pcl::ESFSignature640>::CloudVectorType signatures;
        std::vector<Eigen::Vector3f> centroids;
        estimator->estimate (vxgrided, processed, signatures, centroids);

        //std::cout << signatures.size() << std::endl;
        boost::replace_all (ply_name, "/", "_aitor_");

        std::stringstream out_file;
        out_file << out_dir << "/" << ply_name << ".txt";

        for (size_t idx = 0; idx < signatures.size (); idx++)
        {
            float* hist = signatures[idx].points[0].histogram;
            int size_feat = sizeof(signatures[idx].points[0].histogram) / sizeof(float);
            writeHistogramToFile(out_file.str(), hist, size_feat);
        }

        std::cout << model_files[i] << " npoints:" << vxgrided->points.size() << " " << out_file.str() << std::endl;

        count++;
    }

    std::cout << "total:" << count << std::endl;
    return 0;

    /*std::string ply_name;
  std::vector<std::string> strs;
  boost::split (strs, ply_file, boost::is_any_of ("/"));
  ply_name = strs[strs.size() - 1];
  boost::replace_all (ply_name, ".ply", "");4

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
  }*/
}

