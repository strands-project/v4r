/**
 *  Copyright (C) 2012  
 *    Andreas Richtsfeld, Johann Prankl, Thomas Mörwald
 *    Automation and Control Institute
 *    Vienna University of Technology
 *    Gusshausstraße 25-29
 *    1170 Vienn, Austria
 *    ari(at)acin.tuwien.ac.at
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see http://www.gnu.org/licenses/
 */

/**
 * @file SegmenterLight.cpp
 * @author Andreas Richtsfeld
 * @date January 2012
 * @version 0.1
 * @brief Segment images efficiently
 */

#include "SegmenterLight.h"

#include <stdio.h>
#include <pcl/io/pcd_io.h>

namespace segment
{
  
  /* --------------- SegmenterLight --------------- */

  SegmenterLight::SegmenterLight (std::string _model_path)
  {
    model_path = _model_path;
    useStructuralLevel = true;
    detail = 2;
  }

  SegmenterLight::~SegmenterLight ()
  {
  }

  void
  SegmenterLight::computeNormals (pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud_in,
                             pcl::PointCloud<pcl::Normal>::Ptr &normals_out)
  {
    normals_out.reset (new pcl::PointCloud<pcl::Normal>);
    surface::ZAdaptiveNormals<pcl::PointXYZRGB>::Parameter za_param;
    za_param.adaptive = true;
    surface::ZAdaptiveNormals<pcl::PointXYZRGB> nor (za_param);
    nor.setInputCloud (cloud_in);
    nor.compute ();
    nor.getNormals (normals_out);
  }

  void
  SegmenterLight::computePlanes (pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud_in,
                            pcl::PointCloud<pcl::Normal>::Ptr &normals_in,
                            std::vector<surface::SurfaceModel::Ptr> &surfaces_out)
  {
    surface::View view;
    view.normals = normals_in;
    surface::ClusterNormalsToPlanes::Parameter param;
    param.adaptive = true;
    if(detail == 1) {
      param.epsilon_c = 0.58;
      param.omega_c = -0.002;
    } else if (detail == 2) {
      param.epsilon_c = 0.62;
      param.omega_c = 0.0;
    } 
    surface::ClusterNormalsToPlanes clusterNormals (param);
    clusterNormals.setInputCloud (cloud_in);
    clusterNormals.setView (&view);
    clusterNormals.setPixelCheck (true, 5);
    clusterNormals.compute ();
    surfaces_out = view.surfaces;
  }

  void
  SegmenterLight::computeSurfaces (pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud_in,
                              std::vector<surface::SurfaceModel::Ptr> &surfaces_in_out)
  {
    surface::View view;
    view.surfaces = surfaces_in_out;
    pcl::on_nurbs::SequentialFitter::Parameter nurbsParams;
    nurbsParams.order = 3;
    nurbsParams.refinement = 0;
    nurbsParams.iterationsQuad = 0;
    nurbsParams.iterationsBoundary = 0;
    nurbsParams.iterationsAdjust = 0;
    nurbsParams.iterationsInterior = 3;
    nurbsParams.forceBoundary = 100.0;
    nurbsParams.forceBoundaryInside = 300.0;
    nurbsParams.forceInterior = 1.0;
    nurbsParams.stiffnessBoundary = 0.1;
    nurbsParams.stiffnessInterior = 0.1;
    nurbsParams.resolution = 16;
    surface::SurfaceModeling::Parameter sfmParams;
    sfmParams.nurbsParams = nurbsParams;
    sfmParams.sigmaError = 0.003;
    sfmParams.kappa1 = 0.008;
    sfmParams.kappa2 = 1.0;
    sfmParams.planePointsFixation = 8000;
    sfmParams.z_max = 0.01;
    surface::SurfaceModeling surfModeling (sfmParams);
    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity ();
    surfModeling.setIntrinsic (525., 525., 320., 240.);
    surfModeling.setExtrinsic (pose);
    surfModeling.setInputCloud (cloud_in);
    surfModeling.setView (&view);
    surfModeling.compute ();
    surfaces_in_out = view.surfaces;
  }

  void
  SegmenterLight::computeObjects (pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud_in,
                             std::vector<surface::SurfaceModel::Ptr> &surfaces_in_out,
                             pcl::PointCloud<pcl::PointXYZRGBL>::Ptr &cloud_out)
  {
    // contour detector
    surface::View view;
    view.width = cloud_in->width;
    view.height = cloud_in->height;
    view.surfaces = surfaces_in_out;
    
    surface::ContourDetector contourDet;
    contourDet.setInputCloud(cloud_in);
    contourDet.setView(&view);
    contourDet.computeContours();
    
    surface::StructuralRelationsLight stRel;
    stRel.setInputCloud(cloud_in);
    stRel.setView(&view);
    stRel.computeRelations();

    // init model path
    std::string svmStructuralModel;
    std::string svmStructuralScaling;
    svmStructuralModel = "PP-Trainingsset.txt.scaled.model";
    svmStructuralScaling = "param.txt";
  
    svm::SVMPredictorSingle svm_structural(model_path, svmStructuralModel);
    svm_structural.setScaling(true, model_path, svmStructuralScaling);
    svm_structural.classify(&view, 1);
    
    gc::GraphCut graphCut;
    #ifdef DEBUG
      graphCut.printResults(true);
    #endif
    if(graphCut.init(&view))
      graphCut.process();
    
     for (unsigned i = 0; i < view.graphCutGroups.size (); i++)
       for (unsigned j = 0; j < view.graphCutGroups[i].size (); j++)
         surfaces_in_out[view.graphCutGroups[i][j]]->label = i;

     for (unsigned i = 0; i < surfaces_in_out.size (); i++) {
       surface::SurfaceModel::Ptr s = surfaces_in_out[i];
       for (unsigned j = 0; j < s->indices.size (); j++)
         cloud_out->at (s->indices[j]).label = s->label;
     }
  }

  pcl::PointCloud<pcl::PointXYZRGBL>::Ptr
  SegmenterLight::processPointCloud (pcl::PointCloud<pcl::PointXYZRGB>::Ptr &pcl_cloud)
  {
    surface::View view;
    view.width = pcl_cloud->width;
    view.height = pcl_cloud->height;

    pcl::PointCloud<pcl::PointXYZRGBL>::Ptr result (new pcl::PointCloud<pcl::PointXYZRGBL>);
    pcl::copyPointCloud (*pcl_cloud, *result);

    // calcuate normals
    view.normals.reset (new pcl::PointCloud<pcl::Normal>);
    surface::ZAdaptiveNormals<pcl::PointXYZRGB>::Parameter za_param;
    za_param.adaptive = true;
    surface::ZAdaptiveNormals<pcl::PointXYZRGB> nor (za_param);
    nor.setInputCloud (pcl_cloud);
    nor.compute ();
    nor.getNormals (view.normals);

    // adaptive clustering
    surface::ClusterNormalsToPlanes::Parameter param;
    param.adaptive = true;
    if(detail == 1) {
      param.epsilon_c = 0.58;
      param.omega_c = -0.002;
    } else if (detail == 2) {
      param.epsilon_c = 0.62;
      param.omega_c = 0.0;
    } 
    surface::ClusterNormalsToPlanes clusterNormals (param);
    clusterNormals.setInputCloud (pcl_cloud);
    clusterNormals.setView (&view);
    clusterNormals.setPixelCheck (true, 5);
    clusterNormals.compute ();

    // model abstraction
    if(!fast) {
      pcl::on_nurbs::SequentialFitter::Parameter nurbsParams;
      nurbsParams.order = 3;
      nurbsParams.refinement = 0;
      nurbsParams.iterationsQuad = 0;
      nurbsParams.iterationsBoundary = 0;
      nurbsParams.iterationsAdjust = 0;
      nurbsParams.iterationsInterior = 3;
      nurbsParams.forceBoundary = 100.0;
      nurbsParams.forceBoundaryInside = 300.0;
      nurbsParams.forceInterior = 1.0;
      nurbsParams.stiffnessBoundary = 0.1;
      nurbsParams.stiffnessInterior = 0.1;
      nurbsParams.resolution = 16;
      surface::SurfaceModeling::Parameter sfmParams;
      sfmParams.nurbsParams = nurbsParams;
      sfmParams.sigmaError = 0.003;
      sfmParams.kappa1 = 0.008;
      sfmParams.kappa2 = 1.0;
      sfmParams.planePointsFixation = 8000;
      sfmParams.z_max = 0.01;
      surface::SurfaceModeling surfModeling (sfmParams);
      Eigen::Matrix4d pose = Eigen::Matrix4d::Identity ();
      surfModeling.setIntrinsic (525., 525., 320., 240.);
      surfModeling.setExtrinsic (pose);
      surfModeling.setInputCloud (pcl_cloud);
      surfModeling.setView (&view);
      surfModeling.compute ();
    }

    // contour detector
    surface::ContourDetector contourDet;
    contourDet.setInputCloud(pcl_cloud);
    contourDet.setView(&view);
    contourDet.computeContours();
    
    // relations
    surface::StructuralRelationsLight stRel;
    stRel.setInputCloud(pcl_cloud);
    stRel.setView(&view);
    stRel.computeRelations();
 
    // init svm model
    std::string svmStructuralModel;
    std::string svmStructuralScaling;
    if(!fast) {
      svmStructuralModel = "PP-Trainingsset.txt.scaled.model";
      svmStructuralScaling = "param.txt";
    } else {
      svmStructuralModel = "PP-Trainingsset.txt.scaled.model.fast";
      svmStructuralScaling = "param.txt.fast";
    }

    // svm classification
    svm::SVMPredictorSingle svm_structural(model_path, svmStructuralModel);
    svm_structural.setScaling(true, model_path, svmStructuralScaling);
    svm_structural.classify(&view, 1);
    
    // graph cut
    gc::GraphCut graphCut;
    #ifdef DEBUG
      graphCut.printResults(true);
    #endif
    if(graphCut.init(&view))
      graphCut.process();

    // copy results
    for (unsigned i = 0; i < view.graphCutGroups.size (); i++)
      for (unsigned j = 0; j < view.graphCutGroups[i].size (); j++)
        view.surfaces[view.graphCutGroups[i][j]]->label = i;
    for (unsigned i = 0; i < view.surfaces.size (); i++) {
      for (unsigned j = 0; j < view.surfaces[i]->indices.size (); j++) {
        result->points[view.surfaces[i]->indices[j]].label = view.surfaces[i]->label;
      }
    }

    return result;
  }

  
  std::vector<pcl::PointIndices>
  SegmenterLight::processPointCloudV (pcl::PointCloud<pcl::PointXYZRGB>::Ptr &pcl_cloud)
  {
    surface::View view;
    view.width = pcl_cloud->width;
    view.height = pcl_cloud->height;
    
    pcl::PointCloud<pcl::PointXYZRGBL>::Ptr result (new pcl::PointCloud<pcl::PointXYZRGBL>);
    pcl::copyPointCloud (*pcl_cloud, *result);

    // calcuate normals
    view.normals.reset (new pcl::PointCloud<pcl::Normal>);
    surface::ZAdaptiveNormals<pcl::PointXYZRGB>::Parameter za_param;
    za_param.adaptive = true;
    surface::ZAdaptiveNormals<pcl::PointXYZRGB> nor (za_param);
    nor.setInputCloud (pcl_cloud);
    nor.compute ();
    nor.getNormals (view.normals);

    // adaptive clustering
    surface::ClusterNormalsToPlanes::Parameter param;
    param.adaptive = true;
    if(detail == 1) {
      param.epsilon_c = 0.58;
      param.omega_c = -0.002;
    } else if (detail == 2) {
      param.epsilon_c = 0.62;
      param.omega_c = 0.0;
    } 
    surface::ClusterNormalsToPlanes clusterNormals (param);
    clusterNormals.setInputCloud (pcl_cloud);
    clusterNormals.setView (&view);
    clusterNormals.setPixelCheck (true, 5);
    clusterNormals.compute ();

    // model abstraction
    if(!fast) {
      pcl::on_nurbs::SequentialFitter::Parameter nurbsParams;
      nurbsParams.order = 3;
      nurbsParams.refinement = 0;
      nurbsParams.iterationsQuad = 0;
      nurbsParams.iterationsBoundary = 0;
      nurbsParams.iterationsAdjust = 0;
      nurbsParams.iterationsInterior = 3;
      nurbsParams.forceBoundary = 100.0;
      nurbsParams.forceBoundaryInside = 300.0;
      nurbsParams.forceInterior = 1.0;
      nurbsParams.stiffnessBoundary = 0.1;
      nurbsParams.stiffnessInterior = 0.1;
      nurbsParams.resolution = 16;
      surface::SurfaceModeling::Parameter sfmParams;
      sfmParams.nurbsParams = nurbsParams;
      sfmParams.sigmaError = 0.003;
      sfmParams.kappa1 = 0.008;
      sfmParams.kappa2 = 1.0;
      sfmParams.planePointsFixation = 8000;
      sfmParams.z_max = 0.01;
      surface::SurfaceModeling surfModeling (sfmParams);
      Eigen::Matrix4d pose = Eigen::Matrix4d::Identity ();
      surfModeling.setIntrinsic (525., 525., 320., 240.);
      surfModeling.setExtrinsic (pose);
      surfModeling.setInputCloud (pcl_cloud);
      surfModeling.setView (&view);
      surfModeling.compute ();
    }

    // contour detector
    surface::ContourDetector contourDet;
    contourDet.setInputCloud(pcl_cloud);
    contourDet.setView(&view);
    contourDet.computeContours();
    
    // relations
    surface::StructuralRelationsLight stRel;
    stRel.setInputCloud(pcl_cloud);
    stRel.setView(&view);
    stRel.computeRelations();
 
    // init svm model
    std::string svmStructuralModel;
    std::string svmStructuralScaling;
    if(!fast) {
      svmStructuralModel = "PP-Trainingsset.txt.scaled.model";
      svmStructuralScaling = "param.txt";
    } else {
      svmStructuralModel = "PP-Trainingsset.txt.scaled.model.fast";
      svmStructuralScaling = "param.txt.fast";
    }
    
    // svm classification
    svm::SVMPredictorSingle svm_structural(model_path, svmStructuralModel);
    svm_structural.setScaling(true, model_path, svmStructuralScaling);
    svm_structural.classify(&view, 1);
    
    // graph cut
    gc::GraphCut graphCut;
    #ifdef DEBUG
      graphCut.printResults(true);
    #endif
    if(graphCut.init(&view))
      graphCut.process();

    std::vector<pcl::PointIndices> results;
    results.resize (view.graphCutGroups.size ());
    for (unsigned i = 0; i < view.graphCutGroups.size (); i++)
      for (unsigned j = 0; j < view.graphCutGroups[i].size (); j++)
        for (unsigned k = 0; k < view.surfaces[view.graphCutGroups[i][j]]->indices.size (); k++)
          results[i].indices.push_back(view.surfaces[view.graphCutGroups[i][j]]->indices[k]);
    return results;
  }

} // end segment
