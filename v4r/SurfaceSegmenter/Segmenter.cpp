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
 * @file Segmenter.cpp
 * @author Andreas Richtsfeld
 * @date July 2012
 * @version 0.1
 * @brief Segment images
 */

#include "Segmenter.h"

#include <stdio.h>
#include <iostream>
#include <pcl/io/pcd_io.h>

#include <v4r/utils/timehdl.h>
#include "v4r/SurfaceUtils/KinectData.h"
#include "v4r/SurfaceClustering/ClusterNormalsToPlanes.hh"
#include "v4r/SurfaceClustering/ZAdaptiveNormals.hh"
#include "v4r/SurfaceModeling/SurfaceModeling.hh"
#include "v4r/SurfaceRelations/PatchRelations.h"
#include "v4r/svm/SVMPredictorSingle.h"
#include "v4r/GraphCut/GraphCut.h"

namespace segment
{

  /* --------------- Segmenter --------------- */

  Segmenter::Segmenter (std::string _model)
  {
    model_path = _model;
    useStructuralLevel = true;
    useAssemblyLevel = false;
    detail = 0;
  }

  Segmenter::~Segmenter ()
  {}

  void Segmenter::setInputCloud(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &_cloud)
  {
    if (_cloud->height<=1 || _cloud->width<=1 || !_cloud->isOrganized())
      throw std::runtime_error("[Segmenter::setInputCloud] Invalid point cloud (height must be > 1)");

    cloud = _cloud;
  }

  void Segmenter::computeNormals ()
  {
    view.normals.reset (new pcl::PointCloud<pcl::Normal>);
    surface::ZAdaptiveNormals<pcl::PointXYZRGB>::Parameter za_param;
    za_param.adaptive = true;
    surface::ZAdaptiveNormals<pcl::PointXYZRGB> nor (za_param);
    nor.setInputCloud (cloud);
    nor.compute ();
    nor.getNormals (view.normals);
  }

  void Segmenter::clusterNormals ()
  {
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
    clusterNormals.setInputCloud (cloud);
    clusterNormals.setView (&view);
    clusterNormals.setPixelCheck (true, 5);
    clusterNormals.compute ();
  }

  void Segmenter::computeSurfaces ()
  {
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
    surfModeling.setInputCloud (cloud);
    surfModeling.setView (&view);
    surfModeling.compute ();
  }

  void Segmenter::computeContours ()
  {
    surface::ContourDetector contourDet;
    contourDet.setInputCloud(cloud);
    contourDet.setView(&view);
    contourDet.computeContours();
  }

  void Segmenter::computeStructuralRelations ()
  {
    surface::StructuralRelations stRel;
    stRel.setInputCloud(cloud);
    stRel.setView(&view);
    stRel.computeRelations();
  }

  void Segmenter::computeAssemblyRelations ()
  {
    if(useAssemblyLevel) {
      surface::AssemblyRelations asRel;
      asRel.setInputCloud(cloud);
      asRel.setView(&view);
      asRel.computeRelations();
    }
  }

  void Segmenter::classifyRelations ()
  {
    // init svm model
    std::string svmStructuralModel, svmAssemblyModel;
    std::string svmStructuralScaling, svmAssemblyScaling;
    svmStructuralModel = "PP-Trainingsset.txt.scaled.model";
    svmAssemblyModel = "PP2-Trainingsset.txt.scaled.model";
    svmStructuralScaling = "param.txt";
    svmAssemblyScaling = "param2.txt";

    // svm classification
    svm::SVMPredictorSingle svm_structural(model_path, svmStructuralModel);
    svm_structural.setScaling(true, model_path, svmStructuralScaling);
    svm_structural.classify(&view, 1);

    if(useAssemblyLevel) {
      svm::SVMPredictorSingle svm_assembly(model_path, svmAssemblyModel);
      svm_assembly.setScaling(true, model_path, svmAssemblyScaling);
      svm_assembly.classify(&view, 2);
    }
  }

  void Segmenter::computeGraphCut ()
  {
    // graph cut
    gc::GraphCut graphCut;
    #ifdef DEBUG
      graphCut.printResults(true);
    #endif
    if(graphCut.init(&view))
      graphCut.process();
    for (unsigned i = 0; i < view.graphCutGroups.size (); i++)
       for (unsigned j = 0; j < view.graphCutGroups[i].size (); j++)
         view.surfaces[view.graphCutGroups[i][j]]->label = i;
  }

  void Segmenter::compute ()
  {

    view.Reset();
    view.width = cloud->width;
    view.height = cloud->height;
    
    V4R::ThreadTimer timer;
    timer.start("0-processPointCloudV");

    timer.start("2-calc-normals");
    computeNormals();
    timer.stop("2-calc-normals");

    timer.start("3-cluster-normals");
    clusterNormals();
    timer.stop("3-cluster-normals");

    timer.start("4-calc-surfaces");
    computeSurfaces();
    timer.stop("4-calc-surfaces");

    timer.start("5-comp-contours");
    computeContours();
    timer.stop("5-comp-contours");
    
    timer.start("61-struc-relations");
    computeStructuralRelations();
    timer.stop("61-struc-relations");

    timer.start("62-ass-relations");
    computeAssemblyRelations();
    timer.stop("62-ass-relations");
 
    timer.start("7-classify-relations");
    classifyRelations();
    timer.stop("7-classify-relations");
    
    timer.start("8-graph-cut");
    computeGraphCut();
    timer.stop("8-graph-cut");

    timer.stop("0-processPointCloudV");
    std::cout << "Segmenter::processPointCloudV: timings\n" << timer.summary() << "\n";
  }

  pcl::PointCloud<pcl::PointXYZRGBL>::Ptr Segmenter::getLabels ()
  {
    pcl::PointCloud<pcl::PointXYZRGBL>::Ptr result (new pcl::PointCloud<pcl::PointXYZRGBL>);
    pcl::copyPointCloud (*cloud, *result);

    // copy results
    for (unsigned i = 0; i < view.surfaces.size (); i++) {
      for (unsigned j = 0; j < view.surfaces[i]->indices.size (); j++) {
        result->points[view.surfaces[i]->indices[j]].label = view.surfaces[i]->label;
      }
    }
    return result;
  }

  std::vector<pcl::PointIndices> Segmenter::getSegments ()
  {
    std::vector<pcl::PointIndices> results;
    results.resize (view.graphCutGroups.size ());
    for (unsigned i = 0; i < view.graphCutGroups.size (); i++)
      for (unsigned j = 0; j < view.graphCutGroups[i].size (); j++)
        for (unsigned k = 0; k < view.surfaces[view.graphCutGroups[i][j]]->indices.size (); k++)
          results[i].indices.push_back(view.surfaces[view.graphCutGroups[i][j]]->indices[k]);
    return results;
  }

  std::vector<surface::SurfaceModel::Ptr> Segmenter::getSurfaces ()
  {
    return view.surfaces;

    /* NOTE: It does not really matter in which order we return the surfaces:
     * a) in packages according to label (as in the commented out code below)
     * b) as they are in the view
     * Option (b) however has the advantage that the indices in a surface's
     * neighbor list are still valid, i.e. point to the correct surface in the
     * returned list. This would not be the case for (a).
    std::vector<surface::SurfaceModel::Ptr> results;
    for (unsigned i = 0; i < view.graphCutGroups.size (); i++)
      for (unsigned j = 0; j < view.graphCutGroups[i].size (); j++)
        results.push_back(view.surfaces[view.graphCutGroups[i][j]]);
    return results;*/
  }

} // end segment
