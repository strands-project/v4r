/**
 *  Copyright (C) 2014  
 *    Ekaterina Potapova
 *    Automation and Control Institute
 *    Vienna University of Technology
 *    Gusshausstraße 25-29
 *    1170 Vienna, Austria
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
 * @file segmentation.cpp
 * @author Ekaterina Potapova
 * @date January 2014
 * @version 0.1
 * @brief segmentation.cpp
 */


#include "segmentation.hpp"

namespace segmentation
{

Segmenter::Segmenter()
{
  have_cloud = false;
//   have_normals = false;
  have_saliencyMaps = false;
  
  ClassName = "Segmenter";
}

Segmenter::~Segmenter()
{
}

void Segmenter::setPointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr _pcl_cloud)
{
  pcl_cloud = _pcl_cloud;
  have_cloud = true;
}

// void Segmenter::setNormals(pcl::PointCloud<pcl::Normal>::Ptr _normals)
// {
//   normals = _normals;
//   have_normals = true;
// }

void Segmenter::setSaliencyMaps(std::vector<cv::Mat> _saliencyMaps)
{
  saliencyMaps = _saliencyMaps;
  have_saliencyMaps = true;
}

void Segmenter::calculateNormals()
{
  // calcuate normals
  normals.reset(new pcl::PointCloud<pcl::Normal>);
  surface::ZAdaptiveNormals<pcl::PointXYZRGB>::Parameter param;
  param.adaptive = true;
  surface::ZAdaptiveNormals<pcl::PointXYZRGB> nor(param);
  nor.setInputCloud(pcl_cloud);
  nor.compute();
  nor.getNormals(normals);
}

void Segmenter::calculatePatches()
{
  surface::ClusterNormalsToPlanes::Parameter param;
  param.adaptive = true;         // use adaptive thresholds
  param.epsilon_c = 0.58;//0.62;//
  param.omega_c = -0.002;

  clusterNormals = surface::ClusterNormalsToPlanes::Ptr(new surface::ClusterNormalsToPlanes(param));
  // adaptive clustering
  clusterNormals->setInputCloud(pcl_cloud);
  clusterNormals->setNormals(normals);
  clusterNormals->setPixelCheck(true, 5);
  clusterNormals->compute();
  normals = clusterNormals->getNormals();
  surfaces = clusterNormals->getSurfaces();
}

void Segmenter::initModelSurfaces()
{
  // init nurbsfitting & model-selection
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
  sfmParams.planePointsFixation = 8000;       // 8000
  sfmParams.z_max = 0.01;
  surfModeling = surface::SurfaceModeling::Ptr(new surface::SurfaceModeling(sfmParams));
  surfModeling->setIntrinsic(570.3, 570.3, 320., 240.);
//   surfModeling->setIntrinsic(525., 525., 320., 240.);
  Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
  surfModeling->setExtrinsic(pose);
  
  surfModeling->setInputCloud(pcl_cloud);
//   surfModeling->setNormals(normals);
  
  surfModeling->setMinSurfaceSize(50);
  
  surfModeling->setTryMergePlanes(false);
  surfModeling->setTryMergeNurbs(true);
}

void Segmenter::modelSurfaces()
{
  surfModeling->setBoundary2D(ngbr2D_map);
  surfModeling->setBoundary3D(ngbr3D_map);

  surfModeling->setNormals(normals);
  surfModeling->setSurfaces(surfaces);
  surfModeling->compute();
  normals = surfModeling->getNormals();
  surfaces = surfModeling->getSurfaces();
  ngbr3D_map = surfModeling->getBoundary3D();
  ngbr2D_map = surfModeling->getBoundary2D();
}

void Segmenter::preComputeRelations()
{ 
  structuralRelations.setInputCloud(pcl_cloud);
  structuralRelations.setNormals(normals);
  structuralRelations.setSurfaces(surfaces);
  //   structuralRelations.setRelations(relations);
  structuralRelations.setNeighbours2D(ngbr2D_map);
  structuralRelations.setNeighbours3D(ngbr3D_map);
  //   structuralRelations.setUsedRelations(0x0001);
  structuralRelations.setTrainingMode(false);
  structuralRelations.init();
}

void Segmenter::computeRelations()
{
  structuralRelations.setSurfaces(surfaces);
  //   structuralRelations.setRelations(relations);
  structuralRelations.setNeighbours2D(ngbr2D_map);
  structuralRelations.setNeighbours3D(ngbr3D_map);
  structuralRelations.compute();
  validRelations = structuralRelations.getValidRelations();
  
  //   assemblyRelations.setInputCloud(pcl_cloud);
  //   assemblyRelations.setNormals(normals);
  //   assemblyRelations.setSurfaces(surfaces);
  //   assemblyRelations.setRelations(relations);
  //   assemblyRelations.setNeighbours2D(ngbr2D);
  //   assemblyRelations.setNeighbours3D(ngbr3D);
  //   assemblyRelations.compute();
  //
  //   relations = assemblyRelations.getRelations();
  //   view.relations = relations;
}

void Segmenter::graphBasedSegmentation()
{
  svmPredictorSingle.setPredictProbability(true);
  svmPredictorSingle.setModelFilename("./ST-TrainAll.model.txt");
  svmPredictorSingle.setSurfaces(surfaces);
  svmPredictorSingle.setRelations(validRelations);
  svmPredictorSingle.setType(1);
  svmPredictorSingle.setScaling(true,"./ST-TrainAll.scalingparams.txt");
  svmPredictorSingle.compute();
  validRelations = svmPredictorSingle.getRelations();
  
  //   svmPredictorSingle.setPredictProbability(true);
  //   svmPredictorSingle.setModelFilename("./AS-TrainALL.model.txt");
  //   svmPredictorSingle.setSurfaces(surfaces);
  //   svmPredictorSingle.setRelations(relations);
  //   svmPredictorSingle.setType(2);
  //   svmPredictorSingle.setScaling(true,"./AS-TrainALL.scalingparams.txt");
  //   svmPredictorSingle.compute();
  //   relations = svmPredictorSingle.getRelations();
  //   view.relations = relations;
  
  graphCut.setSurfaces(surfaces);
  graphCut.setRelations(validRelations);
//   graphCut.printResults(true);
//   if(graphCut.init())
//   {
//     printf("graph cut initialized\n");
//     graphCut.process();
//     surfaces = graphCut.getSurfaces();
//   }
//   else
//   {
//     printf("cant init graph cut\n");
//   }
  graphCut.process2();
  surfaces = graphCut.getSurfaces();
}

// check changes in segmentation
bool Segmenter::checkSegmentation(cv::Mat &mask, int originalIndex, int salMapNumber)
{ 
  int label = surfaces.at(originalIndex)->label;
  cv::Mat new_mask = cv::Mat_<uchar>::zeros(mask.size());
  for(int i = 0; i < surfaces.size(); i++)
  {
    if(surfaces.at(i)->label == label)
    {
      for(int j = 0; j < surfaces.at(i)->indices.size(); j++)
      {
        int idx = surfaces.at(i)->indices.at(j);
        int row = idx / pcl_cloud->width;
        int col = idx % pcl_cloud->width;
        new_mask.at<uchar>(row,col) = 1;
      }
    }
  }
    
  assert(new_mask.size() == mask.size());
    
  bool different = false;
  for(int i = 0; i < new_mask.rows; ++i)
  {
    for(int j = 0; j < new_mask.cols; ++j)
    {
      if(new_mask.at<uchar>(i,j) != mask.at<uchar>(i,j))
      {
        different = true;
        break;
      }
    }
    if(different)
      break;
  }
    
  if(different)
  {
    new_mask.copyTo(mask);
    return false;
  }
  else
  {
    for(int i = 0; i < surfaces.size(); i++)
    {
      if(surfaces.at(i)->label == label)
      {
        surfaces.at(i)->valid = false;
        surfaces.at(i)->segmented_number = salMapNumber;
      }
    }
    return true;
  }
    
}

void Segmenter::attentionSegment()
{
  if( (!have_cloud) || (!have_saliencyMaps) )
  {
    char* error_message = new char[200];
    sprintf(error_message,"[%s::segment()]: I suggest you first set the point cloud.",ClassName.c_str()); // and normals
    throw std::runtime_error(error_message);
  }
  
  masks.resize(saliencyMaps.size());

EPUtils::TimeEstimationClass timeEstimationClass_All(CLOCK_THREAD_CPUTIME_ID);//CLOCK_THREAD_CPUTIME_ID);//CLOCK_PROCESS_CPUTIME_ID
EPUtils::TimeEstimationClass timeEstimationClass_Custom(CLOCK_THREAD_CPUTIME_ID);

timeEstimationClass_All.countingStart();

timeEstimationClass_Custom.countingStart();
  calculateNormals();
timeEstimationClass_Custom.countingEnd();
timeEstimates.time_normalsCalculation = timeEstimationClass_Custom.getWorkTimeInNanoseconds();

timeEstimationClass_Custom.countingStart();
  calculatePatches();
timeEstimationClass_Custom.countingEnd();
timeEstimates.time_patchesCalculation = timeEstimationClass_Custom.getWorkTimeInNanoseconds();

timeEstimationClass_Custom.countingStart();
  surface::View view;
  view.Reset();
  view.setPointCloud(pcl_cloud);
  view.normals = normals;
  view.setSurfaces(surfaces);
  surfaces = view.surfaces;
  
  view.createPatchImage();
timeEstimationClass_Custom.countingEnd();
timeEstimates.time_patchImageCalculation = timeEstimationClass_Custom.getWorkTimeInNanoseconds();

timeEstimationClass_Custom.countingStart();  
  view.computeNeighbors();
timeEstimationClass_Custom.countingEnd();
timeEstimates.time_neighborsCalculation = timeEstimationClass_Custom.getWorkTimeInNanoseconds();

timeEstimationClass_Custom.countingStart();
  surfaces = view.surfaces;
  view.calculateBorders(view.cloud);
  ngbr3D_map = view.ngbr3D_map;
  ngbr2D_map = view.ngbr2D_map;
timeEstimationClass_Custom.countingEnd();
timeEstimates.time_borderCalculation = timeEstimationClass_Custom.getWorkTimeInNanoseconds();

timeEstimationClass_Custom.countingStart();
  preComputeRelations();
timeEstimationClass_Custom.countingEnd();
timeEstimates.time_relationsPreComputation = timeEstimationClass_Custom.getWorkTimeInNanoseconds();

timeEstimationClass_Custom.countingStart();
  initModelSurfaces();
timeEstimationClass_Custom.countingEnd();
timeEstimates.time_initModelSurfaces = timeEstimationClass_Custom.getWorkTimeInNanoseconds();

timeEstimates.times_saliencySorting.resize(saliencyMaps.size());
timeEstimates.times_surfaceModelling.resize(saliencyMaps.size());
timeEstimates.times_relationsComputation.resize(saliencyMaps.size());
timeEstimates.times_graphBasedSegmentation.resize(saliencyMaps.size());
timeEstimates.times_maskCreation.resize(saliencyMaps.size());
timeEstimates.times_neigboursUpdate.resize(saliencyMaps.size());
timeEstimates.time_totalPerSegment.resize(saliencyMaps.size());

  for(int i = 0; i < saliencyMaps.size(); ++i)
  {

EPUtils::TimeEstimationClass timeEstimationClass_CustomLoopSegment(CLOCK_THREAD_CPUTIME_ID);
timeEstimationClass_CustomLoopSegment.countingStart();
    
EPUtils::TimeEstimationClass timeEstimationClass_CustomLoop(CLOCK_THREAD_CPUTIME_ID);
timeEstimationClass_CustomLoop.countingStart();
    view.setSaliencyMap(saliencyMaps.at(i));
//     cv::imshow("saliencyMaps.at(i)",saliencyMaps.at(i));
//     cv::waitKey(-1);
    view.sortPatches();
    surfaces = view.surfaces;
    
    for(int i = 0; i < surfaces.size(); i++)
    {
      surfaces.at(i)->selected = false;
      surfaces.at(i)->isNew = false;
    }
  
    int originalIndex = view.sortedSurfaces.at(0);

timeEstimationClass_CustomLoop.countingEnd();
timeEstimates.times_saliencySorting.at(i) = timeEstimationClass_CustomLoop.getWorkTimeInNanoseconds();

    if(surfaces.at(originalIndex)->segmented_number != -1)
    {

timeEstimates.times_surfaceModelling.at(i) = 0;
timeEstimates.times_relationsComputation.at(i) = 0;
timeEstimates.times_graphBasedSegmentation.at(i) = 0;
timeEstimates.times_maskCreation.at(i) = 0;
timeEstimates.times_neigboursUpdate.at(i) = 0;

      masks.at(surfaces.at(originalIndex)->segmented_number).copyTo(masks.at(i));
      //continue;
    }
    else
    {

    surfaces.at(originalIndex)->selected = true;
    surfaces.at(originalIndex)->isNew = true;
    view.surfaces = surfaces;

    cv::Mat object_mask = cv::Mat_<uchar>::zeros(480,640);
    originalIndex = attentionSegment(object_mask, originalIndex, i);
    object_mask.copyTo(masks.at(i));
    view.surfaces = surfaces;
    }

timeEstimationClass_CustomLoopSegment.countingEnd();
timeEstimates.time_totalPerSegment.at(i) = timeEstimationClass_CustomLoopSegment.getWorkTimeInNanoseconds();
    
  }

timeEstimationClass_All.countingEnd();
timeEstimates.time_total = timeEstimationClass_All.getWorkTimeInNanoseconds();
  
}

int Segmenter::attentionSegment(cv::Mat &object_mask, int originalIndex, int salMapNumber)
{

EPUtils::TimeEstimationClass timeEstimationClass_CustomLocal(CLOCK_THREAD_CPUTIME_ID);
timeEstimates.times_surfaceModelling.at(salMapNumber) = 0;
timeEstimates.times_relationsComputation.at(salMapNumber) = 0;
timeEstimates.times_graphBasedSegmentation.at(salMapNumber) = 0;
timeEstimates.times_maskCreation.at(salMapNumber) = 0;
timeEstimates.times_neigboursUpdate.at(salMapNumber) = 0;
  
  while(true)
  {
timeEstimationClass_CustomLocal.countingStart();
    modelSurfaces();

    //update original index
    std::vector<int> addedTo = surfModeling->getAddedTo();
    if(addedTo.at(originalIndex) >= 0)
    {
      originalIndex = addedTo.at(originalIndex);
      while(addedTo.at(originalIndex) != -1)
      {
        originalIndex = addedTo.at(originalIndex);
      }
    }
timeEstimationClass_CustomLocal.countingEnd();
timeEstimates.times_surfaceModelling.at(salMapNumber) += timeEstimationClass_CustomLocal.getWorkTimeInNanoseconds();

timeEstimationClass_CustomLocal.countingStart();
    computeRelations();
timeEstimationClass_CustomLocal.countingEnd();
timeEstimates.times_relationsComputation.at(salMapNumber) += timeEstimationClass_CustomLocal.getWorkTimeInNanoseconds();

timeEstimationClass_CustomLocal.countingStart();
    graphBasedSegmentation();
timeEstimationClass_CustomLocal.countingEnd();
timeEstimates.times_graphBasedSegmentation.at(salMapNumber) += timeEstimationClass_CustomLocal.getWorkTimeInNanoseconds();

//     printf("Before check object!");
timeEstimationClass_CustomLocal.countingStart();
    if(checkSegmentation(object_mask,originalIndex,salMapNumber))
    {
      printf("Object was segmented!\n");
timeEstimationClass_CustomLocal.countingEnd();
timeEstimates.times_maskCreation.at(salMapNumber) += timeEstimationClass_CustomLocal.getWorkTimeInNanoseconds();
      return(originalIndex);
    }
timeEstimationClass_CustomLocal.countingEnd();
timeEstimates.times_maskCreation.at(salMapNumber) += timeEstimationClass_CustomLocal.getWorkTimeInNanoseconds();

//     printf("After check object!");
timeEstimationClass_CustomLocal.countingStart();
    std::vector<int> selected_surfaces;
    for(int i = 0; i < surfaces.size(); ++i)
    {
      if((surfaces.at(i)->selected) && (surfaces.at(i)->valid))
      {
        selected_surfaces.push_back(i);
        if(surfaces.at(i)->isNew)
          surfaces.at(i)->isNew = false;
      }
    }

    for(int i = 0; i < selected_surfaces.size(); ++i)
    {
      int idx = selected_surfaces.at(i);
//       printf("Surface %u is selected \n",idx);

      for(std::set<unsigned>::iterator itr = surfaces.at(idx)->neighbors3D.begin(); itr != surfaces.at(idx)->neighbors3D.end(); itr++)
      {
        if( (surfaces.at(*itr)->valid) && (!(surfaces.at(*itr)->selected)) )
        {
//           printf("Adding neigbour %u is selected \n",*itr);
          surfaces.at(*itr)->selected = true;
          surfaces.at(*itr)->isNew = true;
        }
      }
    }
timeEstimationClass_CustomLocal.countingEnd();
timeEstimates.times_neigboursUpdate.at(salMapNumber) += timeEstimationClass_CustomLocal.getWorkTimeInNanoseconds();

  }

  return(-1);
}

void Segmenter::segment()
{
  if( (!have_cloud) )//|| (!have_normals) )
  {
    char* error_message = new char[200];
    sprintf(error_message,"[%s::segment()]: I suggest you first set the point cloud.",ClassName.c_str()); // and normals
    throw std::runtime_error(error_message);
  }

EPUtils::TimeEstimationClass timeEstimationClass_All(CLOCK_THREAD_CPUTIME_ID);
EPUtils::TimeEstimationClass timeEstimationClass_Custom(CLOCK_THREAD_CPUTIME_ID);

timeEstimationClass_All.countingStart();

timeEstimationClass_Custom.countingStart();
  calculateNormals();
timeEstimationClass_Custom.countingEnd();
timeEstimates.time_normalsCalculation = timeEstimationClass_Custom.getWorkTimeInNanoseconds();

timeEstimationClass_Custom.countingStart();
  calculatePatches();
timeEstimationClass_Custom.countingEnd();
timeEstimates.time_patchesCalculation = timeEstimationClass_Custom.getWorkTimeInNanoseconds();

timeEstimationClass_Custom.countingStart();
  surface::View view;
  view.Reset();
  view.setPointCloud(pcl_cloud);
  view.normals = normals;
  view.setSurfaces(surfaces);
  surfaces = view.surfaces;

  view.createPatchImage();
timeEstimationClass_Custom.countingEnd();
timeEstimates.time_patchImageCalculation = timeEstimationClass_Custom.getWorkTimeInNanoseconds();

timeEstimationClass_Custom.countingStart();
  view.computeNeighbors();
timeEstimationClass_Custom.countingEnd();
timeEstimates.time_neighborsCalculation = timeEstimationClass_Custom.getWorkTimeInNanoseconds();

timeEstimationClass_Custom.countingStart();
  surfaces = view.surfaces;
  view.calculateBorders(view.cloud);
  ngbr3D_map = view.ngbr3D_map;
  ngbr2D_map = view.ngbr2D_map;
timeEstimationClass_Custom.countingEnd();
timeEstimates.time_borderCalculation = timeEstimationClass_Custom.getWorkTimeInNanoseconds();

timeEstimationClass_Custom.countingStart();
  preComputeRelations();
timeEstimationClass_Custom.countingEnd();
timeEstimates.time_relationsPreComputation = timeEstimationClass_Custom.getWorkTimeInNanoseconds();

timeEstimationClass_Custom.countingStart();
  initModelSurfaces();
timeEstimationClass_Custom.countingEnd();
timeEstimates.time_initModelSurfaces = timeEstimationClass_Custom.getWorkTimeInNanoseconds();

timeEstimationClass_Custom.countingStart();
  modelSurfaces();
timeEstimationClass_Custom.countingEnd();
timeEstimates.times_surfaceModelling.clear();
timeEstimates.times_surfaceModelling.push_back(timeEstimationClass_Custom.getWorkTimeInNanoseconds());

timeEstimationClass_Custom.countingStart();
  computeRelations();
timeEstimationClass_Custom.countingEnd();
timeEstimates.times_relationsComputation.clear();
timeEstimates.times_relationsComputation.push_back(timeEstimationClass_Custom.getWorkTimeInNanoseconds());

timeEstimationClass_Custom.countingStart();
  graphBasedSegmentation();
timeEstimationClass_Custom.countingEnd();
timeEstimates.times_graphBasedSegmentation.clear();
timeEstimates.times_graphBasedSegmentation.push_back(timeEstimationClass_Custom.getWorkTimeInNanoseconds());

timeEstimationClass_Custom.countingStart();
  createMasks();
timeEstimationClass_Custom.countingEnd();
timeEstimates.times_maskCreation.clear();
timeEstimates.times_maskCreation.push_back(timeEstimationClass_Custom.getWorkTimeInNanoseconds());

timeEstimationClass_All.countingEnd();
timeEstimates.time_total = timeEstimationClass_All.getWorkTimeInNanoseconds();
  
}

void Segmenter::createMasks()
{
  masks.clear();
  cv::Mat mask = cv::Mat_<uchar>::zeros(pcl_cloud->height,pcl_cloud->width);
  
  for(int i = 0; i < surfaces.size(); i++)
  {
    if(surfaces.at(i)->label == -1)
      continue;
    
    for(int j = 0; j < surfaces.at(i)->indices.size(); j++)
    {
      int row = surfaces.at(i)->indices.at(j) / pcl_cloud->width;
      int col = surfaces.at(i)->indices.at(j) % pcl_cloud->width;
      
      mask.at<uchar>(row,col) = (surfaces.at(i)->label + 1);
    }
  }
  
  masks.push_back(mask);
}

} // end segmentation