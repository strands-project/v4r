/**
 *  Copyright (C) 2012  
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
 * @file segmenter.h
 * @author Ekaterina Potapova
 * @date January 2014
 * @version 0.1
 * @brief Library to call segmentation from outside
 */


#ifndef SEGMENTATION_MODULES
#define SEGMENTATION_MODULES

#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <fstream>

#include <pcl/io/pcd_io.h>

#include <boost/filesystem/fstream.hpp>
#include <boost/filesystem.hpp>

#include "v4r/PCLAddOns/PCLUtils.h"
#include "v4r/SurfaceUtils/SurfaceModel.hpp"
#include "v4r/SurfaceClustering/ZAdaptiveNormals.hh"
#include "v4r/SurfaceClustering/ClusterNormalsToPlanes.hh"
#include "v4r/SurfaceModeling/SurfaceModeling.hh"

//#include "v4r/SurfaceUtils/AddGroundTruth.h"
#include "v4r/SurfaceRelations/BoundaryRelationsMeanDepth.hpp"
#include "v4r/SurfaceRelations/BoundaryRelationsMeanCurvature.hpp"
#include "v4r/SurfaceRelations/BoundaryRelationsMeanColor.hpp"
#include "v4r/SurfaceRelations/StructuralRelations.h"
//#include "v4r/SurfaceRelations/AssemblyRelations.h"

//#include "v4r/SurfaceUtils/ContourDetector.h"
//#include "v4r/SurfaceUtils/BoundaryDetector.h"

//#include "v4r/svm/SVMFileCreator.h"
#include "v4r/svm/SVMScale.h"
#include "v4r/svm/SVMPredictorSingle.h"
//#include "v4r/svm/SVMTrainModel.h"

#include "v4r/GraphCut/GraphCut.h"

#include "v4r/EPUtils/EPUtils.hpp"
#include "v4r/AttentionModule/AttentionModule.hpp"

namespace segmentation
{

struct TimeEstimates {

  unsigned long long time_normalsCalculation;
  unsigned long long time_patchesCalculation;
  unsigned long long time_patchImageCalculation;
  unsigned long long time_neighborsCalculation;
  unsigned long long time_borderCalculation;
  unsigned long long time_relationsPreComputation;
  unsigned long long time_initModelSurfaces;

  std::vector<unsigned long long> times_saliencySorting;
  std::vector<unsigned long long> times_surfaceModelling;
  std::vector<unsigned long long> times_relationsComputation;
  std::vector<unsigned long long> times_graphBasedSegmentation;
  std::vector<unsigned long long> times_maskCreation;
  std::vector<unsigned long long> times_neigboursUpdate;
  std::vector<unsigned long long> time_totalPerSegment;

  unsigned long long time_total;
};
  
/**
 * @class Segmenter
 */
class Segmenter
{
private:
  
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud;                 ///< original pcl point cloud
  pcl::PointCloud<pcl::Normal>::Ptr normals;
  surface::ClusterNormalsToPlanes::Ptr clusterNormals;
  std::vector<surface::SurfaceModel::Ptr> surfaces;
  surface::SurfaceModeling::Ptr surfModeling;
  std::map<surface::borderIdentification,std::vector<surface::neighboringPair> > ngbr2D_map;
  std::map<surface::borderIdentification,std::vector<surface::neighboringPair> > ngbr3D_map;
  std::vector<surface::Relation> validRelations;
  surface::StructuralRelations structuralRelations;
  svm::SVMPredictorSingle svmPredictorSingle;
  gc::GraphCut graphCut;
  svm::SVMScale svmScale;
  std::vector<cv::Mat> saliencyMaps;
  
  std::vector<cv::Mat> masks;
  std::vector<std::vector<int> > segmentedObjectsIndices;
  
  bool have_cloud;
  //bool have_normals;
  bool have_saliencyMaps;

  std::string model_file_name, scaling_file_name;
  
  std::string ClassName;

  TimeEstimates timeEstimates;

public:

private:
  
  void calculatePatches();
  void calculateNormals();
  void initModelSurfaces();
  void modelSurfaces();
  void preComputeRelations();
  void computeRelations();
  void graphBasedSegmentation();
  bool checkSegmentation(cv::Mat &mask, int originalIndex, int salMapNumber);
  int attentionSegment(cv::Mat &object_mask, int originalIndex, int salMapNumber);
  void createMasks();
  
public:
  Segmenter();
  virtual ~Segmenter();
  
  /** Run the pre-segmenter **/
  void segment();
  void attentionSegment();
  
  void setPointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr _pcl_cloud);
  //void setNormals(pcl::PointCloud<pcl::Normal>::Ptr _normals);
  void setSaliencyMaps(std::vector<cv::Mat> _saliencyMaps);

  void setModelFilename(std::string _model_file_name);
  void setScaling(std::string _scaling_file_name);
  
  inline std::vector<surface::SurfaceModel::Ptr> getSurfaces();
  inline std::vector<cv::Mat> getMasks();
  inline std::vector<std::vector<int> > getSegmentedObjectsIndices();
  inline TimeEstimates getTimeEstimates();

};

inline void Segmenter::setModelFilename(std::string _model_file_name)
{
  model_file_name = _model_file_name;

}

inline void Segmenter::setScaling(std::string _scaling_file_name)
{
  scaling_file_name = _scaling_file_name;
}

inline std::vector<surface::SurfaceModel::Ptr> Segmenter::getSurfaces()
{
  return(surfaces);
}

inline std::vector<cv::Mat> Segmenter::getMasks()
{
  return(masks);
}

inline std::vector<std::vector<int> > Segmenter::getSegmentedObjectsIndices()
{
  return(segmentedObjectsIndices);
}

inline TimeEstimates Segmenter::getTimeEstimates()
{
  return timeEstimates;
}

} //namespace segmentation

#endif //SEGMENTATION_MODULES
