/**
 *  Copyright (C) 2012  
 *    Ekaterina Potapova, Andreas Richtsfeld, Johann Prankl, Thomas Mörwald, Michael Zillich
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

#include <v4r/core/macros.h>

#include "v4r/attention_segmentation/PCLUtils.h"
#include "v4r/attention_segmentation/SurfaceModel.h"
#include "v4r/attention_segmentation/ZAdaptiveNormals.h"
#include "v4r/attention_segmentation/ClusterNormalsToPlanes.h"
#include "v4r/attention_segmentation/SurfaceModeling.h"

#include "v4r/attention_segmentation/AddGroundTruth.h"
#include "v4r/attention_segmentation/StructuralRelations.h"
#include "v4r/attention_segmentation/SVMFileCreator.h"
#include "v4r/attention_segmentation/SVMPredictorSingle.h"

#include "v4r/attention_segmentation/GraphCut.h"
#include "v4r/attention_segmentation/EPUtils.h"

namespace v4r
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
class V4R_EXPORTS Segmenter
{
private:
  
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud;                 ///< original pcl point cloud
  pcl::PointCloud<pcl::PointXYZRGBL>::Ptr pcl_cloud_l;              ///< labeled pcl point cloud
  pcl::PointCloud<pcl::Normal>::Ptr normals;
  v4r::ClusterNormalsToPlanes::Ptr clusterNormals;
  std::vector<v4r::SurfaceModel::Ptr> surfaces;
  v4r::SurfaceModeling::Ptr surfModeling;
  std::map<v4r::borderIdentification,std::vector<v4r::neighboringPair> > ngbr2D_map;
  std::map<v4r::borderIdentification,std::vector<v4r::neighboringPair> > ngbr3D_map;
  std::vector<v4r::Relation> validRelations;
  v4r::StructuralRelations structuralRelations;
  svm::SVMPredictorSingle svmPredictorSingle;
  gc::GraphCut graphCut;
  std::vector<cv::Mat> saliencyMaps;
  
  std::vector<cv::Mat> masks;
  std::vector<std::vector<int> > segmentedObjectsIndices;
  
  v4r::View view;
  
  bool have_cloud;
  bool have_cloud_l;
  //bool have_normals;
  bool have_saliencyMaps;

  bool use_planes;

  std::string model_file_name, scaling_file_name;
  
  std::string ClassName;

  TimeEstimates timeEstimates;
  
  //only for training
  v4r::AddGroundTruth addGroundTruth;
  svm::SVMFileCreator svmFileCreator;
  std::string train_ST_file_name, train_AS_file_name;

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
  //void attentionSegment(int &objNumber);
  
  void attentionSegmentInit();
  bool attentionSegmentNext();
  
  //only for training
  void createTrainFile();
  
  void setPointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr _pcl_cloud);
  void setPointCloud(pcl::PointCloud<pcl::PointXYZRGBL>::Ptr _pcl_cloud_l);
  void setSaliencyMaps(std::vector<cv::Mat> _saliencyMaps);

  inline void setUsePlanesNotNurbs(bool _use_planes);

  inline void setModelFilename(std::string _model_file_name);
  inline void setScaling(std::string _scaling_file_name);
  
  inline std::vector<v4r::SurfaceModel::Ptr> getSurfaces();
  inline std::vector<cv::Mat> getMasks();
  inline std::vector<std::vector<int> > getSegmentedObjectsIndices();
  inline TimeEstimates getTimeEstimates();
  
  //only for training
  inline void setTrainSTFilename(std::string _train_ST_file_name);
  inline void setTrainASFilename(std::string _train_AS_file_name);

};

inline void Segmenter::setUsePlanesNotNurbs(bool _use_planes)
{
  use_planes = _use_planes;
}

inline void Segmenter::setModelFilename(std::string _model_file_name)
{
  model_file_name = _model_file_name;

}

inline void Segmenter::setTrainSTFilename(std::string _train_ST_file_name)
{
  train_ST_file_name = _train_ST_file_name;
}

inline void Segmenter::setTrainASFilename(std::string _train_AS_file_name)
{
  train_AS_file_name = _train_AS_file_name;
}

inline void Segmenter::setScaling(std::string _scaling_file_name)
{
  scaling_file_name = _scaling_file_name;
}

inline std::vector<v4r::SurfaceModel::Ptr> Segmenter::getSurfaces()
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
