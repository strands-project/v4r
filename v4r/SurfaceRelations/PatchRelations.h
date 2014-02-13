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
 * @file PatchRelations.h
 * @author Richtsfeld
 * @date November 2011, July 2012
 * @version 0.1
 * @brief Calculate patch relations.
 */

#ifndef SURFACE_PATCH_RELATION_HH
#define SURFACE_PATCH_RELATION_HH

#include <omp.h>
#include <vector>
#include <utility>
#include <cstdio>
#include <opencv2/highgui/highgui.hpp>

#include "v4r/PCLAddOns/PCLCommonHeaders.h"
#include "v4r/PCLAddOns/PCLUtils.h"
#include "v4r/PCLAddOns/PCLFunctions.h"
#include "v4r/PCLAddOns/NormalsEstimationNR.hh"
#include "v4r/SurfaceUtils/SurfaceModel.hpp"
//#include "v4r/SurfaceModeling/PrincipalCurvature.h"

#include "v4r/vs3/VisionCore.hh"
#include "v4r/vs3/LJunction.hh"
#include "v4r/vs3/TJunction.hh"
#include "v4r/vs3/Collinearity.hh"
#include "v4r/vs3/Draw.hh"
#include "v4r/vs3/Math.hh"
#include "v4r/vs3/Vector2.hh"

#include "v4r/SurfaceUtils/Relation.h"
//#include "ColorHistogram.h"
//#include "ColorHistogram2D.h"
#include "ColorHistogram3D.h"
#include "Surf.h"
#include "Fourier.h"
#include "Gabor.h"
#include "ContourNormalsDistance.hh"
#include "CEdgeExtractor.h"

namespace surface
{
  
class PatchRelations
{
public:
EIGEN_MAKE_ALIGNED_OPERATOR_NEW     /// for 32-bit systems for pcl mandatory
  
protected:

private:

  // parameter    
  double max3DDistance;                                             ///< Maximum z-distance for point-neighbors
  int learn_size_1st;                                               ///< Minimum size for a 1nd-level patch for learning relation 
  int learn_size_2nd;                                               ///< Minimum size for a 2nd-level patch for learning relation 
  int train_size_2nd;                                               ///< Minimum size for a 2nd-level patch for prediction!
  
  double z_max;                                                     ///< Maximum z-value between neighboring pixels
  
  bool calculateStructuralRelations;                                ///< Calculate relations for neighbouring patches (structural level)
  bool calculateAssemblyRelations;                                  ///< Calculate relations for non-neighbouring patches (assembly level)
  bool compute_patch_models;                                        ///< Compute optimal patch models before relation calculation

  bool have_preprocessed;                                           ///< already preprocessed
  bool have_learn_relations;                                        ///< Learn relations for 1st level already calculated
    
  bool have_input_cloud;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud;           	    ///< Input cloud
  cv::Mat_<cv::Vec3b> matImage;                                     ///< Image as Mat
  IplImage *iplImage;                                               ///< Image as IplImage

  bool have_normals;
  pcl::PointCloud<pcl::Normal>::Ptr pcl_normals;                    ///< Original normals of the point cloud

  bool have_patches;
  unsigned nr_patches;                                              ///< Number of patches
  cv::Mat_<cv::Vec3b> patches;                                      ///< Patch indices (+1 !) on 2D image grid
  pcl::PointCloud<pcl::Normal>::Ptr pcl_model_normals;              ///< Recalculated optimal normals of the models
  std::vector<surface::SurfaceModel::Ptr> surfaces;                 ///< Surface models (input and output)

//   surface::PrincipalCurvature *pc;                                  ///< Principal curvature calculation
  
  std::vector<Relation> relations;                                  ///< Relations between patches

  Z::VisionCore *vs3;                                               ///< vs3 interface for line segment calculations

  /// TODO Diese 3 models sollen nicht mehr verwendet werden => Alle Infos in surfaces!
  std::vector<int> pcl_model_types;                                 ///< Type of model                                          /// TODO Weg damit
  std::vector<pcl::ModelCoefficients::Ptr> model_coefficients;      ///< model coeffficients                                    /// TODO Weg damit
  std::vector<pcl::PointIndices::Ptr> pcl_model_cloud_indices;      ///< pcl_model_cloud_indices                                /// TODO Weg damit

  bool have_neighbors;
  std::vector< std::vector<unsigned> > neighbors2D;                 ///< Neighboring patches in image space
  std::vector< std::vector<unsigned> > neighbors3D;                 ///< Neighboring patches (with z_max value)
  
  int nr_hist_bins;                                                 ///< Number of color histogram bins
  double uvThreshold;                                               ///< Threshold of UV pruning
//   std::vector<ColorHistogram*> hist;                                ///< Color histogram of each patch
//   std::vector<ColorHistogram2D*> hist2D;                            ///< Color histogram of each patch (Max version)
//   std::vector<ColorHistogram2DMax*> hist2D;                         ///< Color histogram of each patch (Max version)
//   std::vector<ColorHistogram3D*> hist3D;                            ///< Color histogram of each patch (Max version)
  std::vector<ColorHistogram3D> hist3D;                            ///< Color histogram of each patch (Max version)

  bool have_annotation;                                             ///< Annotation of 1st level svm
  std::vector< std::vector<int> > annotation;                       ///< Annotation: Assigned patches of each patch
  std::vector<int> anno_bg_list;                                    ///< Annotation: List of patches in background
  
  bool have_annotation2;                                            ///< Annotation of 2nd level svm
  std::vector< std::vector<int> > annotation2;                      ///< Annotation2: Relations of each patch
  std::vector<int> anno_bg_list2;                                   ///< Annotation2: List of patches in background
  
  std::vector<bool> texture;                                        ///< Texture in image space
  std::vector<double> textureRate;                                  ///< Texture rate for each surface
  
  bool have_fourier;                                                ///< Fourier values already calculated
  Fourier *fourier;

  bool have_gabor;                                                  ///< Gabor values already calculated
  Gabor *gabor;
  
  bool compute_feature_normals;
  std::vector<Eigen::Vector3d> normals_mean;                        ///< Mean of surface normals of patch                       /// TODO Müssen die global definiert sein?
  std::vector<double> normals_var;                                  ///< Variance of surface normals of patch                   /// TODO

  bool have_border_indices;                                         ///< True, if we have calculated border indices     /// TODO Not inplemented right now
  std::vector< std::vector<int> > border_indices;
  
  ContourNormalsDistance *cnd;                                      ///< angle and distance between nearest points of contour
  
  /** Preprocess point cloud for feature extraction **/
  void preprocess();
  
  /** Check, if pair is 2D or 3D neigbor **/
  bool Is2DNeigbor(int p0, int p1);
  bool Is3DNeighbor(int p0, int p1);
  
  /** Calculate relations on borders of patchtes **/
  bool CalculateBorderRelation(int p0, int p1, 
                               std::vector<double> &rel_value);
  
  /** Returns true, if a annotation relation is available (1=structural/2=assembly **/
  bool haveAnnoRelation(int i, int j);
  bool haveAnnoRelation2(int i, int j);
  
  /** Returns true, if relation is between background patches (no annotation available) **/
  bool BackgroundRelation(int i, int j);
  bool BackgroundRelation2(int i, int j);
  
  /** Calculate the texture relation **/
  bool calculateTextureRelation(int i, int j,
                                double &_textureRate);              ///< Texture relation based on canny edges
  bool calculateSurfRelation(int i, int j, 
                             double &_surfRelation);                ///< Relation, based on surf-features
  bool calculateFourierRelation(int i, int j, 
                                double &_fourierRelation);          ///< Texture relation based on DFT
  bool calculateGaborRelation(int i, int j, 
                              double &_gaborRelation);              ///< Texture relation based on Gabor filter
  
  bool calculateNormalRelations(int i, int j,
                                double &_nor_mean, 
                                double &_nor_var);                  /// Normal relations (mean and variance)

  bool calculateBoundaryRelations(int i, int j,
                                  double &_vs3_col0, 
                                  double &_vs3_col1, 
                                  double &_vs3_1,
                                  double &_vs3_2,
                                  double &_vs3_3,
                                  double &_vs3_4);                  /// Calculate vs3 boundary relations

  void DrawLine(int x1, int y1, int x2, int y2, 
                std::vector<int> &_x, std::vector<int> &_y);        /// Get 2D line points in an array
    
public:
  PatchRelations();
  ~PatchRelations();

  /** Set depth limits for neighborhood calculation **/
  void setZLimit(double _z_max);

  /** Set caculation of relations of non-neighbouring patches **/
  void setAssemblyLevel(bool _on) {calculateAssemblyRelations = _on;}
  
  /** Set caculation of relations of neighbouring patches **/
  void setStructuralLevel(bool _on) {calculateStructuralRelations = _on;}
  
  /** Set input point cloud **/
  void setInputCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr & _pcl_cloud);
  
  /** Set input surface patches **/
  void setSurfaceModels(std::vector< surface::SurfaceModel::Ptr > & _surfaces);

  /** Compute neigbors between planes **/
  void setNeighbors(std::vector< std::vector<unsigned> > &n) {neighbors3D = n; have_neighbors = true;}
  
  /** Set annotation (from each patch) for 1st and 2nd level svm **/
  void setAnnotion(std::vector< std::vector<int> > &_anno,
                   std::vector<int> &_anno_bg_list);
  void setAnnotion2(std::vector< std::vector<int> > &_anno,
                    std::vector<int> &_anno_bg_list);
  
  /** Compute patch models: project point cloud to model and recalculate optimal model normals **/
  void setOptimalPatchModels(bool cpm);

  /** Compute neigboring patches **/
  void computeNeighbors();
  
  /** Compute relations for learning structural and assembly level svm **/
  void computeLearnRelations();
//   void computeLearnRelations3();

  /** Compute relations for testing the svm **/
  void computeTestRelations();

  /** Compute relations for the segmenter **/
  void computeSegmentRelations();
  
   /** Get the results from the neighborhood processing **/
  void getNeighbors(std::vector< std::vector<unsigned> > &n) {n = neighbors3D;}

  /** getRelations **/
  void getRelations(std::vector<surface::Relation> &r) {r = relations;}
       
  /** Get patches after calculation **/
  void getSurfaceModels(std::vector< surface::SurfaceModel::Ptr > &_surfaces);
  
  /** Print neighbors **/
  void printNeighbors();
};

/*************************** INLINE METHODES **************************/

} //--END--

#endif

