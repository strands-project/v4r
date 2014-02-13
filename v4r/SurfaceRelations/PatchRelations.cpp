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
 * @file PatchRelations.cpp
 * @author Richtsfeld
 * @date November 2011, July 2012
 * @version 0.1
 * @brief Calculate patch relations.
 */

#include "PatchRelations.h"

namespace surface
{

/** Upscale indices by a scale of 2 **/
std::vector<int> UpscaleIndices(std::vector<int> &_indices,
                                int new_image_width)
{
  static bool first = true;
  if(first)
    printf("[Patches: UpscaleIndices] Warning: Experimental function: Check this first!\n");
  first = false;
  std::vector<int> new_indices;
  for(unsigned i=0; i<_indices.size(); i++) {
    new_indices.push_back(_indices[i]*2);
    new_indices.push_back(_indices[i]*2 + 1);
    new_indices.push_back(_indices[i]*2 + new_image_width);
    new_indices.push_back(_indices[i]*2 + new_image_width + 1);
  }
  return new_indices;
}


void ConvertIpl2MatImage(IplImage &iplImage, cv::Mat_<cv::Vec3b> &image)
{
  image = cv::Mat_<cv::Vec3b>(iplImage.height, iplImage.width); 
  for (int v = 0; v < iplImage.height; ++v)
  {
    uchar *d = (uchar*) iplImage.imageData + v*iplImage.widthStep;
    for (int u = 0; u < iplImage.width; ++u, d+=3)
    {
      cv::Vec3b &ptCol = image(v,u);
      ptCol[0] = d[0];
      ptCol[1] = d[1];
      ptCol[2] = d[2];
    }
  }
}

void ConvertMat2IplImage(cv::Mat_<cv::Vec3b> & image, IplImage &iplImage)
{
  for(int row=0; row<image.rows; row++) {
    uchar *d = (uchar*)(iplImage.imageData + row*iplImage.widthStep);
    for(int col=0; col<image.cols; col++, d+=3) {
      cv::Vec3b &ptCol = image(col, row);
      d[0] = ptCol[0];
      d[1] = ptCol[1];
      d[2] = ptCol[2];
    }
  }
}

/************************************************************************************
 * Constructor/Destructor
 */

PatchRelations::PatchRelations()
{
  z_max = 0.01;               // depth limit for 3D neighborhood
  nr_hist_bins = 4;           // Number of color histogram bins
  uvThreshold = 0.0f;         // Threshold of UV pruning @ center of histogram
  max3DDistance = z_max; //0.015;      // TODO maximum distance for 3D neighborhood of border relations => TODO Kann man hier nicht z_max verwenden?
  learn_size_1st = 30;        // minimum patch size for learning in structural level
  learn_size_2nd = 300;       // minimum patch size for learning in assambly level
  train_size_2nd = 300;       // predict only 2nd level for big patches => Does avoid some long calculations and does not change the results (for 300)
  
  calculateStructuralRelations = true;
  calculateAssemblyRelations = true;
  have_input_cloud = false;
  have_normals = false;
  have_patches = false;
  have_neighbors = false;
  have_annotation = false;
  have_annotation2 = false;
  have_fourier = false;
  have_gabor = false;
  compute_patch_models = false;
  compute_feature_normals = true;
  have_preprocessed = false;
  have_border_indices = false;
  
  pcl_model_normals.reset(new pcl::PointCloud<pcl::Normal>);
  
  fourier = new Fourier();
  gabor = new Gabor();
  
  cnd = new ContourNormalsDistance();
  ContourNormalsDistance::Parameter cndParam;
  cndParam.pcntContourPoints = 0.2;
  cnd->setParameter(cndParam);
  
  vs3 = new Z::VisionCore();
  vs3->EnableGestaltPrinciple(Z::GestaltPrinciple::FORM_SEGMENTS);            // 0
//   vs3->EnableGestaltPrinciple(Z::GestaltPrinciple::FORM_ARCS);                // 2
//   vs3->EnableGestaltPrinciple(Z::GestaltPrinciple::FORM_ARC_JUNCTIONS);       // 9
//   vs3->EnableGestaltPrinciple(Z::GestaltPrinciple::FORM_CONVEX_ARC_GROUPS);   // 3
//   vs3->EnableGestaltPrinciple(Z::GestaltPrinciple::FORM_ELLIPSES);            // 4
  vs3->EnableGestaltPrinciple(Z::GestaltPrinciple::FORM_LINES);               // 2
  vs3->EnableGestaltPrinciple(Z::GestaltPrinciple::FORM_JUNCTIONS);           // col=6, t=7, l=8
  vs3->EnableGestaltPrinciple(Z::GestaltPrinciple::FORM_CLOSURES);
  
  iplImage = 0;
}

PatchRelations::~PatchRelations()
{
  delete fourier;
  delete gabor;
  delete cnd;
  delete vs3;
//   delete pc;
  cvReleaseImageHeader(&iplImage);
}

// ================================= Private functions ================================= //

void PatchRelations::preprocess()
{
  hist3D.clear();
  for(unsigned i=0; i<surfaces.size(); i++) { 
//     hist3D.push_back(new ColorHistogram3D(nr_hist_bins, uvThreshold));                               
    hist3D.push_back(ColorHistogram3D(nr_hist_bins, uvThreshold));                               
    hist3D[i].setInputCloud(pcl_cloud);
    hist3D[i].setIndices(pcl_model_cloud_indices[i]);
    hist3D[i].compute();
  }
  
  // invert plane coefficients, if normals point to background  // TODO Sollte schon in modelAbstraction überprüft worden sein
//   if(compute_patch_models) {
//     for(unsigned i=0; i<surfaces.size(); i++) {
//       pcl_normals = pcl_model_normals;  // copy normals
// 
//       // copy points of point cloud to model surface.
//       if(surfaces[i]->type == pcl::SACMODEL_NORMAL_PLANE || 
//          surfaces[i]->type == pcl::SACMODEL_PLANE)
//       {
//         if(model_coefficients[i]->values[3] < 0.) {
//            model_coefficients[i]->values[0] = -model_coefficients[i]->values[0];
//            model_coefficients[i]->values[1] = -model_coefficients[i]->values[1];
//            model_coefficients[i]->values[2] = -model_coefficients[i]->values[2];
//            model_coefficients[i]->values[3] = -model_coefficients[i]->values[3];
//         }
//         pclA::ProjectPC2Model(pcl::SACMODEL_PLANE, 
//                               pcl_cloud, 
//                               pcl_model_cloud_indices[i],
//                               model_coefficients[i]);
//       }
//       else if(surfaces[i]->type == MODEL_NURBS) {
// //         printf("[PatchRelations::preprocess] Warning: NURBS points are not projected to NURBS model.\n");
//       }
//       else
//         printf("[PatchRelations::preprocess] Warning: Reculculation of model type not supported: model_type: %u\n", surfaces[i]->type);
//     }
//   }
  
  // calculate texture for each patch (canny edge extractor)
  CEdgeExtractor cedge;
  cedge.extract(iplImage);
  cedge.getTexture(texture);
  
  // calculate texture-rates for all surfaces, if we have texture (dilation is not a good idea!)
  // MaskDilationErosion *dil = new MaskDilationErosion();
  if(pcl_cloud->width != (pcl_cloud->height *4/3))
    printf("[PatchRelations::preprocess] Warning: Point cloud not organized.\n");
  // dil->setImageSize(pcl_cloud->width, pcl_cloud->height);
  // dil->setSize(1);
  // dil->showMasks(false);
  textureRate.resize(surfaces.size());
  for(unsigned i=0; i<surfaces.size(); i++) {
    std::vector<int> mask = surfaces[i]->indices;
    // dil->compute(mask);
    double area = mask.size();
    int tex_area = 0;
    for(unsigned j=0; j<mask.size(); j++) {
      if(texture[mask[j]])
        tex_area++;
    }
    if(area == 0) 
      textureRate[i] = 0.;
    else
      textureRate[i] = (double) (tex_area / area);
    // printf("[Patches::preprocess] Texture Rate [%u]: %4.3f/%u = %4.3f\n", i, area, tex_area, textureRate[i]);
  }

  // calculate mean and variance of surface normals
  if(compute_feature_normals) {
    normals_mean.clear();
    normals_var.clear();
    for(unsigned i=0; i<surfaces.size(); i++) {
      Eigen::Vector3d mean;
      mean[0]=0.;
      mean[1]=0.;
      mean[2]=0.;
      double var = 0;
      for(unsigned j=0; j<surfaces[i]->normals.size(); j++)
        mean += surfaces[i]->normals[j];
      mean /= surfaces[i]->normals.size();
      normals_mean.push_back(mean);
      
      // calculate variance
      for(unsigned j=0; j<surfaces[i]->normals.size(); j++) {
        double x = surfaces[i]->normals[j].dot(mean) / (surfaces[i]->normals[j].norm() * mean.norm());
        if(x>1.0) {
//           printf("[PatchRelations::preprocess] Warning: Value too high (%8.8f).\n", x);
          x = 1.0;
        }
        var += acos(x);
      }
      var /= surfaces[i]->normals.size();
      normals_var.push_back(var);
      // printf("Normals sum & var of patch %u: %4.3f (size: %lu)\n", i, var, surfaces[i]->normals.size());
    }
  }
  
//   if(true)  /// TODO Calculte only, if we use this feature => time consuming
//     calculateBorderIndices();
 
  cnd->setInputCloud(pcl_cloud);
  
  // compute point principle curvature
//   pc = new surface::PrincipalCurvature(surface::PrincipalCurvature::Parameter());
//   pc->setInputCloud(pcl_cloud);
//   pc->setInputNormals(pcl_model_normals);
//   pc->setSurfaceModels(surfaces);
  
  have_preprocessed = true;
}


bool PatchRelations::Is2DNeigbor(int p0, int p1)
{
  if(!have_neighbors)
    computeNeighbors();
  for(unsigned i=0; i<neighbors2D[p0].size(); i++)
    if((int) neighbors2D[p0][i] == p1)
      return true;
  return false;
}

bool PatchRelations::Is3DNeighbor(int p0, int p1)
{
  if(!have_neighbors)
    computeNeighbors();
  for(unsigned i=0; i<neighbors3D[p0].size(); i++)
    if((int) neighbors3D[p0][i] == p1)
      return true;
  return false;
}


/** Calculate relations on the 2D border of patches **/
/** color, depth, mask, curvature **/
/// @param p0 Index of first patch
/// @param p1 Index of second patch
/// @param rel_value Vector of relation values
bool PatchRelations::CalculateBorderRelation(int p0, int p1, 
                                             std::vector<double> &rel_value)
{
  rel_value.resize(0);
  p0++; p1++;  // indices on patches starting with 1! (0 is invalid!)
  
  int first = 0;
  int second = 0;
  std::vector<int> first_ngbr;
  std::vector<int> second_ngbr;
  std::vector<int> first_3D_ngbr;
  std::vector<int> second_3D_ngbr;
  
  // get neighbouring pixel-pairs in 2D and in 3D
  for(int row=1; row<patches.rows; row++) {
    for(int col=1; col<patches.cols; col++) {
      bool found = false;
      double distance = 10.;
      if((patches.at<cv::Vec3b>(row, col)[0] == p0 && patches.at<cv::Vec3b>(row-1, col-1)[0] == p1) ||  // left-upper pixel
         (patches.at<cv::Vec3b>(row, col)[0] == p1 && patches.at<cv::Vec3b>(row-1, col-1)[0] == p0)) {
        first = row*patches.cols + col;
        second = (row-1)*patches.cols + col-1;
        distance = fabs(pcl_cloud->points[row*pcl_cloud->width + col].z - pcl_cloud->points[(row-1)*pcl_cloud->width + col-1].z);
        found = true;
      }
      if((patches.at<cv::Vec3b>(row, col)[0] == p0 && patches.at<cv::Vec3b>(row-1, col+1)[0] == p1) ||  // right-upper pixel
         (patches.at<cv::Vec3b>(row, col)[0] == p1 && patches.at<cv::Vec3b>(row-1, col+1)[0] == p0)) {
        first = row*patches.cols + col;
        second = (row-1)*patches.cols + col+1;
        distance = fabs(pcl_cloud->points[row*pcl_cloud->width + col].z - pcl_cloud->points[(row-1)*pcl_cloud->width + col+1].z);
        found = true;
      }
      if((patches.at<cv::Vec3b>(row, col)[0] == p0 && patches.at<cv::Vec3b>(row, col-1)[0] == p1) ||    // left pixel
         (patches.at<cv::Vec3b>(row, col)[0] == p1 && patches.at<cv::Vec3b>(row, col-1)[0] == p0)) {
        first = row*patches.cols + col;
        second = row*patches.cols + col-1;
        distance = fabs(pcl_cloud->points[row*pcl_cloud->width + col].z - pcl_cloud->points[row*pcl_cloud->width + col-1].z);
        found = true;
      }
      if((patches.at<cv::Vec3b>(row, col)[0] == p0 && patches.at<cv::Vec3b>(row-1, col)[0] == p1) ||    // upper pixel
         (patches.at<cv::Vec3b>(row, col)[0] == p1 && patches.at<cv::Vec3b>(row-1, col)[0] == p0)) {
        first = row*patches.cols + col;
        second = (row-1)*patches.cols + col;
        distance = fabs(pcl_cloud->points[row*pcl_cloud->width + col].z - pcl_cloud->points[(row-1)*pcl_cloud->width + col].z);
        found = true;
      }
      if(found) {
        first_ngbr.push_back(first);
        second_ngbr.push_back(second);
        if(distance < max3DDistance) {
          first_3D_ngbr.push_back(first);
          second_3D_ngbr.push_back(second);
        }
      }
    }
  }
  

  int nr_valid_points_color = 0;
  int nr_valid_points_depth = 0;
  double sum_uv_color_distance = 0.0f;    // 
  double sum_2D_curvature = 0.0f;
  double sum_depth = 0.0f;
  double sum_depth_var = 0.0f;
  std::vector<double> depth_vals;         // depth values on neighboring surface borders
  
  for(unsigned i=0; i<first_ngbr.size(); i++)
  {
    // calculate mean depth
    double p0_z = pcl_cloud->points[first_ngbr[i]].z;
    double p1_z = pcl_cloud->points[second_ngbr[i]].z;
    double depth = fabs(p0_z - p1_z);
    depth_vals.push_back(depth);
    if(depth == depth) { // no nan
      nr_valid_points_depth++;
      sum_depth += depth;
    }
#ifdef DEBUG      
    else 
      printf("[PatchRelations::CalculateBorderRelation] Warning: Invalid depht points (nan): Should not happen! Why?\n");
#endif
  } 
  
  // normalize depth sum and calculate depth variance
  if(nr_valid_points_depth != 0) {
    sum_depth /= nr_valid_points_depth;
    for(unsigned i=0; i<depth_vals.size(); i++)
      sum_depth_var += fabs(depth_vals[i] - sum_depth);
    sum_depth_var /= nr_valid_points_depth;
  }
#ifdef DEBUG      
  else 
    std::printf("[PatchRelations::CalculateBorderRelation] Warning: Number of valid depth points is zero: sum_depth: %4.3f\n", sum_depth);
#endif
    
// normalize color sum and calculate variance /// TODO Variance of color???
//   if(nr_valid_points_color != 0)
//     sum_uv_color_distance /= nr_valid_points_color;
//   else 
//     printf("[Patches::CalculateBorderRelation] Warning: Number of valid color points is zero: sum_color: %4.3f\n", sum_uv_color_distance);
  
  // normalize curvature sum and calculate variance /// Curvature on 2D Border ist nicht besonders
//   if(nr_valid_points_curvature != 0)
//     sum_2D_curvature /= nr_valid_points_curvature;
//   else 
//     printf("[Patches::CalculateBorderRelation] Warning: Number of valid curvature points is zero: sum_2D_curvature: %4.3f\n", sum_2D_curvature);

  
  /// calcuate curvature / depth
  int nr_valid_points_curvature3D = 0;
  double sum_3D_curvature = 0.0f;
  double sum_3D_curvature_var = 0.0f;
  std::vector<double> curvature_vals;         // single curvature values
  for(unsigned i=0; i<first_3D_ngbr.size(); i++)
  {
    /// calculate color similarity in 3D
    nr_valid_points_color++;
    pclA::RGBValue p0_color, p1_color;
    p0_color.float_value = pcl_cloud->points[first_3D_ngbr[i]].rgb;
    p1_color.float_value = pcl_cloud->points[second_3D_ngbr[i]].rgb;

// //     double p0_Y =  (0.257 * p0_color.r) + (0.504 * p0_color.g) + (0.098 * p0_color.b) + 16;
//     double p0_U = -(0.148 * p0_color.r) - (0.291 * p0_color.g) + (0.439 * p0_color.b) + 128;
//     double p0_V =  (0.439 * p0_color.r) - (0.368 * p0_color.g) - (0.071 * p0_color.b) + 128;
// //     double p1_Y =  (0.257 * p1_color.r) + (0.504 * p1_color.g) + (0.098 * p1_color.b) + 16;
//     double p1_U = -(0.148 * p1_color.r) - (0.291 * p1_color.g) + (0.439 * p1_color.b) + 128;
//     double p1_V =  (0.439 * p1_color.r) - (0.368 * p1_color.g) - (0.071 * p1_color.b) + 128;

    /// TODO rgb or bgr => For the eucledian distance it seems not that important
// // printf("rgb[%u][%u] an [%u][%u]: %u - %u - %u vs. %u - %u - %u\n", p0, p1, first_ngbr[i], second_ngbr[i], p0_color.r, p0_color.g, p0_color.b, p1_color.r, p1_color.g, p1_color.b);
    //     double p0_Y =  (0.257 * p0_color.b) + (0.504 * p0_color.g) + (0.098 * p0_color.r) + 16;
    double p0_U = -(0.148 * p0_color.b) - (0.291 * p0_color.g) + (0.439 * p0_color.r) + 128;
    double p0_V =  (0.439 * p0_color.b) - (0.368 * p0_color.g) - (0.071 * p0_color.r) + 128;
//     double p1_Y =  (0.257 * p1_color.b) + (0.504 * p1_color.g) + (0.098 * p1_color.r) + 16;
    double p1_U = -(0.148 * p1_color.b) - (0.291 * p1_color.g) + (0.439 * p1_color.r) + 128;
    double p1_V =  (0.439 * p1_color.b) - (0.368 * p1_color.g) - (0.071 * p1_color.r) + 128;
    
//     double y_1 = p0_Y/255 - p1_Y/255;
//     double y_2 = y_1 * y_1 * 100;
    double u_1 = p0_U/255 - p1_U/255;
    double u_2 = u_1 * u_1;
    double v_1 = p0_V/255 - p1_V/255;
    double v_2 = v_1 * v_1;
    double cDist = sqrt(u_2 + v_2);
// printf("u: %3.0f-%3.0f => %4.3f => %4.3f // v: %3.0f-%3.0f => %4.3f => %4.3f   and cDist: %4.3f\n", p0_U, p1_U, u_1, u_2, p0_V, p1_V, v_1, v_2, cDist);
    sum_uv_color_distance += cDist;
    
    /// calculate mean curvature
    cv::Vec3f pt0, pt1;
    pt0[0]= pcl_cloud->points[first_ngbr[i]].x;
    pt0[1]= pcl_cloud->points[first_ngbr[i]].y;
    pt0[2]= pcl_cloud->points[first_ngbr[i]].z;
    pt1[0]= pcl_cloud->points[second_ngbr[i]].x;
    pt1[1]= pcl_cloud->points[second_ngbr[i]].y;
    pt1[2]= pcl_cloud->points[second_ngbr[i]].z;
    
    if(pt0 == pt0 || pt1 == pt1)
    {
      cv::Vec3f p0_normal;
      p0_normal[0] = pcl_model_normals->points[first_ngbr[i]].normal_x;                 /// TODO Remove pcl_model_normals and use surface->normals
      p0_normal[1] = pcl_model_normals->points[first_ngbr[i]].normal_y;
      p0_normal[2] = pcl_model_normals->points[first_ngbr[i]].normal_z;
      cv::Vec3f p1_normal;
      p1_normal[0] = pcl_model_normals->points[second_ngbr[i]].normal_x;
      p1_normal[1] = pcl_model_normals->points[second_ngbr[i]].normal_y;
      p1_normal[2] = pcl_model_normals->points[second_ngbr[i]].normal_z;
      cv::Vec3f pp = pt1 - pt0;

      double norm_pp = cv::norm(pp);
      cv::Vec3f pp_dir;
      pp_dir[0] = pp[0]/norm_pp;
      pp_dir[1] = pp[1]/norm_pp;
      pp_dir[2] = pp[2]/norm_pp;  

      double a_p0_pp = acos(p0_normal.ddot(pp_dir));
      pp_dir = -pp_dir; // invert direction between points
      double a_p1_pp = acos(p1_normal.ddot(pp_dir));
      double curvature = 0.0;
      if(a_p0_pp == a_p0_pp && a_p1_pp == a_p1_pp) {
        nr_valid_points_curvature3D++;
        curvature = a_p0_pp+a_p1_pp - M_PI;
        curvature_vals.push_back(curvature);
        sum_3D_curvature += curvature;
      }
#ifdef DEBUG      
      else
        printf("[PatchRelations::CalculateBorderRelation] Warning: Invalid curvature points (nan): Should not happen! DO SOMETHING!\n");
#endif
    }
  }

  // normalise
  if(nr_valid_points_color != 0)
    sum_uv_color_distance /= nr_valid_points_color;
#ifdef DEBUG      
  else 
    printf("[PatchRelations::CalculateBorderRelation] Warning: Number of valid color points is zero: sum_color: %4.3f\n", sum_uv_color_distance);
#endif
    
  if(nr_valid_points_curvature3D != 0) {
    sum_3D_curvature /= nr_valid_points_curvature3D;
    for(unsigned i=0; i<curvature_vals.size(); i++)
      sum_3D_curvature_var += fabs(curvature_vals[i] - sum_3D_curvature);
    sum_3D_curvature_var /= nr_valid_points_depth;
  }
#ifdef DEBUG      
  else 
    printf("[PatchRelations::CalculateBorderRelation] Warning: Number of valid 3D curvature points is zero: sum_3D_curvature: %4.3f\n", sum_3D_curvature);
#endif
    
  rel_value.push_back(1.-sum_uv_color_distance);
  rel_value.push_back(sum_depth);
  rel_value.push_back(sum_depth_var);
  rel_value.push_back(sum_2D_curvature);            /// TODO We do not use that: Remove that at one point
  rel_value.push_back(sum_3D_curvature);
  rel_value.push_back(sum_3D_curvature_var);
  return true;
}


bool PatchRelations::haveAnnoRelation(int i, int j)
{
  if((int) annotation.size() < i) {
    printf("[Patches::haveAnnoRelation] Error: Requested patch higher than number of annotations!\n");
    return false;
  }
  if(annotation[i].size() == 0)
    return false;

  for(unsigned idx=0; idx<annotation[i].size(); idx++)
    if(annotation[i][idx] == j)
      return true;
  return false;
}

bool PatchRelations::haveAnnoRelation2(int i, int j)
{
  if((int) annotation2.size() < i) {
    printf("[Patches::haveAnnoRelation2] Error: Requested patch higher than number of annotations!\n");
    return false;
  }
  if(annotation2[i].size() == 0)
    return false;

  for(unsigned idx=0; idx<annotation2[i].size(); idx++)
    if(annotation2[i][idx] == j)
      return true;
  return false;
}

bool PatchRelations::BackgroundRelation(int i, int j)
{
  bool found_i = false;
  bool found_j = false;
  for(unsigned idx=0; idx<anno_bg_list.size(); idx++) {
    if(anno_bg_list[idx] == i) 
      found_i = true;
    if(anno_bg_list[idx] == j) 
      found_j = true;
  }
  if(found_i && found_j)
    return true;
  else
    return false;
}

bool PatchRelations::BackgroundRelation2(int i, int j)
{
  bool found_i = false;
  bool found_j = false;
  for(unsigned idx=0; idx<anno_bg_list2.size(); idx++) {
    if(anno_bg_list2[idx] == i) 
      found_i = true;
    if(anno_bg_list2[idx] == j) 
      found_j = true;
  }
  if(found_i && found_j)
    return true;
  else
    return false;
}

bool PatchRelations::calculateTextureRelation(int i, int j, double &_textureRate)
{
    _textureRate = 1. - fabs(textureRate[i] - textureRate[j]);
    return true;
}


bool PatchRelations::calculateSurfRelation(int i, int j, double &_surfRelation)
{
  if(!have_input_cloud) {
    printf("[PatchRelations::calculateSurfRelation] Error: No input image available. Abort.\n");
    return false;
  }
  
  Surf surf;    // TODO calculate only once, if you want to use it!
  surf.setInputImage(iplImage);
  surf.compute();
  
  /// Check UpscaleIndices
  std::vector<int> indices_0, indices_1;
  unsigned image_width = cvGetSize(iplImage).width;
  if(pcl_cloud->width != image_width) {
    printf("[Patches::calculateSurfRelation] Warning: Upscale indices untested!\n");
    indices_0 = UpscaleIndices(pcl_model_cloud_indices[i]->indices, image_width);
    indices_1 = UpscaleIndices(pcl_model_cloud_indices[j]->indices, image_width);
  }
  else {
    indices_0 = pcl_model_cloud_indices[i]->indices;
    indices_1 = pcl_model_cloud_indices[j]->indices;
  }
  _surfRelation = surf.compare(indices_0, indices_1);
  
  // Draw surf features of a certain pair
//   printf("[Patches::calculateSurfRelation] Draw surf features.\n");
//   if(i == 2 && j == 3) {
//     cv::Mat_<cv::Vec3b> feature_image;
//     ConvertImage(*iplImage, feature_image);  
//     surf.drawFeatures(feature_image);
//   }
  
  return true;
}


bool PatchRelations::calculateFourierRelation(int i, int j, double &_fourierRelation)
{
  if(!have_input_cloud) {
    printf("[Patches::calculateFourierRelation] Error: No input image available. Abort.\n");
    return false;
  }
  
  if(!have_fourier) {
    fourier->setInputImage(matImage);
    fourier->compute();
    have_fourier = true;
  }
  
  std::vector<int> indices_0, indices_1;
  unsigned image_width = cvGetSize(iplImage).width;
  if(pcl_cloud->width != image_width) {
    printf("[Patches::calculateFourierRelation] Warning: Upscale indices untested!\n");
    indices_0 = UpscaleIndices(pcl_model_cloud_indices[i]->indices, image_width);
    indices_1 = UpscaleIndices(pcl_model_cloud_indices[j]->indices, image_width);
  }
  else {
    indices_0 = pcl_model_cloud_indices[i]->indices;
    indices_1 = pcl_model_cloud_indices[j]->indices;
  }

  _fourierRelation = fourier->compare(indices_0, indices_1);
  return true;
}


bool PatchRelations::calculateGaborRelation(int i, int j, double &_gaborRelation)
{
  if(!have_input_cloud) {
    printf("[Patches::calculateGaborRelation] Error: No input image available. Abort.\n");
    return false;
  }

  if(!have_gabor) {
    // gabor->setDilation(1); // bad results with dilation
//     gabor->setInputImage(matImage);       // TODO Reimplement with cv::Mat instead of iplImage
    gabor->setInputImage(iplImage);
    gabor->compute();
    have_gabor = true;
  }
  
  std::vector<int> mask_0, mask_1;
  unsigned image_width = cvGetSize(iplImage).width;
  if(pcl_cloud->width != image_width) {
    printf("[PatchRelations::calculateGaborRelation] Warning: Upscaling of indices untested!\n");
    mask_0 = UpscaleIndices(pcl_model_cloud_indices[i]->indices, image_width);
    mask_1 = UpscaleIndices(pcl_model_cloud_indices[j]->indices, image_width);
  }
  else {
    mask_0 = surfaces[i]->indices;
    mask_1 = surfaces[j]->indices;
  }
  _gaborRelation = gabor->compare(mask_0, mask_1);
  return true;
}


bool PatchRelations::calculateNormalRelations(int i, int j, double &_nor_mean, double &_nor_var)
{
  if(!compute_feature_normals) {
    printf("[PatchRelations::calculateNormalRelations] Error: No normals calculation enabled. Abort.\n");
    return false;
  }

  _nor_mean = acos(normals_mean[i].dot(normals_mean[j]) / (normals_mean[i].norm() * normals_mean[j].norm()));
  _nor_var = fabs(normals_var[i] - normals_var[j]);
  return true;
}


bool PatchRelations::calculateBoundaryRelations(int i, int j, double &_vs3_col0, double &_vs3_col1, double &_vs3_1, double &_vs3_2, double &_vs3_3, double &_vs3_4)
{
  _vs3_col0 = 1.0;        // Value, if it is a 2D neighbour
  _vs3_col1 = -1.0;
  _vs3_1 = 0.0;           // area support
  _vs3_2 = 1.0;           // TODO unused
  _vs3_3 = 0.0;           // lines support
  _vs3_4 = 1.0;           // gap/lines
  if(Is2DNeigbor(i, j))
    return true;
  
  _vs3_col0 = 2.0;        // Value, if it is no 2D neighbour and no collinearities found

  vs3->NewImage(iplImage);
  vs3->SetBoundarySegments(i, surfaces[i]->contours[0]);
  vs3->SetBoundarySegments(j, surfaces[j]->contours[0]);
  vs3->ProcessBoundarySegments(200);

  /// Collinearities
  int k=0;
  for(unsigned l = 0; l < vs3->Gestalts(Z::Gestalt::COLLINEARITY).Size(); l++) {
    if(((Z::Collinearity*) vs3->Gestalts(Z::Gestalt::COLLINEARITY, l))->line[0]->seg->b_id != 
      ((Z::Collinearity*) vs3->Gestalts(Z::Gestalt::COLLINEARITY, l))->line[1]->seg->b_id) {
      k++;
    
      // Calculate distance of end-point to line
      Eigen::Vector3f p00, p01, p10, p11;
      int near_point_0 = ((Z::Collinearity*) vs3->Gestalts(Z::Gestalt::COLLINEARITY, l))->near_point[0];
      int near_point_1 = ((Z::Collinearity*) vs3->Gestalts(Z::Gestalt::COLLINEARITY, l))->near_point[1];
    
      int x = ((Z::Collinearity*) vs3->Gestalts(Z::Gestalt::COLLINEARITY, l))->line[0]->point[near_point_0].x;
      int y = ((Z::Collinearity*) vs3->Gestalts(Z::Gestalt::COLLINEARITY, l))->line[0]->point[near_point_0].y;
      p00 = pcl_cloud->points[y*pcl_cloud->width+x].getVector3fMap();
      int x1 = ((Z::Collinearity*) vs3->Gestalts(Z::Gestalt::COLLINEARITY, l))->line[0]->point[Other(near_point_0)].x;
      int y1 = ((Z::Collinearity*) vs3->Gestalts(Z::Gestalt::COLLINEARITY, l))->line[0]->point[Other(near_point_0)].y;
      p01 = pcl_cloud->points[y1*pcl_cloud->width+x1].getVector3fMap();
      int x2 = ((Z::Collinearity*) vs3->Gestalts(Z::Gestalt::COLLINEARITY, l))->line[1]->point[near_point_1].x;
      int y2 = ((Z::Collinearity*) vs3->Gestalts(Z::Gestalt::COLLINEARITY, l))->line[1]->point[near_point_1].y;
      p10 = pcl_cloud->points[y2*pcl_cloud->width+x2].getVector3fMap();
      int x3 = ((Z::Collinearity*) vs3->Gestalts(Z::Gestalt::COLLINEARITY, l))->line[1]->point[Other(near_point_1)].x;
      int y3 = ((Z::Collinearity*) vs3->Gestalts(Z::Gestalt::COLLINEARITY, l))->line[1]->point[Other(near_point_1)].y;
      p11 = pcl_cloud->points[y3*pcl_cloud->width+x3].getVector3fMap();

      // Distanz von Endpunkt zu Linie (p00) zu Linie (p10-p11)
      Eigen::Vector3f dir = p11-p10;
      Eigen::Vector3f v0 = p00-p10;
      Eigen::Vector3f v1 = p00-p11;
      double height_dist0 = (v0.cross(dir)).norm()/(dir.norm());
      double height_dist1 = (v1.cross(dir)).norm()/(dir.norm());

      // Add angle between collinearities
      Eigen::Vector3f dir0 = p01-p00;
      double delta_dir = acos(fabs((dir.normalized()).dot(dir0.normalized())))/(M_PI/2.);
      double height_dist = sqrt(height_dist0*height_dist1);
      if((delta_dir + height_dist) < _vs3_col0) {
        _vs3_col0 = delta_dir + height_dist;

        /// occlusion of collinearity feature (z-distance of line to real points)
        unsigned nr_valid_pts = 0;          // number of valid 3d points (if occlusion nan's are in the point cloud)
        double z_dist_mean = 0.0f;          // mean z-distance of hidden line (gap) to orginal point cloud
        Eigen::Vector3f dir3D;              // direction of 3D line
        std::vector<int> x_2D, y_2D;        // x and y of 2D line
        DrawLine(x, y, x2, y2, x_2D, y_2D);
        dir3D = (p10-p00).normalized();
        
        // calculate mean of z-distance from hypothesized 3D line to orignal point cloud data
        for(unsigned m=1; m<x_2D.size()-1; m++) {
          Eigen::Vector3f p3D_org = pcl_cloud->points[y_2D[m]*pcl_cloud->width+x_2D[m]].getVector3fMap();
          if(p3D_org[2] == p3D_org[2]) {
            Eigen::Vector3f p3D = p00 + dir3D*m/x_2D.size();
            z_dist_mean += p3D[2] - p3D_org[2];
            nr_valid_pts++;
          }
        }
        if(nr_valid_pts != 0)
          z_dist_mean /= nr_valid_pts;
        _vs3_col1 = z_dist_mean;
      }
      
    }
    else
      vs3->Gestalts(Z::Gestalt::COLLINEARITY, l)->Mask(1000);
  }
 
  /// Closure Calculations
  int seg_size_i = ((Z::Segment*) vs3->Gestalts(Z::Gestalt::SEGMENT, 0))->edgels.Size();  // number segment pixels
  int seg_size_j = ((Z::Segment*) vs3->Gestalts(Z::Gestalt::SEGMENT, 1))->edgels.Size();
  double area_support = 0.0f;     // support of area of the two surfaces
  for(unsigned l = 0; l < vs3->Gestalts(Z::Gestalt::CLOSURE).Size(); l++) 
  {
    bool found = false;
    for(unsigned m = 0; m < ((Z::Closure*) vs3->Gestalts(Z::Gestalt::CLOSURE, l))->lines.Size(); m++) {
      for(unsigned n = m+1; n < ((Z::Closure*) vs3->Gestalts(Z::Gestalt::CLOSURE, l))->lines.Size(); n++) {
   
        if( ((Z::Closure*) vs3->Gestalts(Z::Gestalt::CLOSURE, l))->lines[m]->seg->b_id != 
            ((Z::Closure*) vs3->Gestalts(Z::Gestalt::CLOSURE, l))->lines[n]->seg->b_id) {
// printf("  Found a closure over both boundaries: %u\n", ((Z::Closure*) vs3->Gestalts(Z::Gestalt::CLOSURE, l))->ID());
          found = true;
          break;
        }
      }
      if(found) break;
    }
    
    // closure over both segments found => line and area support
    if(found) 
    {
      /// calculate line support of closure
      int clos_size_i = 0;
      int clos_size_j = 0;
      for(unsigned m = 0; m < ((Z::Closure*) vs3->Gestalts(Z::Gestalt::CLOSURE, l))->lines.Size(); m++) {
        if( ( ( (Z::Closure*) vs3->Gestalts(Z::Gestalt::CLOSURE, l))->lines[m]->seg->b_id) == i)
          clos_size_i += ((Z::Closure*) vs3->Gestalts(Z::Gestalt::CLOSURE, l))->lines[m]->len;
        if( ( ( (Z::Closure*) vs3->Gestalts(Z::Gestalt::CLOSURE, l))->lines[m]->seg->b_id) == j)
          clos_size_j += ((Z::Closure*) vs3->Gestalts(Z::Gestalt::CLOSURE, l))->lines[m]->len;
      }

      double line_support = ((double)clos_size_i/(double)seg_size_i + (double)clos_size_j/(double)seg_size_j)/2.;      
      if(line_support > _vs3_3) 
      {
        _vs3_3 = line_support;
        
        /// sum_gaps/sum_lines for closure
        double lines = ((Z::Closure*) vs3->Gestalts(Z::Gestalt::CLOSURE, l))->SumLines();
        double gaps = ((Z::Closure*) vs3->Gestalts(Z::Gestalt::CLOSURE, l))->SumGaps();
        _vs3_2 = gaps/lines;        
      }

      // best area support
      /// Area support
      unsigned nr_points_i = surfaces[i]->indices.size();
      unsigned inside_i = 0;
      for(unsigned m=0; m<nr_points_i; m++) {
        VEC::Vector2 p;
        p.x = surfaces[i]->indices[m] % pcl_cloud->width;
        p.y = surfaces[i]->indices[m] / pcl_cloud->width;
        if(((Z::Closure*) vs3->Gestalts(Z::Gestalt::CLOSURE, l))->Inside(p))
          inside_i++;
      }
      unsigned nr_points_j = surfaces[j]->indices.size();
      unsigned inside_j = 0;
      for(unsigned m=0; m<nr_points_j; m++) {
        VEC::Vector2 p;
        p.x = surfaces[j]->indices[m] % pcl_cloud->width;
        p.y = surfaces[j]->indices[m] / pcl_cloud->width;
        if(((Z::Closure*) vs3->Gestalts(Z::Gestalt::CLOSURE, l))->Inside(p))
          inside_j++;
      }

      area_support = (double) (inside_i+inside_j) / (double) (nr_points_i + nr_points_j);
      if(_vs3_1 < area_support)      // best area support
        _vs3_1 = area_support;
// printf(" => Area support: %4.5f\n", area_support);
      
      // smallest gaps/lines
      double lines = ((Z::Closure*) vs3->Gestalts(Z::Gestalt::CLOSURE, l))->SumLines();
      double gaps = ((Z::Closure*) vs3->Gestalts(Z::Gestalt::CLOSURE, l))->SumGaps();
      double gap2lines = gaps/lines;
//   printf("  gaps/lines %4.0f-%4.0f prob: %4.3f\n", gaps, lines, gap2lines);
      if(gap2lines < _vs3_4)
        _vs3_4 = gap2lines;
    }
  }
  
  /// Arcs, convex arc groups and ellipses

  
  /// Result of relations
//   printf("    => vs3_col0: %5.4f\n", _vs3_col0);
//   printf("    => vs3_col1: %4.3f\n", _vs3_col1);
//   printf("    => vs3_1: %4.3f\n", _vs3_1);
//   printf("    => Best gap2lines: vs3_2: %5.5f\n\n", _vs3_2);
   
  
  /// Print all Gestalts
//   for(int i = 0; i < Z::GestaltPrinciple::MAX_TYPE; i++)
//     if(vs3->NumGestalts((Z::Gestalt::Type) i) != 0)
//       printf("  Number of Gestalts before masking [%u]: %u\n", i, vs3->NumGestalts((Z::Gestalt::Type) i));

    
  /// Wieviele L-Junctions gibt es zwischen den verschiedenen Boundaries
//   int k=0;
//   for(unsigned l = 0; l < vs3->Gestalts(Z::Gestalt::L_JUNCTION).Size(); l++) {
//     if(((Z::LJunction*) vs3->Gestalts(Z::Gestalt::L_JUNCTION, l))->line[0]->seg->b_id != 
//       ((Z::LJunction*) vs3->Gestalts(Z::Gestalt::L_JUNCTION, l))->line[1]->seg->b_id) {
//       k++; printf("%u - ", l);
//     }
//     else
//       vs3->Gestalts(Z::Gestalt::L_JUNCTION, l)->Mask(1000);
//   }
//   printf("\nValid L-junction: %i\n", k);

      
  /// T-Junctions
//   k=0;
//   for(unsigned l = 0; l < vs3->Gestalts(Z::Gestalt::T_JUNCTION).Size(); l++) {
//     if(((Z::TJunction*) vs3->Gestalts(Z::Gestalt::T_JUNCTION, l))->line[0]->seg->b_id != 
//       ((Z::TJunction*) vs3->Gestalts(Z::Gestalt::T_JUNCTION, l))->line[2]->seg->b_id) {
//       k++; printf("%u - ", l);
//     }
//     else
//       vs3->Gestalts(Z::Gestalt::T_JUNCTION, l)->Mask(1000);
//   }
//   printf("\nValid T-junction: %i\n", k);

//printf("[Patches::CalculateBoundaryRelation] done\n");
  // Anzeige auf Bildschirm
//   int num = 0;
//   int detail = 0;
//   bool do_it = true;
//   IplImage *drawImage = cvCreateImage(cvSize(640, 480), IPL_DEPTH_8U, 3);
//   while(do_it)
//   {
//     int key = cvWaitKey(50);
//     switch((char) key)
//     {
//       case '-':
//         if(detail > 0)
//           detail--;
//         printf("detail: %u\n", detail);
//         break;
//       case '+':
//         detail++;
//         printf("detail: %u\n", detail);
//         break;        
// 
//         case ',':
//         if(num > 0)
//           num--;
//         printf("num: %u\n", num);
//         break;
//       case '.':
//         num++;
//         printf("num: %u\n", num);
//         break;        
// 
//         
//       case 'a':
//         cvCopy(iplImage, drawImage);
//         Z::SetActiveDrawArea(drawImage);
//         vs3->DrawGestalts(Z::Gestalt::SEGMENT, detail, true);
//         cvShowImage("Debug image", drawImage);
//         break;
//       case 's':
//         cvCopy(iplImage, drawImage);
//         Z::SetActiveDrawArea(drawImage);
//         vs3->DrawGestalts(Z::Gestalt::ARC, detail, true);
//         cvShowImage("Debug image", drawImage);
//         break;
//       case 'd':
//         cvCopy(iplImage, drawImage);
//         Z::SetActiveDrawArea(drawImage);
//         vs3->DrawGestalts(Z::Gestalt::A_JUNCTION, detail, true);
//         cvShowImage("Debug image", drawImage);
//         break;
//       case 'f':
//         cvCopy(iplImage, drawImage);
//         Z::SetActiveDrawArea(drawImage);
//         vs3->DrawGestalts(Z::Gestalt::CONVEX_ARC_GROUP, detail, true);
//         cvShowImage("Debug image", drawImage);
//         break;
//       case 'g':
//         cvCopy(iplImage, drawImage);
//         Z::SetActiveDrawArea(drawImage);
//         vs3->DrawGestalts(Z::Gestalt::ELLIPSE, detail, true);
//         cvShowImage("Debug image", drawImage);
//         break;
//       case 'h':
//         cvCopy(iplImage, drawImage);
//         Z::SetActiveDrawArea(drawImage);
//         vs3->DrawGestalts(Z::Gestalt::CLOSURE, detail, true);
//         cvShowImage("Debug image", drawImage);
//         break;
//         
//         
//       case 'y':
//         cvCopy(iplImage, drawImage);
//         Z::SetActiveDrawArea(drawImage);
//         vs3->DrawGestalt(Z::Gestalt::SEGMENT, num, detail, true);
//         cvShowImage("Debug image", drawImage);
//         break;
//       case 'x':
//         cvCopy(iplImage, drawImage);
//         Z::SetActiveDrawArea(drawImage);
//         vs3->DrawGestalt(Z::Gestalt::LINE, num, detail, true);
//         cvShowImage("Debug image", drawImage);
//         break;
//       case 'c':
//         cvCopy(iplImage, drawImage);
//         Z::SetActiveDrawArea(drawImage);
//         vs3->DrawGestalt(Z::Gestalt::COLLINEARITY, num, detail, true);
//         cvShowImage("Debug image", drawImage);
//         break;
//       case 'v':
//         cvCopy(iplImage, drawImage);
//         Z::SetActiveDrawArea(drawImage);
//         vs3->DrawGestalt(Z::Gestalt::CONVEX_ARC_GROUP, num, detail, true);
//         cvShowImage("Debug image", drawImage);
//         break;
//       case 'b':
//         cvCopy(iplImage, drawImage);
//         Z::SetActiveDrawArea(drawImage);
//         vs3->DrawGestalt(Z::Gestalt::ELLIPSE, num, detail, true);
//         cvShowImage("Debug image", drawImage);
//         break;
//       case 'n':
//         cvCopy(iplImage, drawImage);
//         Z::SetActiveDrawArea(drawImage);
//         vs3->DrawGestalt(Z::Gestalt::CLOSURE, num, detail, true);
//         cvShowImage("Debug image", drawImage);
//         break;
//         
//       case 'w':
//         do_it = false;
//         break;
//     }
//   }
  
  return true;
}


void PatchRelations::DrawLine(int x1, int y1, int x2, int y2, std::vector<int> &_x, std::vector<int> &_y)
{
  int dx, dy, inc_x, inc_y, x, y, err;

  if(!ClipLine(iplImage->width-1, iplImage->height-1, &x1, &y1, &x2, &y2))
    return;
  dx = x2 - x1;
  dy = y2 - y1;
  if(dx == 0 && dy == 0)  // line might be clipped to length 0
    return;
  x = x1;
  y = y1;
  if(dx >= 0)
    inc_x = 1;
  else
  {
    dx = -dx;
    inc_x = -1;
  }
  if(dy >= 0)
    inc_y = 1;
  else
  {
    dy = -dy;
    inc_y = -1;
  }
  // octants 1,4,5,8

  if(dx >= dy)
  {
    // first octant bresenham
    err = -dx/2;
    do
    {
//       SetPixel(x, y, /*type,*/ id);
      _x.push_back(x);
      _y.push_back(y);
      err += dy;
      if(err >= 0)
      {
        y += inc_y;
        err -= dx;
        if(x + inc_x != x2)
        {
          // make line dense
//           SetPixel(x, y, /*type,*/ id);
          _x.push_back(x);
          _y.push_back(y);
        }
      }
      x += inc_x;
    } while(x != x2); // TODO: x2 is not coloured!
  }
  // octants 2,3,6,7
  else // dx < dy
  {
    // second octant bresenham
    err = -dy/2;
    do
    {
//       SetPixel(x, y, /*type,*/ id);
      _x.push_back(x);
      _y.push_back(y);
      err += dx;
      if(err >= 0)
      {
        x += inc_x;
        err -= dy;
        if(y + inc_y != y2) {
          // make line dense
//           SetPixel(x, y, /*type,*/ id);
          _x.push_back(x);
          _y.push_back(y);
        }
      }
      y += inc_y;
    } while(y != y2);
  }
}


// ================================= Public functions ================================= //

void PatchRelations::setZLimit(double _z_max)
{
  z_max = _z_max;
}  

void PatchRelations::setInputCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr & _pcl_cloud)
{
  relations.resize(0);
  neighbors3D.resize(0);
  
  if(_pcl_cloud.get() == 0 || _pcl_cloud->points.size() == 0) {
    printf("[PatchRelations::setInputCloud] Error: Empty or invalid pcl-point-cloud. Abort.\n");
    return;
  } 
  pcl_cloud = _pcl_cloud;
  if(pcl_cloud->width <= 1 || pcl_cloud->height <=1)
    printf("[PatchRelations::setInputCloud] Warning: Point cloud is not ordered.\n");

  // convert pcl cloud to cv::Mat image and to iplImage
  matImage = cv::Mat_<cv::Vec3b>(pcl_cloud->height, pcl_cloud->width); 
  pclA::ConvertPCLCloud2Image(pcl_cloud, matImage);
  
  if(iplImage != 0)
    cvReleaseImageHeader(&iplImage);
  iplImage = cvCreateImageHeader(cvSize(pcl_cloud->width, pcl_cloud->height), IPL_DEPTH_8U, 3);
  *iplImage = matImage;
  //ConvertMat2IplImage(matImage, *iplImage);
    
  have_input_cloud = true;
  have_patches = false;
  have_neighbors = false;
  have_annotation = false;
  have_annotation2 = false;
//   have_texture = false;
  have_preprocessed = false;
  have_learn_relations = false;
  have_border_indices = false;
}


void PatchRelations::setSurfaceModels(std::vector< surface::SurfaceModel::Ptr > & _surfaces)
{
  if(!have_input_cloud) {
  printf("[PatchRelations::setSurfaceModels] Error: Set input cloud with correct resolution. Abort.\n");
  return;
  }
  if(pcl_cloud->width <= 1 || pcl_cloud->height <= 1) {
    printf("[Patches::setSurfaceModels] Error: Input cloud is not a correct matrix. Abort.\n");
    return;
  }

  for(unsigned i=0; i<_surfaces.size(); i++)
    if(_surfaces[i]->type != pcl::SACMODEL_PLANE && _surfaces[i]->type != MODEL_NURBS)
      printf("[PatchRelations::setSurfaceModels] Warning: Unknown patch model type!\n");
  
  patches = cv::Mat_<cv::Vec3b>(pcl_cloud->height, pcl_cloud->width);                   /// TODO Muss kein Vec3b sein => unsigned genügt?
  patches.setTo(0);
  for(unsigned i=0; i<_surfaces.size(); i++) {
    for(unsigned j=0; j<_surfaces[i]->indices.size(); j++) {
      int row = _surfaces[i]->indices[j] / pcl_cloud->width;
      int col = _surfaces[i]->indices[j] % pcl_cloud->width;
      patches.at<cv::Vec3b>(row, col)[0] = i+1;  /// plane 1,2,...,n
    }
  }
  nr_patches = _surfaces.size();
 
  pcl_model_types.resize(_surfaces.size());                                             /// TODO All diese Kopien entfernen und direkt aus surfaces verwenden!!!
  model_coefficients.resize(_surfaces.size());
  pcl_model_cloud_indices.resize(_surfaces.size());

  for(unsigned i=0; i<_surfaces.size(); i++)                                            /// TODO All diese Kopien entfernen und direkt aus surfaces verwenden!!!
  {
    // copy model_coefficients
    pcl_model_types[i] = _surfaces[i]->type;
    if(pcl_model_types[i] < MODEL_NURBS) {
      pcl::ModelCoefficients::Ptr mc (new pcl::ModelCoefficients);
      mc->values = _surfaces[i]->coeffs;
      model_coefficients[i] = mc;
    }
    
    // copy point indices
    pcl::PointIndices::Ptr indices (new pcl::PointIndices);
    indices->indices = _surfaces[i]->indices;
    pcl_model_cloud_indices[i] = indices;
    
    // create pcl_model_normals
//     if(!have_normals)
    //   pcl_model_normals->points = pcl_normals->points;

    pcl_model_normals->points.resize(pcl_cloud->width * pcl_cloud->height);                                     /// TODO is it neccessary to copy the normals?
    for(unsigned j=0; j<_surfaces[i]->indices.size(); j++) {
      pcl_model_normals->points[_surfaces[i]->indices[j]].normal_x = _surfaces[i]->normals[j][0];
      pcl_model_normals->points[_surfaces[i]->indices[j]].normal_y = _surfaces[i]->normals[j][1];
      pcl_model_normals->points[_surfaces[i]->indices[j]].normal_z = _surfaces[i]->normals[j][2];
    }
  }
  surfaces = _surfaces;
  have_patches = true;
}


void PatchRelations::getSurfaceModels(std::vector< surface::SurfaceModel::Ptr > &_surfaces)
{
  if(!have_patches)
    printf("[PatchRelations::getSurfaceModels] Error: No surface models available.\n");
  _surfaces = surfaces;
}

void PatchRelations::setAnnotion(std::vector< std::vector<int> > &_anno,
                          std::vector<int> &_anno_bg_list)
{
  annotation = _anno;
  anno_bg_list = _anno_bg_list;
  have_annotation = true;
}

void PatchRelations::setAnnotion2(std::vector< std::vector<int> > &_anno,
                           std::vector<int> &_anno_bg_list)
{
  annotation2 = _anno;
  anno_bg_list2 = _anno_bg_list;
  have_annotation2 = true;
}

void PatchRelations::setOptimalPatchModels(bool cpm)
{
  compute_patch_models = cpm;
}

void PatchRelations::computeNeighbors()
{
  if(!have_input_cloud || !have_patches) {
    printf("[Patches::computeNeighbors] Error: No input cloud or patches available. Abort.\n");
    return;
  }
  neighbors2D.resize(0);
  neighbors3D.resize(0);

  bool nbgh_matrix3D[nr_patches+1][nr_patches+1];
  bool nbgh_matrix2D[nr_patches+1][nr_patches+1];
  for(unsigned i=0; i<nr_patches+1; i++)
    for(unsigned j=0; j<nr_patches+1; j++) {
      nbgh_matrix3D[i][j] = false;
      nbgh_matrix2D[i][j] = false;
    }
  
  for(int row=1; row<patches.rows; row++) {
    for(int col=1; col<patches.cols; col++) {
      if(patches.at<cv::Vec3b>(row, col)[0] != 0) {
        if(patches.at<cv::Vec3b>(row, col)[0] != patches.at<cv::Vec3b>(row-1, col)[0]) {
          if(patches.at<cv::Vec3b>(row-1, col)[0] != 0) {
            int pos_0 = row*pcl_cloud->width+col;
            int pos_1 = (row-1)*pcl_cloud->width+col;
            double dis = fabs(pcl_cloud->points[pos_0].z - pcl_cloud->points[pos_1].z);
            if( dis < z_max) {
              nbgh_matrix3D[patches.at<cv::Vec3b>(row-1, col)[0]][patches.at<cv::Vec3b>(row, col)[0]] = true;
              nbgh_matrix3D[patches.at<cv::Vec3b>(row, col)[0]][patches.at<cv::Vec3b>(row-1, col)[0]] = true;
            }
            nbgh_matrix2D[patches.at<cv::Vec3b>(row-1, col)[0]][patches.at<cv::Vec3b>(row, col)[0]] = true;
            nbgh_matrix2D[patches.at<cv::Vec3b>(row, col)[0]][patches.at<cv::Vec3b>(row-1, col)[0]] = true;
          }
        }
        if(patches.at<cv::Vec3b>(row, col)[0] != patches.at<cv::Vec3b>(row, col-1)[0]) {
          if(patches.at<cv::Vec3b>(row, col-1)[0] != 0) {
            int pos_0 = row*pcl_cloud->width+col;
            int pos_1 = row*pcl_cloud->width+col-1;
            double dis = fabs(pcl_cloud->points[pos_0].z - pcl_cloud->points[pos_1].z);
            if( dis < z_max) {
              nbgh_matrix3D[patches.at<cv::Vec3b>(row, col-1)[0]][patches.at<cv::Vec3b>(row, col)[0]] = true;
              nbgh_matrix3D[patches.at<cv::Vec3b>(row, col)[0]][patches.at<cv::Vec3b>(row, col-1)[0]] = true;
            }
            nbgh_matrix2D[patches.at<cv::Vec3b>(row, col-1)[0]][patches.at<cv::Vec3b>(row, col)[0]] = true;
            nbgh_matrix2D[patches.at<cv::Vec3b>(row, col)[0]][patches.at<cv::Vec3b>(row, col-1)[0]] = true;
          }
        }
        if(patches.at<cv::Vec3b>(row, col)[0] != patches.at<cv::Vec3b>(row-1, col-1)[0]) {
          if(patches.at<cv::Vec3b>(row-1, col-1)[0] != 0) {
            int pos_0 = row*pcl_cloud->width+col;
            int pos_1 = (row-1)*pcl_cloud->width+col-1;
            double dis = fabs(pcl_cloud->points[pos_0].z - pcl_cloud->points[pos_1].z);
            if( dis < z_max) {
              nbgh_matrix3D[patches.at<cv::Vec3b>(row-1, col-1)[0]][patches.at<cv::Vec3b>(row, col)[0]] = true;
              nbgh_matrix3D[patches.at<cv::Vec3b>(row, col)[0]][patches.at<cv::Vec3b>(row-1, col-1)[0]] = true;
            }
            nbgh_matrix2D[patches.at<cv::Vec3b>(row-1, col-1)[0]][patches.at<cv::Vec3b>(row, col)[0]] = true;
            nbgh_matrix2D[patches.at<cv::Vec3b>(row, col)[0]][patches.at<cv::Vec3b>(row-1, col-1)[0]] = true;
          }
        }
      }
    }
  }
  
  for(unsigned i=1; i<nr_patches+1; i++) { //[Planes::computeNeighbors] Error: No input cloud and patches available. Abort.
    std::vector<unsigned> neighbor;
    for(unsigned j=1; j<nr_patches+1; j++) {
      if(nbgh_matrix3D[i][j])
        neighbor.push_back(j-1);
    }
    neighbors3D.push_back(neighbor);
  }
  for(unsigned i=1; i<nr_patches+1; i++) {
    std::vector<unsigned> neighbor;
    for(unsigned j=1; j<nr_patches+1; j++) {
      if(nbgh_matrix2D[i][j])
        neighbor.push_back(j-1);
    }
    neighbors2D.push_back(neighbor);
  }

  /* TODO: this was the version before MZ changes, and it does not look correct
  // Add 3D neighbors to surface models
  for(unsigned i=0; i< neighbors3D.size(); i++)
    surfaces[i]->neighbors3D = neighbors3D[i];
  */
  for(unsigned i=0; i< neighbors3D.size(); i++)
    surfaces[i]->neighbors3D.insert(neighbors3D[i].begin(), neighbors3D[i].end());

  have_neighbors = true;
}


void PatchRelations::computeLearnRelations()
{
  if(!have_input_cloud || !have_patches) {
    printf("[Patches::computeLearnRelations] Error: No input cloud or patches available.\n");
    return;
  }
  if(!have_annotation) {
    printf("[Patches::computeLearnRelations] Error: No structural level annotation available.\n");
    return;
  }
 
  if(!have_neighbors)
    computeNeighbors();

  if(!have_preprocessed)
    preprocess();
  
  printf("Relations r_st={r_co, r_tr, r_ga, r_fo, r_rs, r_co3, r_cu3, r_di2, r_vd2, r_cv3}\n");

  for(unsigned i=0; i<neighbors3D.size(); i++) {
    for(unsigned j=0; j<neighbors3D[i].size(); j++) {
      bool valid_relation = true;
      int p0 = i;
      int p1 = neighbors3D[i][j];

      if(p0 > p1)                         // check for double entries
        continue;
      
      if(BackgroundRelation(p0, p1))      // check forground-background constraint
        continue;
 
      // check learn size
      if((int) surfaces[p0]->indices.size() < learn_size_1st || (int) surfaces[p1]->indices.size() < learn_size_1st)
        continue;

      double colorSimilarity = hist3D[p0].compare(hist3D[p1]);
      
      double textureRate = 0.0;
      if(!calculateTextureRelation(p0, p1, textureRate))
        valid_relation = false;

      double gaborRelation;
      if(!valid_relation || !calculateGaborRelation(p0, p1, gaborRelation))
        valid_relation = false;
        
      double fourierRelation;
      if(!valid_relation || !calculateFourierRelation(p0, p1, fourierRelation))
        valid_relation = false;
      
      double relSize = std::min((double)surfaces[p0]->indices.size()/(double)surfaces[p1]->indices.size(), 
                                (double)surfaces[p1]->indices.size()/(double)surfaces[p0]->indices.size());

      std::vector<double> border_relations;
      if(!CalculateBorderRelation(p0, p1, border_relations))
        valid_relation = false;

//       double surfRelation;
//       if(!valid_relation || !calculateSurfRelation(p0, p1, surfRelation))
//         valid_relation = false;
//         
//       if(!valid_relation)
//         printf("[PatchRelations::computeLearnRelations] Warning: Relation not valid.\n");
      
      // Learn only relations with at least one foreground patch!
      if(valid_relation)
      {
        Relation r;
        r.groundTruth = haveAnnoRelation(p0, p1);
        r.prediction = -1;
        r.type = 1;                                     // 1st level svm type
        r.id_0 = p0;
        r.id_1 = p1;
  
        r.rel_value.push_back(colorSimilarity);         // r_co ... color similarity (histogram) of the patch
        r.rel_value.push_back(textureRate);             // r_tr ... difference of texture rate
        r.rel_value.push_back(gaborRelation);           // r_ga ... Gabor similarity of patch texture
        r.rel_value.push_back(fourierRelation);         // r_fo ... Fourier similarity of patch texture
        r.rel_value.push_back(relSize);                 // r_rs ... relative patch size difference

        r.rel_value.push_back(border_relations[0]);     // r_co3 ... color similarity on 3D border
// //         r.rel_value.push_back(border_relations[3]);     // curvature of 2D neighboring points
        r.rel_value.push_back(border_relations[4]);     // r_cu3 ... mean curvature of 3D neighboring points
        
        r.rel_value.push_back(border_relations[1]);     // r_di2 ... depth mean value between border points (2D)
        r.rel_value.push_back(border_relations[2]);     // r_vd2 ... depth variance value

        r.rel_value.push_back(border_relations[5]);     // r_cv3 ... curvature variance of 3D neighboring points
       
// //         r.rel_value.push_back(surfRelation);            // Texture similarity from surf features
// //         r.rel_value.push_back(cosDeltaAngle);           // Angle between normals of contour-neighbors
// //         r.rel_value.push_back(distNormal);              // distance in normal direction of surface-neighbors
        relations.push_back(r);

        if(r.groundTruth)
          printf("r_st [%u][%u] (true) : ", r.id_0, r.id_1);
        else
          printf("r_st [%u][%u] (false): ", r.id_0, r.id_1);
        for(unsigned ridx=0; ridx<r.rel_value.size(); ridx++)
          printf("%4.3f-", r.rel_value[ridx]);
        printf("\n");
              }
    }
  }

  have_learn_relations = true;
 
  if(!calculateAssemblyRelations) 
    return;

  if(!have_annotation2) {
    printf("[PatchRelations::computeLearnRelations] Error: No annotation for 2nd level available.\n");
    return;
  }

  printf("Relations 2: {r_co, r_tr, r_ga, r_fo, r_rs, r_md, r_nm, r_nv, r_ac, r_dn, r_cs, r_oc, r_as, r_ls, r_lg} \n"); // [surf] 
  for(unsigned i=0; i<nr_patches; i++) {
    for(unsigned j=i+1; j<nr_patches; j++) {
      int p0 = i; 
      int p1 = j;
      bool valid_relation = true;
      
      if(Is3DNeighbor(p0, p1))            // 3D neighbors in the first level
        continue;

      if((int) surfaces[p0]->indices.size() < learn_size_2nd || (int)surfaces[p1]->indices.size() < learn_size_2nd)
        continue;

      // true relations:
      //   -- true foreground-foreground in the second annotation (non-neighboring)
      // false relations:
      //   -- forground-background in the second annotation
      //   -- not foreground-foreground in the first annotation (same object?)
      if(BackgroundRelation2(p0, p1))      // check forground-background constraint
        continue;

      bool groundTruth = false;
      if(haveAnnoRelation2(p0, p1))
        groundTruth = true;
      else {
        if(haveAnnoRelation(p0, p1))    // is foreground-foreground in the first
           valid_relation = false;
        groundTruth = false;
      }
      double colorSimilarity = hist3D[p0].compare(hist3D[p1]);
      
      double textureRate = 0.0;
      if(!calculateTextureRelation(p0, p1, textureRate))
        valid_relation = false;
        
      double gaborRelation;
      if(!valid_relation || !calculateGaborRelation(p0, p1, gaborRelation))
        valid_relation = false;

      double fourierRelation;
      if(!valid_relation || !calculateFourierRelation(p0, p1, fourierRelation))
        valid_relation = false;

      double relSize = std::min((double)surfaces[p0]->indices.size()/(double)surfaces[p1]->indices.size(), 
                                (double)surfaces[p1]->indices.size()/(double)surfaces[p0]->indices.size());

      float cosDeltaAngle, distNormal, minDist, occl;
      if(!valid_relation || !cnd->compute(surfaces[p0], surfaces[p1], cosDeltaAngle, distNormal, minDist, occl))
        valid_relation = false;

      double nor_mean, nor_var;
      if(!valid_relation || !calculateNormalRelations(p0, p1, nor_mean, nor_var))
        valid_relation = false;

      double vs3_col0, vs3_col1, vs3_1, vs3_2, vs3_3, vs3_4;
      if(!valid_relation || !calculateBoundaryRelations(p0, p1, vs3_col0, vs3_col1, vs3_1, vs3_2, vs3_3, vs3_4))
         valid_relation = false;
 
// // //       double surfRelation;
// // //       if(!valid_relation || !calculateSurfRelation(p0, p1, surfRelation))
// // //         valid_relation = false;
        
      if(!valid_relation)
        printf("[PatchRelations::computeLearnRelations] Warning: Relation not valid.\n");
        
      // Learn only relations with at least one foreground patch!
      if(valid_relation)
      {
        Relation r;
        r.groundTruth = groundTruth; //haveAnnoRelation(p0, p1);
        r.prediction = -1;
        r.type = 2;                                     // 2nd level svm type
        r.id_0 = p0;
        r.id_1 = p1;
        r.rel_value.push_back(colorSimilarity);         // r_co ... color similarity (histogram) of the patch 
        r.rel_value.push_back(textureRate);             // r_tr ... difference of texture rate
        r.rel_value.push_back(gaborRelation);           // r_ga ... Gabor similarity of patch texture
        r.rel_value.push_back(fourierRelation);         // r_fo ... Fourier similarity of patch texture
        r.rel_value.push_back(relSize);                 // r_rs .. Relative size between patches

        r.rel_value.push_back(minDist);                 // r_md ... proximity: minimum distance between two surfaces
        r.rel_value.push_back(nor_mean);                // r_nm ... Mean value of the normals
        r.rel_value.push_back(nor_var);                 // r_nv ... Variance of the normals
        
        r.rel_value.push_back(cosDeltaAngle);           // r_ac ... Angle between normals of contour-neighbors
        r.rel_value.push_back(distNormal);              // r_dn ... distance in normal direction of surface-neighbors
        
        r.rel_value.push_back(vs3_col0);                // r_cs ... Colliniarity distance*angle measurement
        r.rel_value.push_back(vs3_col1);                // r_oc ... Mean depth (occlusion) value of collinearity
        r.rel_value.push_back(vs3_1);                   // r_as ... Closure: max area support
// // //         r.rel_value.push_back(vs3_2);                   // Closure gap-line for vs3_1
        r.rel_value.push_back(vs3_3);                   // r_ls ... Closure: max line support
        r.rel_value.push_back(vs3_4);                   // r_lg ... Closure: amx gap2lines

// // //         r.rel_value.push_back(surfRelation);            // Texture similarity from surf features

        relations.push_back(r);

        if(r.groundTruth)
          printf("r_as [%u][%u](tr) : ", r.id_0, r.id_1);
        else
          printf("r_as [%u][%u](fa): ", r.id_0, r.id_1);
        for(unsigned ridx=0; ridx<r.rel_value.size(); ridx++)
          printf("%4.3f-", r.rel_value[ridx]);
        printf("\n");
      }
    }
  }
  printf("[PatchRelations::computeLearnRelations] Computing relations: done.\n");
}


void PatchRelations::computeTestRelations()
{
printf("PatchRelations::computeTestRelations: start\n");
  if(!have_input_cloud || !have_patches) {
    printf("[PatchRelations::computeTestRelations] Error: No input cloud and patches available.\n");
    return;
  }

  if(!have_neighbors)
    computeNeighbors();

  if(!have_preprocessed)
    preprocess();
  
printf("PatchRelations::computeTestRelations: 1\n");

  if(calculateStructuralRelations)
    for(unsigned i=0; i<neighbors3D.size(); i++) 
    {
      for(unsigned j=0; j<neighbors3D[i].size(); j++) 
      {
        bool valid_relation = true;
        int p0 = i;
        int p1 = neighbors3D[i][j];

        if(p0 > p1) 
          continue;
        
        double colorSimilarity = hist3D[p0].compare(hist3D[p1]);
        
        double textureRate = 0.0;
        if(!calculateTextureRelation(p0, p1, textureRate))
          valid_relation = false;

        double gaborRelation;
        if(!valid_relation || !calculateGaborRelation(p0, p1, gaborRelation))
          valid_relation = false;

        double fourierRelation;

        if(!valid_relation || !calculateFourierRelation(p0, p1, fourierRelation))
          valid_relation = false;
        
        double relSize = std::min((double)surfaces[p0]->indices.size()/(double)surfaces[p1]->indices.size(), 
                                  (double)surfaces[p1]->indices.size()/(double)surfaces[p0]->indices.size());

        std::vector<double> border_relations;
        if(!CalculateBorderRelation(p0, p1, border_relations))
          valid_relation = false;
  // 
  // //       double surfRelation;
  // //       if(!valid_relation || !calculateSurfRelation(p0, p1, surfRelation))
  // //         valid_relation = false;
        

        if(!valid_relation)
          printf("[PatchRelations::computeTestRelations] Warning: Relation not valid.\n");
        
        // Learn only relations with at least one foreground patch!
        if(valid_relation)
        {
          Relation r;
          if(BackgroundRelation(p0, p1))
            r.groundTruth = -1;
          else
            r.groundTruth = haveAnnoRelation(p0, p1);
          r.prediction = -1;
          r.type = 1;                                     // 1st level svm type
          r.id_0 = p0;
          r.id_1 = p1;

          r.rel_value.push_back(colorSimilarity);         // r_co ... color similarity (histogram) of the patch
          r.rel_value.push_back(textureRate);             // r_tr ... difference of texture rate
          r.rel_value.push_back(gaborRelation);           // r_ga ... Gabor similarity of patch texture
          r.rel_value.push_back(fourierRelation);         // r_fo ... Fourier similarity of patch texture
          r.rel_value.push_back(relSize);                 // r_rs ... relative patch size difference

          r.rel_value.push_back(border_relations[0]);     // r_co3 ... color similarity on 3D border
//   //         r.rel_value.push_back(border_relations[3]);     // curvature of 2D neighboring points
          r.rel_value.push_back(border_relations[4]);     // r_cu3 ... mean curvature of 3D neighboring points
          
          r.rel_value.push_back(border_relations[1]);     // r_di2 ... depth mean value between border points (2D)
          r.rel_value.push_back(border_relations[2]);     // r_vd2 ... depth variance value

          r.rel_value.push_back(border_relations[5]);     // r_cu3 ... curvature variance of 3D neighboring points

          relations.push_back(r);
        
          /// print patch relations
          printf("r_st [%u][%u]", r.id_0, r.id_1);
          if(r.groundTruth == 1)
            printf(" (tr)  => ");
          else if(r.groundTruth == 0)
            printf(" (fa) => ");
          else
            printf(" (uk) => ");
          for(unsigned ridx=0; ridx<r.rel_value.size(); ridx++)
            printf("%4.3f  ", r.rel_value[ridx]);
          printf("\n");
        }
      }
    }
  
printf("PatchRelations::computeTestRelations: 2\n");

  if(!calculateAssemblyRelations) 
    return;
  
  for(unsigned i=0; i<nr_patches; i++) 
  {
    for(unsigned j=i+1; j<nr_patches; j++) 
    {
      int p0 = i; 
      int p1 = j;
      bool valid_relation = true;
      
      if(Is3DNeighbor(p0, p1))
        continue;
      
      if((int) surfaces[p0]->indices.size() < train_size_2nd || (int)surfaces[p1]->indices.size() < train_size_2nd)
        continue;
     
      double colorSimilarity = hist3D[p0].compare(hist3D[p1]);
      
      double textureRate = 0.0;
      if(!calculateTextureRelation(p0, p1, textureRate))
        valid_relation = false;

      double gaborRelation;
      if(!valid_relation || !calculateGaborRelation(p0, p1, gaborRelation))
        valid_relation = false;
        
      double fourierRelation;
      if(!valid_relation || !calculateFourierRelation(p0, p1, fourierRelation))
        valid_relation = false;

      double relSize = std::min((double)surfaces[p0]->indices.size()/(double)surfaces[p1]->indices.size(), 
                                (double)surfaces[p1]->indices.size()/(double)surfaces[p0]->indices.size());

      float cosDeltaAngle, distNormal, minDist, occl;
      if(!valid_relation || !cnd->compute(surfaces[p0], surfaces[p1], cosDeltaAngle, distNormal, minDist, occl))
        valid_relation = false;

      double nor_mean, nor_var;
      if(!valid_relation || !calculateNormalRelations(p0, p1, nor_mean, nor_var))
        valid_relation = false;
      
      double vs3_col0, vs3_col1, vs3_1, vs3_2, vs3_3, vs3_4;
      if(!valid_relation || !calculateBoundaryRelations(p0, p1, vs3_col0, vs3_col1, vs3_1, vs3_2, vs3_3, vs3_4))
         valid_relation = false;

// // //       double surfRelation;
// // //       if(!valid_relation || !calculateSurfRelation(p0, p1, surfRelation))
// // //         valid_relation = false;

      bool groundTruth = false;
      if(haveAnnoRelation2(p0, p1))
        groundTruth = true;
      else {
        if(haveAnnoRelation(p0, p1))    // is foreground-foreground in the first
           groundTruth = -1;
        groundTruth = false;
      }
      
      if(valid_relation) // if relation not valid, ignore!
      {
        Relation r;
        r.groundTruth = groundTruth;
        r.prediction = -1;
        r.type = 2;                                     // 2nd level svm type
        r.id_0 = p0;
        r.id_1 = p1;

        r.rel_value.push_back(colorSimilarity);         // color similarity (histogram) of the patch 
        r.rel_value.push_back(textureRate);             // difference of texture rate
        r.rel_value.push_back(gaborRelation);           // Gabor similarity of patch texture
        r.rel_value.push_back(fourierRelation);         // Fourier similarity of patch texture
        r.rel_value.push_back(relSize);                 // Relative size between patches

        r.rel_value.push_back(minDist);                 // proximity: minimum distance between two surfaces
        r.rel_value.push_back(nor_mean);                // Mean value of the normals
        r.rel_value.push_back(nor_var);                 // Variance of the normals
        r.rel_value.push_back(cosDeltaAngle);           // Angle between normals of contour-neighbors
        r.rel_value.push_back(distNormal);              // distance in normal direction of surface-neighbors

        r.rel_value.push_back(vs3_col0);                // Colliniarity distance*angle measurement
        r.rel_value.push_back(vs3_col1);                // Mean depth (occlusion) value of collinearity
        r.rel_value.push_back(vs3_1);                   // Closure: line support
// // //         r.rel_value.push_back(vs3_2);                   // Closure gap-line for vs3_1
        r.rel_value.push_back(vs3_3);                   // Closure gap-line max
        r.rel_value.push_back(vs3_4);                   // Closure: area support

// // // //         r.rel_value.push_back(surfRelation);            // Texture similarity from surf features
        relations.push_back(r);
        
        if(r.groundTruth == 1)
          printf("r_as [%u][%u](tr): ", r.id_0, r.id_1);
        else if(r.groundTruth == 0)
          printf("r_as [%u][%u](fa): ", r.id_0, r.id_1);
        else
          printf("r_as [%u][%u](uk): ", r.id_0, r.id_1);
        for(unsigned ridx=0; ridx<r.rel_value.size(); ridx++)
          printf("%4.3f  ", r.rel_value[ridx]);
        printf("\n");
      }
      else       
        printf("[PatchRelations::computeTestRelations] Warning: Relation not valid.\n");

      printf("[PatchRelations::computeTestRelations] 2nd level: %u-%u done\n", i, j);
    }
  }
}


void PatchRelations::computeSegmentRelations()
{
  if(!have_input_cloud || !have_patches) {
    printf("[PatchRelations::computeSegmentRelations] Error: No input cloud and patches available.\n");
    return;
  }

  if(!have_neighbors)
    computeNeighbors();

  if(!have_preprocessed)
    preprocess();
  
  if(calculateStructuralRelations) {

    unsigned i,j;
    bool valid_relation;
    
//     #pragma omp parallel for //private(j)
    for(i=0; i<neighbors3D.size(); i++) {
      for(j=0; j<neighbors3D[i].size(); j++) {
        valid_relation = true;
        int p0 = i;
        int p1 = neighbors3D[i][j];

        if(p0 > p1) 
          continue;

	double colorSimilarity = hist3D[p0].compare(hist3D[p1]);
        
        double textureRate = 0.0;
        if(!calculateTextureRelation(p0, p1, textureRate))
          valid_relation = false;
	
        double fourierRelation;
        if(!valid_relation || !calculateFourierRelation(p0, p1, fourierRelation))
          valid_relation = false;

	double relSize = std::min((double)surfaces[p0]->indices.size()/(double)surfaces[p1]->indices.size(), 
                                  (double)surfaces[p1]->indices.size()/(double)surfaces[p0]->indices.size());

        std::vector<double> border_relations;
        if(!CalculateBorderRelation(p0, p1, border_relations))
          valid_relation = false;

        if(!valid_relation)
          printf("[PatchRelations::computeSegmentRelations] Warning: Relation not valid.\n");

        double gaborRelation;
// #pragma omp critical
        {
          if(!valid_relation || !calculateGaborRelation(p0, p1, gaborRelation))
            valid_relation = false;
        }
// #pragma omp end critical

// printf("i-j: %u-%u => %u-%u with thread %i: %4.2f-%4.2f-%4.2f-%4.2f-%4.2f-%4.2f-%4.2f-%4.2f-%4.2f-%4.2f\n", i, j, p0, p1, omp_get_thread_num(), colorSimilarity, textureRate, gaborRelation, fourierRelation, relSize, border_relations[0], border_relations[4], border_relations[1], border_relations[2], border_relations[5]);

	// Learn only relations with at least one foreground patch!
        if(valid_relation)
        {
          Relation r;
          r.groundTruth = -1;
          r.prediction = -1;
          r.type = 1;                                     // 1st level svm type
          r.id_0 = p0;
          r.id_1 = p1;

          r.rel_value.push_back(colorSimilarity);         // r_co ... color similarity (histogram) of the patch
          r.rel_value.push_back(textureRate);             // r_tr ... difference of texture rate
          r.rel_value.push_back(gaborRelation);           // r_ga ... Gabor similarity of patch texture
          r.rel_value.push_back(fourierRelation);         // r_fo ... Fourier similarity of patch texture
          r.rel_value.push_back(relSize);                 // r_rs ... relative patch size difference

          r.rel_value.push_back(border_relations[0]);     // r_co3 ... color similarity on 3D border
          r.rel_value.push_back(border_relations[4]);     // r_cu3 ... mean curvature of 3D neighboring points
          r.rel_value.push_back(border_relations[1]);     // r_di2 ... depth mean value between border points (2D)
          r.rel_value.push_back(border_relations[2]);     // r_vd2 ... depth variance value
          r.rel_value.push_back(border_relations[5]);     // r_cu3 ... curvature variance of 3D neighboring points

// #pragma omp critical
          relations.push_back(r);
// #pragma omp end critical
        
#ifdef DEBUG
          printf("r_st: [%u][%u]: ", r.id_0, r.id_1);
          for(unsigned ridx=0; ridx<r.rel_value.size(); ridx++)
            printf("%4.3f ", r.rel_value[ridx]);
          printf("\n");
#endif
	}
      }
    }
  }
  
  if(calculateAssemblyRelations) {
    unsigned i, j;
    
//     #pragma omp parallel for// private(j)
    for(i=0; i<nr_patches; i++) {
      for(j=i+1; j<nr_patches; j++) {
        int p0 = i; 
        int p1 = j;
        bool valid_relation = true;
        
        if(Is3DNeighbor(p0, p1))            // check 3D neighborhood
          continue;
        
        if((int) surfaces[p0]->indices.size() < train_size_2nd || (int)surfaces[p1]->indices.size() < train_size_2nd)
          continue;  
      
        double colorSimilarity = hist3D[p0].compare(hist3D[p1]);
        
        double textureRate = 0.0;
        if(!calculateTextureRelation(p0, p1, textureRate))
          valid_relation = false;

        double fourierRelation;
        if(!valid_relation || !calculateFourierRelation(p0, p1, fourierRelation))
          valid_relation = false;

        double relSize = std::min((double)surfaces[p0]->indices.size()/(double)surfaces[p1]->indices.size(), 
                                  (double)surfaces[p1]->indices.size()/(double)surfaces[p0]->indices.size());

        float cosDeltaAngle, distNormal, minDist, occl;
        if(!valid_relation || !cnd->compute(surfaces[p0], surfaces[p1], cosDeltaAngle, distNormal, minDist, occl))
          valid_relation = false;

        double nor_mean, nor_var;
        if(!valid_relation || !calculateNormalRelations(p0, p1, nor_mean, nor_var))
          valid_relation = false;

        double vs3_col0, vs3_col1, vs3_1, vs3_2, vs3_3, vs3_4;
        double gaborRelation;

// #pragma omp critical
        {
        if(!valid_relation || !calculateGaborRelation(p0, p1, gaborRelation))
            valid_relation = false;

        if(!valid_relation || !calculateBoundaryRelations(p0, p1, vs3_col0, vs3_col1, vs3_1, vs3_2, vs3_3, vs3_4))
            valid_relation = false;
        }
// #pragma omp end critical

        if(valid_relation) // if relation not valid, ignore!
        {
          Relation r;
          r.groundTruth = -1;
          r.prediction = -1;
          r.type = 2;                                     // assembly level svm type
          r.id_0 = p0;
          r.id_1 = p1;

          r.rel_value.push_back(colorSimilarity);         // color similarity (histogram) of the patch 
          r.rel_value.push_back(textureRate);             // difference of texture rate
          r.rel_value.push_back(gaborRelation);           // Gabor similarity of patch texture
          r.rel_value.push_back(fourierRelation);         // Fourier similarity of patch texture
          r.rel_value.push_back(relSize);                 // Relative size between patches

          r.rel_value.push_back(minDist);                 // proximity: minimum distance between two surfaces
          r.rel_value.push_back(nor_mean);                // Mean value of the normals
          r.rel_value.push_back(nor_var);                 // Variance of the normals
          r.rel_value.push_back(cosDeltaAngle);           // Angle between normals of contour-neighbors
          r.rel_value.push_back(distNormal);              // distance in normal direction of surface-neighbors

          r.rel_value.push_back(vs3_col0);                // Colliniarity distance*angle measurement
          r.rel_value.push_back(vs3_col1);                // Mean depth (occlusion) value of collinearity
          r.rel_value.push_back(vs3_1);                   // Closure: line support
          r.rel_value.push_back(vs3_3);                   // Closure gap-line max
          r.rel_value.push_back(vs3_4);                   // Closure: area support

// #pragma omp critical
          relations.push_back(r);
// #pragma omp end critical
          
#ifdef DEBUG        
          printf("r_as: [%u][%u]: ", r.id_0, r.id_1);
          for(unsigned ridx=0; ridx<r.rel_value.size(); ridx++)
            printf("%4.3f ", r.rel_value[ridx]);
          printf("\n");
#endif
        }
        else       
          printf("[PatchRelations::computeSegmentRelations] Warning: Relation not valid.\n");
      }
    }
  }
}

void PatchRelations::printNeighbors()
{
  if(!have_neighbors) {
    printf("[PatchRelations::printNeighbors] Error: No neighbors available.\n");
    return;
  }
  printf("[PatchRelations::printNeighbors] neighbors2D:\n");
  for(unsigned i=0; i<neighbors2D.size(); i++) {
    printf("  %u: ", i);
    for(unsigned j=0; j<neighbors2D[i].size(); j++) {
      printf(" %u ", neighbors2D[i][j]);
    }
    printf("\n");
  }
  printf("[PatchRelations::printNeighbors] neighbors3D:\n");
  for(unsigned i=0; i<neighbors3D.size(); i++) {
    printf("  %u: ", i);
    for(unsigned j=0; j<neighbors3D[i].size(); j++) {
      printf(" %u ", neighbors3D[i][j]);
    }
    printf("\n");
  }    
}

}





