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
 * @file UniformSurfaces.cpp
 * @author Andreas Richtsfeld
 * @date December 2012
 * @version 0.1
 * @brief Get uniform patches from depth image by detection of discontinuities.
 */

#include "UniformSurfaces.h"

namespace surface 
{
  
double timespec_diff5(struct timespec *x, struct timespec *y)
{
  if(x->tv_nsec < y->tv_nsec)
  {
    int nsec = (y->tv_nsec - x->tv_nsec) / 1000000000 + 1;
    y->tv_nsec -= 1000000000 * nsec;
    y->tv_sec += nsec;
  }
  if(x->tv_nsec - y->tv_nsec > 1000000000)
  {
    int nsec = (y->tv_nsec - x->tv_nsec) / 1000000000;
    y->tv_nsec += 1000000000 * nsec;
    y->tv_sec -= nsec;
  }
  return (double)(x->tv_sec - y->tv_sec) +
    (double)(x->tv_nsec - y->tv_nsec)/1000000000.;
}
  

/********************** UniformSurfaces ************************
 * Constructor/Destructor
 */
// UniformSurfaces::UniformSurfaces(Parameter p)
UniformSurfaces::UniformSurfaces()
{
  initialized = false;
//   setParameter(p);
}

UniformSurfaces::~UniformSurfaces()
{
}



/************************** PRIVATE ************************/

void UniformSurfaces::Initialize()
{
  cv::Mat_<float> camR = cv::Mat::eye(3, 3, CV_32F);
  cv::Mat_<float> camT = cv::Mat::zeros(3, 1, CV_32F);
  cv::Mat_<double> camI = cv::Mat::zeros(3, 3, CV_64F);
  camI.at<double> (0, 0) = camI.at<double> (1, 1) = 525;
  camI.at<double> (0, 2) = 320;
  camI.at<double> (1, 2) = 240;
  camI.at<double> (2, 2) = 1;
  
  m_engine = new TomGine::tgEngine(pcl_cloud->width, pcl_cloud->height);
  glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
  m_engine->SetCamera(camI, pcl_cloud->width, pcl_cloud->height, camR, camT);
  m_ip = new TomGine::tgImageProcessor(pcl_cloud->width, pcl_cloud->height);
//   m_ip->setFBO(true);
}


void UniformSurfaces::NormaliseDepthAndCurvature(const cv::Mat_<cv::Vec4f> &cloud, 
                                                 const cv::Mat_<cv::Vec4f> &normals,
                                                 cv::Mat_<float> &depth, 
                                                 cv::Mat_<float> &curvature)
{
  float depth_max(4.0);
  float depth_min(0.0);      // INFINITY);
  float max_curv(1.57f);
  float min_curv(0.0);          // INFINITY);
//   for( int j = 0; j < cloud.cols; j++ ) {
//     for( int i = 0; i < cloud.rows; i++ ) {
//       if( depth_max < cloud(i, j)[2] )
//         depth_max = cloud(i, j)[2];
//       if( depth_min > cloud(i, j)[2] )
//         depth_min = cloud(i, j)[2];
// 
//       cv::Vec4f n = normals(i, j);
//       float &curv = curvature(i, j);
//       curv = n[3];
//       if( max_curv < curv )
//         max_curv = curv;
//       if( min_curv > curv )
//         min_curv = curv;
//     }
//   }

  float dcurv = 1.0f / (max_curv - min_curv);
  float ddepth = 1.0f / (depth_max - depth_min);
  for( int j = 0; j < cloud.cols; j++ ) {
    for( int i = 0; i < cloud.rows; i++ ) {
      depth(i, j) = (cloud(i, j)[2] - depth_min) * ddepth;

      cv::Vec4f n = normals(i, j);
      float &curv = curvature(i, j);
      curv = n[3];
      if( n[0] == 0.0f && n[1] == 0.0f && n[2] == 0.0f )
        curv = 0.0f;
      else
        curv = (curv - min_curv) * dcurv;
    }
  }
}


void UniformSurfaces::EdgeDetectionRGBDC(TomGine::tgEngine* engine,
                                         TomGine::tgImageProcessor* m_ip,
                                         const cv::Mat_<cv::Vec3b> &image, 
                                         const cv::Mat_<float> &depth,
                                         cv::Mat_<float> &curvature, 
                                         const cv::Mat_<uchar> &mask, 
                                         cv::Mat_<float> &color_edges,
                                         cv::Mat_<float> &depth_edges, 
                                         cv::Mat_<float> &curvature_edges, 
                                         cv::Mat_<float> &mask_edges, 
                                         cv::Mat_<float> &edges)
                                          
{  
  TomGine::tgTexture2D texEdgesDepth, texEdgesCurvature, texEdges, texZero;

  /// depth edges
  TomGine::tgTexture2D texDepth;
  texDepth.Load(&depth(0, 0), depth.cols, depth.rows, GL_LUMINANCE, GL_LUMINANCE, GL_FLOAT);
  m_ip->sobelAdaptive(texDepth, texEdgesDepth, 0.028f, 0.0, true, true);
//   texEdgesDepth.Bind();
//   glGetTexImage(GL_TEXTURE_2D, 0, GL_BLUE, GL_FLOAT, &depth_edges(0, 0));
  
  /// curvature edges
  TomGine::tgTexture2D texCurvature;
  texCurvature.Load(&curvature(0, 0), curvature.cols, curvature.rows, GL_LUMINANCE, GL_LUMINANCE, GL_FLOAT);
  m_ip->gauss(texEdgesCurvature, texEdgesCurvature);
  m_ip->add(texCurvature, texZero, texEdgesCurvature, 20.0);
//   texEdgesCurvature.Bind();
//   glGetTexImage(GL_TEXTURE_2D, 0, GL_BLUE, GL_FLOAT, &curvature(0, 0));
//   m_ip->threshold(texEdgesCurvature, texEdgesCurvature, 0.4);
  texEdgesCurvature.Bind();
  glGetTexImage(GL_TEXTURE_2D, 0, GL_BLUE, GL_FLOAT, &curvature_edges(0, 0));
  
  // curvature NEW
//   TomGine::tgTexture2D normals, conv_normals;
//   normals.Load(cv_normals.data, cv_normals.cols, cv_normals.rows, GL_RGB32F, GL_RGB, GL_FLOAT);
//   m_ip->dotConvolute(normals, conv_normals, kernel, 0.0f);
  TomGine::tgTexture2D tex_normals, tex_normals2, tex_curvature, tex_curvature2;
  
  /// KERNEL
  std::vector<float> kernel;    /// TODO Kernel später wo anders anlegen
  kernel.resize(25);
  for(unsigned i=0; i<25; i++)
    kernel[i] = (0.0);
  kernel[6] =  (1.0);   kernel[7] = (2.0);   kernel[8] = (1.0);
  kernel[11] = (0.0);  kernel[12] = (0.0);  kernel[13] = (0.0);
  kernel[16] = (1.0);  kernel[17] = (2.0);  kernel[18] = (1.0);
//   kernel[0] =  (.0);  kernel[1] =  (.0);  kernel[2] =  (.0);  kernel[3] =  (.0);  kernel[4] =  (.0);
//   kernel[5] =  (.0);  kernel[6] =  (2.0);  kernel[7] =  (2.0);  kernel[8] =  (2.0);  kernel[9] =  (.0);
//   kernel[10] = (.0);  kernel[11] = (2.0);  kernel[12] = (0.0);  kernel[13] = (2.0);  kernel[14] = (.0);
//   kernel[15] = (.0);  kernel[16] = (2.0);  kernel[17] = (2.0);  kernel[18] = (2.0);  kernel[19] = (.0);
//   kernel[20] = (.0);  kernel[21] = (.0);  kernel[22] = (.0);  kernel[23] = (.0);  kernel[24] = (.0);
  float kernel_sum = 0.0f;
  for(unsigned i=0; i<25; i++)
    kernel_sum += kernel[i];
  for(unsigned i=0; i<25; i++)
    kernel[i] /= kernel_sum;
    
  /// KERNEL 2
  std::vector<float> kernel2;    /// TODO Kernel später wo anders anlegen
  kernel2.resize(25);
  for(unsigned i=0; i<25; i++)
    kernel2[i] = (0.0);
  kernel2[6] =  (1.0);   kernel2[7] = (0.0);   kernel2[8] = (1.0);
  kernel2[11] = (2.0);  kernel2[12] = (0.0);  kernel2[13] = (2.0);
  kernel2[16] = (1.0);  kernel2[17] = (0.0);  kernel2[18] = (1.0);
//   kernel2[0] =  (.0);  kernel2[1] =  (.0);  kernel2[2] =  (.0);  kernel2[3] =  (.0);  kernel2[4] =  (.0);
//   kernel2[5] =  (.0);  kernel2[6] =  (2.0);  kernel2[7] =  (2.0);  kernel2[8] =  (2.0);  kernel2[9] =  (.0);
//   kernel2[10] = (.0);  kernel2[11] = (2.0);  kernel2[12] = (0.0);  kernel2[13] = (2.0);  kernel2[14] = (.0);
//   kernel2[15] = (.0);  kernel2[16] = (2.0);  kernel2[17] = (2.0);  kernel2[18] = (2.0);  kernel2[19] = (.0);
//   kernel2[20] = (.0);  kernel2[21] = (.0);  kernel2[22] = (.0);  kernel2[23] = (.0);  kernel2[24] = (.0);
  float kernel2_sum = 0.0f;
  for(unsigned i=0; i<25; i++)
    kernel2_sum += kernel2[i];
  for(unsigned i=0; i<25; i++)
    kernel2[i] /= kernel2_sum;
    
  /// Calculate curvature NEW
  tex_normals.Load(cv_normals.data, cv_normals.cols, cv_normals.rows, GL_RGBA32F, GL_RGBA, GL_FLOAT);
  tex_curvature.Load(0, cv_normals.cols, cv_normals.rows, GL_RGBA32F, GL_RGB, GL_FLOAT);
  m_ip->setFBO(true);
//   m_ip->copy(tex_curvature, tex_curvature2);
  
  /// CURVATURE: sobel, thinning, thresholding
  m_ip->sobel(tex_normals, tex_curvature, 0.1, true);
  m_ip->thinning(tex_curvature, tex_curvature, true, false);
  m_ip->threshold(tex_curvature, tex_curvature2, 0.1);
//   m_ip->spreading(tex_curvature2, tex_curvature2, true, 1.0f, 0.0f);
//   m_ip->dilate(tex_curvature2, tex_curvature2, 1.0);

//     /// CURVATURE: sobel, no-thinning?, thresholding
//   m_ip->sobel(tex_normals, tex_curvature, 0.1, true);
//   m_ip->thinning(tex_curvature, tex_curvature, true, false);
//   m_ip->threshold(tex_curvature, tex_curvature2, 0.1);
//   m_ip->spreading(tex_curvature2, tex_curvature2, true, 1.0f, 0.0f);
  
  
//   m_ip->dotConvolute(tex_normals, tex_curvature, kernel, 0.0f);              // kernel 1
//   m_ip->invert(tex_curvature, tex_curvature);
// //   m_ip->threshold(tex_curvature, tex_curvature, 0.2);
//   m_ip->add(tex_curvature, texZero, tex_curvature, 50.0, 0.0);
  
  
//   m_ip->dotConvolute(tex_normals, tex_curvature2, kernel2, 0.0f);            // kernel 2
//   m_ip->invert(tex_curvature2, tex_curvature2);
// //   m_ip->threshold(tex_curvature, tex_curvature, 0.2);
//   m_ip->add(tex_curvature2, texZero, tex_curvature2, 50.0, 0.0);

//   m_ip->add(tex_curvature, tex_curvature2, tex_curvature2, 50.0, 0.0);       // add

  tex_curvature.Bind();
  glGetTexImage(GL_TEXTURE_2D, 0, GL_BLUE, GL_FLOAT, &curvature(0, 0));
  tex_curvature2.Bind();
  glGetTexImage(GL_TEXTURE_2D, 0, GL_BLUE, GL_FLOAT, &mask_edges(0, 0));
  
  
  
  /// depth edge + mask edge
  m_ip->add(texEdgesDepth, tex_curvature2, texEdges);
//   m_ip->add(texEdgesDepth, texEdgesCurvature2, texEdges);
  texEdges.Bind();
  glGetTexImage(GL_TEXTURE_2D, 0, GL_BLUE, GL_FLOAT, &edges(0, 0));  
  

  /// TODO Remove mask edges from depth edges
//   TomGine::tgTexture2D texMask;
//   texMask.Load(&mask(0, 0), mask.cols, mask.rows, GL_LUMINANCE, GL_LUMINANCE, GL_UNSIGNED_BYTE);
//   m_ip->invert(texMask, texMaskInvert);
//   m_ip->sobel(texDepth, texEdgesDepth, texMaskInvert, 0.0f, false, true);
//   m_ip->multiply(texEdgesDepth, 15.0f, texEdgesDepth);
//   texEdgesDepth.Bind();
//   glGetTexImage(GL_TEXTURE_2D, 0, GL_BLUE, GL_FLOAT, &depth_edges(0, 0));
//   //   glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_RGB, &gradients(0, 0));

  /// mask edges
//   m_ip->sobel(texMask, texEdgesMask, 0.03f, true, true);
//   texEdgesMask.Bind();
//   glGetTexImage(GL_TEXTURE_2D, 0, GL_BLUE, GL_FLOAT, &mask_edges(0, 0));
  
  /// color edges
//   TomGine::tgTexture2D texColor;
//   texColor.Load(&image(0, 0), image.cols, image.rows, GL_RGB, GL_RGB, GL_UNSIGNED_BYTE);
//   m_ip->gauss(texColor, texEdgesGauss);
//   m_ip->sobel(texEdgesGauss, texEdgesColor);
//   m_ip->thinning(texEdgesColor, texEdgesColorThin);
//   m_ip->add(texEdgesColorThin, texZero, texEdgesColorThin, 7);
//   texEdgesColorThin.Bind();
//   glGetTexImage(GL_TEXTURE_2D, 0, GL_BLUE, GL_FLOAT, &color_edges(0, 0));

}

void UniformSurfaces::RecursiveClustering(const cv::Mat_<float> &_edges)
{
  std::queue< std::pair<int, int> > queue;    
  std::pair<int, int> p[4];
  p[0] = std::make_pair<int, int>(-1, 0);
  p[1] = std::make_pair<int, int>( 1, 0);
  p[2] = std::make_pair<int, int>( 0,-1);
  p[3] = std::make_pair<int, int>( 0, 1);
  std::vector<int> patch;
  cv::Mat_<float> patch_mask;
  _edges.copyTo(patch_mask);
   
  CopyToLabeledCloud(pcl_cloud, pcl_cloud_labeled);
          
  int patch_idx = 0;
  for(unsigned row=0; row<pcl_cloud->height; row++) {
    for(unsigned col=0; col<pcl_cloud->width; col++) {
      int idx = GetIdx(col, row);
      if(!(patch_mask(row, col) >= 0.001) && !isnan(pcl_cloud->points[idx].x)) {
        patch_idx++;
        pcl_cloud_labeled->points[idx].label = patch_idx;
        patch.clear();
        patch.push_back(idx);
        patch_mask(row, col) = 1;
        queue.push(std::make_pair<int, int>(col, row));
        while (queue.size()>0) {
          std::pair<int, int> x_y = queue.front();
          int x = x_y.first;
          int y = x_y.second;
          queue.pop();

          for(unsigned i=0; i<4; i++) {
            int u = x + p[i].first;
            int v = y + p[i].second;
            if (v>=0 && u>=0 && v<(int)pcl_cloud->height && u<(int)pcl_cloud->width) {
              idx = GetIdx(u, v);
              if (!(patch_mask(v, u) >= 0.001)) {
                if(!isnan(pcl_cloud->points[idx].x)) {
                  patch_mask(v, u) = 1;
                  patch.push_back(idx);
                  queue.push(std::make_pair<int, int>(u, v));
                  pcl_cloud_labeled->points[idx].label = patch_idx;
                }
              }
            }
          }
        }
      }
    }
  }
}

void UniformSurfaces::CopyToLabeledCloud(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &_in, 
                          pcl::PointCloud<pcl::PointXYZRGBL>::Ptr &_out)
{
  _out.reset(new pcl::PointCloud<pcl::PointXYZRGBL>);
  _out->points.resize(_in->width*_in->height);
  _out->width = _in->width;
  _out->height = _in->height;
  for(unsigned row=0; row<_in->height; row++) {
    for(unsigned col=0; col<_in->width; col++) {
      int idx = GetIdx(col, row);
      _out->points[idx].x = _in->points[idx].x;
      _out->points[idx].y = _in->points[idx].y;
      _out->points[idx].z = _in->points[idx].z;
      _out->points[idx].rgb = _in->points[idx].rgb;
      _out->points[idx].label = 0;                                 // Label 0 == unlabeled
    }
  }
}

void UniformSurfaces::ReassignMaskPoints(cv::Mat_<float> &_edges) 
{
  float max_z = 0.007;   // 0.01 = 1cm @ 1m                                                     /// TODO Als Parameter einstellen!!!
  
  bool done = false;
  std::queue< std::pair<int, int> > queue;    
  std::pair<int, int> p[4];
  p[0] = std::make_pair<int, int>(-1, 0);
  p[1] = std::make_pair<int, int>( 1, 0);
  p[2] = std::make_pair<int, int>( 0,-1);
  p[3] = std::make_pair<int, int>( 0, 1);
    
  while(!done) {
    done = true;
    std::vector< std::pair<int, int> > reassign_pts;
    for(unsigned row=0; row<pcl_cloud_labeled->height; row++) {
      for(unsigned col=0; col<pcl_cloud_labeled->width; col++) {
        int idx = GetIdx(col, row);
        if(_edges(row, col) == 1 && !isnan(pcl_cloud_labeled->points[idx].x)) {
          
          std::map<double, int> cands;
          for(unsigned i=0; i<4; i++) {
            int x = col + p[i].first;
            int y = row + p[i].second;
            int idx_n = GetIdx(x, y);
            int label = pcl_cloud_labeled->points[idx_n].label;
            if(!isnan(pcl_cloud_labeled->points[idx].x) && label != 0) {
              float max_dist =  pcl_cloud_labeled->points[idx].z * max_z * pcl_cloud_labeled->points[idx].z * max_z;
              double dist = (pcl_cloud_labeled->points[idx].getVector3fMap() - pcl_cloud_labeled->points[idx_n].getVector3fMap()).squaredNorm();                        /// TODO Check validity of distance! /// TODO Check curvature
              if(dist < max_dist)
                cands.insert(std::make_pair<double, int>(dist, label));
            }
          }
          /// get best neighbor (if there is one)
          if(cands.size() > 0) {
            reassign_pts.push_back(std::make_pair<int, int>(idx, cands.begin()->second));
            _edges(row, col) = 0;
            done = false;
          }
        }
      }
    }
    
    // Reassign points
    for(unsigned i=0; i<reassign_pts.size(); i++)
      pcl_cloud_labeled->points[reassign_pts[i].first].label = reassign_pts[i].second;
  }
}

void UniformSurfaces::CreateView()
{
  view = new View();
  view->width = pcl_cloud->width;
  view->height = pcl_cloud->height;
    
//   std::map<int, int> patch_nr;          /// patch, idx
  
  std::set<int> nr;
  for(unsigned i=0; i<pcl_cloud_labeled->points.size(); i++)
    nr.insert(pcl_cloud_labeled->points[i].label);
  
//   std::vector<int> patch_nr;
//   std::set <int >::iterator it;
//   for(it = nr.begin(); it != nr.end(); it++)
//     if(*it != 0)
//       patch_nr.push_back(*it);
//     
//   std::vector<surface::SurfaceModel::Ptr> models;
//   models.resize(nr.size()-1);
//   for(unsigned i=0; i<models.size(); i++)
//     models[i].reset( new surface::SurfaceModel() );
// 
// printf(" number of patches: %u\n", nr.size()-1);
//   for(unsigned i=0; i<pcl_cloud_labeled->points.size(); i++) {
// //     it = nr.find(pcl_cloud_labeled->points[i].label);
// // printf("%u - ", it - nr.begin());
// //     models[*find]->indices.push_back(i);
//   }
//   
// printf(" - done!\n");
//   
// printf(" number of patches: %u\n", nr.size()-1);
//   for(unsigned i=0; i<models.size(); i++)
//     if(i != 0 || *nr.begin() != 0)
//       view->surfaces.push_back(models[i]);


//   for(unsigned i=0; i<pcl_cloud_labeled->points.size(); i++)
//     view->surfaces[pcl_cloud_labeled->points[i].label]->indices.push_back(i);
//   for(unsigned i=1000; i>=0; i--)
//     if(view->surfaces[i]->indices.size() == 0)
//       view->surfaces.erase(view->surfaces.end());
}

/************************** PUBLIC *************************/

/**
 * set the input cloud for detecting planes
 */
void UniformSurfaces::setInputCloud(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &_pcl_cloud)
{
  if (_pcl_cloud->height<=1 || _pcl_cloud->width<=1 || !_pcl_cloud->isOrganized())
    throw std::runtime_error("[UniformSurfaces::setInputCloud] Error: Unorganized point cloud.");

  pcl_cloud = _pcl_cloud;
  if(!initialized)
    Initialize();
}

/** 
 * Set input normals 
 */
void UniformSurfaces::setInputNormals(const pcl::PointCloud<pcl::Normal>::Ptr &_pcl_normals)
{
//   pcl_normals = _pcl_normals;
}

/**
 * setParameter
 */
// void UniformSurfaces::setParameter(Parameter p)
// {
//   param = p;
//   cosThrAngleNC = cos(param.thrAngle);
// }

/**
 * Compute
 */
void UniformSurfaces::compute()
{
  /// we calculate the normals, which we need for that approach
  pcl_normals.reset(new pcl::PointCloud<pcl::Normal>);
  ZAdaptiveNormals<pcl::PointXYZRGB>::Parameter param;
  param.adaptive = true;
  param.kappa = 0.005125; //0.005125;
//   param.d = 0.008;
  param.kernel_radius[0] = 4;
  param.kernel_radius[1] = 5;
  param.kernel_radius[2] = 6;
  param.kernel_radius[3] = 7;
  param.kernel_radius[4] = 7;
  param.kernel_radius[5] = 7;
  param.kernel_radius[6] = 7;
  param.kernel_radius[7] = 7;
  ZAdaptiveNormals<pcl::PointXYZRGB> nor;
  nor.setParameter(param);
  nor.setInputCloud(pcl_cloud);
  nor.compute();
  nor.getNormals(pcl_normals); 
  pclA::ConvertPCLNormals2CvMat(pcl_normals, cv_normals);
  
  /// TODO tomGine dbgWin => Remove
#ifdef DEBUG
  cv::Mat_<float> camR = cv::Mat::eye(3, 3, CV_32F);
  cv::Mat_<float> camT = cv::Mat::zeros(3, 1, CV_32F);
  cv::Mat_<double> camI = cv::Mat::zeros(3, 3, CV_64F);
  camI.at<double> (0, 0) = camI.at<double> (1, 1) = 525;
  camI.at<double> (0, 2) = 320;
  camI.at<double> (1, 2) = 240;
  camI.at<double> (2, 2) = 1;
  TomGine::tgTomGineThreadPCL* dbgWin = new TomGine::tgTomGineThreadPCL(pcl_cloud->width, pcl_cloud->height);
  dbgWin->SetClearColor(1.0, 1.0, 1.0);
  dbgWin->SetCamera(camI);
  dbgWin->SetCamera(camR, camT);
  dbgWin->SetInputSpeeds(0.5f, 0.25f, 0.25f);
#endif
  
  /// create image containers
  cv::Mat_<cv::Vec4f> cloud;
  cv::Mat_<cv::Vec4f> normals;
  cv::Mat_<cv::Vec3b> image;
  cv::Mat_<uchar> mask;

  pclA::ConvertPCLCloud2CvMat(pcl_cloud, cloud);
  pclA::ConvertPCLCloud2Image(pcl_cloud, image);
  pclA::ConvertPCLCloud2Mask(pcl_cloud, mask);
  pclA::ConvertPCLNormals2CvMat(pcl_normals, normals);

  cv::Mat_<float> depth = cv::Mat_<float>(cloud.rows, cloud.cols);
  cv::Mat_<float> curvature = cv::Mat_<float>(cloud.rows, cloud.cols);
  cv::Mat_<float> color_edges = cv::Mat_<float>(cloud.rows, cloud.cols);
  cv::Mat_<float> depth_edges = cv::Mat_<float>(cloud.rows, cloud.cols);
  cv::Mat_<float> curvature_edges = cv::Mat_<float>(cloud.rows, cloud.cols);
  cv::Mat_<float> mask_edges = cv::Mat_<float>(cloud.rows, cloud.cols);
  cv::Mat_<float> edges = cv::Mat_<float>(cloud.rows, cloud.cols);
  
  
static struct timespec start, current;
clock_gettime(CLOCK_THREAD_CPUTIME_ID, &start);
  
  NormaliseDepthAndCurvature(cloud, normals, depth, curvature);
  
clock_gettime(CLOCK_THREAD_CPUTIME_ID, &current);
printf("[Segmenter::process] Runtime for NormaliseDepthAndCurvature: %4.3f\n", timespec_diff5(&current, &start));
start = current;

  EdgeDetectionRGBDC(m_engine, m_ip, image, depth, curvature, mask, color_edges, depth_edges, curvature_edges, mask_edges, edges);

clock_gettime(CLOCK_THREAD_CPUTIME_ID, &current);
printf("[Segmenter::process] Runtime for EdgeDetectionRGBDC: %4.3f\n", timespec_diff5(&current, &start));
start = current;

//   cv::imshow("color edges", color_edges);
//   cv::imshow("depth edges", depth_edges);
cv::imshow("curvature", curvature);

int dilation_size = 2;
int dilation_elem = 0;
int dilation_type;
if( dilation_elem == 0 ){ dilation_type = cv::MORPH_RECT; }
else if( dilation_elem == 1 ){ dilation_type = cv::MORPH_CROSS; }
else if( dilation_elem == 2) { dilation_type = cv::MORPH_ELLIPSE; }
cv::Mat element = cv::getStructuringElement( dilation_type,
                                     cv::Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                     cv::Point( dilation_size, dilation_size ) );
cv::dilate(mask_edges, mask_edges, element);
cv::imshow("curvature2", mask_edges);

cv::imshow("curvature edges", curvature_edges);
cv::imshow("edges", edges);
//   cv::waitKey(100);
  

  RecursiveClustering(edges);  

clock_gettime(CLOCK_THREAD_CPUTIME_ID, &current);
printf("[Segmenter::process] Runtime for RecursiveClustering: %4.3f\n", timespec_diff5(&current, &start));
start = current;

  ReassignMaskPoints(edges);

clock_gettime(CLOCK_THREAD_CPUTIME_ID, &current);
printf("[Segmenter::process] Runtime for ReassignMaskPoints: %4.3f\n", timespec_diff5(&current, &start));
start = current;

  CreateView();

clock_gettime(CLOCK_THREAD_CPUTIME_ID, &current);
printf("[Segmenter::process] Runtime for CreateView: %4.3f\n", timespec_diff5(&current, &start));
start = current;

#ifdef DEBUG
  dbgWin->AddPointCloudPCL(*pcl_cloud_labeled);
#endif

  /// 2. Möglichkeit: Impelementierung über pcl-> edge_extractor
  
printf("[UniformSurfaces::compute] Computation done!\n");

//   pcl::OrganizedEdgeFromNormals<pcl::PointXYZRGB, pcl::Normal, pcl::Label> oed;
//   oed.setInputNormals(pcl_normals);
//   oed.setInputCloud(pcl_cloud);
//   oed.setDepthDisconThreshold(0.01);
// //   oed.setEdgeType(pcl::EDGELABEL_HIGH_CURVATURE);
//   oed.setMaxSearchNeighbors (50);
//   pcl::PointCloud<pcl::Label> labels;
//   std::vector<pcl::PointIndices> label_indices;
//   oed.compute (labels, label_indices);
//   
//   pcl::PointCloud<pcl::PointXYZRGB>::Ptr high_curvature_edges (new pcl::PointCloud<pcl::PointXYZRGB>); 
//   pcl::copyPointCloud (*pcl_cloud, label_indices[3].indices, *high_curvature_edges);
// 
//   dbgWin->AddPointCloudPCL(*high_curvature_edges);
}



} //-- THE END --

