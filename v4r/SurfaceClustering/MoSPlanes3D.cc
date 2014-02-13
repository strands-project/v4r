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


#include "MoSPlanes3D.hh"

namespace surface 
{

using namespace std;

template<typename T1,typename T2>
inline T1 Dot3(const T1 v1[3], const T2 v2[3])
{
  return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2];
}

template<typename T1,typename T2, typename T3>
inline void Mul3(const T1 v[3], T2 s, T3 r[3])
{
  r[0] = v[0]*s;
  r[1] = v[1]*s;
  r[2] = v[2]*s;
}

template<typename T1,typename T2, typename T3>
inline void Add3(const T1 v1[3], const T2 v2[3], T3 r[3])
{
  r[0] = v1[0]+v2[0];
  r[1] = v1[1]+v2[1];
  r[2] = v1[2]+v2[2];
}

#ifdef DEBUG
  double timespec_diff(struct timespec *x, struct timespec *y)
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
#endif

static const bool CmpNumPoints(const SurfaceModel::Ptr &a, const SurfaceModel::Ptr &b)
{
  return (a->indices.size() > b->indices.size());
}


/********************** MoSPlanes3D ************************
 * Constructor/Destructor
 */
MoSPlanes3D::MoSPlanes3D(Parameter p)
{
  setParameter(p);
  line_check = false;
}

MoSPlanes3D::~MoSPlanes3D()
{
}



/************************** PRIVATE ************************/
/**
 * Init
 */
void MoSPlanes3D::Init(pcl::PointCloud<pcl::PointXYZRGB> &cloud)
{
  if (cloud.height<=1 || cloud.width<=1)
    throw std::runtime_error("[MoSPlanes3D::Init] Invalid point cloud (height must be > 1)");

  width = cloud.width;
  height = cloud.height;

  planes.reset( new std::vector<SurfaceModel::Ptr>() );
  tmpPlanes.reset( new std::vector<SurfaceModel::Ptr>() );

  mask.reserve(width*height);
}

/**
 * ClusterNormals
 * @param normal Surface normal
 */
void MoSPlanes3D::ClusterNormals(unsigned idx, 
                                 pcl::PointCloud<pcl::PointXYZRGB> &cloud, 
                                 pcl::PointCloud<pcl::Normal> &normals, 
                                 vector<int> &pts,
                                 pcl::Normal &normal)
{
  float *n;
  short x,y;
  normal = normals.points[idx];
  pts.clear();
  queue.clear();

  if (normal.normal[0]!=normal.normal[0] || cloud.points[idx].x!=cloud.points[idx].x)
    return;

  mask[idx] = 1;
  pts.push_back(idx);
  queue.push_back(idx);

  while (queue.size()>0)
  {
    idx = queue.back();
    queue.pop_back();
    x = X(idx);
    y = Y(idx);
    pcl::PointXYZRGB &pt = cloud.points[idx];

    for (int v=y-1; v<=y+1; v++)
    {
      for (int u=x-1; u<=x+1; u++)
      {
        if (v>0 && u>0 && v<height && u<width)
        {
          idx = GetIdx(u,v);
          if (mask[idx]==0)
          {
            n = &normals.points[idx].normal[0];                                 // actual normal of u,v

            if (n[0]!= n[0])
              continue;

            float optCosThrAngleNC = cosThrAngleNC;
#ifdef USE_NOISE_MODEL
            if(cloud.points[idx].z == cloud.points[idx].z) {
              float *n0;
              n0 = new float[3]; 
              n0[0] = 0.0; n0[1] = 0.0; n0[2]= -1.0;
              
              float d_z = 1.36364*cloud.points[idx].z + 0.34;              
              float d_a_phi = Dot3(&normal.normal[0], n0);                      // Öffnungswinkel zwischen z-Achse und Flächennormale
              float d_a_phi2 = 0. - (d_a_phi * d_a_phi)/0.7854;                 // Umrechnung mittels Gauß'scher Glockenkurve (Exponent)
              float d_a = 1.0 + exp(d_a_phi2);                                  // 1 + e^(-phi²/sigma)
              optCosThrAngleNC = cos(param.thrAngleNormalClustering * d_z);     // Neuberechnung des Winkel-thresholds
              
              printf("[MoSPlanes3D::ClusterNormals] dz: %4.3f * da: %4.3f::%4.3f::%4.3f =>  cosThrAngleNC: %4.4f => %4.4f\n", d_z, d_a_phi, d_a_phi2, d_a, cosThrAngleNC, optCosThrAngleNC);
            }
#endif

            if (Dot3(&normal.normal[0], n) > optCosThrAngleNC && 
                fabs(Plane::NormalPointDist(&pt.x, &normal.normal[0], &cloud.points[idx].x)) < param.inlDist) 
            {
              mask[idx] = 1;
              
              Mul3(&normal.normal[0], pts.size(), &normal.normal[0]);
              Add3(&normal.normal[0], &normals.points[idx].normal[0], &normal.normal[0]);

              pts.push_back(idx);
              queue.push_back(idx);

              Mul3(&normal.normal[0], 1./(float)pts.size(), &normal.normal[0]);
            }
          }
        }
      }
    }
  }
}


/**
 * ClusterNormals
 */
void MoSPlanes3D::ClusterNormals(pcl::PointCloud<pcl::PointXYZRGB> &cloud, 
        pcl::PointCloud<pcl::Normal> &normals, 
        std::vector<SurfaceModel::Ptr> &planes, int level)
{
  SurfaceModel::Ptr plane;
  mask.clear();
  mask.resize(cloud.width*cloud.height,0);
  unsigned idx;
  pcl::Normal normal;

  int minPts = pow((float) param.minPoints, 1./pow(2.,level));
printf("[MoSPlanes3D::ClusterNormals] minPts errechnet: %u ", minPts);
  minPts = (minPts<4?4:minPts);
printf(" => gesetzt: %u (level %u)\n", minPts, level);

  for (unsigned v=0; v<cloud.height; v++)
  {
    for (unsigned u=0; u<cloud.width; u++)
    {
      idx = GetIdx(u,v);

      if (mask[idx]==0)
      {
        plane.reset(new SurfaceModel());
        plane->type = pcl::SACMODEL_PLANE;
        plane->level=level;

        ClusterNormals(idx, cloud, normals, plane->indices, normal);

        if (((int)plane->indices.size())>=minPts)
        {
          plane->coeffs.resize(3);
          float *n = &plane->coeffs[0];
          n[0]=normal.normal[0]; 
          n[1]=normal.normal[1]; 
          n[2]=normal.normal[2];
          planes.push_back(plane);
        }
      }
    }
  }
}

/**
 * ComputeLSPlanes
 */
void MoSPlanes3D::ComputeLSPlanes(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud, std::vector<SurfaceModel::Ptr> &planes)
{
  pcl::SampleConsensusModelPlane<pcl::PointXYZRGB> lsPlane(cloud);
  Eigen::VectorXf coeffs(4);
  Eigen::Vector3d n0(0., 0., 1.);

  for (unsigned i=0; i<planes.size(); i++)
  {
    SurfaceModel &plane = *planes[i];

    lsPlane.optimizeModelCoefficients(plane.indices, coeffs, coeffs);

    if (Dot3(&coeffs[0], &n0[0]) > 0)
    {
      coeffs*=-1.;
    }

    if (Dot3(&coeffs[0], &plane.coeffs[0]) <= cosThrAngleNC)
    {
      //cout<<"level="<<plane.level<<" id="<<i<<", ----> invalid plane!"<<", size="<<plane.indices.size()<<endl;
      planes.erase(planes.begin()+i);
      i--;
    }
    else
    {
      plane.coeffs.resize(4);
      plane.coeffs[0] = coeffs[0];
      plane.coeffs[1] = coeffs[1];
      plane.coeffs[2] = coeffs[2];
      plane.coeffs[3] = coeffs[3];
    }
  }
}

/**
 * ComputePointProbs
 */
void MoSPlanes3D::ComputePointProbs(pcl::PointCloud<pcl::PointXYZRGB> &cloud, std::vector<SurfaceModel::Ptr> &planes)
{
  float a, b, c, d;
  float err;

  for (unsigned j=0; j<planes.size(); j++)
  {
    SurfaceModel &plane = *planes[j];

    plane.error.resize(plane.indices.size());
    plane.probs.resize(plane.indices.size());
    a=plane.coeffs[0], b=plane.coeffs[1], c=plane.coeffs[2], d=plane.coeffs[3];

    for (unsigned i=0; i<plane.indices.size(); i++)
    {
      err = Plane::ImpPointDist(a,b,c,d, &cloud.points[plane.indices[i]].x);
      plane.error[i] = err;
      plane.probs[i] = exp(-(pow(err, 2) * invSqrSigma));
    }
  }
}

/**
 * PseudoUpsample
 */
void MoSPlanes3D::PseudoUpsample(pcl::PointCloud<pcl::PointXYZRGB> &cloud0, pcl::PointCloud<pcl::PointXYZRGB> &cloud1, std::vector<SurfaceModel::Ptr> &planes, float nbDist)
{
  int u0, v0;
  vector<int> indices;
  float sqrDist = pow(nbDist, 2);

  for (unsigned i=0; i<planes.size(); i++)
  {
    indices.clear();
    SurfaceModel &plane = *planes[i];

    for (unsigned j=0; j<plane.indices.size(); j++)
    {
      u0 = 2*X(plane.indices[j], cloud1.width);;
      v0 = 2*Y(plane.indices[j], cloud1.width);;
      pcl::PointXYZRGB &pt0 = cloud0(u0, v0);

      if (!IsNaN(pt0))
      {
        for (int y = v0; y<=v0+1; y++)
        {
          for (int x = u0; x<=u0+1; x++)
          {
            if (x>=0 && x<(int)cloud0.width && y>=0 && y<(int)cloud0.height)
            {
              pcl::PointXYZRGB &pt = cloud0(x,y);
              if (!IsNaN(pt) && SqrDistance(pt0,pt) < sqrDist)
              {
                indices.push_back(GetIdx(x,y,cloud0.width));
              }
            }
          }
        }
      }
    }
    plane.indices = indices;
  }
}

/**
 * CCFilter
 */
void MoSPlanes3D::CCFilter(std::vector<SurfaceModel::Ptr> &planes)
{
  std::vector<int> queue;
  std::vector<unsigned> used;
  std::vector<unsigned> mask;
  unsigned idx;
  short x,y;
  used.resize(cloud->points.size(),0);
  mask.resize(cloud->points.size(),0);

  unsigned z,mcnt=0, ucnt=0;

  for (unsigned i=0; i<planes.size(); i++)
  {
    std::vector<int> &planeIndices = planes[i]->indices;
    std::vector<int> nbs = planes[i]->indices;

    if (nbs.size()>2)
    {
      //set plane
      mcnt++;
      for (unsigned j=0; j<nbs.size(); j++)
        mask[nbs[j]]=mcnt;

      z=0;
      do{
        ucnt++;
        idx = nbs[ rand()%nbs.size() ];
        queue.clear();
        queue.push_back(idx);
        used[idx] = ucnt;

        planeIndices.clear();
        planeIndices.push_back(idx);

        //cluster
        while (queue.size()>0)
        {
          idx = queue.back();
          queue.pop_back();
          x = X(idx);
          y = Y(idx);

          for (int v=y-1; v<=y+1; v++)
          {
            for (int u=x-1; u<=x+1; u++)
            {
              if (v>0 && u>0 && v<height && u<width)
              {
                idx = GetIdx(u,v);
                if (mask[idx] == mcnt && used[idx]!=ucnt)
                {
                  planeIndices.push_back(idx);
                  queue.push_back(idx);
                  used[idx] = ucnt;
                }
              }
            }
          }
        }
        z++;

      }while(planeIndices.size() < nbs.size()/2 && z<10);
    }
  }
}




/************************** PUBLIC *************************/

/**
 * set the input cloud for detecting planes
 */
void MoSPlanes3D::setInputCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &_cloud)
{
  cloud = _cloud;
}

/**
 * setParameter
 */
void MoSPlanes3D::setParameter(Parameter p)
{
  param = p;
  if (param.pyrLevels<1)
    throw std::runtime_error("[MoSPlanes3D::compute] pyrLevels must be >1");

  cosThrAngleNC = cos(param.thrAngleNormalClustering);
  invSqrSigma = 1./(pow(param.sigma, 2));
}

/**
 * Compute
 */
void MoSPlanes3D::compute()
{
  #ifdef DEBUG
  cout<<"********************** start detecting planar patches **********************"<<endl;
  #endif

  srand(time(NULL));

  if (cloud.get()==0)
    throw std::runtime_error("[MoSPlanes3D::compute] Point cloud or normals not set!"); 

  Init(*cloud);

  #ifdef DEBUG
  struct timespec start1, end1, start2, end2;
  clock_gettime(CLOCK_REALTIME, &start1);
  #endif

  // compute pyramid
  normals.resize(param.pyrLevels);
  clouds.resize(param.pyrLevels);
  clouds[0] = cloud;
  for (int i=1; i<param.pyrLevels; i++) {
    resize.setParameter(SubsamplePointCloud2::Parameter(param.nbDist*i));
    resize.setInputCloud(clouds[i-1]);
    resize.compute();
    resize.getCloud(clouds[i]);
  }

  // detect planes
  planes->clear();
  tmpPlanes->clear();
  pclA::NormalsEstimationNR::Parameter neParam = param.ne;
  for (int i=param.pyrLevels-1; i>=0; i--)
  {
    width = clouds[i]->width;
    height = clouds[i]->height;
    neParam.maxDist = param.ne.maxDist*(i+1);
    
    normalsEstimation.setParameter(neParam);
    normalsEstimation.setInputCloud(clouds[i]);
    normalsEstimation.compute();
    normalsEstimation.getNormals(normals[i]);
    
    tmpPlanes->clear();
    ClusterNormals(*clouds[i], *normals[i], *tmpPlanes, i); 
    ComputeLSPlanes(clouds[i], *tmpPlanes);

  #ifdef DEBUG
    clock_gettime(CLOCK_REALTIME, &start2);
  #endif

    modsel.setParameter(param.mos);
    modsel.setInputCloud(clouds[i]);
    modsel.compute(*tmpPlanes, *planes);

  #ifdef DEBUG
    clock_gettime(CLOCK_REALTIME, &end2);
    cout<<"[MoSPlanes3D] Model selection: "<<timespec_diff(&end2, &start2)<<endl;
  #endif

    if (i>0) 
      PseudoUpsample(*clouds[i-1], *clouds[i], *planes, param.nbDist*i);

  #ifdef DEBUG
    cout<<"[MoSPlanes3D] Level "<<i<<endl;
    cout<<"[MoSPlanes3D] Number of planes: "<<planes->size()<<endl;
  #endif

  #ifdef DEBUG_WIN      // TODO funktioniert nicht: nicht initialisiert?
    dbgWin->Clear();
    //DrawPointCloud(clouds[(i-1>=0?i-1:0)]);
    //DrawNormals(clouds[(i-1>=0?i-1:0)], normals[(i-1>=0?i-1:0)]);
    DrawPlanePoints(*clouds[(i-1>=0?i-1:0)], *planes);
    /*CreateMeshModel createMesh(CreateMeshModel::Parameter(.1));
    createMesh.setInputCloud(clouds[(i-1>=0?i-1:0)]);
    createMesh.compute(*planes);
    DrawSurfaces(*planes);*/ 
    width = clouds[(i-1>=0?i-1:0)]->width;
    height = clouds[(i-1>=0?i-1:0)]->height;
    for (unsigned j=0; j<planes->size(); j++)
    {
      cv::Point pt=ComputeMean((*planes)[j]->indices);
      pcl::PointXYZRGB &pt3 = (*clouds[(i-1>=0?i-1:0)])(pt.x,pt.y);
      string label = toString((*planes)[j]->level,0);
      dbgWin->AddLabel3D(label, 12, pt3.x, pt3.y, pt3.z);
    }
    dbgWin->WaitForEvent(TomGine::TMGL_Press, TomGine::TMGL_Space);
  #endif
  }

  /// filter and assign points
  #ifdef DEBUG
  clock_gettime(CLOCK_REALTIME, &start2);
  #endif
  ptsToPlane.setParameter(AssignPointsToPlanes::Parameter(param.minPoints, param.inlDist));
  ptsToPlane.setInputCloud(cloud);
  ptsToPlane.setInputNormals(normals[0]);
  ptsToPlane.compute(*planes);

  CCFilter(*planes);

  /*ptsToPlane.setParameter(AssignPointsToPlanes::Parameter(param.minPoints,param.inlDist));
  ptsToPlane.setInputCloud(cloud);
  ptsToPlane.setInputNormals(normals[0]);
  ptsToPlane.compute(*planes);*/

  // sort
  std::sort(planes->begin(), planes->end(), CmpNumPoints);

  #ifdef DEBUG
  if (!dbgWin.empty())
  {
    dbgWin->Clear();
    DrawPlanePoints(*cloud, *planes);
    DrawNormals(clouds[0], normals[0]);
    dbgWin->Update();
  }
  #endif
  
  // recompute probs
  ComputePointProbs(*cloud, *planes);
  
  if(line_check)
    LineCheck();
  
  #ifdef DEBUG
  clock_gettime(CLOCK_REALTIME, &end2);
  cout<<"[MoSPlanes3D] Plane detection post processing: "<<timespec_diff(&end2, &start2)<<endl;
  
  clock_gettime(CLOCK_REALTIME, &end1);
  cout<<"[MoSPlanes3D] Time detect planes: "<<timespec_diff(&end1, &start1)<<endl;
  #endif
}

/**
 * getSurfaceModels
 */
void MoSPlanes3D::getSurfaceModels(std::vector<SurfaceModel::Ptr> &_planes)
{
  _planes = *planes;
}

/**
 * getError
 */
void MoSPlanes3D::getError(std::vector< std::vector<double> > &_error)
{
  printf("[MoSPlanes3D::getError] Warning: Antiquated method. Use getSurfaceModels()\n");

  _error.resize(planes->size());
  for(unsigned i=0; i<planes->size(); i++)
  {
    _error[i] = (*planes)[i]->error;
  }
}

/**
 * @brief Check if there are patch models with "line"-style (not more than n neighbors)
 * @param check True to check
 * @param neighbors Threshold for line_check neighbors
 */
void MoSPlanes3D::setLineCheck(bool check, int neighbors)
{
  line_check = check;
  lc_neighbors = neighbors;
}


/**
 * @brief Check if there are patch models with "line"-style
 */
void MoSPlanes3D::LineCheck()
{
  int minSurfacePoints = param.minPoints;
  int minAssignNeighbors = 2; // number of neighbors to assign point

  // create patch-image
  cv::Mat_<cv::Vec3b> patches;  //< Patch indices(+1) on 2D image grid
  patches = cv::Mat_<cv::Vec3b>(cloud->height, cloud->width);
  patches.setTo(0);
  for(unsigned i=0; i<(*planes).size(); i++) {
    for(unsigned j=0; j<(*planes)[i]->indices.size(); j++) {
      int row = (*planes)[i]->indices[j] / cloud->width;
      int col = (*planes)[i]->indices[j] % cloud->width;
      patches.at<cv::Vec3b>(row, col)[0] = i+1;   /// plane 1,2,...,n
    }
  }
  
  // go through image and count neighbors in the 8-neighborhood
  std::vector<int> bad_patch_indices[(*planes).size()];  // bad indices list for each patch
  for(int row=1; row<patches.rows-1; row++) {
    for(int col=1; col<patches.cols-1; col++) {
      int counter = 0;    // counts the same neighbors
      bool neighbors[8] = {false, false, false, false, false, false, false, false};
      if(patches.at<cv::Vec3b>(row, col)[0] == 0) continue; // do not consider 0-points (not assigned)
      if(patches.at<cv::Vec3b>(row, col)[0] == patches.at<cv::Vec3b>(row, col-1)[0]) {counter++; neighbors[0] = true;}
      if(patches.at<cv::Vec3b>(row, col)[0] == patches.at<cv::Vec3b>(row-1, col-1)[0]) {counter++; neighbors[1] = true;}
      if(patches.at<cv::Vec3b>(row, col)[0] == patches.at<cv::Vec3b>(row-1, col)[0]) {counter++; neighbors[2] = true;}
      if(patches.at<cv::Vec3b>(row, col)[0] == patches.at<cv::Vec3b>(row-1, col+1)[0]) {counter++; neighbors[3] = true;}
      if(patches.at<cv::Vec3b>(row, col)[0] == patches.at<cv::Vec3b>(row, col+1)[0]) {counter++; neighbors[4] = true;}
      if(patches.at<cv::Vec3b>(row, col)[0] == patches.at<cv::Vec3b>(row+1, col+1)[0]) {counter++; neighbors[5] = true;}
      if(patches.at<cv::Vec3b>(row, col)[0] == patches.at<cv::Vec3b>(row+1, col)[0]) {counter++; neighbors[6] = true;}
      if(patches.at<cv::Vec3b>(row, col)[0] == patches.at<cv::Vec3b>(row+1, col-1)[0]) {counter++; neighbors[7] = true;}
        
      // neighboring neighbors increase counter by one
      for(unsigned i=0; i<8; i++) {
        int j = i+1;
        if(j > 7) j=0;
        if(neighbors[i] == true && neighbors[j] == true)
          counter++;
      }
      
      // number of required neighbors (neighboring neighbors count one more)
      if(counter <= lc_neighbors)
        bad_patch_indices[patches.at<cv::Vec3b>(row, col)[0] - 1].push_back(row*patches.cols + col);
    }
  }  
  
  // and now check how many bad surface points we have
  for(unsigned i=0; i<(*planes).size(); i++) {
    if(((*planes)[i]->indices.size() - bad_patch_indices[i].size()) < minSurfacePoints)
      bad_patch_indices[i] = (*planes)[i]->indices;
    else
      bad_patch_indices[i].clear();;
  }

  bool ready = false;
  int circle = 0;
  bool assigned = false;
  while(!ready)
  {
    assigned = false;
    for(int i=0; i< (int)(*planes).size(); i++)
    {
      std::vector<int> not_assigned_indexes;
      for(int j=0; j< (int) bad_patch_indices[i].size(); j++)
      {
        // Take one point after another and assign it to the best surrounding 
        unsigned row = bad_patch_indices[i][j] / cloud->width;
        unsigned col = bad_patch_indices[i][j] % cloud->width;
        if(row > 0 && row < cloud->height && col > 0 && col < cloud->width)
        {
          int surounding[(*planes).size()];
          for(unsigned s=0; s<(*planes).size(); s++)
            surounding[s] = 0;
        
          int idx = bad_patch_indices[i][j]-1;
          if(patches.at<cv::Vec3b>(row, col-1)[0] != 0 && 
             patches.at<cv::Vec3b>(row, col-1)[0] != patches.at<cv::Vec3b>(row, col)[0] &&
             fabs(cloud->points[bad_patch_indices[i][j]].z - cloud->points[idx].z) < param.inlDist)
            surounding[patches.at<cv::Vec3b>(row, col-1)[0]]++;
          idx = bad_patch_indices[i][j]+1;
          if(patches.at<cv::Vec3b>(row, col+1)[0] != 0 && 
             patches.at<cv::Vec3b>(row, col+1)[0] != patches.at<cv::Vec3b>(row, col)[0] &&
             fabs(cloud->points[bad_patch_indices[i][j]].z - cloud->points[idx].z) < param.inlDist) 
            surounding[patches.at<cv::Vec3b>(row, col+1)[0]]++;
          idx = bad_patch_indices[i][j]-cloud->width;
          if(patches.at<cv::Vec3b>(row-1, col)[0] != 0 && 
             patches.at<cv::Vec3b>(row-1, col)[0] != patches.at<cv::Vec3b>(row, col)[0] &&
             fabs(cloud->points[bad_patch_indices[i][j]].z - cloud->points[idx].z) < param.inlDist) 
            surounding[patches.at<cv::Vec3b>(row-1, col)[0]]++;
          idx = bad_patch_indices[i][j]+cloud->width;
          if(patches.at<cv::Vec3b>(row+1, col)[0] != 0 && 
             patches.at<cv::Vec3b>(row+1, col)[0] != patches.at<cv::Vec3b>(row, col)[0] &&
             fabs(cloud->points[bad_patch_indices[i][j]].z - cloud->points[idx].z) < param.inlDist) 
            surounding[patches.at<cv::Vec3b>(row+1, col)[0]]++;
          idx = bad_patch_indices[i][j]+cloud->width+1;
          if(patches.at<cv::Vec3b>(row+1, col+1)[0] != 0 && 
             patches.at<cv::Vec3b>(row+1, col+1)[0] != patches.at<cv::Vec3b>(row, col)[0] &&
             fabs(cloud->points[bad_patch_indices[i][j]].z - cloud->points[idx].z) < param.inlDist) 
            surounding[patches.at<cv::Vec3b>(row+1, col+1)[0]]++;
          idx = bad_patch_indices[i][j]+cloud->width-1;
          if(patches.at<cv::Vec3b>(row+1, col-1)[0] != 0 && 
             patches.at<cv::Vec3b>(row+1, col-1)[0] != patches.at<cv::Vec3b>(row, col)[0] &&
             fabs(cloud->points[bad_patch_indices[i][j]].z - cloud->points[idx].z) < param.inlDist) 
            surounding[patches.at<cv::Vec3b>(row+1, col-1)[0]]++;
          idx = bad_patch_indices[i][j]-cloud->width+1;
          if(patches.at<cv::Vec3b>(row-1, col+1)[0] != 0 && 
             patches.at<cv::Vec3b>(row-1, col+1)[0] != patches.at<cv::Vec3b>(row, col)[0] &&
             fabs(cloud->points[bad_patch_indices[i][j]].z - cloud->points[idx].z) < param.inlDist) 
            surounding[patches.at<cv::Vec3b>(row-1, col+1)[0]]++;
          idx = bad_patch_indices[i][j]-cloud->width-1;
          if(patches.at<cv::Vec3b>(row-1, col-1)[0] != 0 && 
             patches.at<cv::Vec3b>(row-1, col-1)[0] != patches.at<cv::Vec3b>(row, col)[0] &&
             fabs(cloud->points[bad_patch_indices[i][j]].z - cloud->points[idx].z) < param.inlDist) 
            surounding[patches.at<cv::Vec3b>(row-1, col-1)[0]]++;

          // Where are the most points and more than 3?
          int max_neighbors = 0;
          int most_id = 0;
          for(unsigned idx=0; idx<(*planes).size(); idx++) {
            if(max_neighbors < surounding[idx]) {
              max_neighbors = surounding[idx];
              most_id = idx;
            }
          }
          if(max_neighbors > minAssignNeighbors) {
            assigned = true;
            
            (*planes)[most_id-1]->indices.push_back(bad_patch_indices[i][j]); // assign index to new surface
            (*planes)[most_id-1]->error.push_back((*planes)[i]->error[j]);    // assign error to new surface
            
            std::vector<int> surfaces_indices_copy;                           // delete index from surface patch
            for(int su=0; su < (int)(*planes)[i]->indices.size(); su++)
              if((*planes)[i]->indices[su] != bad_patch_indices[i][j])
                surfaces_indices_copy.push_back((*planes)[i]->indices[su]);
            (*planes)[i]->indices = surfaces_indices_copy;
              
            patches.at<cv::Vec3b>(row, col)[0] = most_id;                     // change entry in patches
          } 
          else
            not_assigned_indexes.push_back(bad_patch_indices[i][j]);
        }
      }
      bad_patch_indices[i] = not_assigned_indexes;
    }
    circle++;
    if(circle > minSurfacePoints || !assigned)
      ready = true;
  }

  #ifdef DEBUG
    printf("[MoSPlanes3D::LineCheck] Surfaces before deletion: %lu\n", (*planes).size());
  #endif
    
  // delete the empty surface patches
  std::vector<surface::SurfaceModel::Ptr> planes_copy;
  for(int su=0; su<(int)(*planes).size(); su++)
    if((int)(*planes)[su]->indices.size() >= minSurfacePoints)
      planes_copy.push_back((*planes)[su]);
  (*planes) = planes_copy;

  #ifdef DEBUG
    printf("[MoSPlanes3D::LineCheck] Surfaces after deletion: %lu\n", (*planes).size());
  #endif
}





/******************************* DEBUG ************************/

#ifdef DEBUG
/**
 * DrawPointCloud
 */
void MoSPlanes3D::DrawPointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud)
{
  if (dbgWin.empty())
    return;

  cv::Mat_<cv::Vec4f> cvCloud;
  pclA::ConvertPCLCloud2CvMat(cloud, cvCloud);
  dbgWin->AddPointCloud(cvCloud);
  dbgWin->Update();
}

/**
 * DrawNormals
 */
void MoSPlanes3D::DrawNormals(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud,pcl::PointCloud<pcl::Normal>::Ptr &normals)
{
  if (dbgWin.empty())
    return;

  float pt2[3];
  for (unsigned i=0; i<cloud->points.size() && i<normals->points.size(); i++)
  {
    pcl::PointXYZRGB &pt1 = cloud->points[i];
    if (i%4 && pt1.x==pt1.x && normals->points[i].normal[0]==normals->points[i].normal[0])
    {
      Mul3(&normals->points[i].normal[0],0.004, &pt2[0]);
      Add3(&pt1.x, &pt2[0], &pt2[0]);
      dbgWin->AddLine3D(pt1.x,pt1.y,pt1.z, pt2[0], pt2[1], pt2[2]);
    }
  }

  dbgWin->Update();
}

/**
 * DrawPlanePoints
 */
void MoSPlanes3D::DrawPlanePoints(pcl::PointCloud<pcl::PointXYZRGB> &cloud, std::vector<SurfaceModel::Ptr> &planes)
{
  if (dbgWin.empty())
    return;

  srand(time(NULL));
  cv::Mat_<cv::Vec4f> cvCloud;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudPatch(new pcl::PointCloud<pcl::PointXYZRGB>());

  for (unsigned i=0; i<planes.size(); i++)
  {
      if (planes[i]->indices.size()<=1)
        continue;

      pcl::copyPointCloud(cloud, planes[i]->indices, *cloudPatch);
      pcl::PointXYZRGB col;
      col.rgb = GetRandomColor();
      for (unsigned j=0; j< cloudPatch->points.size(); j++)
        cloudPatch->points[j].rgb = col.rgb;
      pclA::ConvertPCLCloud2CvMat(cloudPatch, cvCloud);
      dbgWin->AddPointCloud(cvCloud);
  }
  dbgWin->Update();
}

/**
 * DrawSurfaces
 */
void MoSPlanes3D::DrawSurfaces(std::vector<SurfaceModel::Ptr> &surfaces)
{
  if (dbgWin.empty())
    return;

  dbgWin->ClearModels();

  for (unsigned i=0; i<surfaces.size(); i++)
    dbgWin->AddModel(&surfaces[i]->mesh);

  dbgWin->Update();
}

#endif

/**
 * ComputeMean 
 */
cv::Point MoSPlanes3D::ComputeMean(std::vector<int> &indices)
{
  cv::Point mean(0., 0.);

  if( indices.size() == 0 )
    return mean;

  for( unsigned i = 0; i < indices.size(); i++ ) {
    mean += cv::Point(X(indices[i]), Y(indices[i]));
  }

  mean.x /= indices.size();
  mean.y /= indices.size();

  return mean;
}



} //-- THE END --

