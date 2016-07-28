/**
 * $Id$
 * Johann Prankl, 2015-2
 * prankl@acin.tuwien.ac.at
 */


#include <v4r/segmentation/SlicRGBD.h>

#include <cfloat>
#include <cmath>
#include <iostream>
#include <fstream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "pcl/features/integral_image_normal.h"

namespace v4r
{

using namespace std;

SlicRGBD::SlicRGBD(const Parameter &p)
  : param(p), num_superpixel(-1)
{
}

SlicRGBD::~SlicRGBD()
{
}

/**
 * convertRGBtoLAB
 */
void SlicRGBD::convertRGBtoLAB(const pcl::PointCloud<pcl::PointXYZRGB> &cloud, cv::Mat_<cv::Vec3d> &im_lab)
{
  im_lab = cv::Mat_<cv::Vec3d>(cloud.height, cloud.width);

  double R, G, B, r, g, b;
  double X, Y, Z, xr, yr, zr;
  double fx, fy, fz;

  double epsilon = 0.008856;  //actual CIE standard
  double kappa   = 903.3;   //actual CIE standard

  const double inv_Xr = 1./0.950456; //reference white
  //const double inv_Yr = 1./1.0;    //reference white
  const double inv_Zr = 1./1.088754; //reference white
  const double inv_255 = 1./255;
  const double inv_12 = 1./12.92;
  const double inv_1 = 1./1.055;
  const double inv_3 = 1./3.0;
  const double inv_116 = 1./116.0;

  #pragma omp parallel for private(R,G,B,r,g,b,X,Y,Z,xr,yr,zr,fx,fy,fz)
  for (int v=0; v<(int)cloud.height; v++)
  {
    for (int u=0; u<(int)cloud.width; u++)
    {
      const pcl::PointXYZRGB &pt = cloud(u,v);
      cv::Vec3d &lab = im_lab(v,u);

      R = pt.r*inv_255;
      G = pt.g*inv_255;
      B = pt.b*inv_255;

      if(R <= 0.04045)  r = R*inv_12;
      else        r = pow((R+0.055)*inv_1,2.4);
      if(G <= 0.04045)  g = G*inv_12;
      else        g = pow((G+0.055)*inv_1,2.4);
      if(B <= 0.04045)  b = B*inv_12;
      else        b = pow((B+0.055)*inv_1,2.4);

      X = r*0.4124564 + g*0.3575761 + b*0.1804375;
      Y = r*0.2126729 + g*0.7151522 + b*0.0721750;
      Z = r*0.0193339 + g*0.1191920 + b*0.9503041;

      xr = X*inv_Xr;
      yr = Y;//*inv_Yr;
      zr = Z*inv_Zr;

      if(xr > epsilon)  fx = pow(xr, inv_3);
      else        fx = (kappa*xr + 16.0)*inv_116;
      if(yr > epsilon)  fy = pow(yr, inv_3);
      else        fy = (kappa*yr + 16.0)*inv_116;
      if(zr > epsilon)  fz = pow(zr, inv_3);
      else        fz = (kappa*zr + 16.0)*inv_116;

      lab[0] = 116.0*fy-16.0;
      lab[1] = 500.0*(fx-fy);
      lab[2] = 200.0*(fy-fz);
    }
  }
}

/**
 * drawContours
 */
void SlicRGBD::drawContours(cv::Mat_<cv::Vec3b> &im_rgb, const cv::Mat_<int> &labels, int r, int g, int b)
{
	const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
	const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};

  bool have_col=false;
  if (r!=-1 && g!=-1 && b!=-1) have_col=true;

  int width = im_rgb.cols;
  int height = im_rgb.rows;
	int sz = width*height;
	vector<bool> istaken(sz, false);
	vector<int> contourx(sz);
  vector<int> contoury(sz);
	int mainindex(0);int cind(0);

	for( int j = 0; j < height; j++ )
	{
		for( int k = 0; k < width; k++ )
		{
			int np(0);
			for( int i = 0; i < 8; i++ )
			{
				int x = k + dx8[i];
				int y = j + dy8[i];

				if( (x >= 0 && x < width) && (y >= 0 && y < height) )
				{
					int index = y*width + x;

					if( false == istaken[index] )
					{
						if( labels(mainindex) != labels(index) ) np++;
					}
				}
			}
			if( np > 1 )
			{
				contourx[cind] = k;
				contoury[cind] = j;
				istaken[mainindex] = true;
				cind++;
			}
			mainindex++;
		}
	}

	int numboundpix = cind;
	for( int j = 0; j < numboundpix; j++ )
	{
		int ii = contoury[j]*width + contourx[j];
    if (have_col) im_rgb(ii) = cv::Vec3b(b,g,r);
		else im_rgb(ii) = cv::Vec3b(255,255,255);

		for( int n = 0; n < 8; n++ )
		{
			int x = contourx[j] + dx8[n];
			int y = contoury[j] + dy8[n];
			if( (x >= 0 && x < width) && (y >= 0 && y < height) )
			{
				int ind = y*width + x;
				if(!have_col && !istaken[ind]) im_rgb(ind) = cv::Vec3b(0,0,0);
			}
		}
	}
}

/**
 * getSeeds2
 */
void SlicRGBD::getSeeds2(const pcl::PointCloud<pcl::PointXYZRGB> &cloud, const pcl::PointCloud<pcl::Normal> &normals, const cv::Mat_<cv::Vec3d> &im_lab, const std::vector<bool> &valid, std::vector<SlicRGBDPoint> &seeds, const int &step)
{
  int numseeds(0);
  int xe, n(0);
  int width = im_lab.cols;
  int height = im_lab.rows;

  int xstrips = (0.5+double(width)/double(step));
  int ystrips = (0.5+double(height)/double(step));

  int xerr = width - step*xstrips;
  if(xerr < 0){xstrips--; xerr = width - step*xstrips;}
  int yerr = height - step*ystrips;
  if(yerr < 0){ystrips--; yerr = height - step*ystrips;}

  double xerrperstrip = double(xerr)/double(xstrips);
  double yerrperstrip = double(yerr)/double(ystrips);

  int xoff = step/2;
  int yoff = step/2;

  numseeds = xstrips*ystrips;

  seeds.resize(numseeds);

  cv::Mat_<unsigned char> im_gray(im_lab.size());

  for (int i=0; i<im_lab.rows*im_lab.cols; i++)
    im_gray(i) = (unsigned char)im_lab(i)[0];
  
  cv::Sobel( im_gray, grad_x, CV_16S, 1, 0, 3, 1,0, cv::BORDER_DEFAULT );
  cv::Sobel( im_gray, grad_y, CV_16S, 0, 1, 3, 1,0, cv::BORDER_DEFAULT );

  int sx, sy, tmp_x, tmp_y;
  int h_seed_win = (step/4>0?step/4:1);
  int min_grad, grad;
  //cout<<"h_seed_win="<<h_seed_win<<endl;

  for( int y = 0; y < ystrips; y++ )
  {
    int ye = y*yerrperstrip;
    for( int x = 0; x < xstrips; x++ )
    {
      SlicRGBDPoint &pt = seeds[n];

      min_grad = INT_MAX; 
      xe = x*xerrperstrip;
      sx = (x*step+xoff+xe);
      sy = (y*step+yoff+ye);

      for (int v=-h_seed_win; v<=h_seed_win; v++)
      {
        for (int u=-h_seed_win; u<=h_seed_win; u++)
        {
          tmp_x = sx+u;
          tmp_y = sy+v;

          if (valid[tmp_y*width+tmp_x])
          {
            grad = abs(grad_x.at<short>(tmp_y,tmp_x))+abs(grad_y.at<short>(tmp_y,tmp_x)) ;
            if (grad < min_grad)
            {
              min_grad = grad;
              pt.x = tmp_x;
              pt.y = tmp_y;
            }
          }
        }
      }

      if (min_grad != INT_MAX)
      {
        const cv::Vec3d &lab = im_lab(pt.y,pt.x);
        pt.l = lab[0];
        pt.a = lab[1];
        pt.b = lab[2];
        pt.pt = cloud(pt.x,pt.y).getVector3fMap().cast<double>();
        pt.n = normals(pt.x,pt.y).getNormalVector3fMap().cast<double>();
        n++;
      }
    }
  }

  seeds.resize(n);
}

/**
 * getSeeds
 */
void SlicRGBD::getSeeds(const pcl::PointCloud<pcl::PointXYZRGB> &cloud, const pcl::PointCloud<pcl::Normal> &normals, const cv::Mat_<cv::Vec3d> &im_lab, const std::vector<bool> &valid, std::vector<SlicRGBDPoint> &seeds, const int &step)
{
  int numseeds(0);
  int xe, n(0);
  int width = im_lab.cols;
  int height = im_lab.rows;

  int xstrips = (0.5+double(width)/double(step));
  int ystrips = (0.5+double(height)/double(step));

  int xerr = width - step*xstrips;
  if(xerr < 0){xstrips--; xerr = width - step*xstrips;}
  int yerr = height - step*ystrips;
  if(yerr < 0){ystrips--; yerr = height - step*ystrips;}

  double xerrperstrip = double(xerr)/double(xstrips);
  double yerrperstrip = double(yerr)/double(ystrips);

  int xoff = step/2;
  int yoff = step/2;

  numseeds = xstrips*ystrips;

  seeds.resize(numseeds);

  for( int y = 0; y < ystrips; y++ )
  {
    int ye = y*yerrperstrip;
    for( int x = 0; x < xstrips; x++ )
    {
      SlicRGBDPoint &pt = seeds[n];
      xe = x*xerrperstrip;
      pt.x = (x*step+xoff+xe);
      pt.y = (y*step+yoff+ye);

      if (valid[pt.y*width+pt.x])
      {
        const cv::Vec3d &lab = im_lab(pt.y,pt.x);
        pt.l = lab[0];
        pt.a = lab[1];
        pt.b = lab[2];
        pt.pt = cloud(pt.x,pt.y).getVector3fMap().cast<double>();
        pt.n = normals(pt.x,pt.y).getNormalVector3fMap().cast<double>();
        n++;
      }
    }
  }

  seeds.resize(n);
}

/**
 * performSlicRGBD
 * Performs k mean segmentation. It is fast because it looks locally, not over the entire image.
 */
void SlicRGBD::performSlicRGBD(const pcl::PointCloud<pcl::PointXYZRGB> &cloud, const pcl::PointCloud<pcl::Normal> &normals, const cv::Mat_<cv::Vec3d> &im_lab,
                               const std::vector<bool> &valid, std::vector<SlicRGBDPoint> &seeds, cv::Mat_<int> &labels, const int &step)
{
  int width = im_lab.cols;
  int height = im_lab.rows;
  int sz = width*height;
  const int numk = seeds.size();
  int offset = step;
	
  clustersize.clear();
  clustersize.resize(numk,0);

  sigma.clear();
  sigma.resize(numk, SlicRGBDPoint());
  dists.resize(sz);

  double invwt_xy = 1.0/((step/param.compactness_image)*(step/param.compactness_image));
  double wt_xyz = param.compactness_xyz*param.compactness_xyz;
  double wt_cosa = param.weight_diff_normal_angle; 

  int idx, x1, y1, x2, y2;
  double dist;
  double dist_xy;
  double dist_xyz;
  double dist_cosa;
  double inv;
  const cv::Vec3d *ptr_lab = &im_lab(0);
  int *ptr_labels = &labels(0);

  for( int itr = 0; itr < 10; itr++ )
  {
    dists.assign(sz, DBL_MAX);

    for( int n = 0; n < numk; n++ )
    {
      const SlicRGBDPoint &pt = seeds[n];
      y1 = max(0.0, pt.y-offset);
      y2 = min((double)height, pt.y+offset);
      x1 = max(0.0, pt.x-offset);
      x2 = min((double)width, pt.x+offset);

      #pragma omp parallel for private(idx,dist,dist_xy,dist_xyz,dist_cosa)
      for( int y = y1; y < y2; y++ )
      {
        for( int x = x1; x < x2; x++ )
        {
          idx = y*width+x;

          if (valid[idx])
          {
            const cv::Vec3d &lab = ptr_lab[idx];
            const pcl::PointXYZRGB &pt3 = cloud.points[idx];
            const pcl::Normal &normal = normals.points[idx];

            dist = sqr(lab[0] - pt.l) + sqr(lab[1] - pt.a) + sqr(lab[2] - pt.b);
            dist_xy = sqr((x - pt.x)) + sqr(y - pt.y);
            //dist_xyz = fabs(pt.n.dot(pt3.getVector3fMap().cast<double>()-pt.pt));
            dist_xyz = (pt.pt-pt3.getVector3fMap().cast<double>()).squaredNorm();  // just a test
            dist_cosa = 1.-pt.n.dot(normal.getNormalVector3fMap().cast<double>());
//cout<<dist<<" "<<dist_xy<<" "<<dist_xyz<<" "<<dist_cosa<<", "<<dist_xy*invwt_xy<<" "<<dist_xyz*wt_xyz<<" "<<dist_cosa*wt_cosa<<endl;
            dist += (dist_xy*invwt_xy + dist_xyz*wt_xyz + dist_cosa*wt_cosa);

            //dist =  sqr(lab[0] - pt.l) + sqr(lab[1] - pt.a) + sqr(lab[2] - pt.b);
            //distxy =		sqr((x - pt.x)) + sqr(y - pt.y);
//            dist = (lab[0]-pt.l)*(lab[0]-pt.l) + (lab[1]-pt.a)*(lab[1]-pt.a) + (lab[2]-pt.b)*(lab[2]-pt.b);
//            distxy = (x-pt.x)*(x-pt.x) + (y-pt.y)*(y-pt.y);
//            dist += distxy*invwt;

            if( dist < dists[idx] )
            {
              dists[idx] = dist;
              ptr_labels[idx] = n;
            }
          }
        }
      }
    }

    sigma.assign(numk, SlicRGBDPoint());
    clustersize.assign(numk, 0);

    idx = 0;
    for( int r = 0; r < height; r++ )
    {
      for( int c = 0; c < width; c++, idx++ )
      {
        if (valid[idx] && ptr_labels[idx]!=-1)
        {
          SlicRGBDPoint &sig = sigma[ptr_labels[idx]];
          const cv::Vec3d &lab = ptr_lab[idx];
          const pcl::PointXYZRGB &pt3 = cloud.points[idx];
          const pcl::Normal &normal = normals.points[idx];
          sig.l += lab[0];
          sig.a += lab[1];
          sig.b += lab[2];
          sig.x += c;
          sig.y += r;
          sig.pt += pt3.getVector3fMap().cast<double>();
          sig.n += normal.getNormalVector3fMap().cast<double>();
          clustersize[ptr_labels[idx]] += 1.0;
        }
      }
    }

    //#pragma omp parallel for
    for( int k = 0; k < numk; k++ )
    {
      if( clustersize[k] <= 0 ) clustersize[k] = 1;
      inv = 1./clustersize[k];
      SlicRGBDPoint &pt = seeds[k];
      const SlicRGBDPoint &sig = sigma[k];

      pt.l = sig.l*inv;
      pt.a = sig.a*inv;
      pt.b = sig.b*inv;
      pt.x = sig.x*inv;
      pt.y = sig.y*inv;
      pt.pt = sig.pt*inv;
      pt.n = sig.n*inv;
      pt.n.normalize();
    }
  }
}



/**
 * enforceLabelConnectivity
 * 1. finding an adjacent label for each new component at the start
 * 2. if a certain component is too small, assigning the previously found
 *    adjacent label to this component, and not incrementing the label.
 */
void SlicRGBD::enforceLabelConnectivity(cv::Mat_<int> &labels, cv::Mat_<int> &out_labels, int& numlabels, const int& K)
{
	const int dx4[4] = {-1,  0,  1,  0};
	const int dy4[4] = { 0, -1,  0,  1};

  int width = labels.cols;
  int height = labels.rows;
	const int sz = width*height;
	const int SUPSZ = sz/K;
  out_labels = cv::Mat_<int>(height,width);
  out_labels.setTo(-1);
	int label(0);
	int* xvec = new int[sz];
	int* yvec = new int[sz];
	int oindex(0);
	int adjlabel(0);//adjacent label
  int *ol = &out_labels(0);
  int *il = &labels(0);

	for( int j = 0; j < height; j++ )
	{
		for( int k = 0; k < width; k++ )
		{
			if( 0 > ol[oindex] )
			{
				ol[oindex] = label;
				xvec[0] = k;
				yvec[0] = j;
				// Quickly find an adjacent label for use later if needed
				for( int n = 0; n < 4; n++ )
				{
					int x = xvec[0] + dx4[n];
					int y = yvec[0] + dy4[n];
					if( (x >= 0 && x < width) && (y >= 0 && y < height) )
					{
						int nindex = y*width + x;
						if(ol[nindex] >= 0) adjlabel = ol[nindex];
					}
				}

				int count(1);
				for( int c = 0; c < count; c++ )
				{
					for( int n = 0; n < 4; n++ )
					{
						int x = xvec[c] + dx4[n];
						int y = yvec[c] + dy4[n];

						if( (x >= 0 && x < width) && (y >= 0 && y < height) )
						{
							int nindex = y*width + x;

							if( 0 > ol[nindex] && il[oindex] == il[nindex] )
							{
								xvec[count] = x;
								yvec[count] = y;
								ol[nindex] = label;
								count++;
							}
						}

					}
				}
				// If segment size is less then a limit, assign an
				// adjacent label found before, and decrement label count.
				if(count <= SUPSZ >> 2)
				{
					for( int c = 0; c < count; c++ )
					{
						int ind = yvec[c]*width+xvec[c];
						ol[ind] = adjlabel;
					}
					label--;
				}
				label++;
			}
			oindex++;
		}
	}
	numlabels = label;

	if(xvec) delete [] xvec;
	if(yvec) delete [] yvec;
}

/**
 * @brief SlicRGBD::segmentSuperpixel
 * @param labels
 * @param numlabels
 */
void SlicRGBD::segmentSuperpixel(cv::Mat_<int> &labels, int& numlabels)
{
  if (cloud.get()==0) return;

  const int superpixelsize = (num_superpixel==-1?param.superpixelsize:0.5+double(cloud->width*cloud->height)/double(num_superpixel));

  int width  = cloud->width;
  int height = cloud->height;
  int sz = width*height;
  const int step = sqrt(double(superpixelsize))+0.5;
  seeds.clear();

  labels = cv::Mat_<int>(height,width);
  labels.setTo(-1);
  SlicRGBD::convertRGBtoLAB(*cloud, im_lab);

  getSeeds2(*cloud, *normals, im_lab, valid, seeds, step);
  performSlicRGBD(*cloud, *normals, im_lab, valid, seeds, labels, step);
  numlabels = seeds.size();

  cv::Mat_<int> new_labels;
  enforceLabelConnectivity(labels, new_labels, numlabels, double(sz)/double(step*step));
  new_labels.copyTo(labels);
}

/**
 * @brief SlicRGBD::setNumberOfSuperpixel
 * @param N
 */
void SlicRGBD::setNumberOfSuperpixel(int N)
{
  num_superpixel = N;//0.5+double(im_rgb.rows*im_rgb.cols)/double(K);
}

/**
 * @brief SlicRGBD::setSuperpixeSize
 * @param size
 */
void SlicRGBD::setSuperpixeSize(int size)
{
  param.superpixelsize = size;
  num_superpixel = -1;
}

/**
 * @brief SlicRGBD::setCloud
 * @param _cloud
 * @param _normals
 */
void SlicRGBD::setCloud(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &_cloud, const pcl::PointCloud<pcl::Normal>::Ptr &_normals)
{
  if (_cloud.get()==0) return;

  cloud = _cloud;

  if (_normals.get()==0)
  {
    normals.reset(new pcl::PointCloud<pcl::Normal>());
    pcl::IntegralImageNormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
    ne.setNormalEstimationMethod (ne.AVERAGE_3D_GRADIENT);
    ne.setMaxDepthChangeFactor(param.normals_max_depth_change_factor);//0.02f);
    ne.setNormalSmoothingSize(param.normals_smoothing_size);//20.0f);
    ne.setDepthDependentSmoothing(false);//param.normals_depth_dependent_smoothing);
    ne.setInputCloud(cloud);
    ne.setViewPoint(0,0,0);
    ne.compute(*normals);
  }
  else
  {
    normals = _normals;
  }

  const pcl::PointCloud<pcl::PointXYZRGB> &ref_cloud = *cloud;
  const pcl::PointCloud<pcl::Normal> &ref_normals = *normals;

  valid.clear();
  valid.resize(ref_cloud.points.size(), true);

  for (unsigned i=0; i<ref_cloud.points.size(); i++)
    if (isnan(ref_cloud.points[i].getVector3fMap()) || isnan(ref_normals.points[i].getNormalVector3fMap()))
      valid[i] = false;
}

}

