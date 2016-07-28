/**
 * $Id$
 * Johann Prankl, 2014-3
 * johann.prankl@josephinum.at
 */


#include <cfloat>
#include <cmath>
#include <iostream>
#include <fstream>
#include <opencv2/imgproc/imgproc.hpp>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <v4r/segmentation/Slic.h>

namespace v4r
{

using namespace std;

Slic::Slic()
{
}

Slic::~Slic()
{
}

/**
 * convertRGBtoLAB
 */
void Slic::convertRGBtoLAB(const cv::Mat_<cv::Vec3b> &im_rgb, cv::Mat_<cv::Vec3d> &im_lab)
{
  im_lab = cv::Mat_<cv::Vec3d>(im_rgb.size());

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
  for (int v=0; v<im_rgb.rows; v++)
  {
    for (int u=0; u<im_rgb.cols; u++)
    {
      const cv::Vec3b &rgb = im_rgb(v,u);
      cv::Vec3d &lab = im_lab(v,u);

      R = rgb[0]*inv_255;
      G = rgb[1]*inv_255;
      B = rgb[2]*inv_255;

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
 * convertRGBtoLAB
 */
void Slic::convertRGBtoLAB(double r, double g, double b, double &labL, double &labA, double &labB)
{
  double R, G, B;
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

  R = r*inv_255;
  G = g*inv_255;
  B = b*inv_255;

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

  labL = 116.0*fy-16.0;
  labA = 500.0*(fx-fy);
  labB = 200.0*(fy-fz);
}

/**
 * drawContours
 */
void Slic::drawContours(cv::Mat_<cv::Vec3b> &im_rgb, const cv::Mat_<int> &labels, int r, int g, int b)
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
 * getSeeds
 */
void Slic::getSeeds(const cv::Mat_<cv::Vec3d> &_im_lab, std::vector<SlicPoint> &_seeds, const int &step)
{
  int numseeds(0);
  int xe, n(0);
  int width = _im_lab.cols;
  int height = _im_lab.rows;

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

  _seeds.resize(numseeds);

  for( int y = 0; y < ystrips; y++ )
  {
    int ye = y*yerrperstrip;
    for( int x = 0; x < xstrips; x++ )
    {
      SlicPoint &pt = _seeds[n];
      xe = x*xerrperstrip;
      pt.x = (x*step+xoff+xe);
      pt.y = (y*step+yoff+ye); 
      const cv::Vec3d &lab = _im_lab(pt.y,pt.x);
      pt.l = lab[0];
      pt.a = lab[1];
      pt.b = lab[2];
			n++;
		}
	}
}

/**
 * performSlic
 * Performs k mean segmentation. It is fast because it looks locally, not over the entire image.
 */
void Slic::performSlic(const cv::Mat_<cv::Vec3d> &im_lab, std::vector<SlicPoint> &_seeds, cv::Mat_<int> &labels, const int &step, const double &m)
{
  int width = im_lab.cols;
  int height = im_lab.rows;
  int sz = width*height;
    const int numk = _seeds.size();
	int offset = step;
	
  clustersize.clear();
  clustersize.resize(numk,0);

  sigma.clear();
  sigma.resize(numk, SlicPoint()); 
  dists.resize(sz);

	double invwt = 1.0/((step/m)*(step/m));

	int idx, x1, y1, x2, y2;
	double dist;
	double distxy;
  double inv;
  const cv::Vec3d *ptr_lab = &im_lab(0);
  int *ptr_labels = &labels(0);

	for( int itr = 0; itr < 10; itr++ )
	{
    dists.assign(sz, DBL_MAX);

		for( int n = 0; n < numk; n++ )
		{
      const SlicPoint &pt = _seeds[n];
      y1 = max(0.0, pt.y-offset);
      y2 = min((double)height, pt.y+offset);
      x1 = max(0.0, pt.x-offset);
      x2 = min((double)width, pt.x+offset);

      //#pragma omp parallel for private(idx,dist,distxy)
			for( int y = y1; y < y2; y++ )
			{
				for( int x = x1; x < x2; x++ )
				{
          idx = y*width+x;
          const cv::Vec3d &lab = ptr_lab[idx];

          //dist =  sqr(lab[0] - pt.l) + sqr(lab[1] - pt.a) + sqr(lab[2] - pt.b);
					//distxy =		sqr((x - pt.x)) + sqr(y - pt.y);
          dist = (lab[0]-pt.l)*(lab[0]-pt.l) + (lab[1]-pt.a)*(lab[1]-pt.a) + (lab[2]-pt.b)*(lab[2]-pt.b);
          distxy = (x-pt.x)*(x-pt.x) + (y-pt.y)*(y-pt.y);
					dist += distxy*invwt;

					if( dist < dists[idx] )
					{
						dists[idx] = dist;
						ptr_labels[idx] = n;  
          }
        }
      }
    }

    sigma.assign(numk, SlicPoint());
		clustersize.assign(numk, 0);

    idx = 0;
		for( int r = 0; r < height; r++ )
		{
			for( int c = 0; c < width; c++, idx++ )
			{
        SlicPoint &sig = sigma[ptr_labels[idx]];
        const cv::Vec3d &lab = ptr_lab[idx];
        sig.l += lab[0];
        sig.a += lab[1];
        sig.b += lab[2];
        sig.x += c;
        sig.y += r;
				clustersize[ptr_labels[idx]] += 1.0;
			}
		}

		for( int k = 0; k < numk; k++ )
		{
      if( clustersize[k] <= 0 ) clustersize[k] = 1;
      inv = 1./clustersize[k];
      SlicPoint &pt = _seeds[k];
      const SlicPoint &sig = sigma[k];

			pt.l = sig.l*inv;
			pt.a = sig.a*inv;
			pt.b = sig.b*inv;
			pt.x = sig.x*inv;
			pt.y = sig.y*inv;
		}
	}
}



/**
 * enforceLabelConnectivity
 * 1. finding an adjacent label for each new component at the start
 * 2. if a certain component is too small, assigning the previously found
 *    adjacent label to this component, and not incrementing the label.
 */
void Slic::enforceLabelConnectivity(cv::Mat_<int> &labels, cv::Mat_<int> &out_labels, int& numlabels, const int& K)
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
 * segmentSuperpixelSize
 * given an desired size
 */
void Slic::segmentSuperpixelSize(const cv::Mat_<cv::Vec3b> &im_rgb, cv::Mat_<int> &labels, int &numlabels, const int &superpixelsize, const double& compactness)
{
	int width  = im_rgb.cols;
	int height = im_rgb.rows;
	int sz = width*height;
  const int step = sqrt(double(superpixelsize))+0.5;
  seeds.clear();

  labels = cv::Mat_<int>(im_rgb.size());
  labels.setTo(-1);
  Slic::convertRGBtoLAB(im_rgb, im_lab);
  //cv::cvtColor(im_rgb, im_lab, CV_RGB2Lab);

	getSeeds(im_lab, seeds, step);
	performSlic(im_lab, seeds, labels, step, compactness);
	numlabels = seeds.size();

  cv::Mat_<int> new_labels;
	enforceLabelConnectivity(labels, new_labels, numlabels, double(sz)/double(step*step));
  new_labels.copyTo(labels);
}

/**
 * segmentSuperpixelNumber
 * given a desired number of superpixel
 */
void Slic::segmentSuperpixelNumber(const cv::Mat_<cv::Vec3b> &im_rgb,
        cv::Mat_<int> &labels, int& numlabels, const int& K, const double& compactness)
{
  const int superpixelsize = 0.5+double(im_rgb.rows*im_rgb.cols)/double(K);
  segmentSuperpixelSize(im_rgb, labels, numlabels, superpixelsize, compactness);
}


}

