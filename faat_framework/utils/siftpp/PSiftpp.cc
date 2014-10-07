/**
 * $Id$
 *
 * Copyright (c) 2014, Johann Prankl
 * @author Johann Prankl (prankl@acin.tuwien.ac.at)
 */

#include "PSiftpp.hh"

namespace P 
{

using namespace std;

PSiftpp::PSiftpp(const Parameter &p)
 : param(p), width(0), height(0)
{
}

PSiftpp::~PSiftpp()
{
}




/************************************** PRIVATE ************************************/

/**
 * initSift
 */
void PSiftpp::initSift(int w, int h)
{
  width = w;
  height = h;
  O      = param.octaves ;
  S      = param.levels ;
  omin   = param.first_level ;
  sigman = .5 ;
  sigma0 = 1.6 * powf(2.0f, 1.0f / S) ;

  if(O < 1) 
  {
    O = std::max( int( std::floor(log2(std::min(width, height))) - omin -3), 1);
  }

  sift = new VL::Sift((const VL::pixel_t*)0,0,0, sigman, sigma0, O, S, omin, -1, S+1);
}


/**
 * TransformToRootSIFT
 * computes the square root of the L1 normalized SIFT vectors.
 * Then the Euclidean distance is equivalent to using the Hellinger kernel 
 *  to compare the original SIFT vectors
 */
void PSiftpp::TransformToRootSIFT(cv::Mat& descriptors) const
{
  float norm;

  for (int i=0; i<descriptors.rows; i++)
  {
    Eigen::Map<Eigen::VectorXf> desc(&descriptors.at<float>(i,0), descriptors.cols);
    norm = desc.lpNorm<1>();
    desc.array() /= norm;
    desc.array() = desc.array().sqrt();
  }
}

/**
 * convertGrayToFloat
 */
void PSiftpp::convertGrayToFloat(const cv::Mat_<unsigned char> &im_gray, std::vector<float> &im_float) const 
{
  unsigned size = im_gray.rows*im_gray.cols;
  im_float.resize(size);
  
  for (unsigned i=0; i<size; i++)
    im_float[i] = float(im_gray(i))/255.;
}

void PSiftpp::detectImpl( const cv::Mat& image, vector<cv::KeyPoint>& keypoints, const cv::Mat& mask) const
{
  keypoints.clear();

  cv::Mat_<unsigned char> im_gray;
  std::vector<float> im_float;

  if( image.type() != CV_8U ) cv::cvtColor( image, im_gray, CV_RGB2GRAY );
  else im_gray = image;

  convertGrayToFloat(im_gray, im_float);
  VL::Sift* s = (VL::Sift*)&(*sift);
  s->process(&im_float[0], im_gray.cols, im_gray.rows) ;

  // compute sift
  s->detectKeypoints(param.threshold, param.edge_threshold) ;
  s->setNormalizeDescriptor( true ) ;
  s->setMagnification( param.magnif ) ;

  VL::float_t angles[4];
  int z=0,nangles;
  keypoints.reserve( s->keypointsEnd()-s->keypointsBegin() );

  if (mask.size() != image.size() || mask.type()!=CV_8U)
  {
    for(VL::Sift::KeypointsConstIter iter=s->keypointsBegin(); iter!=s->keypointsEnd(); ++iter) 
    {
      nangles = s->computeKeypointOrientations(angles, *iter);

      for (int i=0; i<nangles; i++)
      {
        keypoints.push_back( cv::KeyPoint( iter->x,iter->y, iter->sigma*3, 
                                           -angles[i]*180/M_PI, 1,iter->o,z ) );
        z++;
      }
    }
  }
  else
  {
    for(VL::Sift::KeypointsConstIter iter=s->keypointsBegin(); iter!=s->keypointsEnd(); ++iter) 
    {
      if (mask.at<unsigned char>((int)(iter->y+.5),(int)(iter->x+.5)) > 0)
      {
        nangles = s->computeKeypointOrientations(angles, *iter);

        for (int i=0; i<nangles; i++)
        {
          keypoints.push_back( cv::KeyPoint( iter->x,iter->y, iter->sigma*3, 
                                             -angles[i]*180/M_PI, 1,iter->o,z ) );
          z++;
        }
      }
    }

  }
}

/**
 * compute descriptors for given keypoints
 */
void PSiftpp::computeImpl(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) const
{
  if (keypoints.size()==0)
  {
    descriptors = cv::Mat();
    return;
  }

  cv::Mat_<unsigned char> im_gray;
  std::vector<float> im_float;

  if( image.type() != CV_8U ) cv::cvtColor( image, im_gray, CV_RGB2GRAY );
  else im_gray = image;

  convertGrayToFloat(im_gray, im_float);
  VL::Sift* s = (VL::Sift*)&(*sift);
  s->process(&im_float[0], im_gray.cols, im_gray.rows) ;

  VL::float_t desc[128];
  VL::Sift::Keypoint key;
  descriptors = cv::Mat_<float>(keypoints.size(),128);

  for (unsigned i=0; i<keypoints.size(); i++) 
  {
    cv::KeyPoint &k = keypoints[i];
    key = s->getKeypoint(k.pt.x,k.pt.y, k.size/3.);
    s->computeKeypointDescriptor(desc, key, -k.angle*M_PI/180.) ;
    float *d = descriptors.ptr<float>(i,0);
    for (unsigned j=0; j<128; j++) d[j] = desc[j];
  }

  if (param.computeRootSIFT) TransformToRootSIFT(descriptors);
}




/************************************** PUBLIC ************************************/

/** 
 * compute dog keypoints and sift descriptor
 */
void PSiftpp::detect( const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors, const cv::Mat& mask)
{
  keypoints.clear();
  if( image.type() != CV_8U ) cv::cvtColor( image, im_gray, CV_RGB2GRAY );
  else im_gray = image;

  if (sift.empty() || width!=image.cols || height!=image.rows)
    initSift(image.cols, image.rows);

  convertGrayToFloat(im_gray, im_float);
  sift->process(&im_float[0], width, height) ;

  // compute sift
  sift->detectKeypoints(param.threshold, param.edge_threshold) ;
  sift->setNormalizeDescriptor( true ) ;
  sift->setMagnification( param.magnif ) ;

  VL::Sift &s = *sift;
  VL::float_t angles[4];
  int z=0,nangles;
  VL::float_t desc[128];
  cv::Mat tmp_desc = cv::Mat_<float>(0,128);
  keypoints.reserve( s.keypointsEnd()-s.keypointsBegin() );

  if (mask.size() != image.size() || mask.type()!=CV_8U)
  {
    for(VL::Sift::KeypointsConstIter iter=s.keypointsBegin(); iter!=s.keypointsEnd(); ++iter) 
    {
      nangles = s.computeKeypointOrientations(angles, *iter);

      for (int i=0; i<nangles; i++)
      {
        keypoints.push_back( cv::KeyPoint( iter->x,iter->y, iter->sigma*3, 
                                           -angles[i]*180/M_PI, 1,iter->o,z ) );
        s.computeKeypointDescriptor(desc, *iter, angles[i]) ;
        tmp_desc.push_back(cv::Mat(1,128,CV_32F));
        float *d = tmp_desc.ptr<float>(tmp_desc.rows-1,0);
        for (unsigned j=0; j<128; j++) d[j] = desc[j];
        z++;
      }
    }
  }
  else
  {
    for(VL::Sift::KeypointsConstIter iter=s.keypointsBegin(); iter!=s.keypointsEnd(); ++iter) 
    {
      if (mask.at<unsigned char>((int)(iter->y+.5),(int)(iter->x+.5)) > 0)
      {
        nangles = s.computeKeypointOrientations(angles, *iter);

        for (int i=0; i<nangles; i++)
        {
          keypoints.push_back( cv::KeyPoint( iter->x,iter->y, iter->sigma*3, 
                                             -angles[i]*180/M_PI, 1,iter->o,z ) );
          s.computeKeypointDescriptor(desc, *iter, angles[i]) ;
          tmp_desc.push_back(cv::Mat(1,128,CV_32F));
          float *d = tmp_desc.ptr<float>(tmp_desc.rows-1,0);
          for (unsigned j=0; j<128; j++) d[j] = desc[j];
          z++;
        }
      }
    }
  }
  
  tmp_desc.copyTo(descriptors);

  if (param.computeRootSIFT) TransformToRootSIFT(descriptors);
}

/**
 * compute dog keypoints
 */
void PSiftpp::detect( const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, const cv::Mat& mask)
{
  if (sift.empty() || width!=image.cols || height!=image.rows)
    initSift(image.cols, image.rows);

  detectImpl(image, keypoints, mask);
}

/**
 * compute descriptors for given keypoints
 */
void PSiftpp::compute(const cv::Mat& image, vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors )
{
  if (sift.empty() || width!=image.cols || height!=image.rows)
    initSift(image.cols, image.rows);

  computeImpl(image, keypoints, descriptors);
}


}

