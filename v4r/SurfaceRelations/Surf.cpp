/**
 * @file Surf.h
 * @author Richtsfeld
 * @date December 2011
 * @version 0.1
 * @brief Calculate surf features to compare surface texture.
 */

#include "Surf.h"
#include <stdio.h>

namespace surface
{


/************************************************************************************
 * Constructor/Destructor
 */

Surf::Surf()
{
  have_input_image = false;
  storage_compute = cvCreateMemStorage(0);
  storage_compare_1 = cvCreateMemStorage(0);
  storage_compare_2 = cvCreateMemStorage(0);

  int hessian = 20;     // default: 500
  bool extended = 0;    // default: 1
  params  = cvSURFParams(hessian, extended);                           /// TODO parameter
}

Surf::~Surf()
{
}

// ================================= Private functions ================================= //

double Surf::compareSURFDescriptors(const float* descriptor_1, 
                                    const float* descriptor_2, 
                                    int descriptorsCount,
                                    float lastMinSquaredDistance)

{
  double totalCost = 0;
  for (int i = 0; i < descriptorsCount; i += 4) {
    cv::Vec4f descriptor1(descriptor_1[i+0], descriptor_1[i+1], descriptor_1[i+2], descriptor_1[i+3]);
    cv::Vec4f descriptor2(descriptor_2[i+0], descriptor_2[i+1], descriptor_2[i+2], descriptor_2[i+3]);
    totalCost += pow((norm(descriptor1 - descriptor2)), 2);
    if (totalCost > lastMinSquaredDistance)
      break;
  }
  return totalCost;
}


int Surf::findNaiveNearestNeighbor(const float* descriptor_1, 
                             const CvSURFPoint* keypoint_1, 
                             CvSeq* descriptors_2, 
                             CvSeq* keypoints_2,
                             double &distance)

{
  std::map<double, unsigned> best_matches;
  
  int descriptorsCount = (int)(descriptors_2->elem_size/sizeof(float));
  double minSquaredDistance = std::numeric_limits<double>::max();
  double lastMinSquaredDistance = std::numeric_limits<double>::max();

// printf("  findNaiveNearestNeighbor: of %i\n", descriptors_2->total);
  int neighbor;
  for (int i = 0; i < descriptors_2->total; i++) 
  {
    const CvSURFPoint* keypoint_2 = (const CvSURFPoint*) cvGetSeqElem(keypoints_2, i);
    const float* descriptor_2 = (const float*) cvGetSeqElem(descriptors_2, i);

    if (keypoint_1->laplacian != keypoint_2->laplacian)
        continue; // Don't worry about key points unless laplacian signs are equal

    double squaredDistance = compareSURFDescriptors(descriptor_1, 
                                                    descriptor_2, 
                                                    descriptorsCount, 
                                                    lastMinSquaredDistance);

// printf("  squaredDistance: %4.3f", squaredDistance);
    
    if (squaredDistance < minSquaredDistance) {
        neighbor = i;
        lastMinSquaredDistance = minSquaredDistance;
        minSquaredDistance = squaredDistance;
    } else if (squaredDistance < lastMinSquaredDistance) {
        lastMinSquaredDistance = squaredDistance;
    }
    best_matches.insert(std::make_pair(squaredDistance, i));
// printf("   (min: %4.3f)\n", minSquaredDistance);
  }

  matches.push_back(best_matches); // store all matches global
  
  if (minSquaredDistance < 0.7 * lastMinSquaredDistance)                  /// TODO HERE IS A THRESHOLD !!!
  {                                                                       /// The second nearest neighbor muss um ca. ein Drittel schlechter sein!
    distance = minSquaredDistance;
    return neighbor;
  }
  distance = -1;
  return -1;
}

// ================================= Public functions ================================= //


void Surf::setInputImage(IplImage *_image)
{
  gray_image = cvCreateImage(cvGetSize(_image), 8, 1);
// printf("[Surf::setInputImage] Error: Change to new opencv commands! NOW!\n");
  cvCvtColor(_image, gray_image, CV_RGB2GRAY);  
  have_input_image = true;
  width = cvGetSize(gray_image).width;
  height = cvGetSize(gray_image).height;
}

void Surf::compute()
{
  if(!have_input_image) {
    printf("[Surf::compute] Error: No input image available.\n");
    return;
  }  

  keypoints = 0;
  descriptors = 0;
  cvExtractSURF(gray_image, 0, &keypoints, &descriptors, storage_compute, params);
}


double Surf::compare(std::vector<int> indices_0, std::vector<int> indices_1)
{
  if(!have_input_image) {
    printf("[Surf::compute] Error: No input image available.\n");
    return 0.;
  }  
  
  IplImage *mask_0 = cvCreateImage(cvSize(width, height), 8, 1);                 // gray-level image
  IplImage *mask_1 = cvCreateImage(cvSize(width, height), 8, 1);                 // gray-level image
  for(int i=0; i<width*height; i++) {
    mask_0->imageData[i] = 0;
    mask_1->imageData[i] = 0;
  }
  for(unsigned i=0; i<indices_0.size(); i++) {
    mask_0->imageData[indices_0[i]] = 1; 
  }
  for(unsigned i=0; i<indices_1.size(); i++) {
    mask_1->imageData[indices_1[i]] = 1; 
  }
  
  matches.clear();
  first.clear();
  second.clear();
  surfDistance.clear();
  keypoints_1 = 0;
  keypoints_2 = 0;
  descriptors_1 = 0;
  descriptors_2 = 0;
  double surfRelation = 0;
  
  cvExtractSURF(gray_image, mask_0, &keypoints_1, &descriptors_1, storage_compare_1, params);
  cvExtractSURF(gray_image, mask_1, &keypoints_2, &descriptors_2, storage_compare_2, params);
// printf("[Surf::compare] Keypoints.size: %u - %u\n", keypoints_1->total, keypoints_2->total);  

  for(int i=0; i<descriptors_1->total; i++)
  {
    const CvSURFPoint* keypoint_1 = (const CvSURFPoint*) cvGetSeqElem(keypoints_1, i);
    const float* descriptor_1 = (const float*) cvGetSeqElem(descriptors_1, i);

    double distance = 0.0;
    int nngbh = findNaiveNearestNeighbor(descriptor_1, keypoint_1, descriptors_2, keypoints_2, distance);
// printf("Nearest neighbor for %u => %i (%4.3f)\n", i, nngbh, distance);
  }
  
  std::map<double, unsigned>::iterator it;    // iterator for match map
  
  /// remove all matches, where 2nd match is nearer than 0.7
//   for(unsigned i=0; i<matches.size(); i++) {
//     if(matches[i].size() > 1) {
//       it = matches[i].begin();
//       double val_0 = (*it).first;
//       it++;
//       double val_1 = (*it).first;
//       if((val_0 / val_1) > 0.7) {                             /// TODO Threshold
//         matches[i].clear();
//       }
//     }
//   }
  

  bool succeed = true;
  while(succeed)
  {
    succeed = false;
    
//     printf("\nmatch table:\n");
//     for(unsigned i=0; i<matches.size(); i++) {
//       int cnt = 0;
//       for(it = matches[i].begin(); it != matches[i].end(); it++) {
//         printf("%i: %u - %u  => %4.3f\n", cnt++, i, (*it).second, (*it).first);
//       }
//     }

    /// find smallest element
    double smallest = 100.;
    unsigned pos = 1000;
    unsigned sec = 1000;
    for(unsigned i=0; i<matches.size(); i++) {
      if(matches[i].size() > 0) {
        it = matches[i].begin();
        if((*it).first < smallest) {
          smallest = (*it).first;
          sec = (*it).second;
          pos = i;
          succeed = true;
        }
      }
    }
    
// printf("position of smallest: %u => %4.3f\n", pos, smallest);
    /// store smallest element
    if(succeed) 
    {
      first.push_back(pos);
      second.push_back(sec);
      surfDistance.push_back(smallest);
// printf("%i: push back: %u-%u => %4.3f\n", first.size(), pos, sec, smallest);

      /// delete map of biggest element
      matches[pos].clear();
      
      /// go through all maps and delete entrys with sec!
      for(int i=0; i<matches.size(); i++)
      {
        for(it = matches[i].begin(); it != matches[i].end(); it++)
        {
          if((*it).second == sec)
            matches[i].erase(it);
        }
      }
    }
  }

  /// calculate matching relation value
//   printf("matched points: %u\n", first.size());
  for(unsigned i=0; i<first.size(); i++)
//   {
//     printf("  %u-%u with %4.3f\n", first[i], second[i], surfDistance[i]);
    surfRelation += surfDistance[i];  
//   }  
  if(keypoints_1->total < keypoints_2->total)
    surfRelation = surfRelation * keypoints_1->total / (pow(first.size(), 2));
  else
    surfRelation = surfRelation * keypoints_2->total / (pow(first.size(), 2));

//   cvReleaseImage(&gray_image);
//   have_input_image = false;

  if(surfRelation != surfRelation) {
    printf("[Surf::compare] Warning: Surf relation value is nan! Return 0.\n");
    return 0.;
  }
  return surfRelation;
}


/** TODO: For debugging: Draw surfs and connections **/
void Surf::drawFeatures(cv::Mat_<cv::Vec3b> &_feature_image)
{
// printf("Surf::drawFeatures start\n");
  CvSURFPoint* point;
  cv::Scalar color0(0,0,0,0);
  for(int i=0; i < keypoints_1->total; i++)
  {
    point=(CvSURFPoint*)cvGetSeqElem(keypoints_1, i);
    cv::Point center;
    center.x = point->pt.x;
    center.y = point->pt.y;
    cv::circle(_feature_image, center, point->size, color0, 1, 8, 0);
  }  
  cv::Scalar color1(255,0,0,0);
  for(int i=0; i < keypoints_2->total; i++)
  {
    point=(CvSURFPoint*)cvGetSeqElem(keypoints_2, i);
    cv::Point center;
    center.x = point->pt.x;
    center.y = point->pt.y;
    cv::circle(_feature_image, center, point->size, color1, 1, 8, 0);
  }  

  // draw connections
  cv::Scalar color2(0,255,0,0);
  for(unsigned i=0; i<first.size(); i++)
  {
// printf("Draw connection between: %u-%u\n", first[i], second[i]);
    cv::Point pt1, pt2;
    point=(CvSURFPoint*)cvGetSeqElem(keypoints_1, first[i]);
    pt1.x = point->pt.x;
    pt1.y = point->pt.y;
    point=(CvSURFPoint*)cvGetSeqElem(keypoints_2, second[i]);
    pt2.x = point->pt.x;
    pt2.y = point->pt.y; 
    cv::line(_feature_image, pt1, pt2, color2, 1, 8, 0);
  }
  cv::imshow("SURF features", _feature_image);
// printf("Surf::drawFeatures done\n");
}

}












