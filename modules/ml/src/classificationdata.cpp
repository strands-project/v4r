/*
    Copyright (c) 2013, <copyright holder> <email>
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
        * Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.
        * Neither the name of the <organization> nor the
        names of its contributors may be used to endorse or promote products
        derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY <copyright holder> <email> ''AS IS'' AND ANY
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL <copyright holder> <email> BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


#include <v4r/ml/classificationdata.h>
#include "boost/filesystem.hpp"

using namespace boost::filesystem;
using namespace v4r::RandomForest;

ClassificationData::ClassificationData()
{

}

int ClassificationData::GetDimensions()
{
  return dimensions;
}

std::vector<unsigned int> ClassificationData::generateRandomIndices(unsigned int n, unsigned int totalPoints)
{
    std::vector<unsigned int> idx(totalPoints);

    for(unsigned int i=0; i<totalPoints; i++)
        idx[i] = i;

    for (unsigned int i = totalPoints-1; i > 0; i--)
    {
      boost::uniform_int<int> intDist(0, i);
      unsigned int j = intDist(randomGenerator);
      swap(idx, i, j);
    }

    return std::vector<unsigned int>(idx.begin(), idx.begin()+n);
}

void ClassificationData::swap(std::vector<unsigned int>& array, unsigned int idx1, unsigned int idx2)
{
    unsigned int tmp = array[idx1];
    array[idx1] = array[idx2];
    array[idx2] = tmp;
}

std::vector< unsigned int > ClassificationData::NewBag(float baggingRatio)
{
  // creates a new "bag" of training data containing the same number of
  // points of each label
  
  data.clear();
  trainingLabels.clear();   
  
  int nPoints = totalPoints * baggingRatio;
  unsigned int maxPpLabel = nPoints / availableLabels.size();

  nPoints = 0;

  // check how many points are available (some labels might have less points than demanded by bagging ratio)
  for(unsigned int i=0; i<availableLabels.size(); i++)
  {
      nPoints += std::min(pointsPerLabel[availableLabels[i]], maxPpLabel); // ? n : maxPpLabel;
  }

  labelWeights.clear();

  // calculate label weights for inbalanced datasets
  for(unsigned int i=0; i < availableLabels.size(); ++i)
  {
      double n = double (std::min(pointsPerLabel[availableLabels[i]], maxPpLabel));
      double N = double(nPoints);
      labelWeights.push_back(N/n);
  }

  data.assign(nPoints*dimensions, 0.0f);
  trainingLabels.reserve(nPoints);
  std::vector<unsigned int > indices(nPoints);
  
  float value;

  unsigned int cnt = 0;

  for(unsigned int i=0; i<availableLabels.size(); i++)
  {     
    std::string filename = str(boost::format("%1$s/%2$04d.data") % directory % availableLabels[i]);    
    std::ifstream trainingfile;
    trainingfile.open(filename.c_str());

    unsigned int n = std::min(maxPpLabel, pointsPerLabel[availableLabels[i]]);
    std::vector<unsigned int> linenumbers = generateRandomIndices(n, pointsPerLabel[availableLabels[i]]);

    for(unsigned int j=0; j < n; j++)
    {
      long long int idx = linenumbers[j];
      long long int pos = idx* (dimensions*fieldWidth);
      trainingfile.seekg(pos);            
            	  
      for(int k=0; k < dimensions; ++k)      
      {
		trainingfile >> value;
        int o = cnt*dimensions+k;
        data[o] = value;
      }

      trainingLabels.push_back(i);
      indices[cnt] = cnt;

      cnt++;
    }
    
    trainingfile.close();    
  }

  return indices;
}

int ClassificationData::LoadChunkForLabel(int labelID, int nPoints)
{
  data.clear();
    
  std::string filename = str(boost::format("%1$s/%2$04d.data") % directory % labelID);    
  std::ifstream trainingfile;
  trainingfile.open(filename.c_str());
  
  data.reserve(nPoints*dimensions);
  
  trainingfile.seekg(trainingDataFilePos[labelID]);
  
  float value;
  int i=0;
  
  for(; i<nPoints && trainingfile.good(); ++i)
  {
	for(int k=0; k < dimensions; ++k)      
	{
	  trainingfile >> value;
	  data[i*dimensions+k] = value;
	}
  }
  
  trainingDataFilePos[labelID] = trainingfile.tellg();
  
  
  trainingfile.close();  
  return i;
}

// calculates range of selected data points for given dimension
std::pair<float, float> ClassificationData::GetMinMax(std::vector< unsigned int >::iterator startidx, std::vector< unsigned int >::iterator stopidx, int dimension)
{
  float min = GetFeature(*startidx, dimension);
  float max = min;
  
  float f = 0.0f;
  
  for(std::vector<unsigned int>::iterator i = startidx; i != stopidx; i++){
    f = GetFeature(*i,dimension);
    
    if(f < min)
      min = f;
    if(f > max)
      max = f;    
  }
 
  return std::pair<float,float>(min,max);
}

std::pair<float, float> ClassificationData::GetMinMax(int dimension)
{
  float min = data[dimension];
  float max = min;
  
  float f = 0.0f;
  
  for(unsigned int i=0; i<data.size()/dimensions; ++i){
    f = data[i*dimensions + dimension];
    
    if(f < min)
      min = f;
    if(f > max)
      max = f;    
  }
 
  return std::pair<float,float>(min,max);
}

int ClassificationData::GetCount()
{
  return totalPoints;
}

float ClassificationData::GetFeature(int pointIdx, int featureIdx)
{
  return data[pointIdx*dimensions + featureIdx];
}

std::vector< float > ClassificationData::GetFeatures(int pointIdx)
{  
  std::vector<float> p(&data[pointIdx*dimensions], &data[(pointIdx+1)*dimensions]);
  return p;
}


void ClassificationData::LoadDemoSpiral(int nPoints, float noise)
{
  trainingLabels.clear();
  data.clear();
  directory = "";
  
  srand (time(NULL));
  
  float pi = 3.14159;    
  float step = (2*pi-0.5) / (nPoints-1);
  
  float xsum = 0.0f;
  float ysum = 0.0f;
  float x, y;
  
  for(float t=0.5; t < 2*pi; t+=step){
    trainingLabels.push_back(0);
    x = t * cos(t) + ((float)rand()) / RAND_MAX * noise - noise/2.0;
    xsum += x;
    data.push_back(x);      
    y = t * sin(t) + ((float)rand()) / RAND_MAX * noise - noise/2.0;
    ysum += y;
    data.push_back(y);
    
    trainingLabels.push_back(1);
    x = t * cos(t+2) + ((float)rand()) / RAND_MAX * noise - noise/2.0;
    xsum += x;
    data.push_back(x);
    y = t * sin(t+2) + ((float)rand()) / RAND_MAX * noise - noise/2.0;
    ysum += y;
    data.push_back(y);
    
    trainingLabels.push_back(2);
    x = t * cos(t+4) + ((float)rand()) / RAND_MAX * noise - noise/2.0;
    xsum += x;
    data.push_back(x);
    y = t * sin(t+4) + ((float)rand()) / RAND_MAX * noise - noise/2.0;
    ysum += y;
    data.push_back(y);    
  }
  
  xsum /= nPoints;
  ysum /= nPoints;
  
  for(unsigned int i=0; i < data.size() / 2; i++){
    data[i*2] = data[i*2]-xsum;
    data[i*2+1] = data[i*2+1]-ysum;
  }
  
  labelStatus = LABELED;
  this->dimensions = 2;
}

// splits selected data points into "left" and "right" according to given threshold and feature dimension
std::vector< unsigned int >::iterator ClassificationData::Partition(std::vector< unsigned int >::iterator startidx, std::vector< unsigned int >::iterator stopidx, int dimension, float threshold)
{
  std::vector<unsigned int>::iterator i = startidx;
  std::vector<unsigned int>::iterator j = stopidx-1;
  unsigned int swapidx;
    
  while(i != j){    
   
    if(GetFeature(*i,dimension) > threshold){
      swapidx = *j;
      *j = *i;
      *i = swapidx;
      j--;
    }
    else{
      i++;
    }          
  }
   
  // return iterator to first element of "right" group to mark the split element
  return GetFeature(*i, dimension) > threshold ? i : i+1;
}

void ClassificationData::ClearHistogram(std::vector<float>& hist)
{
  for(int k=0; k < (int)hist.size(); k++)
    hist[k] = 0;  
}

float ClassificationData::GetInformationGain(std::vector< unsigned int >::const_iterator startidx, std::vector< unsigned int >::const_iterator stopidx, std::vector< unsigned int >::const_iterator divider)
{
  std::vector<float> hist(availableLabels.size()); // histogram for all labels
  float entropyBefore  = 0.0f;
  float entropyLeft = 0.0f;
  float entropyRight = 0.0f;

  hist.assign(availableLabels.size(), 0.0f);
  // initialize histogram

  // before split
  for(std::vector<unsigned int>::const_iterator i = startidx; i != stopidx; i++){
      hist[trainingLabels[*i]] += labelWeights[trainingLabels[*i]];
  }

  // normalize histogram
  float sum = 0;
  for(int l=0; l < (int)availableLabels.size(); l++)
      sum += hist[l];

  if(sum == 0)
      return -1;

  for(int l=0; l < (int)availableLabels.size(); l++){
    hist[l] = (hist[l]+1) / (sum+1);
    entropyBefore -= hist[l] * log2(hist[l]);
  }

  hist.assign(availableLabels.size(), 0.0f);

  // left side
  for(std::vector<unsigned int>::const_iterator i = startidx; i != divider; i++){
      hist[trainingLabels[*i]] += labelWeights[trainingLabels[*i]];
  }

  //normalize histogram
  float sumleft = 0;
  for(int l=0; l < (int)availableLabels.size(); l++)
      sumleft += hist[l];

  if(sumleft == 0)
      return -1;

  for(unsigned int l=0; l < availableLabels.size(); l++){
    hist[l] = (hist[l]+1) / (sumleft+1);
    entropyLeft -= hist[l] * log2(hist[l]);
  }

  hist.assign(availableLabels.size(), 0.0f);

  // right side
  for(std::vector<unsigned int>::const_iterator i=divider; i != stopidx; i++){
      hist[trainingLabels[*i]] += labelWeights[trainingLabels[*i]];
  }

  // normalize histogram
  float sumright = 0;
  for(int l=0; l < (int)availableLabels.size(); l++)
      sumright += hist[l];

  if(sumright == 0)
      return -1;

  for(unsigned int l=0; l < availableLabels.size(); l++){
    hist[l] = (hist[l]+1) / (sumright+1);
    entropyRight -= hist[l] * log2(hist[l]);
  }

  // calculate information gain
  return entropyBefore - (entropyLeft*sumleft + entropyRight*sumright) / sum;
}

//float ClassificationData::GetInformationGain(std::vector< unsigned int >::const_iterator startidx, std::vector< unsigned int >::const_iterator stopidx, std::vector< unsigned int >::const_iterator divider)
//{
//  std::vector<float> hist(availableLabels.size()); // histogram for all labels
//  float entropyBefore  = 0.0f;
//  float entropyLeft = 0.0f;
//  float entropyRight = 0.0f;

//  hist.assign(availableLabels.size(), 0.0f);
//  // initialize histogram
//  //ClearHistogram(hist);
  
//  // before split
//  for(std::vector<unsigned int>::const_iterator i = startidx; i != stopidx; i++){
//    hist[trainingLabels[*i]]++;
//    //  hist[trainingLabels[*i]] += labelWeights[trainingLabels[*i]];
//  }
  
//  // normalize histogram
//  int nPtsTotal = std::distance(startidx, stopidx);
////  float sum = 0;
////  for(int l=0; l < (int)availableLabels.size(); l++)
////      sum += hist[l];

////  if(sum == 0)
////      return -1;
//  if(nPtsTotal == 0)
//    return -1;

//  for(int l=0; l < (int)availableLabels.size(); l++){
//    hist[l] = (hist[l]+1) / (nPtsTotal+1); // (sum+1);
//    entropyBefore -= hist[l] * log2(hist[l]);
//  }
    
//  //ClearHistogram(hist);
//  hist.assign(availableLabels.size(), 0.0f);
  
//  // left side
//  for(std::vector<unsigned int>::const_iterator i = startidx; i != divider; i++){
//    hist[trainingLabels[*i]]++;
//    //  hist[trainingLabels[*i]] += labelWeights[trainingLabels[*i]];
//  }
  
//  //normalize histogram
//  int nPtsLeft = std::distance(startidx, divider);
//  if(nPtsLeft == 0)
//    return -1;
////  float sumleft = 0;
////  for(int l=0; l < (int)availableLabels.size(); l++)
////      sumleft += hist[l];

////  if(sumleft == 0)
////      return -1;
  
//  for(unsigned int l=0; l < availableLabels.size(); l++){
//    hist[l] = (hist[l]+1) / (nPtsLeft+1); //(sumleft+1);
//    entropyLeft -= hist[l] * log2(hist[l]);
//  }
  
//  //ClearHistogram(hist);
//  hist.assign(availableLabels.size(), 0.0f);
  
//  // right side
//  for(std::vector<unsigned int>::const_iterator i=divider; i != stopidx; i++){
//     hist[trainingLabels[*i]]++;
////      hist[trainingLabels[*i]] += labelWeights[trainingLabels[*i]];
//  }
  
//  // normalize histogram
//  int nPtsRight = std::distance(divider, stopidx);
//  if(nPtsRight == 0)
//    return -1;
  
////  float sumright = 0;
////  for(int l=0; l < (int)availableLabels.size(); l++)
////      sumright += hist[l];

////  if(sumright == 0)
////      return -1;

//  for(unsigned int l=0; l < availableLabels.size(); l++){
//    hist[l] = (hist[l]+1) / (nPtsRight+1); // (sumright+1);
//    entropyRight -= hist[l] * log2(hist[l]);
//  }

//  // calculate information gain
//  return entropyBefore - (entropyLeft*nPtsLeft + entropyRight*nPtsRight) / nPtsTotal;
////  return entropyBefore - (entropyLeft*sumleft + entropyRight*sumright) / sum;
//}

//std::map< int, std::string > ClassificationData::GetStringLabels()
//{
//  return stringLabels;
//}

// std::vector< float > ClassificationData::CalculateNormalizedHistogram(std::vector< unsigned int >::const_iterator startidx, std::vector< unsigned int >::const_iterator stopidx)
// {
//   std::vector<float> hist(availableLabels.size());
//   std::vector<float> normalized(availableLabels.size());
//   ClearHistogram(hist);
//   
//   for(std::vector<unsigned int>::const_iterator i=startidx; i != stopidx; i++){
//     hist[trainingLabels[*i]]++;    
//   }
//  
//   // normalize histogram
//   int nPts = std::distance(startidx, stopidx);
//   
//   for(int l=0; l < (int)availableLabels.size(); l++)
//     normalized[l] = hist[l] / (float)nPts;    
//   
//   return normalized;
// }

std::pair<std::vector<unsigned int >, std::vector< float > > ClassificationData::CalculateNormalizedHistogram(std::vector< unsigned int >::const_iterator startidx, std::vector< unsigned int >::const_iterator stopidx)
{  
  int nLabels = availableLabels.size();
  std::vector<unsigned int > hist(nLabels);
  std::vector<float> normalized(nLabels);
  hist.assign(nLabels, 0);

  for(std::vector<unsigned int>::const_iterator i=startidx; i != stopidx; i++){
//    hist[trainingLabels[*i]]++;
      hist[trainingLabels[*i]] += labelWeights[trainingLabels[*i]];
  }  

  // normalize histogram
  float nPts = 0.0f;
  for(int l=0; l < (int)availableLabels.size(); l++)
      nPts += hist[l];

  // normalize histogram
//  int nPts = std::distance(startidx, stopidx);
  
  for(int l=0; l < (int)availableLabels.size(); l++)     
    normalized[l] = ((float)hist[l]) / (float)nPts;
  
  return std::pair<std::vector<unsigned int >, std::vector< float > >(hist, normalized);
}

void ClassificationData::SaveToFile(std::string filepath)
{
  std::ofstream myfile;
  myfile.open(filepath.c_str());
  
  for(int i=0; i < (float)data.size()/dimensions; i++){
    myfile << trainingLabels[i];
    
    for(int j=0; j < dimensions; j++)
      myfile << "   " << data[i*dimensions + j];
    
    myfile << std::endl;
  }
  
  myfile.close();
}
std::vector< int >& ClassificationData::GetAvailableLabels()
{
  return availableLabels;
}

unsigned int ClassificationData::LoadFromDirectory(std::string directory, std::vector< int > labelIDs)
{
  // loads files specified by directory and the labelIDs and counts the number of available
  // data points per label
  
    path p(directory);
    if(!exists(p))
    {
        std::cout << "Training data directory " << directory << " does not exist!" << std::endl;
        return 0;
    }

    totalPoints = 0;
    this->directory = directory;
    availableLabels = labelIDs;

  for(unsigned int i=0; i < labelIDs.size(); ++i)
  {
	trainingDataFilePos[labelIDs[i]] = 0;
	
    std::string filename = str(boost::format("%1$s/%2$04d.data") % this->directory % labelIDs[i]);
    std::ifstream inFile(filename.c_str()); 
    
    if(i == 0)
    {
      // get dimension in first run
      std::string line;      
      std::getline(inFile, line);
      std::stringstream s(line);
      float value;
      int cnt = 0;
      
      // count how often a float can be read in
      while(s.good())
      {
		s >> value;
		cnt++;
      }
      
      dimensions = cnt;
      
      // constant fieldWidth is assumed throughout the file!
      fieldWidth = (line.length()+1) / dimensions;
    }
    
    // count lines -> number of data points for label
    inFile.seekg(0, inFile.end);    
    pointsPerLabel[labelIDs[i]] = inFile.tellg()/(dimensions*fieldWidth);
    
    // sum up total number of available data points
    totalPoints += pointsPerLabel[labelIDs[i]];    
    inFile.close();
  }   
  
  return totalPoints;
}

std::map<int, unsigned int >& ClassificationData::GetCountPerLabel()
{
  return pointsPerLabel;
}

void ClassificationData::LoadFromFile(std::string trainingFilePath, std::string categoryFilePath)
{
  trainingLabels.clear();
  data.clear();
  
  directory = "";
  
  std::ifstream trainingfile;
  trainingfile.open(trainingFilePath.c_str());
  
  std::string line;
  int label;
  float d;
  int labeled = 0;
  
  std::stringstream ss; 
  
  if(trainingfile.is_open()){
    while(trainingfile.good()){
      std::getline(trainingfile, line);      
      ss.clear();
      
      if(ss << line){
	if(ss >> label){
	  trainingLabels.push_back(label);
	  
	  if(std::find(availableLabels.begin(), availableLabels.end(), label) == availableLabels.end())
	    availableLabels.push_back(label);
	  
	  if(label >= 0)	    
	    labeled++;
      
	  while(ss >> d)	
	    data.push_back(d);      	      
	}
      }
    }      
  }
  trainingfile.close();

  if(labeled == 0)
  {
    labelStatus = UNLABELED;
  }
  else if(labeled < (int)trainingLabels.size())
  {
    labelStatus = PARTIALLY_LABELED;
  }
  else
  {
    labelStatus = LABELED;
  }
  
  dimensions = data.size() / trainingLabels.size();
}

ClassificationData::~ClassificationData()
{

}

