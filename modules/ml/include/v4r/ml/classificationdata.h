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


#ifndef CLASSIFICATIONDATA_H
#define CLASSIFICATIONDATA_H

#include <vector>
#include <string>
#include <map>
#include <math.h>
#include <algorithm>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <boost/format.hpp>
#include <boost/random.hpp>

#include <v4r/core/macros.h>

namespace v4r{
namespace RandomForest{

enum LabelStatus{
    LABELED,
    UNLABELED,
    PARTIALLY_LABELED
  };
  
class V4R_EXPORTS ClassificationData
{
private:
  std::string directory;
  std::vector<float> data;
  std::vector<int> trainingLabels;
  std::vector<float> labelWeights;
  std::vector<int> availableLabels;
  std::map<int, long long int > trainingDataFilePos;
  int fieldWidth;
  int dimensions;
  void ClearHistogram(std::vector< float >& hist);
  LabelStatus labelStatus;
  std::vector<unsigned int> generateRandomIndices(unsigned int n, unsigned int totalPoints);
  void swap(std::vector<unsigned int>& array, unsigned int idx1, unsigned int idx2);

  std::map<int, unsigned int> pointsPerLabel;
  unsigned int totalPoints;
  boost::mt19937 randomGenerator;
  
public:
    
  ClassificationData();
  std::vector<unsigned int> NewBag(float baggingRatio);
  int LoadChunkForLabel(int labelID, int nPoints);
  int GetDimensions();
  std::map< int, unsigned int >& GetCountPerLabel();
  int GetCount();
  std::vector<int>& GetAvailableLabels();
  std::pair<float, float> GetMinMax(std::vector< unsigned int >::iterator startidx, std::vector< unsigned int >::iterator stopidx, int dimension);
  std::pair<float, float> GetMinMax(int dimension);
  std::vector< unsigned int >::iterator Partition(std::vector< unsigned int >::iterator startidx, std::vector< unsigned int >::iterator stopidx, int dimension, float threshold);
  float GetFeature(int pointIdx, int featureIdx);
  std::vector<float> GetFeatures(int pointIdx);
  float GetInformationGain(std::vector< unsigned int >::const_iterator startidx, std::vector< unsigned int >::const_iterator stopidx, std::vector< unsigned int >::const_iterator divider);
  std::pair<std::vector<unsigned int >, std::vector< float > > CalculateNormalizedHistogram(std::vector<unsigned int>::const_iterator startidx, std::vector<unsigned int>::const_iterator stopidx);
  void LoadDemoSpiral(int nPoints, float noise);
  void SaveToFile(std::string filepath);
  void LoadFromFile(std::string trainingFilePath, std::string categoryFilePath);
  unsigned int LoadFromDirectory(std::string directory, std::vector< int > labelIDs);
  virtual ~ClassificationData();
};

}
}
#endif // CLASSIFICATIONDATA_H
