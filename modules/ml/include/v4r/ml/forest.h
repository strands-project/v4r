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


#ifndef FOREST_H
#define FOREST_H

#include <algorithm>
#include <time.h>
#include "boost/random.hpp"
#include <fstream>
#include <string.h>
#include "classificationdata.h"
#include "tree.h"

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/map.hpp>

#include <v4r/core/macros.h>

#ifdef _OPENMP
   #include <omp.h>
#else
   #define omp_get_thread_num() 0
#endif

namespace v4r {
namespace RandomForest {

// how many data points to load into RAM at once
#define MAX_DATAPOINTS_TO_LOAD 1000000

class V4R_EXPORTS Forest
{   
private:
  friend class boost::serialization::access;  
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version)
  {
    ar & trees;    
    ar & maxDepth;
    ar & nTrees;
    ar & testedSplittingFunctions;
    ar & minInformationGain;
    ar & minPointsForSplit;
    ar & baggingRatio;
	ar & labels;
    ar & splitNodesStoreLabelDistribution;
  }
  
  boost::mt19937 randomGenerator;
  std::vector<Tree> trees;
  int maxDepth;
  int nTrees;
  int testedSplittingFunctions;
  float minInformationGain;
  int minPointsForSplit;
  float baggingRatio;

  // for evaluation, split nodes can also store label distributions, to be able to traverse
  // trees only down to a certain depth for classification
  bool splitNodesStoreLabelDistribution;

  std::vector<int> labels;
  void RefineLeafNodes(ClassificationData& data, int verbosityLevel = 1);
  
public:
  Forest();
  Forest(std::string filename);
  Forest(int nTrees, int maxDepth = 8, float baggingRatio = 0.5, int testedSplittingFunctions = 100, float minInformationGain = 0.02, int minPointsForSplit = 5);
  void Train(ClassificationData& trainingData, int verbosityLevel = 1);
  void TrainLarge(ClassificationData& trainingData, bool allNodesStoreLabelDistribution, bool refineWithAllTrainingData = false, int verbosityLevel = 1);
  std::vector<float> SoftClassify(std::vector<float>& point, int depth = -1, int useNTrees = -1);
  void EraseSplitNodeLabelDistributions();
  int ClassifyPoint(std::vector<float>& point, int depth = -1, int useNTrees = -1);
  void SaveToFile(std::string filename);
  void LoadFromFile(std::string filename);
  void CreateVisualizations(std::string directory);
  std::vector<int> GetLabels();
  virtual ~Forest();
};

}
}
#endif // FOREST_H
