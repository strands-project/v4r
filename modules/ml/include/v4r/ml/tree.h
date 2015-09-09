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


#ifndef TREE_H
#define TREE_H

#include <vector>
#include <stdlib.h>
#include <time.h>
#include "boost/random.hpp"
#include <math.h>
#include <fstream>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/vector.hpp>

#include "node.h"
#include "classificationdata.h"

#include <v4r/core/macros.h>

namespace v4r
{

namespace RandomForest {

class V4R_EXPORTS Tree
{
private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version)
  {
    ar & nodes;
    ar & rootNodeIdx;
    ar & maxDepth;
    ar & testedSplittingFunctions;
  }
  
  int verbosityLevel;
  boost::mt19937* randomGenerator;  
  std::vector<Node> nodes;
  int rootNodeIdx;
  int maxDepth;
  int testedSplittingFunctions;
  
  int trainRecursively(ClassificationData& data, std::vector< unsigned int > indices, int maxDepth, int testedSplittingFunctions, float minInformationGain, int minPointsForSplit, bool allNodesStoreLabelDistribution, int currentDepth);
  int trainRecursivelyParallel(ClassificationData& data, std::vector< unsigned int > indices, int maxDepth, int testedSplittingFunctions, float minInformationGain, int minPointsForSplit, bool allNodesStoreLabelDistribution, int currentDepth);
  
public:
  Tree();
  Tree(boost::mt19937* randomGenerator);
  inline Node* GetRootNode();
  std::vector< float >& Classify(std::vector< float >& point);
  std::vector< float >& Classify(std::vector< float >& point, int depth);
  int GetResultingLeafNode(std::vector< float > point);
  void ClearLeafNodes();
  void RefineLeafNodes(ClassificationData& data, int nPoints, int labelIdx);
  void UpdateLeafNodes(std::vector<int> labels, std::map<int, unsigned int >& pointsPerLabel);
  void EraseSplitNodeLabelDistributions();
  void Train(ClassificationData& data, std::vector< unsigned int >& indices, int maxDepth, int testedSplittingFunctions, float minInformationGain, int minPointsForSplit, bool allNodesStoreLabelDistribution, int verbosityLevel = 1);
  void TrainParallel(ClassificationData& data, std::vector< unsigned int >& indices, int maxDepth, int testedSplittingFunctions, float minInformationGain, int minPointsForSplit, bool allNodesStoreLabelDistribution, int verbosityLevel = 1);
  void CreateVisualization(std::string filename);
  void CreateNodeVisualization(Node* node, int idx, std::ofstream& os);  
  virtual ~Tree();
};

inline Node* Tree::GetRootNode()
{
  return &nodes[rootNodeIdx];
}

}

}
#endif // TREE_H
