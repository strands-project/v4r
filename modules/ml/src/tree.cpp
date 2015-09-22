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


#include <v4r/ml/tree.h>

using namespace v4r::RandomForest;

Tree::Tree()
{

}

Tree::Tree(boost::mt19937* randomGenerator)
{   
  this->randomGenerator = randomGenerator;
}

// compare function for Partition function
bool SmallerThan(float x, float t) { return x<t; }

void Tree::Train(ClassificationData& data, std::vector< unsigned int >& indices, int maxDepth, int testedSplittingFunctions, float minInformationGain, int minPointsForSplit, bool allNodesStoreLabelDistribution, int verbosityLevel)
{   
  this->verbosityLevel = verbosityLevel;
  // start recursion with depth 0
  trainRecursively(data, indices, maxDepth, testedSplittingFunctions, minInformationGain, minPointsForSplit, allNodesStoreLabelDistribution, 0);
}

void Tree::TrainParallel(ClassificationData& data, std::vector< unsigned int >& indices, int maxDepth, int testedSplittingFunctions, float minInformationGain, int minPointsForSplit, bool allNodesStoreLabelDistribution, int verbosityLevel)
{   
  this->verbosityLevel = verbosityLevel;
  // start recursion with depth 0
  trainRecursivelyParallel(data, indices, maxDepth, testedSplittingFunctions, minInformationGain, minPointsForSplit, allNodesStoreLabelDistribution, 0);
}

void Tree::EraseSplitNodeLabelDistributions()
{
    for(unsigned int n=0; n<nodes.size(); n++)
    {
        nodes[n].ClearSplitNodeLabelDistribution();
    }
}

std::vector< float >& Tree::Classify(std::vector< float >& point)
{
  // get root node and traverse through tree until leaf node is reached
  Node* curNode = GetRootNode();
  
  while(curNode->IsSplitNode())
    curNode = &nodes[curNode->EvaluateNode(point)];      
    
  return curNode->GetLabelDistribution();
}

// to classify only down to a certain depth level (for evaluation)
std::vector< float >& Tree::Classify(std::vector< float >& point, int depth)
{
  // get root node and traverse through tree until leaf node is reached
  Node* curNode = GetRootNode();

  for(int d=0; d <= depth && curNode->IsSplitNode(); ++d)
     curNode = &nodes[curNode->EvaluateNode(point)];

  return curNode->GetLabelDistribution();
}

int Tree::GetResultingLeafNode(std::vector< float > point)
{
  // get root node and traverse through tree until leaf node is reached
  Node* curNode = GetRootNode();
  
  int idx = rootNodeIdx;
  
  while(curNode->IsSplitNode())
  {
	idx = curNode->EvaluateNode(point);
	curNode = &nodes[idx];      
  }
  
  // return index of leaf node
  return idx;
}

void Tree::ClearLeafNodes()
{
  // go through all leaf nodes and reset their label distributions
  for(unsigned int i=0; i<nodes.size(); ++i)
  {
	if(!nodes[i].IsSplitNode())
      nodes[i].ResetLabelDistribution();
  }
}

void Tree::RefineLeafNodes(ClassificationData& data, int nPoints, int labelIdx)
{
  // for all available points in data, traverse through tree and add one point to
  // label distribution of resulting leaf node
  for(int i=0; i<nPoints; ++i)
  {
	int idx = GetResultingLeafNode(data.GetFeatures(i));
	nodes[idx].AddToAbsLabelDistribution(labelIdx);	
  }  
}

void Tree::UpdateLeafNodes(std::vector< int > labels, std::map< int, unsigned int >& pointsPerLabel)
{
  // for all leaf nodes in the tree, normalize label distributions
  for(unsigned int i=0; i<nodes.size(); ++i)
  {
	if(!nodes[i].IsSplitNode())
	  nodes[i].UpdateLabelDistribution(labels, pointsPerLabel);
  }
}

int Tree::trainRecursively(ClassificationData& data, std::vector<unsigned int > indices, int maxDepth, int testedSplittingFunctions, float minInformationGain, int minPointsForSplit, bool allNodesStoreLabelDistribution, int currentDepth)
{
  std::vector< unsigned int >::iterator startidx = indices.begin();
  std::vector< unsigned int >::iterator stopidx = indices.end();

  // abort conditions for recursion:
  if(currentDepth == maxDepth){
    // max depth reached, stop with leaf node

    if(verbosityLevel > 2)
    {
      std::cout << "  - Create LEAF node at depth " << currentDepth << " (max depth reached)" << std::endl;
    }

    // create leaf node with current label distribution
    nodes.push_back(Node(data.CalculateNormalizedHistogram(startidx, stopidx)));
    return (int)nodes.size()-1;		// return index of new node
  }
  if(std::distance(startidx, stopidx) < minPointsForSplit){ //indices.size() < 5){
    // too few data points, no point in splitting any more

    if(verbosityLevel > 2)
    {
      std::cout << "  - Create LEAF node at depth " << currentDepth << " (too few points)" << std::endl;
    }

    // create leaf node with current label distribution
    nodes.push_back(Node(data.CalculateNormalizedHistogram(startidx, stopidx)));
    return (int)nodes.size()-1;		// return index of new node
  }

  // distributions for random generator
  boost::uniform_real<float> realDist(0.0f, 1.0f);
  boost::uniform_int<int> intDist(0, data.GetDimensions()-1);

  // vectors to save different parameter sets
  std::vector<float> gains(testedSplittingFunctions);
  std::vector<int> features(testedSplittingFunctions);
  std::vector<float> thresholds(testedSplittingFunctions);

  for(int i=0; i < testedSplittingFunctions; i++)
  {
    // reinitialize iterators because every thread has its own copy of the indices array
    std::vector< unsigned int >::iterator start  = indices.begin();
    std::vector< unsigned int >::iterator stop = indices.end();

    std::pair<float,float> minmax;
    std::vector<unsigned int>::iterator divider;

    // randomly choose one of the offered features
    int curFeature = intDist(*randomGenerator);

    // get range of data points for selected feature
    minmax = data.GetMinMax(start, stop, curFeature);

    if(minmax.first == minmax.second)
    {
      // all datapoints have same value, no split possible for this feature
      gains[i] = -1;
      continue;
    }

    // randomly choose a threshold in the range of the datapoints for selected feature
    float threshold = realDist(*randomGenerator) * (minmax.second-minmax.first) + minmax.first;

    if(threshold == minmax.first || threshold == minmax.second)
    {
      // don't split on max or min value of data points
      gains[i] = -1;
      continue;
    }

    // do left/right split with threshold
    divider = data.Partition(start, stop, curFeature, threshold);

    if(divider == start || divider == stop)
    {
      // prevent splits at first or last element
      gains[i] = -1;
      continue;
    }

    // evaluate split (calc information gain) and store corresponding feature index and threshold
    gains[i] = data.GetInformationGain(start, stop, divider);
    features[i] = curFeature;
    thresholds[i] = threshold;
  }

  // find best split (max information gain)
  std::vector<float>::iterator it = std::max_element(gains.begin(), gains.end());
  float bestIGain = *it;
  // get corresponding feature index and threshold
  int idx = std::distance(gains.begin(), it);
  int bestFeature = features[idx];
  float bestThreshold = thresholds[idx];

  // another abort condition for the recursion
  if(bestIGain < minInformationGain)
  {
    // too little information gain at this point, no sense in splitting data any further

    if(verbosityLevel > 2)
    {
      std::cout << "  - Create LEAF node at depth " << currentDepth << " (too little info gain: " << bestIGain << ")" << std::endl;
    }

    // create leaf node with current label distribution
    nodes.push_back(Node(data.CalculateNormalizedHistogram(startidx, stopidx)));

    if(currentDepth == 0)
      rootNodeIdx = (int)nodes.size()-1;	// set root node of tree

    return (int)nodes.size()-1;		// return index of current node
  }

  // perform split again with best parameters found
  std::vector<unsigned int>::iterator divider = data.Partition(startidx, stopidx, bestFeature, bestThreshold);

  if(verbosityLevel > 2)
  {
    std::cout << "  - Create SPLIT node at depth " << currentDepth << " on feature " << bestFeature << ". Threshold: " << bestThreshold << " GAIN: " << bestIGain << std::endl;
    std::cout << "    - " << std::distance(startidx, divider) << " points left, " << std::distance(divider, stopidx) << " points right." << std::endl;
  }

  // repeat process for left and right child nodes
  std::vector<unsigned int > left(startidx, divider);	// all indices of points going to left child node
  std::vector<unsigned int > right(divider, stopidx);	// all indices of points going to right child node
  int leftChildIdx = trainRecursivelyParallel(data,left, maxDepth, testedSplittingFunctions, minInformationGain, minPointsForSplit, allNodesStoreLabelDistribution, currentDepth+1);
  int rightChildIdx = trainRecursivelyParallel(data,right, maxDepth, testedSplittingFunctions, minInformationGain, minPointsForSplit, allNodesStoreLabelDistribution, currentDepth+1);

  // create and add split node with learned parameters
  Node splitNode;

  if(allNodesStoreLabelDistribution)
  {
    splitNode = Node(bestFeature,bestThreshold,leftChildIdx,rightChildIdx, data.CalculateNormalizedHistogram(startidx, stopidx));
  }
  else
  {
    splitNode = Node(bestFeature,bestThreshold,leftChildIdx,rightChildIdx);
  }

  nodes.push_back(splitNode);

  if(currentDepth == 0)
    rootNodeIdx = (int)nodes.size()-1;	// set root node of tree

  return (int)nodes.size()-1;		// return index of current node
}

int Tree::trainRecursivelyParallel(ClassificationData& data, std::vector<unsigned int > indices, int maxDepth, int testedSplittingFunctions, float minInformationGain, int minPointsForSplit, bool allNodesStoreLabelDistribution, int currentDepth)
{
  std::vector< unsigned int >::iterator startidx = indices.begin();
  std::vector< unsigned int >::iterator stopidx = indices.end();
  
  // abort conditions for recursion:  
  if(currentDepth == maxDepth){
	// max depth reached, stop with leaf node
	
	if(verbosityLevel > 2)
	{
	  std::cout << "  - Create LEAF node at depth " << currentDepth << " (max depth reached)" << std::endl; 
	}
	
	// create leaf node with current label distribution
	nodes.push_back(Node(data.CalculateNormalizedHistogram(startidx, stopidx)));
	return (int)nodes.size()-1;		// return index of new node
  } 
  if(std::distance(startidx, stopidx) < minPointsForSplit){
	// too few data points, no point in splitting any more
  
	if(verbosityLevel > 2)
	{
	  std::cout << "  - Create LEAF node at depth " << currentDepth << " (too few points)" << std::endl;
	}
	
	// create leaf node with current label distribution
    Node q( data.CalculateNormalizedHistogram(startidx, stopidx)) ;
    nodes.push_back(q);
	return (int)nodes.size()-1;		// return index of new node
  }    
  
  // distributions for random generator
  boost::uniform_real<float> realDist(0.0f, 1.0f);
  boost::uniform_int<int> intDist(0, data.GetDimensions()-1);
  
  // vectors to save different parameter sets
  std::vector<float> gains(testedSplittingFunctions);
  std::vector<int> features(testedSplittingFunctions);
  std::vector<float> thresholds(testedSplittingFunctions);
  
#pragma omp parallel
  {
  #pragma omp for nowait firstprivate(indices)
  for(int i=0; i < testedSplittingFunctions; i++)
  {
	// reinitialize iterators because every thread has its own copy of the indices array
	std::vector< unsigned int >::iterator start  = indices.begin();
	std::vector< unsigned int >::iterator stop = indices.end();
  
	std::pair<float,float> minmax;
	std::vector<unsigned int>::iterator divider;	
	
	// randomly choose one of the offered features
	int curFeature = intDist(*randomGenerator);
	
	// get range of data points for selected feature
	minmax = data.GetMinMax(start, stop, curFeature);
	
	if(minmax.first == minmax.second)
	{
	  // all datapoints have same value, no split possible for this feature
	  gains[i] = -1;
	  continue;
	}
	
	// randomly choose a threshold in the range of the datapoints for selected feature
	float threshold = realDist(*randomGenerator) * (minmax.second-minmax.first) + minmax.first;
	
	if(threshold == minmax.first || threshold == minmax.second)
	{
	  // don't split on max or min value of data points
	  gains[i] = -1;
	  continue;
	}
	
	// do left/right split with threshold
	divider = data.Partition(start, stop, curFeature, threshold);
	
    if(std::distance(start, divider) < minPointsForSplit || std::distance(divider, stop) < minPointsForSplit)
    {
      // prevent splits causing too few points on one side
      gains[i] = -1;
      continue;
    }
//	if(divider == start || divider == stop)
//	{
//	  // prevent splits at first or last element
//	  gains[i] = -1;
//	  continue;
//	}
	
	// evaluate split (calc information gain) and store corresponding feature index and threshold 
	gains[i] = data.GetInformationGain(start, stop, divider);
	features[i] = curFeature;
	thresholds[i] = threshold;	
  }
}

  // find best split (max information gain)
  std::vector<float>::iterator it = std::max_element(gains.begin(), gains.end());
  float bestIGain = *it;
  // get corresponding feature index and threshold
  int idx = std::distance(gains.begin(), it);
  int bestFeature = features[idx];
  float bestThreshold = thresholds[idx];
    
  // another abort condition for the recursion
  if(bestIGain < minInformationGain)
  {
	// too little information gain at this point, no sense in splitting data any further
	
	if(verbosityLevel > 2)
	{
	  std::cout << "  - Create LEAF node at depth " << currentDepth << " (too little info gain: " << bestIGain << ")" << std::endl;
	}
	
	// create leaf node with current label distribution
	nodes.push_back(Node(data.CalculateNormalizedHistogram(startidx, stopidx)));
	
	if(currentDepth == 0)
	  rootNodeIdx = (int)nodes.size()-1;	// set root node of tree
	  
	return (int)nodes.size()-1;		// return index of current node
  }
  
  // perform split again with best parameters found
  std::vector<unsigned int>::iterator divider = data.Partition(startidx, stopidx, bestFeature, bestThreshold);
  
  if(verbosityLevel > 2)
  {
	std::cout << "  - Create SPLIT node at depth " << currentDepth << " on feature " << bestFeature << ". Threshold: " << bestThreshold << " GAIN: " << bestIGain << std::endl;
	std::cout << "    - " << std::distance(startidx, divider) << " points left, " << std::distance(divider, stopidx) << " points right." << std::endl;
  }
  
  // repeat process for left and right child nodes 
  std::vector<unsigned int > left(startidx, divider);	// all indices of points going to left child node
  std::vector<unsigned int > right(divider, stopidx);	// all indices of points going to right child node
  int leftChildIdx = trainRecursivelyParallel(data,left, maxDepth, testedSplittingFunctions, minInformationGain, minPointsForSplit, allNodesStoreLabelDistribution, currentDepth+1);
  int rightChildIdx = trainRecursivelyParallel(data,right, maxDepth, testedSplittingFunctions, minInformationGain, minPointsForSplit, allNodesStoreLabelDistribution, currentDepth+1);
  
  // create and add split node with learned parameters
  Node splitNode;

  if(allNodesStoreLabelDistribution)
  {
    splitNode = Node(bestFeature,bestThreshold,leftChildIdx,rightChildIdx, data.CalculateNormalizedHistogram(startidx, stopidx));
  }
  else
  {
    splitNode = Node(bestFeature,bestThreshold,leftChildIdx,rightChildIdx);
  }

  nodes.push_back(splitNode);
  
  if(currentDepth == 0)
	rootNodeIdx = (int)nodes.size()-1;	// set root node of tree
	
  return (int)nodes.size()-1;		// return index of current node
}
	
void Tree::CreateVisualization(std::string filename)
{
  std::ofstream os(filename.c_str());
  
  os << "digraph G {" << std::endl;
  
  // recursively add node visualizations to file
  CreateNodeVisualization(GetRootNode(), 0, os);
  
  os << "}" << std::endl;
  os.close();
}

void Tree::CreateNodeVisualization(Node* node, int idx, std::ofstream& os)
{
  if(node->IsSplitNode())
  {
    os << "n" << idx << " [label=\"" << node->GetSplitFeatureIdx() << "\\n" << node->GetThreshold() << "\"];" << std::endl;
    Node* left = &nodes[node->GetLeftChildIdx()];
    Node* right = &nodes[node->GetRightChildIdx()];
    CreateNodeVisualization(left, idx*2+1, os);
    CreateNodeVisualization(right, idx*2+2, os);
    os << "n" << idx << " -> " << "n" << idx*2+1 << ";" << std::endl;
    os << "n" << idx << " -> " << "n" << idx*2+2 << ";" << std::endl;
  }
  else	
  {	
    os << "n" << idx << " [shape=box,label=\"";
    std::vector<float> d = node->GetLabelDistribution();
	std::vector<float>::iterator it = std::max_element(d.begin(), d.end());
	
	for(std::vector<float>::iterator i=d.begin(); i!=d.end(); i++)
    {
	  if(i!=it)
		os << std::setprecision(3) << *i;
	  else
		os << ">" << std::setprecision(3) << *i << "<";
		  
      if(std::distance(i, d.end()) > 1)
		os << "\\n";
      else
		os << "\"];" << std::endl;
    }    
  }
}


Tree::~Tree()
{

}
