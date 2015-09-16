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


#include <v4r/ml/forest.h>
#include <iostream>
#include <vector>

using namespace v4r::RandomForest;

Forest::Forest()
{
  // initialize random generator once
  randomGenerator = boost::mt19937(time(0));
  nTrees = 5;
  maxDepth = 8;
  testedSplittingFunctions = 100;
  minInformationGain = 0.02;
  minPointsForSplit = 5;
  baggingRatio = 0.5;
}

Forest::Forest(std::string filename)
{  
  LoadFromFile(filename);
  randomGenerator = boost::mt19937(time(0));
}

Forest::Forest(int nTrees, int maxDepth, float baggingRatio, int testedSplittingFunctions, float minInformationGain, int minPointsForSplit)
{
  // initialize random generator once
  randomGenerator = boost::mt19937(time(0));
  this->nTrees = nTrees;
  this->maxDepth = maxDepth;
  this->testedSplittingFunctions = testedSplittingFunctions;
  this->minInformationGain = minInformationGain;
  this->minPointsForSplit = minPointsForSplit;
  this->baggingRatio = baggingRatio;
}

std::vector< float > Forest::SoftClassify(std::vector< float >& point, int depth, int useNTrees)
{
  std::vector<float> labelDist(labels.size());
 
  // initialize label distribution array
  for(unsigned int i=0; i < labelDist.size(); i++)
    labelDist[i] = 0.0f;   
  
  if(useNTrees < 0 || (unsigned int)useNTrees > trees.size())
      useNTrees = trees.size();

  // label distribution for every tree
  std::vector< std::vector<float> > labelDistPerTree(useNTrees);//trees.size());
  
  // initialize for parallelization
  for(int i=0; i < useNTrees; ++i)//nTrees; i++)
	labelDistPerTree[i].reserve(labelDist.size());

  if(depth >= 0 && !splitNodesStoreLabelDistribution)
  {
      // trees should only be traversed down to certain depth, but split nodes do not contain label distributions!
      // So override depth function and evaluate trees down to leaf nodes
      depth = -1;
  }

  if(depth < 0)
  {
        // get label distribution for every tree
        #pragma omp parallel
        {
            #pragma omp for nowait
            for(int i=0; i < useNTrees; ++i)//< nTrees; i++)
            {
              labelDistPerTree[i] = trees[i].Classify(point);
            }
        }
  }
  else
  {
      // get label distribution for every tree, traverse tree max. to defined depth
      #pragma omp parallel
      {
          #pragma omp for nowait
          for(int i=0; i < useNTrees; ++i)//< nTrees; i++)
          {
            labelDistPerTree[i] = trees[i].Classify(point, depth);
          }
      }
  }
  
  // average label distributions of all trees
  for(int i=0; i < useNTrees; ++i)//< nTrees; i++)
  {
    for(unsigned int j=0; j<labelDist.size(); j++)
	  labelDist[j] += labelDistPerTree[i][j];
  }
    
  // normalize by number of trees to get final distribution
  for(unsigned int i=0; i < labelDist.size(); i++)
    labelDist[i] /= useNTrees;//nTrees;
 
  return labelDist;
}

int Forest::ClassifyPoint(std::vector< float >& point, int depth, int useNTrees)
{
  // take max value of label distribution for hard classification
  std::vector<float> labelDist = SoftClassify(point, depth, useNTrees);
  return std::distance(labelDist.begin(), std::max_element(labelDist.begin(), labelDist.end()));
}

void Forest::EraseSplitNodeLabelDistributions()
{
    for(int t=0; t<nTrees; t++)
    {
        trees[t].EraseSplitNodeLabelDistributions();
    }

    splitNodesStoreLabelDistribution = false;
}

void Forest::Train(ClassificationData& trainingData, int verbosityLevel)
{
  // get available labels from training data
  labels = trainingData.GetAvailableLabels();

  if(verbosityLevel > 0)
  {
    std::cout << "Train forest with " << nTrees << " trees..." << std::endl;

    if(verbosityLevel > 1)
    {
      // List all labels the forest is trained for
      std::cout << "Used label IDs:" << std::endl;
      for(unsigned int i=0; i<labels.size(); ++i)
      {
        std::cout << labels[i] << std::endl;
      }
    }
  }

  // create trees
  for(int i=0; i<nTrees; i++)
    trees.push_back(Tree(&randomGenerator));
  
  // random generator for bagging of training data
  boost::uniform_int<int> intDist(0, trainingData.GetCount()-1);
  
  // how many data points for every tree?
  int nDataPoints = floor(trainingData.GetCount() * baggingRatio);
  
  // train every tree independently
  #pragma omp parallel for
  for(int i=0; i < nTrees; i++)
  {
    // storage for the indices of the used data points for each tree (bagging)
    std::vector<unsigned int> dataPointIndices;
    
    #pragma omp critical
    {
        if(verbosityLevel > 0)
        {
          std::cout << "Tree " << i+1 << "/" << nTrees << std::endl;
        }
        if(verbosityLevel > 1)
        {
          std::cout << "- Bag training data..." << std::endl;
          std::cout << "- Train tree with " << nDataPoints << " datapoints..." << std::endl;
        }
    }

    // refill index array for tree
    dataPointIndices.clear();
            
    // randomly select training points for each tree (bagging)
    for(int j=0; j < nDataPoints; j++)
      dataPointIndices.push_back(intDist(randomGenerator));
        
    trees[i].Train(trainingData, dataPointIndices, maxDepth, testedSplittingFunctions, minInformationGain, minPointsForSplit, verbosityLevel);
  }
  
  if(verbosityLevel > 0)
  {
    std::cout << "### TRAINING DONE ###" << std::endl;
  }
}


void Forest::TrainLarge(ClassificationData& trainingData, bool allNodesStoreLabelDistribution, bool refineWithAllTrainingData, int verbosityLevel)
{   
  // get available labels from training data
  labels = trainingData.GetAvailableLabels();
  
  if(verbosityLevel > 0)
  {
	std::cout << "Train forest with " << nTrees << " trees..." << std::endl;  
	
    if(verbosityLevel > 1)
	{
	  // List all labels the forest is trained for
      std::cout << "Used label IDs:" << std::endl;
      for(unsigned int i=0; i<labels.size(); ++i)
	  {
        std::cout << labels[i] << std::endl;
	  }
	}	  
  }
 
  // train every tree independently
  for(int i=0; i < nTrees; ++i)
  { 
	if(verbosityLevel > 0)
	{
	  std::cout << "Tree " << i+1 << "/" << nTrees << std::endl;	  
	}
	
	if(verbosityLevel > 1)
	{
	  std::cout << "- Bag training data..." << std::endl;
	}
	
	// storage for the indices of the used data points for each tree (bagging)    
    std::vector<unsigned int> dataPointIndices = trainingData.NewBag(baggingRatio);	
	
	if(verbosityLevel > 1)
	{
	  std::cout << "- Train tree with " << dataPointIndices.size() << " datapoints..." << std::endl;
	}
	
	// create and train tree
	Tree t(&randomGenerator);
    t.TrainParallel(trainingData, dataPointIndices, maxDepth, testedSplittingFunctions, minInformationGain, minPointsForSplit, allNodesStoreLabelDistribution, verbosityLevel);
	trees.push_back(t);
  }
  
  if(refineWithAllTrainingData)
  {
	if(verbosityLevel > 0)
	{
	  std::cout << "Refine all trees with all available training data..." << std::endl;
	}
	
	RefineLeafNodes(trainingData, verbosityLevel);
  }
  
  splitNodesStoreLabelDistribution = allNodesStoreLabelDistribution;

  if(verbosityLevel > 0)
  {
	std::cout << "### TRAINING DONE ###" << std::endl;
  }
}

std::vector<int> Forest::GetLabels()
{
    return labels;
}

void Forest::RefineLeafNodes(ClassificationData& data, int verbosityLevel)
{
  // reset label distributions of all leaf nodes in the forest
  for(int t=0; t < nTrees; ++t)
	trees[t].ClearLeafNodes();	  
  
  // refine for each label
  for(unsigned int i=0; i<labels.size(); i++)
  {
	int nPoints = 0;
	
	// load training data in chunks
	while((nPoints = data.LoadChunkForLabel(labels[i], MAX_DATAPOINTS_TO_LOAD)) > 0)
	{
	  #pragma omp parallel
	  {		  
		#pragma omp for nowait
		for(int t=0; t < nTrees; ++t)
		{
		  trees[t].RefineLeafNodes(data, nPoints, i);	  
		}
	  }
	}
  }
  
  // normalize distributions (account for inbalanced amount of available data per label)
  for(int t=0; t < nTrees; ++t)
  {
	trees[t].UpdateLeafNodes(labels, data.GetCountPerLabel());
  }
}

void Forest::CreateVisualizations(std::string directory)
{
  // creates text files in DOT format which can be converted to tree visualizations using graphviz command line tool
  for(int i=0; i<nTrees; ++i)
  {
    std::string filename = str(boost::format("%1$s/%2$04d.dot") % directory % i);
    trees[i].CreateVisualization(filename);
  }
}

void Forest::SaveToFile(std::string filename)
{
  // save tree to file
  std::ofstream ofs(filename.c_str());
  boost::archive::text_oarchive oa(ofs);
  oa << *this;
  ofs.close();
}

void Forest::LoadFromFile(std::string filename)
{
  // load tree from file
  std::ifstream ifs(filename.c_str());
  boost::archive::text_iarchive ia(ifs);
  Forest f;
  ia >> f;
  *this = f;
  ifs.close();
}

Forest::~Forest()
{

}

