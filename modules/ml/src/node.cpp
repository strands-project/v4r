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


#include <v4r/ml/node.h>

using namespace v4r::RandomForest;

// default constructor
Node::Node()
{
  isSplitNode_ = false;
  splitOnFeatureIdx_ = -1;
  threshold_ = 0.0f;
  leftChildIdx_ = -1;
  rightChildIdx_ = -1;
}

// constructor for split node
Node::Node(int splitOnFeature, float threshold, int leftChildIdx, int rightChildIdx)
{
  isSplitNode_ = true;
  splitOnFeatureIdx_ = splitOnFeature;
  this->threshold_ = threshold;
  this->leftChildIdx_ = leftChildIdx;
  this->rightChildIdx_ = rightChildIdx;
  labelDistribution_.clear();
}

// constructor for leaf node
Node::Node(std::pair<std::vector<unsigned int >, std::vector< float > > labelDistributions)
{
  isSplitNode_ = false;
  splitOnFeatureIdx_ = -1;
  threshold_ = 0.0f;
  labelDistribution_ = labelDistributions.second;
  leftChildIdx_ = -1;
  rightChildIdx_ = -1;
  absLabelDistribution_ = labelDistributions.first;
}

// constructor for split node, also saving the label distributions for later
Node::Node(int splitOnFeature, float threshold, int leftChildIdx, int rightChildIdx, std::pair<std::vector<unsigned int >, std::vector< float > > labelDistributions)
{
  isSplitNode_ = true;
  splitOnFeatureIdx_ = splitOnFeature;
  this->threshold_ = threshold;
  this->leftChildIdx_ = leftChildIdx;
  this->rightChildIdx_ = rightChildIdx;
  labelDistribution_ = labelDistributions.second;
  absLabelDistribution_ = labelDistributions.first;
}

void Node::ClearSplitNodeLabelDistribution()
{
    if(isSplitNode_)
    {
        labelDistribution_.clear();
        absLabelDistribution_.clear();
    }
}

void Node::ResetLabelDistribution()
{
  labelDistribution_.assign(labelDistribution_.size(), 0.0f);
  absLabelDistribution_.assign(absLabelDistribution_.size(), 0);
}

void Node::AddToAbsLabelDistribution(int labelIdx)
{
  absLabelDistribution_[labelIdx]++;
}

void Node::UpdateLabelDistribution(std::vector<int> labels, std::map<int, unsigned int>& pointsPerLabel)
{
  float sum = 0;
  
  for(unsigned int i=0; i<labels.size(); ++i)
    labelDistribution_[i] = ((float)absLabelDistribution_[i]) / ((float)pointsPerLabel[labels[i]]);
  
  for(unsigned int i=0; i<labelDistribution_.size(); ++i)
    sum += labelDistribution_[i];
  
  for(unsigned int i=0; i<absLabelDistribution_.size(); ++i)
    labelDistribution_[i] /= sum;
}

Node::~Node()
{

}

