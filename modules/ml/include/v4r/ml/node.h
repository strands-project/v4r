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


#ifndef NODE_H
#define NODE_H

#include <vector>
#include <memory>
#include <map>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/vector.hpp>

#include <v4r/core/macros.h>

namespace v4r {
namespace RandomForest {

class V4R_EXPORTS Node
{
private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version)
  {
    ar & isSplitNode_;
    ar & splitOnFeatureIdx_;
    ar & threshold_;
    ar & labelDistribution_;
    ar & leftChildIdx_;
    ar & rightChildIdx_;
  }
  
  bool isSplitNode_;
  int splitOnFeatureIdx_;
  float threshold_;
  std::vector<float> labelDistribution_;
  std::vector<unsigned int > absLabelDistribution_;
  int leftChildIdx_;
  int rightChildIdx_;
  
public:
  Node();
  Node(int splitOnFeature, float threshold_, int leftChildIdx_, int rightChildIdx_);
  Node(int splitOnFeature, float threshold_, int leftChildIdx_, int rightChildIdx_, std::pair< std::vector< unsigned int >, std::vector< float > > labelDistribution_);
  Node(std::pair< std::vector< unsigned int >, std::vector< float > > labelDistribution_);
  void ResetLabelDistribution();
  void ClearSplitNodeLabelDistribution();
  void AddToAbsLabelDistribution(int labelIdx);
  void UpdateLabelDistribution(std::vector< int > labels, std::map< int, unsigned int >& pointsPerLabel);
  inline int GetLeftChildIdx();
  inline int GetRightChildIdx();
  inline std::vector<float>& GetLabelDistribution();
  inline float GetThreshold();
  inline int GetSplitFeatureIdx();
  inline bool IsSplitNode();
  inline int EvaluateNode(std::vector< float >& point);
  virtual ~Node();
};

inline std::vector< float >& Node::GetLabelDistribution()
{
  return labelDistribution_;
}

inline bool Node::IsSplitNode()
{
  return isSplitNode_;
}
 
// for testing, does point go left or right?
inline int Node::EvaluateNode(std::vector< float >& point)
{  
  return point[splitOnFeatureIdx_] > threshold_ ? rightChildIdx_ : leftChildIdx_;
}

inline int Node::GetLeftChildIdx()
{
  return leftChildIdx_;
}

inline int Node::GetRightChildIdx()
{
  return rightChildIdx_;
}

inline int Node::GetSplitFeatureIdx()
{
  return splitOnFeatureIdx_;
}

inline float Node::GetThreshold()
{
  return threshold_;
}

}
}
#endif // NODE_H
