/******************************************************************************
 * Copyright (c) 2017 Daniel Wolf
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 ******************************************************************************/

#pragma once
#include <vector>
#include <memory>
#include <map>
#include <mutex>
#include <random>

#include <boost/serialization/vector.hpp>

#include <v4r/core/macros.h>
#include <v4r/semantic_segmentation/entangled_data.h>
#include <v4r/semantic_segmentation/entangled_split_feature.h>
#include <v4r/semantic_segmentation/entangled_tree.h>

namespace v4r {

    class EntangledForestSplitFeature;
    class EntangledForestNode;
    class EntangledForestTree;

    class V4R_EXPORTS EntangledForestNode
    {
    private:
        friend class boost::serialization::access;
        template<class Archive>
        void serialize(Archive & ar, const unsigned int version __attribute__((unused)))
        {
            ar & mParent;
            ar & mSplitFeature;
            ar & mDepth;
            ar & mLeftChildIdx;
            ar & mRightChildIdx;
            ar & mLabelDistribution;
            ar & mAbsLabelDistribution;
            ar & mIsSplitNode;
            ar & mOnFrontier;

        }

        EntangledForestNode* mParent;

        bool mOnFrontier;

        bool mIsSplitNode;
        EntangledForestSplitFeature* mSplitFeature;

        int mDepth;

        int mLeftChildIdx;
        int mRightChildIdx;

        std::vector<double> mLabelDistribution;
        std::vector<unsigned int> mAbsLabelDistribution;


        ClusterIdxItr SplitOnFeature(EntangledForestSplitFeature* f, EntangledForestData* data, ClusterIdxItr start, ClusterIdxItr end);

    public:
        std::pair<ClusterIdxItr, ClusterIdxItr > mTrainingDataIterators;

        EntangledForestNode();
        EntangledForestNode(EntangledForestData* data);
        EntangledForestNode(EntangledForestData* data, ClusterIdxItr dataStart, ClusterIdxItr dataEnd, EntangledForestNode *parent);

        inline unsigned int GetNrOfPoints() { return std::distance(mTrainingDataIterators.first, mTrainingDataIterators.second); }
        double GetWeightedNrOfPoints(std::vector<double> &classWeights);


        void ResetLabelDistribution();
        void ClearSplitNodeLabelDistribution();
        void AddToAbsLabelDistribution(int labelIdx);
        void UpdateLabelDistribution(std::vector< int > labels, std::map< int, unsigned int >& pointsPerLabel);

        void Split(EntangledForestData* data, EntangledForestSplitFeature* f, EntangledForestNode** leftChild, EntangledForestNode** rightChild);

        inline int GetDepth() { return mDepth; }
        inline int GetLeftChildIdx();
        inline int GetRightChildIdx();
        inline void SetLeftChildIdx(int idx);
        inline void SetRightChildIdx(int idx);
        inline std::vector<double>& GetLabelDistribution();
        inline std::vector<unsigned int>& GetAbsLabelDistribution();
        inline bool IsSplitNode();
        inline bool IsOnFrontier() { return mOnFrontier; }
        void SetAsLeafNode();
        bool IsDescendantOf(EntangledForestNode* n);
        int evaluate(EntangledForestData* data, int imageIdx, int clusterIdx);    // returns child idx
        EntangledForestNode* GetParent();

        EntangledForestSplitFeature* GetSplitFeature();

        void SetParent(EntangledForestNode*par);
        bool IsTopClass(int classIdx);
        bool IsAmongTopN(unsigned int classIdx, unsigned int N);
        void ApplyClusterLabelDistribution(std::vector<double>& labeldist);
        void UpdateLabelDistribution(std::vector<double>& labeldist);

        void SetRandomGenerator(std::mt19937* randomGenerator);    // neccessary after load

        void Clone(EntangledForestNode* n, EntangledForestTree* newTree);
        virtual ~EntangledForestNode();
    };

    inline std::vector< double >& EntangledForestNode::GetLabelDistribution()
    {
        return mLabelDistribution;
    }

    inline std::vector< unsigned int >& EntangledForestNode::GetAbsLabelDistribution()
    {
        return mAbsLabelDistribution;
    }

    inline bool EntangledForestNode::IsSplitNode()
    {
        return mIsSplitNode;
    }

    inline int EntangledForestNode::GetLeftChildIdx()
    {
        return mLeftChildIdx;
    }

    inline int EntangledForestNode::GetRightChildIdx()
    {
        return mRightChildIdx;
    }

    inline void EntangledForestNode::SetLeftChildIdx(int idx)
    {
        mLeftChildIdx = idx;
    }

    inline void EntangledForestNode::SetRightChildIdx(int idx)
    {
        mRightChildIdx = idx;
    }

}