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
#include <array>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <fstream>
#include <memory>
#include <random>

#include <boost/serialization/vector.hpp>

#include <v4r/core/macros.h>
#include <v4r/semantic_segmentation/entangled_definitions.h>
#include <v4r/semantic_segmentation/entangled_split_feature.h>
#include <v4r/semantic_segmentation/entangled_node.h>
#include <v4r/semantic_segmentation/entangled_data.h>


namespace v4r {

    class EntangledForestNode;
    class EntangledForestSplitFeature;

    class V4R_EXPORTS EntangledForestTree
    {
    private:
        friend class boost::serialization::access;
        template<class Archive>
        void serialize(Archive & ar, const unsigned int version __attribute__((unused)))
        {
            ar & mNodes;
            ar & mTreeIdx;
//        ar & mClassWeights;
        }

        int mTreeIdx;

        // uniform distribution for threshold
        std::uniform_real_distribution<double> mUniformDistThresh;

        std::mt19937* mRandomGenerator;
        std::vector<EntangledForestNode*> mNodes;


        // only needed in training
        int mCurrentDepth;

        static bool SmallerThan(float x, float t);

        // weights to rebalance label dist at leaf nodes
//    std::vector<double> mClassWeights;
    public:

        EntangledForestTree(int treeIdx = 0);
        EntangledForestTree(std::mt19937* randomGenerator, int treeIdx = 0);
        inline std::mt19937* GetRandomGenerator() { return mRandomGenerator; }
        void SetRandomGenerator(std::mt19937* randomGenerator);   // neccessary after loading from file
        EntangledForestNode* GetRootNode();

        inline int GetIndex() { return mTreeIdx; }
        void SetTreeIndex(int index);

        void Classify(EntangledForestData* data, std::vector<std::vector<double> > &result, int depth = -1);                             // classify whole image

        void GetFeatureMinMax(EntangledForestSplitFeature *f, EntangledForestData *data, const ClusterIdxItr start, const ClusterIdxItr end, const std::vector<double> &parameters, double &minValue, double &maxValue);
        double GetBestFeatureConfiguration(EntangledForestSplitFeature* f, EntangledForestData* data, ClusterIdxItr start, ClusterIdxItr end, int nParameterSamples, int nThresholdSamples, int minPointsForSplit, double currentEntropy);

        int GetLastNodeIDOfPrevLevel();
        //int GetCurrentDepth();
        void Train(EntangledForestData* trainingData, int maxDepth, int sampledSplitFunctionParameters, int sampledSplitFunctionThresholds, double minInformationGAin, int minPointsForSplit);

        void UpdateLeafs(EntangledForestData* data, int updateDepth, double updateWeight);

        void saveMatlab(std::string filename);

        void Clone(EntangledForestTree *t);
        EntangledForestNode *GetNode(int nodeIdx);
        int GetNrOfNodes();
        bool DoNodesShareAncestor(int nodeIdx1, int nodeIdx2, int maxSteps);
        virtual ~EntangledForestTree();

        inline int GetCurrentDepth()
        {
            /*// if nodes available
            if(nodes.size() > 0)
            {
                // get depth of last node and decrease by 1 (== target depth)
                return nodes[nodes.size()-1]->GetDepth();
            }
            else
            {
                return -1;
            }*/

            return mCurrentDepth;
        }
    };

}