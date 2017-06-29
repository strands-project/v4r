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

#include <algorithm>
#include <time.h>
#include <random>
#include <fstream>
#include <string.h>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/map.hpp>

#include <v4r/core/macros.h>
#include <v4r/semantic_segmentation/entangled_data.h>
#include <v4r/semantic_segmentation/entangled_tree.h>

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_thread_num() 0
#endif

namespace v4r {

// how many data points to load into RAM at once
#define MAX_DATAPOINTS_TO_LOAD 1000000

class EntangledForestTree;

class V4R_EXPORTS EntangledForest
{   
private:
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & mTrees;
        ar & mLabelMap;
        ar & mMaxDepth;
        ar & mNTrees;
        ar & mSampledSplitFunctionParameters;
        ar & mSampledSplitFunctionThresholds;
        ar & mMinInformationGain;
        ar & mMinPointsForSplit;
        ar & mBaggingRatio;

        if(version >= 2)
        {
            ar & mLabelNames;
        }
    }

    std::mt19937 mRandomGenerator;
    std::vector<EntangledForestTree*> mTrees;
    int mMaxDepth;
    int mNTrees;
    int mSampledSplitFunctionParameters;   // new
    int mSampledSplitFunctionThresholds;   // new
    float mMinInformationGain;
    int mMinPointsForSplit;
    float mBaggingRatio;

    // for evaluation, split nodes can also store label distributions, to be able to traverse
    // trees only down to a certain depth for classification
    bool mSplitNodesStoreLabelDistribution;

    int mNLabels;
    std::map<int, int> mLabelMap;
    std::vector<std::string> mLabelNames;

public:
    EntangledForest();
    // new
    EntangledForest(int nTrees, int maxDepth = 8, float baggingRatio = 0.5, int sampledSplitFunctionParameters = 100, int sampledSplitFunctionThresholds = 50, float minInformationGain = 0.02, int minPointsForSplit = 5);
    void Train(EntangledForestData *trainingData, bool tryUniformBags, int verbosityLevel = 1);

    void UpdateRandomGenerator();   // neccessary after load from file

    void Classify(EntangledForestData* data, std::vector<int> &result, int maxDepth = -1, int useNTrees = -1); //, bool reweightLeafDistributions = false);
    void GetHardClassificationResult(std::vector<std::vector<double> > &softResult, std::vector<int> &result);
    void SoftClassify(EntangledForestData *data, std::vector<std::vector<double> > &result, int maxDepth = -1, int useNTrees = -1); //, bool reweightLeafDistributions = false);   // new version)

    void SaveToFile(std::string filename);
    void SaveToBinaryFile(std::string filename);
    static void LoadFromFile(std::string filename, EntangledForest &f);
    static void LoadFromBinaryFile(std::string filename, EntangledForest &f);
    std::vector<int> GetLabels();
    std::map<int, int> GetLabelMap();
    int GetNrOfLabels();
    std::vector<std::string>& GetLabelNames();
    bool SetLabelNames(std::vector<std::string>& names);

    EntangledForestTree* GetTree(int idx);
    int GetNrOfTrees();
    void saveMatlab(std::string filename);
    void Merge(EntangledForest &f);

    void updateLeafs(EntangledForestData *trainingData, int updateDepth, double updateWeight, bool tryUniformBags);

    void correctTreeIndices();
    virtual ~EntangledForest();
};

}

BOOST_CLASS_VERSION(v4r::EntangledForest, 2)