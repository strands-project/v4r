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

#include <math.h>

#include <v4r/semantic_segmentation/entangled_split_feature.h>

using namespace std;

namespace v4r
{

    EntangledForestSplitFeature::EntangledForestSplitFeature() : mName(std::string()), mTree(nullptr), mRandomGenerator(nullptr), mCoinFlip(0, 1),
                                                                 mDynamicThreshold(false), mMaxParameterSamplings(0), mThreshold(0.0f),
                                                                 mMinDepthForFeature(0), mMaxDepthForFeature(std::numeric_limits<int>::max()),
                                                                 mActiveDepthLevels(std::vector < unsigned int>())
    {
    }

    EntangledForestSplitFeature::EntangledForestSplitFeature(string name, EntangledForestTree *tree, std::mt19937 *randomGenerator, std::vector<unsigned int> activeDepthLevels) :
            mName (name), mTree(tree), mRandomGenerator(randomGenerator), mCoinFlip(0, 1), mDynamicThreshold(false),
            mMaxParameterSamplings(0), mThreshold(0.0f), mMinDepthForFeature(0), mMaxDepthForFeature(std::numeric_limits<int>::max()),
            mActiveDepthLevels(activeDepthLevels)
    {
    }

    EntangledForestSplitFeature::EntangledForestSplitFeature(string name, EntangledForestTree *tree, std::mt19937 *randomGenerator, const int minDepth,
                                                             const int maxDepth) : mName(name), mTree(tree),
                                                                                                                     mRandomGenerator(randomGenerator),
                                                                                                                     mCoinFlip(0, 1),
                                                                                                                     mDynamicThreshold(false),
                                                                                                                     mMaxParameterSamplings(0),
                                                                                                                     mThreshold(0.0f),
                                                                                                                     mMinDepthForFeature(minDepth),
                                                                                                                     mMaxDepthForFeature(maxDepth),
                                                                                                                     mActiveDepthLevels(std::vector < unsigned int>())
    {
    }

    EntangledForestSplitFeature::EntangledForestSplitFeature(const EntangledForestSplitFeature &f) : mName(f.mName), mTree(f.mTree), mRandomGenerator(f.mRandomGenerator),
                                                                                                     mCoinFlip(0, 1), mDynamicThreshold(f.mDynamicThreshold),
                                                                                                     mMaxParameterSamplings(f.mMaxParameterSamplings),
                                                                                                     mThreshold(f.mThreshold),
                                                                                                     mMinDepthForFeature(f.mMinDepthForFeature),
                                                                                                     mMaxDepthForFeature(f.mMaxDepthForFeature),
                                                                                                     mActiveDepthLevels(f.mActiveDepthLevels)
    {

    }

    string EntangledForestSplitFeature::GetName()
    {
        return mName;
    }

    void EntangledForestSplitFeature::SetRandomGenerator(std::mt19937 *randomGenerator)
    {
        this->mRandomGenerator = randomGenerator;
    }

/// Checks if label classIdx is among top N in probabilities vector
    bool EntangledForestSplitFeature::IsAmongTopN(const std::vector<double> probabilities, int classIdx, int N)
    {
        int larger = 0;
        double compareWith = probabilities[classIdx];

        for (unsigned int i = 0; i < probabilities.size(); ++i)
        {
            if ((int)i != classIdx && probabilities[i] > compareWith)
            {
                if (++larger >= N)
                {
                    return false;
                }
            }
        }
        return true;
    }

    int EntangledForestSplitFeature::GetMaxParameterSamples()
    {
        return mMaxParameterSamplings;
    }

    bool EntangledForestSplitFeature::HasDynamicThreshold()
    {
        return mDynamicThreshold;
    }

    void EntangledForestSplitFeature::SetThreshold(double t)
    {
        mThreshold = t;
    }

    bool EntangledForestSplitFeature::computeTraining(EntangledForestData*, int /* imageIdx */, int /* clusterIdx */, const std::vector<double>& /* parameters */,
                                                      double& /* value */)
    {
        // TODO: throw exception (function has to be overridden by derived class)
        return false;
    }

    bool EntangledForestSplitFeature::evaluateTraining(EntangledForestData*, int /* imageIdx */, int /* clusterIdx */, const std::vector<double>& /* parameters */,
                                                       bool& /* result */)
    {
        // TODO: throw exception (function has to be overridden by derived class)
        return false;
    }

    EntangledForestUnaryFeature::EntangledForestUnaryFeature(std::string name, EntangledForestTree *tree,
                                                             std::mt19937 *randomGenerator, int featureIdx) :
            EntangledForestSplitFeature(name, tree, randomGenerator), mFeatureIdx(featureIdx)
    {
        mDynamicThreshold = true;
        mMaxParameterSamplings = 1;
    }

    EntangledForestSplitFeature *EntangledForestUnaryFeature::Clone()
    {
        return new EntangledForestUnaryFeature(*this);
    }

    EntangledForestSplitFeature *EntangledForestUnaryFeature::Clone(EntangledForestTree *newTree)
    {
        EntangledForestUnaryFeature *f = new EntangledForestUnaryFeature(*this);
        f->mTree = newTree;
        return f;
    }


    std::string EntangledForestUnaryFeature::ToString()
    {
        return "Unary feature " + std::to_string(mFeatureIdx) + " t=" + std::to_string(mThreshold);
    }

    void EntangledForestUnaryFeature::SetParameters(const std::vector<double> &parameters __attribute__((unused)))
    {
    }

    void EntangledForestUnaryFeature::SampleRandomParameters(std::vector<double> &parameters)
    {
        // actually useless for this feature
        parameters.resize(1);

        // sample parameters //////////////////////////
        parameters[0] = 0;  // no sampling necessary
    }

    bool EntangledForestUnaryFeature::computeTraining(EntangledForestData *data, int imageIdx, int clusterIdx, const std::vector<double> &parameters __attribute__((unused)),
                                                      double &value)
    {
        return compute(data, imageIdx, clusterIdx, value);
    }

    bool EntangledForestUnaryFeature::compute(EntangledForestData *data, int imageIdx, int clusterIdx, double &value)
    {
        value = data->GetUnaryFeature(imageIdx, clusterIdx, mFeatureIdx);
        return true;
    }

    EntangledForestUnaryFeature::EntangledForestUnaryFeature() : EntangledForestSplitFeature()
    {
    }

    EntangledForestUnaryFeature::EntangledForestUnaryFeature(const EntangledForestUnaryFeature &f) : EntangledForestSplitFeature(f), mFeatureIdx(f.mFeatureIdx)
    {

    }

    bool EntangledForestUnaryFeature::evaluateInference(EntangledForestData *data, int imageIdx, int clusterIdx)
    {
        double value(0.0f);
        compute(data, imageIdx, clusterIdx, value);
        return value > mThreshold;
    }

//////////////////////////

    EntangledForestClusterExistsFeature::EntangledForestClusterExistsFeature(string name, EntangledForestTree *tree, std::mt19937 *randomGenerator,
                                                                             bool horizontal) : EntangledForestSplitFeature(name, tree, randomGenerator, 5), mHorizontal(horizontal)
    {
        mDynamicThreshold = false;
        mMaxParameterSamplings = std::numeric_limits<int>::max();
        mUniformDistPtPl = std::uniform_real_distribution<double>(-MAX_PAIRWISE_DISTANCE, MAX_PAIRWISE_DISTANCE);
        mUniformDistAngle = std::uniform_real_distribution<double>(-PI, PI);

        mNormalDistAngle = std::normal_distribution<double>(0.0, SIGMA_PAIRWISE_ANGLE);
        mNormalDistPtPl = std::normal_distribution<double>(0.0, SIGMA_PAIRWISE_DISTANCE);


        mMinAngle = 0.0;
        mMaxAngle = 0.0;
        mMinPtPlDist = 0.0;
        mMaxPtPlDist = 0.0;
        mMaxEDist = 0.0;
    }

    EntangledForestSplitFeature *EntangledForestClusterExistsFeature::Clone()
    {
        return new EntangledForestClusterExistsFeature(*this);
    }

    EntangledForestSplitFeature *EntangledForestClusterExistsFeature::Clone(EntangledForestTree *newTree)
    {
        EntangledForestClusterExistsFeature *f = new EntangledForestClusterExistsFeature(*this);
        f->mTree = newTree;
        return f;
    }


    std::string EntangledForestClusterExistsFeature::ToString()
    {
        if (mHorizontal)
        {
            return "Pairwise feature: H-AngleDiff [" + std::to_string(RAD2DEG(mMinAngle)) + "," +
                   std::to_string(RAD2DEG(mMaxAngle)) + "], ptPl [" + std::to_string(mMinPtPlDist) + "," +
                   std::to_string(mMaxPtPlDist) + "]";
        }
        else
        {
            return "Pairwise feature: V-AngleDiff [" + std::to_string(RAD2DEG(mMinAngle)) + "," +
                   std::to_string(RAD2DEG(mMaxAngle)) + "], ptPl [" + std::to_string(mMinPtPlDist) + "," +
                   std::to_string(mMaxPtPlDist) + "]";
        }
    }

    void EntangledForestClusterExistsFeature::SetParameters(const std::vector<double> &parameters)
    {
        mMinAngle = parameters[0];
        mMaxAngle = parameters[1];
        mMinPtPlDist = parameters[2];
        mMaxPtPlDist = parameters[3];
        mMaxEDist = parameters[4];
    }

    void EntangledForestClusterExistsFeature::SampleRandomParameters(std::vector<double> &parameters)
    {
        // actually useless for this feature
        parameters.resize(5);

        // sample mean angle and mean distance
        double meanAngle = mUniformDistAngle(*mRandomGenerator);
        double meanDist = mUniformDistPtPl(*mRandomGenerator);
        double diffAngle = std::abs(mNormalDistAngle(*mRandomGenerator));
        double distPtPl = std::abs(mNormalDistPtPl(*mRandomGenerator));

        // sample min/max from normal distribution around mean

        // sample parameters //////////////////////////
        parameters[0] = meanAngle - diffAngle;
        parameters[1] = meanAngle + diffAngle;
        parameters[2] = meanDist - distPtPl;
        parameters[3] = meanDist + distPtPl;
        parameters[4] = MAX_PAIRWISE_DISTANCE;     // TODO: change this?
    }

    bool EntangledForestClusterExistsFeature::evaluateTraining(EntangledForestData *data, int imageIdx, int clusterIdx,
                                                               const std::vector<double> &parameters, bool &result)
    {
        return evaluate(data, imageIdx, clusterIdx, parameters[0], parameters[1], parameters[2], parameters[3],
                        parameters[4], result);
    }

    bool EntangledForestClusterExistsFeature::evaluate(EntangledForestData *data, int imageIdx, int clusterIdx, double minangle, double maxangle,
                                                       double minptpl, double maxptpl, double maxeuclid, bool &result)
    {
        std::vector<int> remaining;

        data->FilterClustersByGeometry(imageIdx, clusterIdx, mHorizontal, minangle, maxangle, minptpl, maxptpl, maxeuclid,
                                       remaining);

        result = remaining.size() > 0;
        return true;
    }

    EntangledForestClusterExistsFeature::EntangledForestClusterExistsFeature() : EntangledForestSplitFeature()
    {

    }

    EntangledForestClusterExistsFeature::EntangledForestClusterExistsFeature(const EntangledForestClusterExistsFeature &f) : EntangledForestSplitFeature(f),
                                                                                                                             mMinAngle(f.mMinAngle), mMaxAngle(f.mMaxAngle),
                                                                                                                             mMinPtPlDist(f.mMinPtPlDist), mMaxPtPlDist(f.mMaxPtPlDist),
                                                                                                                             mMaxEDist(f.mMaxEDist), mHorizontal(f.mHorizontal)
    {

    }

    bool EntangledForestClusterExistsFeature::evaluateInference(EntangledForestData *data, int imageIdx, int clusterIdx)
    {
        bool result;
        evaluate(data, imageIdx, clusterIdx, mMinAngle, mMaxAngle, mMinPtPlDist, mMaxPtPlDist, mMaxEDist, result);
        return result;
    }


    EntangledForestTopNFeature::EntangledForestTopNFeature(string name, EntangledForestTree *tree, std::mt19937 *randomGenerator, bool horizontal,
                                                           int nLabels) : EntangledForestSplitFeature(name, tree, randomGenerator, 5),
                                                                          mHorizontal(horizontal)
    {
        mDynamicThreshold = false;
        mMaxParameterSamplings = std::numeric_limits<int>::max();
        mUniformDistPtPl = std::uniform_real_distribution<double>(-MAX_PAIRWISE_DISTANCE, MAX_PAIRWISE_DISTANCE);
        mUniformDistAngle = std::uniform_real_distribution<double>(-PI, PI);

        mNormalDistAngle = std::normal_distribution<double>(0.0, SIGMA_PAIRWISE_ANGLE);
        mNormalDistPtPl = std::normal_distribution<double>(0.0, SIGMA_PAIRWISE_DISTANCE);

        // uniform distribution for class parameters (up to labelnr-1)
        mUniformDistLabel = std::uniform_int_distribution<int>(0, nLabels - 1);
        // uniform distribution for N parameters
        mUniformDistN = std::uniform_int_distribution<int>(1, MAX_N_TOP_CLASSES);


        mDontcare = std::uniform_int_distribution<int>(1, 8);


        mMinAngle = 0.0;
        mMaxAngle = 0.0;
        mMinPtPlDist = 0.0;
        mMaxPtPlDist = 0.0;
        mMaxEDist = 0.0;

        mLabel = 0;
        mN = 0;
    }

    EntangledForestSplitFeature *EntangledForestTopNFeature::Clone()
    {
        return new EntangledForestTopNFeature(*this);
    }

    EntangledForestSplitFeature *EntangledForestTopNFeature::Clone(EntangledForestTree *newTree)
    {
        EntangledForestTopNFeature *f = new EntangledForestTopNFeature(*this);
        f->mTree = newTree;
        return f;
    }


    std::string EntangledForestTopNFeature::ToString()
    {
        if (mHorizontal)
        {
            return "Entangled H feature: TopN [" + std::to_string(RAD2DEG(mMinAngle)) + "," +
                   std::to_string(RAD2DEG(mMaxAngle)) + "], ptPl [" + std::to_string(mMinPtPlDist) + "," +
                   std::to_string(mMaxPtPlDist) +
                   "], lbl " + std::to_string(mLabel) + " in Top " + std::to_string(mN);
        }
        else
        {
            return "Entangled V feature: TopN [" + std::to_string(RAD2DEG(mMinAngle)) + "," +
                   std::to_string(RAD2DEG(mMaxAngle)) + "], ptPl [" + std::to_string(mMinPtPlDist) + "," +
                   std::to_string(mMaxPtPlDist) +
                   "], lbl " + std::to_string(mLabel) + " in Top " + std::to_string(mN);
        }
    }

    void EntangledForestTopNFeature::SetParameters(const std::vector<double> &parameters)
    {
        mMinAngle = parameters[0];
        mMaxAngle = parameters[1];
        mMinPtPlDist = parameters[2];
        mMaxPtPlDist = parameters[3];
        mMaxEDist = parameters[4];
        mLabel = (int) parameters[5];
        mN = (int) parameters[6];
    }

    void EntangledForestTopNFeature::SampleRandomParameters(std::vector<double> &parameters)
    {
        // actually useless for this feature
        parameters.resize(7);

        // sample angle and distance thresholds
        int bla = mDontcare(*mRandomGenerator);

        double meanAngle = mUniformDistAngle(*mRandomGenerator);
        double meanDist = mUniformDistPtPl(*mRandomGenerator);
        double diffAngle = std::abs(mNormalDistAngle(*mRandomGenerator));
        double distPtPl = std::abs(mNormalDistPtPl(*mRandomGenerator));

        switch (bla)
        {
            case 1:
                // care about angle, not dist
                parameters[0] = meanAngle - diffAngle;
                parameters[1] = meanAngle + diffAngle;
                parameters[2] = -100.0;
                parameters[3] = 100.0;
                parameters[4] = MAX_PAIRWISE_DISTANCE;     // TODO: change this?
                parameters[5] = mUniformDistLabel(*mRandomGenerator);
                parameters[6] = mUniformDistN(*mRandomGenerator);
                break;
            case 2: // care about dist, not angle
                parameters[0] = -2.0 * PI;
                parameters[1] = 2.0 * PI;
                parameters[2] = meanDist - distPtPl;
                parameters[3] = meanDist + distPtPl;
                parameters[4] = MAX_PAIRWISE_DISTANCE;     // TODO: change this?
                parameters[5] = mUniformDistLabel(*mRandomGenerator);
                parameters[6] = mUniformDistN(*mRandomGenerator);
                break;
            default:
                // care about both
                // sample parameters //////////////////////////
                parameters[0] = meanAngle - diffAngle;
                parameters[1] = meanAngle + diffAngle;
                parameters[2] = meanDist - distPtPl;
                parameters[3] = meanDist + distPtPl;
                parameters[4] = MAX_PAIRWISE_DISTANCE;     // TODO: change this?
                parameters[5] = mUniformDistLabel(*mRandomGenerator);
                parameters[6] = mUniformDistN(*mRandomGenerator);
                break;
        }

        // sample parameters //////////////////////////
        /*parameters[0] = meanAngle - diffAngle;
        parameters[1] = meanAngle + diffAngle;
        parameters[2] = meanDist - distPtPl;
        parameters[3] = meanDist + distPtPl;
        parameters[4] = MAX_PAIRWISE_DISTANCE;     // TODO: change this?
        parameters[5] = uniformDistLabel(*mRandomGenerator);
        parameters[6] = uniformDistN(*mRandomGenerator);*/
    }

    bool
    EntangledForestTopNFeature::evaluateTraining(EntangledForestData *data, int imageIdx, int clusterIdx, const std::vector<double> &parameters,
                                                 bool &result)
    {
        return evaluate(data, imageIdx, clusterIdx, parameters[0], parameters[1], parameters[2], parameters[3],
                        parameters[4], parameters[5], parameters[6], result);
    }

    bool EntangledForestTopNFeature::evaluate(EntangledForestData *data, int imageIdx, int clusterIdx, double minangle, double maxangle,
                                              double minptpl, double maxptpl, double maxeuclid, unsigned int label, unsigned int N,
                                              bool &result)
    {
        // first filter clusters by geometric constraints
        std::vector<int> remaining;

        data->FilterClustersByGeometry(imageIdx, clusterIdx, mHorizontal, minangle, maxangle, minptpl, maxptpl, maxeuclid,
                                       remaining);

        result = false;

        // then check if at least 1 remaining cluster fulfills the topN criterion
        for (unsigned int i = 0; i < remaining.size(); ++i)
        {
            int nodeidx = data->GetClusterNodeIdx(imageIdx, remaining[i], mTree->GetIndex());
            if (nodeidx != 0)
            {
                EntangledForestNode *n = mTree->GetNode(nodeidx);
                if (n->IsAmongTopN(label, N))
                {
                    result = true;
                    return true;
                }
            }
        }

        return true;
    }

    EntangledForestTopNFeature::EntangledForestTopNFeature() : EntangledForestSplitFeature()
    {

    }

    EntangledForestTopNFeature::EntangledForestTopNFeature(const EntangledForestTopNFeature &f) : EntangledForestSplitFeature(f), mMinAngle(f.mMinAngle),
                                                                                                  mMaxAngle(f.mMaxAngle),
                                                                                                  mMinPtPlDist(f.mMinPtPlDist),
                                                                                                  mMaxPtPlDist(f.mMaxPtPlDist),
                                                                                                  mMaxEDist(f.mMaxEDist), mHorizontal(f.mHorizontal),
                                                                                                  mN(f.mN), mLabel(f.mLabel)

    {

    }

    bool EntangledForestTopNFeature::evaluateInference(EntangledForestData *data, int imageIdx, int clusterIdx)
    {
        bool result;
        evaluate(data, imageIdx, clusterIdx, mMinAngle, mMaxAngle, mMinPtPlDist, mMaxPtPlDist, mMaxEDist, mLabel, mN, result);
        return result;
    }

/////

    EntangledForestInverseTopNFeature::EntangledForestInverseTopNFeature(string name, EntangledForestTree *tree, std::mt19937 *randomGenerator,
                                                                         int nLabels) : EntangledForestSplitFeature(name, tree, randomGenerator, 5)
    {
        mDynamicThreshold = false;
        mMaxParameterSamplings = std::numeric_limits<int>::max();
        mUniformDistPtPl = std::uniform_real_distribution<double>(0.0, MAX_PAIRWISE_DISTANCE); // change?
        mNormalDistPtPl = std::normal_distribution<double>(0.0, SIGMA_PAIRWISE_DISTANCE);
        mUniformDistAngle = std::uniform_real_distribution<double>(-PI, PI);
        mNormalDistAngle = std::normal_distribution<double>(0.0, SIGMA_PAIRWISE_ANGLE);

        // uniform distribution for class parameters (up to labelnr-1)
        mUniformDistLabel = std::uniform_int_distribution<int>(0, nLabels - 1);
        // uniform distribution for N parameters
        mUniformDistN = std::uniform_int_distribution<int>(1, MAX_N_TOP_CLASSES);


        mDontcare = std::uniform_int_distribution<int>(1, 8);


        mMinAngle = 0.0;
        mMaxAngle = 0.0;
        mMinIPtPlDist = 0.0;
        mMaxIPtPlDist = 0.0;
        mMaxEDist = 0.0;

        mLabel = 0;
        mN = 0;
    }

    EntangledForestSplitFeature *EntangledForestInverseTopNFeature::Clone()
    {
        return new EntangledForestInverseTopNFeature(*this);
    }

    EntangledForestSplitFeature *EntangledForestInverseTopNFeature::Clone(EntangledForestTree *newTree)
    {
        EntangledForestInverseTopNFeature *f = new EntangledForestInverseTopNFeature(*this);
        f->mTree = newTree;
        return f;
    }


    std::string EntangledForestInverseTopNFeature::ToString()
    {
        return "Entangled inverse feature: TopN [" + std::to_string(RAD2DEG(mMinAngle)) + "," +
               std::to_string(RAD2DEG(mMaxAngle)) + "], ptPl [" + std::to_string(mMinIPtPlDist) + "," +
               std::to_string(mMaxIPtPlDist) +
               "], lbl " + std::to_string(mLabel) + " in Top " + std::to_string(mN);
    }

    void EntangledForestInverseTopNFeature::SetParameters(const std::vector<double> &parameters)
    {
        mMinAngle = parameters[0];
        mMaxAngle = parameters[1];
        mMinIPtPlDist = parameters[2];
        mMaxIPtPlDist = parameters[3];
        mMaxEDist = parameters[4];
        mLabel = (int) parameters[5];
        mN = (int) parameters[6];
    }

    void EntangledForestInverseTopNFeature::SampleRandomParameters(std::vector<double> &parameters)
    {
        // actually useless for this feature
        parameters.resize(7);

        // sample angle and distance thresholds
        double minDist = 0.0;
        double maxDist = 0.0;

        minDist = mUniformDistPtPl(*mRandomGenerator);
        maxDist = minDist + 2.0 * std::abs(mNormalDistPtPl(*mRandomGenerator));

        if (mCoinFlip(*mRandomGenerator))
        {
            double swap = -minDist;
            minDist = -maxDist;
            maxDist = swap;
        }

        double meanAngle = mUniformDistAngle(*mRandomGenerator);
        double diffAngle = std::abs(mNormalDistAngle(*mRandomGenerator));

        // sample angle and distance thresholds
        int bla = mDontcare(*mRandomGenerator);

        switch (bla)
        {
            case 1:
                // care about angle, not dist
                parameters[0] = meanAngle - diffAngle;
                parameters[1] = meanAngle + diffAngle;
                parameters[2] = -100.0;
                parameters[3] = 100.0;
                parameters[4] = MAX_PAIRWISE_DISTANCE;     // TODO: change this?
                parameters[5] = mUniformDistLabel(*mRandomGenerator);
                parameters[6] = mUniformDistN(*mRandomGenerator);
                break;
            case 2: // care about dist, not angle
                parameters[0] = -2.0 * PI;
                parameters[1] = 2.0 * PI;
                parameters[2] = minDist;
                parameters[3] = maxDist;
                parameters[4] = MAX_PAIRWISE_DISTANCE;     // TODO: change this?
                parameters[5] = mUniformDistLabel(*mRandomGenerator);
                parameters[6] = mUniformDistN(*mRandomGenerator);
                break;
            default:
                // care about both
                // sample parameters //////////////////////////
                parameters[0] = meanAngle - diffAngle;
                parameters[1] = meanAngle + diffAngle;
                parameters[2] = minDist;
                parameters[3] = maxDist;
                parameters[4] = MAX_PAIRWISE_DISTANCE;     // TODO: change this?
                parameters[5] = mUniformDistLabel(*mRandomGenerator);
                parameters[6] = mUniformDistN(*mRandomGenerator);
                break;
        }


        // sample parameters //////////////////////////
        /*parameters[0] = meanAngle - diffAngle;
        parameters[1] = meanAngle + diffAngle;
        parameters[2] = minDist;
        parameters[3] = maxDist;
        parameters[4] = MAX_PAIRWISE_DISTANCE;     // TODO: change this?
        parameters[5] = uniformDistLabel(*mRandomGenerator);
        parameters[6] = uniformDistN(*mRandomGenerator);*/
    }

    bool EntangledForestInverseTopNFeature::evaluateTraining(EntangledForestData *data, int imageIdx, int clusterIdx,
                                                             const std::vector<double> &parameters, bool &result)
    {
        return evaluate(data, imageIdx, clusterIdx, parameters[0], parameters[1], parameters[2], parameters[3],
                        parameters[4], parameters[5], parameters[6], result);
    }

    bool EntangledForestInverseTopNFeature::evaluate(EntangledForestData *data, int imageIdx, int clusterIdx, double minangle, double maxangle,
                                                     double miniptpl, double maxiptpl, double maxeuclid, unsigned int label,
                                                     unsigned int N, bool &result)
    {
        // first filter clusters by geometric constraints
        std::vector<int> remaining;

        data->FilterClustersByInversePtPl(imageIdx, clusterIdx, minangle, maxangle, miniptpl, maxiptpl, maxeuclid,
                                          remaining);

        result = false;

        // then check if at least 1 remaining cluster fulfills the topN criterion
        for (unsigned int i = 0; i < remaining.size(); ++i)
        {
            int nodeidx = data->GetClusterNodeIdx(imageIdx, remaining[i], mTree->GetIndex());
            if (nodeidx != 0)
            {
                EntangledForestNode *n = mTree->GetNode(nodeidx);
                if (n->IsAmongTopN(label, N))
                {
                    result = true;
                    return true;
                }
            }
        }

        return true;
    }

    EntangledForestInverseTopNFeature::EntangledForestInverseTopNFeature() : EntangledForestSplitFeature()
    {

    }

    EntangledForestInverseTopNFeature::EntangledForestInverseTopNFeature(const EntangledForestInverseTopNFeature &f) : EntangledForestSplitFeature(f),
                                                                                                                       mMinAngle(f.mMinAngle),
                                                                                                                       mMaxAngle(f.mMaxAngle),
                                                                                                                       mMinIPtPlDist(f.mMinIPtPlDist),
                                                                                                                       mMaxIPtPlDist(f.mMaxIPtPlDist),
                                                                                                                       mMaxEDist(f.mMaxEDist), mN(f.mN),
                                                                                                                       mLabel(f.mLabel)
    {

    }

    bool EntangledForestInverseTopNFeature::evaluateInference(EntangledForestData *data, int imageIdx, int clusterIdx)
    {
        bool result;
        evaluate(data, imageIdx, clusterIdx, mMinAngle, mMaxAngle, mMinIPtPlDist, mMaxIPtPlDist, mMaxEDist, mLabel, mN, result);
        return result;
    }

    EntangledForestCommonAncestorFeature::EntangledForestCommonAncestorFeature(string name, EntangledForestTree *tree, std::mt19937 *randomGenerator,
                                                                               bool horizontal) : EntangledForestSplitFeature(name, tree, randomGenerator, 5),
                                                                                                  mHorizontal(horizontal)
    {
        mDynamicThreshold = false;
        mMaxParameterSamplings = std::numeric_limits<int>::max();
        mUniformDistPtPl = std::uniform_real_distribution<double>(-MAX_PAIRWISE_DISTANCE, MAX_PAIRWISE_DISTANCE);
        mUniformDistAngle = std::uniform_real_distribution<double>(-PI, PI);

        mNormalDistAngle = std::normal_distribution<double>(0.0, SIGMA_PAIRWISE_ANGLE);
        mNormalDistPtPl = std::normal_distribution<double>(0.0, SIGMA_PAIRWISE_DISTANCE);


        mDontcare = std::uniform_int_distribution<int>(1, 8);


        mMinAngle = 0.0;
        mMaxAngle = 0.0;
        mMinPtPlDist = 0.0;
        mMaxPtPlDist = 0.0;
        mMaxEDist = 0.0;

        mMaxSteps = 0;
    }

    EntangledForestSplitFeature *EntangledForestCommonAncestorFeature::Clone()
    {
        return new EntangledForestCommonAncestorFeature(*this);
    }

    EntangledForestSplitFeature *EntangledForestCommonAncestorFeature::Clone(EntangledForestTree *newTree)
    {
        EntangledForestCommonAncestorFeature *f = new EntangledForestCommonAncestorFeature(*this);
        f->mTree = newTree;
        return f;
    }


    std::string EntangledForestCommonAncestorFeature::ToString()
    {
        if (mHorizontal)
        {
            return "Entangled H feature: CommonAncestor [" + std::to_string(RAD2DEG(mMinAngle)) + "," +
                   std::to_string(RAD2DEG(mMaxAngle)) + "], ptPl [" + std::to_string(mMinPtPlDist) + "," +
                   std::to_string(mMaxPtPlDist) +
                   "], same ancestor in " + std::to_string(mMaxSteps) + " steps";
        }
        else
        {
            return "Entangled V feature: CommonAncestor [" + std::to_string(RAD2DEG(mMinAngle)) + "," +
                   std::to_string(RAD2DEG(mMaxAngle)) + "], ptPl [" + std::to_string(mMinPtPlDist) + "," +
                   std::to_string(mMaxPtPlDist) +
                   "], same ancestor in " + std::to_string(mMaxSteps) + " steps";
        }
    }

    void EntangledForestCommonAncestorFeature::SetParameters(const std::vector<double> &parameters)
    {
        mMinAngle = parameters[0];
        mMaxAngle = parameters[1];
        mMinPtPlDist = parameters[2];
        mMaxPtPlDist = parameters[3];
        mMaxEDist = parameters[4];
        mMaxSteps = (int) parameters[5];
    }

    void EntangledForestCommonAncestorFeature::SampleRandomParameters(std::vector<double> &parameters)
    {
        // actually useless for this feature
        parameters.resize(6);

        // sample angle and distance thresholds
        double meanAngle = mUniformDistAngle(*mRandomGenerator);
        double meanDist = mUniformDistPtPl(*mRandomGenerator);
        double diffAngle = std::abs(mNormalDistAngle(*mRandomGenerator));
        double distPtPl = std::abs(mNormalDistPtPl(*mRandomGenerator));

        // uniform distribution for maxSteps parameters
        std::uniform_int_distribution<int> uniformDistSteps(1, max(1, mTree->GetCurrentDepth() - 1));


        int bla = mDontcare(*mRandomGenerator);

        switch (bla)
        {
            case 1:
                // care about angle, not dist
                parameters[0] = meanAngle - diffAngle;
                parameters[1] = meanAngle + diffAngle;
                parameters[2] = -100.0;
                parameters[3] = 100.0;
                parameters[4] = MAX_PAIRWISE_DISTANCE;     // TODO: change this?
                parameters[5] = uniformDistSteps(*mRandomGenerator);
                break;
            case 2: // care about dist, not angle
                parameters[0] = -2.0 * PI;
                parameters[1] = 2.0 * PI;
                parameters[2] = meanDist - distPtPl;
                parameters[3] = meanDist + distPtPl;
                parameters[4] = MAX_PAIRWISE_DISTANCE;     // TODO: change this?
                parameters[5] = uniformDistSteps(*mRandomGenerator);
                break;
            default:
                // care about both
                // sample parameters //////////////////////////
                parameters[0] = meanAngle - diffAngle;
                parameters[1] = meanAngle + diffAngle;
                parameters[2] = meanDist - distPtPl;
                parameters[3] = meanDist + distPtPl;
                parameters[4] = MAX_PAIRWISE_DISTANCE;     // TODO: change this?
                parameters[5] = uniformDistSteps(*mRandomGenerator);
                break;
        }


        // sample parameters //////////////////////////
/*    parameters[0] = meanAngle - diffAngle;
    parameters[1] = meanAngle + diffAngle;
    parameters[2] = meanDist - distPtPl;
    parameters[3] = meanDist + distPtPl;
    parameters[4] = MAX_PAIRWISE_DISTANCE;     // TODO: change this?
    parameters[5] = uniformDistSteps(*mRandomGenerator);*/
    }

    bool EntangledForestCommonAncestorFeature::evaluateTraining(EntangledForestData *data, int imageIdx, int clusterIdx,
                                                                const std::vector<double> &parameters, bool &result)
    {
        return evaluate(data, imageIdx, clusterIdx, parameters[0], parameters[1], parameters[2], parameters[3],
                        parameters[4], parameters[5], result);
    }

    bool EntangledForestCommonAncestorFeature::evaluate(EntangledForestData *data, int imageIdx, int clusterIdx, double minangle, double maxangle,
                                                        double minptpl, double maxptpl, double maxeuclid, unsigned int maxSteps,
                                                        bool &result)
    {
        // first filter clusters by geometric constraints
        std::vector<int> remaining;

        data->FilterClustersByGeometry(imageIdx, clusterIdx, mHorizontal, minangle, maxangle, minptpl, maxptpl, maxeuclid,
                                       remaining);

        result = false;

        int currentNode = data->GetClusterNodeIdx(imageIdx, clusterIdx, mTree->GetIndex());

        // then check if at least 1 remaining cluster fulfills the topN criterion
        for (unsigned int i = 0; i < remaining.size(); ++i)
        {
            int nodeidx = data->GetClusterNodeIdx(imageIdx, remaining[i], mTree->GetIndex());
            if (nodeidx != 0)
            {
                if (mTree->DoNodesShareAncestor(currentNode, nodeidx, maxSteps))
                {
                    result = true;
                    return true;
                }
            }
        }

        return true;
    }

    EntangledForestCommonAncestorFeature::EntangledForestCommonAncestorFeature() : EntangledForestSplitFeature()
    {

    }

    EntangledForestCommonAncestorFeature::EntangledForestCommonAncestorFeature(const EntangledForestCommonAncestorFeature &f) : EntangledForestSplitFeature(f),
                                                                                                                                mMinAngle(f.mMinAngle), mMaxAngle(f.mMaxAngle),
                                                                                                                                mMinPtPlDist(f.mMinPtPlDist), mMaxPtPlDist(f.mMaxPtPlDist),
                                                                                                                                mMaxEDist(f.mMaxEDist), mHorizontal(f.mHorizontal),
                                                                                                                                mMaxSteps(f.mMaxSteps)
    {

    }

    bool EntangledForestCommonAncestorFeature::evaluateInference(EntangledForestData *data, int imageIdx, int clusterIdx)
    {
        bool result;
        evaluate(data, imageIdx, clusterIdx, mMinAngle, mMaxAngle, mMinPtPlDist, mMaxPtPlDist, mMaxEDist, mMaxSteps, result);
        return result;
    }

/////

    EntangledForestNodeDescendantFeature::EntangledForestNodeDescendantFeature(string name, EntangledForestTree *tree, std::mt19937 *randomGenerator,
                                                                               bool horizontal) : EntangledForestSplitFeature(name, tree, randomGenerator, 5),
                                                                                                  mHorizontal(horizontal)
    {
        mDynamicThreshold = false;
        mMaxParameterSamplings = std::numeric_limits<int>::max();
        mUniformDistPtPl = std::uniform_real_distribution<double>(-MAX_PAIRWISE_DISTANCE, MAX_PAIRWISE_DISTANCE);
        mUniformDistAngle = std::uniform_real_distribution<double>(-PI, PI);

        mNormalDistAngle = std::normal_distribution<double>(0.0, SIGMA_PAIRWISE_ANGLE);
        mNormalDistPtPl = std::normal_distribution<double>(0.0, SIGMA_PAIRWISE_DISTANCE);

        mDontcare = std::uniform_int_distribution<int>(1, 8);

        mMinAngle = 0.0;
        mMaxAngle = 0.0;
        mMinPtPlDist = 0.0;
        mMaxPtPlDist = 0.0;
        mMaxEDist = 0.0;

        mAncestorNode = 0;
    }

    EntangledForestSplitFeature *EntangledForestNodeDescendantFeature::Clone()
    {
        return new EntangledForestNodeDescendantFeature(*this);
    }

    EntangledForestSplitFeature *EntangledForestNodeDescendantFeature::Clone(EntangledForestTree *newTree)
    {
        EntangledForestNodeDescendantFeature *f = new EntangledForestNodeDescendantFeature(*this);
        f->mTree = newTree;
        return f;
    }


    std::string EntangledForestNodeDescendantFeature::ToString()
    {
        if (mHorizontal)
        {
            return "Entangled H feature: NodeDescendant [" + std::to_string(RAD2DEG(mMinAngle)) + "," +
                   std::to_string(RAD2DEG(mMaxAngle)) + "], ptPl [" + std::to_string(mMinPtPlDist) + "," +
                   std::to_string(mMaxPtPlDist) +
                   "], descending from " + std::to_string(mAncestorNode);
        }
        else
        {
            return "Entangled V feature: NodeDescendant [" + std::to_string(RAD2DEG(mMinAngle)) + "," +
                   std::to_string(RAD2DEG(mMaxAngle)) + "], ptPl [" + std::to_string(mMinPtPlDist) + "," +
                   std::to_string(mMaxPtPlDist) +
                   "], descending from " + std::to_string(mAncestorNode);
        }
    }

    void EntangledForestNodeDescendantFeature::SetParameters(const std::vector<double> &parameters)
    {
        mMinAngle = parameters[0];
        mMaxAngle = parameters[1];
        mMinPtPlDist = parameters[2];
        mMaxPtPlDist = parameters[3];
        mMaxEDist = parameters[4];
        mAncestorNode = (unsigned int) parameters[5];
    }

    void EntangledForestNodeDescendantFeature::SampleRandomParameters(std::vector<double> &parameters)
    {
        // actually useless for this feature
        parameters.resize(6);

        // sample angle and distance thresholds
        double meanAngle = mUniformDistAngle(*mRandomGenerator);
        double meanDist = mUniformDistPtPl(*mRandomGenerator);
        double diffAngle = std::abs(mNormalDistAngle(*mRandomGenerator));
        double distPtPl = std::abs(mNormalDistPtPl(*mRandomGenerator));

        // uniform distribution for maxSteps parameters
        std::uniform_int_distribution<int> uniformDistNodeID(1, max(1, min(256, mTree->GetLastNodeIDOfPrevLevel())));

        // sample parameters //////////////////////////
        int nodeIdx = 0;
        do
        {
            nodeIdx = uniformDistNodeID(*mRandomGenerator);
        } while (!(mTree->GetNode(nodeIdx)->IsSplitNode()));


        int bla = mDontcare(*mRandomGenerator);

        switch (bla)
        {
            case 1:
                // care about angle, not dist
                parameters[0] = meanAngle - diffAngle;
                parameters[1] = meanAngle + diffAngle;
                parameters[2] = -100.0;
                parameters[3] = 100.0;
                parameters[4] = MAX_PAIRWISE_DISTANCE;     // TODO: change this?
                parameters[5] = nodeIdx;
                break;
            case 2: // care about dist, not angle
                parameters[0] = -2.0 * PI;
                parameters[1] = 2.0 * PI;
                parameters[2] = meanDist - distPtPl;
                parameters[3] = meanDist + distPtPl;
                parameters[4] = MAX_PAIRWISE_DISTANCE;     // TODO: change this?
                parameters[5] = nodeIdx;
                break;
            default:
                // care about both
                // sample parameters //////////////////////////
                parameters[0] = meanAngle - diffAngle;
                parameters[1] = meanAngle + diffAngle;
                parameters[2] = meanDist - distPtPl;
                parameters[3] = meanDist + distPtPl;
                parameters[4] = MAX_PAIRWISE_DISTANCE;     // TODO: change this?
                parameters[5] = nodeIdx;
                break;
        }


/*    parameters[0] = meanAngle - diffAngle;
    parameters[1] = meanAngle + diffAngle;
    parameters[2] = meanDist - distPtPl;
    parameters[3] = meanDist + distPtPl;
    parameters[4] = MAX_PAIRWISE_DISTANCE;     // TODO: change this?
    parameters[5] = nodeIdx;*/
    }

    bool EntangledForestNodeDescendantFeature::evaluateTraining(EntangledForestData *data, int imageIdx, int clusterIdx,
                                                                const std::vector<double> &parameters, bool &result)
    {
        return evaluate(data, imageIdx, clusterIdx, parameters[0], parameters[1], parameters[2], parameters[3],
                        parameters[4], parameters[5], result);
    }

    bool EntangledForestNodeDescendantFeature::evaluate(EntangledForestData *data, int imageIdx, int clusterIdx, double minangle, double maxangle,
                                                        double minptpl, double maxptpl, double maxeuclid,
                                                        unsigned int ancestorNode, bool &result)
    {
        // first filter clusters by geometric constraints
        std::vector<int> remaining;

        data->FilterClustersByGeometry(imageIdx, clusterIdx, mHorizontal, minangle, maxangle, minptpl, maxptpl, maxeuclid,
                                       remaining);

        result = false;

        EntangledForestNode *ancestor = mTree->GetNode(ancestorNode);

        // then check if at least 1 remaining cluster fulfills the topN criterion
        for (unsigned int i = 0; i < remaining.size(); ++i)
        {
            int nodeidx = data->GetClusterNodeIdx(imageIdx, remaining[i], mTree->GetIndex());
            if (nodeidx != 0)
            {
                EntangledForestNode *n = mTree->GetNode(nodeidx);
                if (n->IsDescendantOf(ancestor))
                {
                    result = true;
                    return true;
                }
            }
        }

        return true;
    }

    EntangledForestNodeDescendantFeature::EntangledForestNodeDescendantFeature() : EntangledForestSplitFeature()
    {

    }

    EntangledForestNodeDescendantFeature::EntangledForestNodeDescendantFeature(const EntangledForestNodeDescendantFeature &f) : EntangledForestSplitFeature(f),
                                                                                                                                mMinAngle(f.mMinAngle), mMaxAngle(f.mMaxAngle),
                                                                                                                                mMinPtPlDist(f.mMinPtPlDist), mMaxPtPlDist(f.mMaxPtPlDist),
                                                                                                                                mMaxEDist(f.mMaxEDist), mHorizontal(f.mHorizontal),
                                                                                                                                mAncestorNode(f.mAncestorNode)
    {

    }

    bool EntangledForestNodeDescendantFeature::evaluateInference(EntangledForestData *data, int imageIdx, int clusterIdx)
    {
        bool result;
        evaluate(data, imageIdx, clusterIdx, mMinAngle, mMaxAngle, mMinPtPlDist, mMaxPtPlDist, mMaxEDist, mAncestorNode, result);
        return result;
    }

}

BOOST_CLASS_EXPORT_GUID(v4r::EntangledForestSplitFeature, "featurebase")
BOOST_CLASS_EXPORT_GUID(v4r::EntangledForestUnaryFeature, "unary")
BOOST_CLASS_EXPORT_GUID(v4r::EntangledForestClusterExistsFeature, "pairwiseexists")
BOOST_CLASS_EXPORT_GUID(v4r::EntangledForestTopNFeature, "pairwisetopn")
BOOST_CLASS_EXPORT_GUID(v4r::EntangledForestCommonAncestorFeature, "pairwisecommonancestor")
BOOST_CLASS_EXPORT_GUID(v4r::EntangledForestNodeDescendantFeature, "pairwisenodedescendant")
BOOST_CLASS_EXPORT_GUID(v4r::EntangledForestInverseTopNFeature, "pairwiseinversetopn")

