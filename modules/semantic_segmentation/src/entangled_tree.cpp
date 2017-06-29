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


#include <ctime>
#include <boost/date_time/posix_time/posix_time_types.hpp>

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_thread_num() 0
#endif

#include <v4r/semantic_segmentation/entangled_tree.h>

#define ROUND_DOWN(x, s) ((x) & ~((s)-1))

using namespace boost::posix_time;

using namespace std;
using namespace cv;

namespace v4r
{

EntangledForestTree::EntangledForestTree(int idx) : mTreeIdx(idx), mUniformDistThresh(0.0f, 1.0f)
{
    mRandomGenerator = nullptr;
    mCurrentDepth = 0;
}

EntangledForestTree::EntangledForestTree(std::mt19937* randomGenerator, int idx) : mTreeIdx(idx), mUniformDistThresh(0.0f, 1.0f)
{   
    this->mRandomGenerator = randomGenerator;
    mCurrentDepth = 0;
}

// compare function for Partition function
bool EntangledForestTree::SmallerThan(float x, float t) { return x<t; }

EntangledForestNode* EntangledForestTree::GetNode(int nodeIdx)
{
    return mNodes[nodeIdx];
}

int EntangledForestTree::GetNrOfNodes()
{
    return mNodes.size();
}

EntangledForestNode* EntangledForestTree::GetRootNode()
{
    return mNodes[0];
}

void EntangledForestTree::SetTreeIndex(int index)
{
    mTreeIdx = index;
}

void EntangledForestTree::GetFeatureMinMax(EntangledForestSplitFeature *f, EntangledForestData *data, const ClusterIdxItr start, const ClusterIdxItr end, const std::vector<double>& parameters, double& minValue, double& maxValue)
{
    long long int npoints = distance(start, end);

    double* mins;
    double* maxs;
    int nthreads = 0;

#ifdef _OPENMP
#pragma omp parallel
    {
        nthreads = omp_get_num_threads();
        const int ithread = omp_get_thread_num();
#else
    {
        nthreads = 1;
        const int ithread = 0;
#endif


#ifdef _OPENMP
#pragma omp single
#endif
        {
            mins = (double*)_mm_malloc(sizeof(double)*nthreads, 4096);
            maxs = (double*)_mm_malloc(sizeof(double)*nthreads, 4096);
        }

#ifdef _OPENMP
#pragma omp for
#endif
        for(int i=0; i<nthreads; ++i)
        {
            mins[i] = std::numeric_limits<double>::max();  //__DBL_MAX__;
            maxs[i] = std::numeric_limits<double>::lowest(); //-__DBL_MAX__;
        }

        double value(0.0f);

#ifdef _OPENMP
#pragma  omp for schedule(dynamic)
#endif
        for(int i=0; i < npoints; ++i)
        {
            ClusterIdx datapoint = *(start+i);
            if(f->computeTraining(data, datapoint[0], datapoint[1], parameters, value))
            {
                if(value < mins[ithread])
                {
                    mins[ithread] = value;
                }
                else if(value > maxs[ithread])
                {
                    maxs[ithread] = value;
                }
            }
        }
    }

    minValue = (double) *(std::min_element(mins, mins+nthreads));
    maxValue = (double) *(std::max_element(maxs, maxs+nthreads));

    delete[] mins;
    delete[] maxs;
}

double EntangledForestTree::GetBestFeatureConfiguration(EntangledForestSplitFeature* f, EntangledForestData* data, ClusterIdxItr start, ClusterIdxItr end, int nParameterSamples, int nThresholdSamples, int minPointsForSplit, double currentEntropy)
{
    // best achieved entropy gain
    double bestGain(std::numeric_limits<double>::lowest());

    // randomly sampled parameters
    vector<double> parameters;

    int nlabels = data->GetNrOfLabels();

    int sampleNParams = min(nParameterSamples, f->GetMaxParameterSamples());
    int sampleNThresholds = f->HasDynamicThreshold() ? nThresholdSamples : 1;

    // randomly sample nParameterSamples different parameter sets and evaluate them
    for (int p = 0; p < sampleNParams; ++p)
    {
        // sample parameters //////////////////////////
        f->SampleRandomParameters(parameters);

        // sample sampleNThresholds different thresholds and scale to minmax range
        std::vector<double> thresholds(sampleNThresholds);

        if (f->HasDynamicThreshold())
        {
            // Only if several threshold have to be sampled

            // get value range for parameter setting to find good threshold /////
            double minv;
            double maxv;
            GetFeatureMinMax(f, data, start, end, parameters, minv, maxv);

            // TODO: smarter sampling according to different distribution?
            for (int j = 0; j < sampleNThresholds; ++j)
            {
                thresholds[j] = mUniformDistThresh(*mRandomGenerator) * (maxv - minv) + minv;
            }

            // sort them to be able to accumulate statistics afterwards
            std::sort(thresholds.begin(), thresholds.end());
        }

        // bin datapoints according to thresholds
        vector<vector<int> > bins(sampleNThresholds + 1, vector<int>(nlabels, 0));

        long long int npoints = distance(start, end);

        int *hista;
        int *validpointsa;

        int nthreads = 0;
        int nbins = (sampleNThresholds + 1) * nlabels;
        int lda = ROUND_DOWN(nbins + 1023, 1024);  //1024 ints = 4096 bytes -> round to a multiple of page size

#ifdef _OPENMP
#pragma  omp parallel
        {
            nthreads = omp_get_num_threads();
            const int ithread = omp_get_thread_num();
#else
            {
                nthreads = 1;
                const int ithread = 0;
#endif
            const int baseidx = ithread * lda;

#ifdef _OPENMP
#pragma  omp single
#endif
            {
                hista = (int *) _mm_malloc(lda * sizeof(int) * nthreads, 4096);
                validpointsa = (int *) _mm_malloc(sizeof(int) * nthreads, 4096);
            }

            for (int i = 0; i < nbins; ++i)
                hista[baseidx + i] = 0;

            validpointsa[ithread] = 0;

#ifdef _OPENMP
#pragma  omp for schedule(dynamic)
#endif
            for (int i = 0; i < npoints; ++i)
            {
                ClusterIdx datapoint = *(start + i);
                int labelIdx = data->GetLabelIdx(datapoint);

                if (f->HasDynamicThreshold())
                {
                    // only if several thresholds have been sampled

                    // compute feature value
                    double value(0.0f);

                    if (f->computeTraining(data, datapoint[0], datapoint[1], parameters, value))
                        validpointsa[ithread]++;

                    // first bin
                    if (value <= thresholds[0])
                    {
                        hista[baseidx + labelIdx]++;
                    }

                    // sweep through data beginning with lowest threshold
                    for (int j = 1; j < sampleNThresholds; ++j)
                    {
                        if (value > thresholds[j - 1] && value <= thresholds[j])
                        {
                            hista[baseidx + j * nlabels + labelIdx]++;
                        }
                    }

                    // last bin
                    if (value > thresholds[sampleNThresholds - 1])
                    {
                        hista[baseidx + sampleNThresholds * nlabels + labelIdx]++;
                    }
                }
                else
                {
                    // no thresholds, evaluate split with sampled parameters

                    // split left/right
                    bool result = false;

                    if (f->evaluateTraining(data, datapoint[0], datapoint[1], parameters, result))
                        validpointsa[ithread]++;

                    if (result)
                    {
                        hista[baseidx + nlabels + labelIdx]++;
                    }
                    else
                    {
                        hista[baseidx + labelIdx]++;
                    }
                }

            }
        }

        for (int i = 0; i <= sampleNThresholds; ++i)
        {
            for (int j = 0; j < nlabels; ++j)
            {
                for (int t = 0; t < nthreads; ++t)
                {
                    bins[i][j] += hista[t * lda + i * nlabels + j];
                }
            }

        }

        int validpoints = 0;
        for (int t = 0; t < nthreads; ++t)
            validpoints += validpointsa[t];

        delete[] validpointsa;
        delete[] hista;

        vector<unsigned int> distributionleft(nlabels, 0);
        vector<unsigned int> distributionright(nlabels, 0);

        // total label distribution
        for (int j = 0; j <= sampleNThresholds; ++j)
        {
            for (int i = 0; i < nlabels; ++i)
            {
                distributionright[i] += bins[j][i];
            }
        }

        vector<double> entropyGains(sampleNThresholds);
        int nleft, nright;
        double dleft, dright;

        vector<double> &classWeights = data->GetClassWeights();

        // accumulate statistics and calculate information gain
        for (int i = 0; i < sampleNThresholds; ++i)
        {
            for (int j = 0; j < nlabels; ++j)
            {
                distributionleft[j] += bins[i][j];
                distributionright[j] -= bins[i][j];
            }

            nleft = 0;
            nright = 0;
            dright = 0.0;
            dleft = 0.0;

            for (int j = 0; j < nlabels; ++j)
            {
                nleft += distributionleft[j];
                nright += distributionright[j];
                dleft += ((double) distributionleft[j]) / classWeights[j];
                dright += ((double) distributionright[j]) / classWeights[j];
            }

            // TODO: Heuristic to avoid splits with only a few points
            //if(nleft < minPointsForSplit || nright < minPointsForSplit || validpoints < minPointsForSplit)
            if (dleft < (double) minPointsForSplit || dright < (double) minPointsForSplit ||
                validpoints < minPointsForSplit)
            {
                // split would cause too few points in one child node
                entropyGains[i] = std::numeric_limits<double>::lowest();
            }
            else
            {
                double entropyleft(0.0f);
                double entropyright(0.0f);
                double weightedEntropy(0.0f);
                double normleft(0.0f);
                double normright(0.0f);
#ifdef USE_RF_ENERGY
                normleft = data->CalculateEnergy(distributionleft, entropyleft);
                normright = data->CalculateEnergy(distributionright, entropyright);
                weightedEntropy = (normleft * entropyleft + normright * entropyright) / (normleft + normright);
#else
                normleft = data->CalculateEntropy(distributionleft, entropyleft);
                normright = data->CalculateEntropy(distributionright, entropyright);
                //        weightedEntropy = ((double)nleft * entropyleft + (double)nright * entropyright) / (double)nsum;
                weightedEntropy = (normleft * entropyleft + normright * entropyright) / (normleft + normright);
#endif

                double gain = currentEntropy - weightedEntropy;
                entropyGains[i] = gain;
            }
        }

        // get max entropy gain
        auto maxEntropy = max_element(entropyGains.begin(), entropyGains.end());

        if (*maxEntropy > bestGain)
        {
            bestGain = *maxEntropy;
            f->SetParameters(parameters);

            if (f->HasDynamicThreshold())
            {
                int maxIdx = distance(entropyGains.begin(), maxEntropy);
                f->SetThreshold(thresholds[maxIdx]);
            }
        }
    }

    return bestGain;
}

    bool EntangledForestTree::DoNodesShareAncestor(int nodeIdx1, int nodeIdx2, int maxSteps)
    {
        EntangledForestNode *n1 = mNodes[nodeIdx1];
        EntangledForestNode *n2 = mNodes[nodeIdx2];

        int depth1 = n1->GetDepth();
        int depth2 = n2->GetDepth();

        if (depth1 != depth2)
        {
            // bring "lower node" up to higher node in tree
            EntangledForestNode *maxNode = depth1 > depth2 ? n1 : n2;

            for (int i = 0; i < abs(depth1 - depth2); ++i)
            {
                maxNode = maxNode->GetParent();
            }

            if (depth1 > depth2)
            {
                n1 = maxNode;
            }
            else
            {
                n2 = maxNode;
            }
        }

        n1 = n1->GetParent();
        n2 = n2->GetParent();

        // now nodes start at same depth level
        for (int steps = 1; steps <= maxSteps && n1 && n2; ++steps)
        {
            if (n1 == n2)
            {
                return true;
            }

            n1 = n1->GetParent();
            n2 = n2->GetParent();
        }

        return false;
    }

    void EntangledForestTree::Train(EntangledForestData *trainingData, int maxDepth, int sampledSplitFunctionParameters,
                     int sampledSplitFunctionThresholds, double minInformationGain, int minPointsForSplit)
    {
        vector<EntangledForestNode *> frontier;

        vector<EntangledForestSplitFeature *> features;

        // unary features
        for (int i = 0; i < 18; ++i)
        {
            features.push_back(new EntangledForestUnaryFeature("Unary feature " + std::to_string(i), this, mRandomGenerator, i));
        }

        int nlabels = trainingData->GetNrOfLabels();
        vector<double> &classWeights = trainingData->GetClassWeights();

        // pairwise features
        features.push_back(
                new EntangledForestClusterExistsFeature("Pairwise feature vertical angle", this, mRandomGenerator, false));
        features.push_back(
                new EntangledForestClusterExistsFeature("Pairwise feature horizontal angle", this, mRandomGenerator, true));
        features.push_back(
                new EntangledForestTopNFeature("Pairwise top n vertical angle", this, mRandomGenerator, false, nlabels));
        features.push_back(
                new EntangledForestTopNFeature("Pairwise top n horizontal angle", this, mRandomGenerator, true, nlabels));
        features.push_back(
                new EntangledForestCommonAncestorFeature("Pairwise common ancestor vertical angle", this, mRandomGenerator,
                                                  false));
        features.push_back(
                new EntangledForestCommonAncestorFeature("Pairwise common ancestor horizontal angle", this, mRandomGenerator,
                                                  true));
        features.push_back(
                new EntangledForestNodeDescendantFeature("Pairwise node descendant vertical angle", this, mRandomGenerator,
                                                  false));
        features.push_back(
                new EntangledForestNodeDescendantFeature("Pairwise node descendant horizontal angle", this, mRandomGenerator,
                                                  true));
        features.push_back(
                new EntangledForestInverseTopNFeature("Pairwise inverse top n feature", this, mRandomGenerator, nlabels));

        std::uniform_int_distribution<int> randomFeature(0, features.size() - 1);

        double bestGain = std::numeric_limits<double>::lowest();
        int bestFeatureIdx = -1;

        if (maxDepth < 0)
        {
            maxDepth = __INT32_MAX__;
        }

        LOG_PLAIN("Tree " << mTreeIdx + 1 << " ### Create root node.");

        // create root node and add it to node list and frontier
        EntangledForestNode *rootNode = new EntangledForestNode(trainingData);
        mNodes.push_back(rootNode);

        for (int d = 0; d < maxDepth; ++d)
        {
            // collect frontier nodes
            LOG_INFO("Tree " << mTreeIdx + 1 << " Level " << d << ": Collect frontier...");
            frontier.clear();

            mCurrentDepth = d;

            for (unsigned int i = 0; i < mNodes.size(); ++i)
            {
                if (mNodes[i]->IsOnFrontier())
                {
                    frontier.push_back(mNodes[i]);

                    // update nodeidxarray for each datapoint of node
                    unsigned int npoints = mNodes[i]->GetNrOfPoints();
                    ClusterIdxItr start = mNodes[i]->mTrainingDataIterators.first;
#if _OPENMP
#pragma omp parallel for
#endif
                    for (unsigned int n = 0; n < npoints; ++n)
                    {
                        ClusterIdx datapoint = *(start + n);
                        trainingData->SetClusterNodeIdx(datapoint, mTreeIdx, i);
                    }
                }
            }

            if (frontier.size() == 0)
            {
                // no more nodes to expand, stop training
                LOG_INFO("Tree " << mTreeIdx + 1 << " Level " << d << ": No more nodes to expand. Stop training.");
                break;
            }

            // expand all nodes at frontier
            for (auto n : frontier)
            {
//            int npoints = n->GetNrOfPoints();
                double npoints = n->GetWeightedNrOfPoints(classWeights);

                LOG_PLAIN("Tree " << mTreeIdx + 1 << " Level " << d << ": Try to split node with " << npoints
                                  << " points.");

                // check if enough points for splitting
                if (npoints < minPointsForSplit)
                {
                    LOG_PLAIN("Level " << d << ": Not enough points for split (" << npoints << "), set as leaf node.");
                    n->SetAsLeafNode();
                    continue;
                }

                double entropyBeforeSplit;

#ifdef USE_RF_ENERGY
                trainingData->CalculateEnergy(n->mTrainingDataIterators.first, n->mTrainingDataIterators.second, entropyBeforeSplit);
#else
                trainingData->CalculateEntropy(n->mTrainingDataIterators.first, n->mTrainingDataIterators.second,
                                               entropyBeforeSplit);
#endif

                LOG_INFO("Current entropy: " << entropyBeforeSplit);

                // go through features

                bestGain = std::numeric_limits<double>::lowest();
                bestFeatureIdx = -1;

                std::vector<unsigned int> depths;
                int mindepth, maxdepth;
                int availfeat = 0;
                for (unsigned int f = 0; f < features.size(); ++f)
                {
                    depths = features[f]->GetActiveDepthLevels();
                    mindepth = features[f]->GetMinDepth();
                    maxdepth = features[f]->GetMaxDepth();

                    if ((depths.size() == 0 && d >= mindepth && d <= maxdepth) ||
                        std::find(depths.begin(), depths.end(), d) != depths.end())
                        availfeat++;
                }

                unsigned int sampleNFeatures = (unsigned int) ceil(0.5 * availfeat);//features.size());



                //for(int f=0; f<features.size(); ++f)
                for (unsigned int f = 0; f < sampleNFeatures; ++f)
                {
                    int featureidx = 0;

                    do
                    {
                        // randomly sample feature
                        featureidx = randomFeature(*mRandomGenerator);
                        // check if sampled feature is available for current depth
                        // e.g. entangled features don't make sense at level 0 or 1

                        depths = features[featureidx]->GetActiveDepthLevels();
                        mindepth = features[featureidx]->GetMinDepth();
                        maxdepth = features[featureidx]->GetMaxDepth();
                    } while ((depths.size() != 0 && std::find(depths.begin(), depths.end(), d) == depths.end()) ||
                             mindepth > d || maxdepth <
                                             d);   //features[featureidx]->GetMinDepth() > d || d > features[featureidx]->GetMaxDepth());

                    // generate random parameter sets
                    //double currentGain = GetBestFeatureConfiguration(features[f], trainingData, n->mTrainingDataIterators.first, n->mTrainingDataIterators.second, sampledSplitFunctionParameters, sampledSplitFunctionThresholds, minPointsForSplit, entropyBeforeSplit);
                    double currentGain = GetBestFeatureConfiguration(features[featureidx], trainingData,
                                                                     n->mTrainingDataIterators.first,
                                                                     n->mTrainingDataIterators.second,
                                                                     sampledSplitFunctionParameters,
                                                                     sampledSplitFunctionThresholds, minPointsForSplit,
                                                                     entropyBeforeSplit);

                    if (currentGain > bestGain)
                    {
                        // new best entropy gain with this feature
                        bestGain = currentGain;
                        //bestFeatureIdx = f;
                        bestFeatureIdx = featureidx;
                    }

                }

                if (bestFeatureIdx >= 0 && bestGain >= minInformationGain)
                {
                    LOG_PLAIN("   Best gain found: " << bestGain);

                    // good split found, partition data and create child nodes
                    EntangledForestNode *leftChild;
                    EntangledForestNode *rightChild;

                    n->Split(trainingData, features[bestFeatureIdx], &leftChild, &rightChild);
                    n->SetLeftChildIdx(mNodes.size());
                    mNodes.push_back(leftChild);
                    n->SetRightChildIdx(mNodes.size());
                    mNodes.push_back(rightChild);

                    LOG_PLAIN("   Split " << leftChild->GetNrOfPoints() << "/" << rightChild->GetNrOfPoints() << ". "
                                          << features[bestFeatureIdx]->ToString());
                }
                else
                {
                    LOG_PLAIN("   No sufficient gain found (" << bestGain << "), set as leaf node.");
                    n->SetAsLeafNode();
                }
            }
        }

        LOG_PLAIN("Tree " << mTreeIdx + 1 << " Expanding done. Set frontier nodes to leaf nodes.");

        // set nodes on last level to leaf nodes
        for (auto n : frontier)
        {
            n->SetAsLeafNode();
        }

        // delete all features
        for (auto f : features)
        {
            delete f;
        }

        LOG_INFO("Training tree " << mTreeIdx + 1 << " DONE.");
    }

    void EntangledForestTree::UpdateLeafs(EntangledForestData *data, int updateDepth, double updateWeight)
    {
        if (updateDepth < 0)
        {
            updateDepth = std::numeric_limits<int>::max();
        }

        // get number of points
        ClusterIdxItr begin, end;
        data->GetBeginAndEndIterator(begin, end);
        int nclusters = std::distance(begin, end);

        int nlabels = data->GetNrOfLabels();

        bool done = false;
        vector<bool> doneArray(nclusters, false);

#ifdef _OPENMP
#pragma  omp parallel for
#endif
        for (int i = 0; i < nclusters; ++i)
        {
            ClusterIdx p = *(begin + i);

            doneArray[i] = false;
            data->SetClusterNodeIdx(p, mTreeIdx, 0);
        }

        std::vector<std::vector<int> > tmpNodeIdxArray;

        // expand nodes in parallel level by level
        for (int d = 0; d < updateDepth && !done; ++d)
        {
            data->GetClusterNodeIndicesPerTree(mTreeIdx, tmpNodeIdxArray);

#ifdef _OPENMP
#pragma  omp parallel for
#endif
            for (int i = 0; i < nclusters; ++i)
            {
                ClusterIdx p = *(begin + i);
                int imageIdx = p[0];
                int clusterIdx = p[1];

                if (!doneArray[i])
                {
                    int nodeIdx = 0;

                    nodeIdx = data->GetClusterNodeIdx(imageIdx, clusterIdx, mTreeIdx);
                    EntangledForestNode *curNode = mNodes[nodeIdx];

                    if (curNode->IsSplitNode())
                    {
                        tmpNodeIdxArray[imageIdx][clusterIdx] = curNode->evaluate(data, imageIdx, clusterIdx);
                    }
                    else
                    {
                        doneArray[i] = true;
                    }
                }
            }

            data->UpdateClusterNodeIndicesPerTree(mTreeIdx, tmpNodeIdxArray);
            done = std::all_of(doneArray.begin(), doneArray.end(), [](bool i)
            { return i; });
        }

        std::map<int, std::vector<double> > oldLeafDistributions;

        // delete all nodes which are too deep, set new leaf nodes
        int firsttodelete = -1;

        for (unsigned int n = 0; n < mNodes.size(); ++n)
        {
            EntangledForestNode *node = mNodes[n];
            int depth = node->GetDepth();
            if (depth == updateDepth)
            {
                node->SetAsLeafNode();
                oldLeafDistributions[n] = node->GetLabelDistribution();
            }
            else if (depth > updateDepth)
            {
                if (firsttodelete < 0)
                    firsttodelete = n;
                delete node;
            }
            else
            {
                if (!node->IsSplitNode())
                {
                    oldLeafDistributions[n] = node->GetLabelDistribution();
                }
            }
        }

        if (firsttodelete >= 0)
            mNodes.erase(mNodes.begin() + firsttodelete, mNodes.end());

        std::map<int, std::vector<double> > newLeafDistributions = oldLeafDistributions;
        for (std::map<int, std::vector<double> >::iterator it = newLeafDistributions.begin();
             it != newLeafDistributions.end(); ++it)
            it->second.assign(nlabels, 0.0);

        // update label distributions of leaf nodes
        for (unsigned int i = 0; i < tmpNodeIdxArray.size(); ++i)
        {
            for (unsigned int c = 0; c < tmpNodeIdxArray[i].size(); ++c)
            {
                // add 1 point to absDist of Node
                int nodeIdx = tmpNodeIdxArray[i][c];
                if (nodeIdx > 0)
                    newLeafDistributions[nodeIdx][data->GetLabelIdx(i, c)]++;
            }
        }

        // weight newLeafDist with classWeights and merge old with new distribution
        vector<double> &classWeights = data->GetClassWeights();

        for (std::map<int, std::vector<double> >::iterator it = newLeafDistributions.begin();
             it != newLeafDistributions.end(); ++it)
        {
            double sum = 0.0f;

            // weight
            for (int i = 0; i < nlabels; ++i)
            {
                it->second[i] /= classWeights[i];
                sum += it->second[i];
            }

            int nodeidx = it->first;

            // normalize and merge
            for (int i = 0; i < nlabels; ++i)
            {
                // debug: 2 steps
                it->second[i] /= sum;
                //it->second[i] = (1.0-updateWeight)*oldLeafDistributions[nodeidx][i] + updateWeight*(it->second[i] / sum);
            }

            for (int i = 0; i < nlabels; ++i)
                it->second[i] = (1.0 - updateWeight) * oldLeafDistributions[nodeidx][i] + updateWeight * it->second[i];

            mNodes[nodeidx]->UpdateLabelDistribution(it->second);
        }
    }

    void EntangledForestTree::saveMatlab(string filename)
    {
        ofstream ofs(filename);

        for (unsigned int i = 0; i < mNodes.size(); ++i)
        {
            EntangledForestNode *n = mNodes[i];

            ofs << setw(13) << n->GetDepth() << setw(13) << n->GetLeftChildIdx() << setw(13) << n->GetRightChildIdx();

            std::vector<double>& dist = n->GetLabelDistribution();
            for (unsigned int j = 0; j < dist.size(); ++j)
            {
                ofs << setw(13) << dist[j];
            }

            std::vector<unsigned int>& absdist = n->GetAbsLabelDistribution();
            for (unsigned int j = 0; j < dist.size(); ++j)
            {
                ofs << setw(13) << absdist[j];
            }
            ofs << std::endl;
        }
        ofs.close();
    }

    void EntangledForestTree::Clone(EntangledForestTree *t)
    {
        t->mTreeIdx = this->mTreeIdx;

        for (unsigned int n = 0; n < mNodes.size(); ++n)
        {
            EntangledForestNode *node = new EntangledForestNode();
            mNodes[n]->Clone(node, t);
            t->mNodes.push_back(node);
        }

        // update parents
        for (unsigned int n = 0; n < mNodes.size(); ++n)
        {
            if (mNodes[n]->GetParent())
            {
                std::vector<EntangledForestNode *>::iterator paritr = std::find(mNodes.begin(), mNodes.end(), mNodes[n]->GetParent());
                int idx = std::distance(mNodes.begin(), paritr);
                t->mNodes[n]->SetParent(t->mNodes[idx]);
            }
        }
    }

    void EntangledForestTree::SetRandomGenerator(std::mt19937 *randomGenerator)
    {
        for (auto n : mNodes)
        {
            n->SetRandomGenerator(randomGenerator);
        }
    }

    void EntangledForestTree::Classify(EntangledForestData *data, std::vector<std::vector<double> > &result, int maxDepth)
    {
        if (maxDepth <= 0)
        {
            maxDepth = std::numeric_limits<int>::max();
        }

        // get number of points
        ClusterIdxItr begin, end;
        data->GetBeginAndEndIterator(begin, end);
        int nclusters = std::distance(begin, end);

        int nlabels = data->GetNrOfLabels();
        result.resize(nclusters, std::vector<double>(nlabels, 0.0));

        bool done = false;
        vector<bool> doneArray(nclusters, false);

#ifdef _OPENMP
#pragma  omp parallel for
#endif
        for (int i = 0; i < nclusters; ++i)
        {
            ClusterIdx p = *(begin + i);
            int clusterIdx = p[1];

            doneArray[clusterIdx] = false;
            data->SetClusterNodeIdx(p, mTreeIdx, 0);
        }

        std::vector<int> tmpNodeIdxArray(nclusters, 0);

        // expand nodes in parallel level by level
        for (int d = 0; d < maxDepth && !done; ++d)
        {

            data->GetClusterNodeIndices(mTreeIdx, tmpNodeIdxArray);
            done = true;

#ifdef _OPENMP
#pragma  omp parallel for reduction(&:done)
#endif
            for (int i = 0; i < nclusters; ++i)
            {
                ClusterIdx p = *(begin + i);
                int clusterIdx = p[1];

                if (!doneArray[clusterIdx])
                {
                    int nodeIdx = 0;

                    nodeIdx = data->GetClusterNodeIdx(0, clusterIdx, mTreeIdx);
                    EntangledForestNode *curNode = mNodes[nodeIdx];

                    if (curNode->IsSplitNode())
                    {
                        tmpNodeIdxArray[clusterIdx] = curNode->evaluate(data, 0, clusterIdx);
                        done = false;
                    }
                    else
                    {
                        doneArray[clusterIdx] = true;
                    }
                }
            }

            data->UpdateClusterNodeIndices(0, mTreeIdx, tmpNodeIdxArray);
        }

        // assign label distributions of leaf nodes
#ifdef _OPENMP
#pragma  omp parallel
#endif
        {
#ifdef _OPENMP
#pragma  omp for
#endif
            for (int i = 0; i < nclusters; ++i)
            {
                ClusterIdx p = *(begin + i);
                int clusterIdx = p[1];

                int nodeIdx = data->GetClusterNodeIdx(0, clusterIdx, mTreeIdx);
                mNodes[nodeIdx]->ApplyClusterLabelDistribution(result[i]);
            }
        }
    }


    int EntangledForestTree::GetLastNodeIDOfPrevLevel()
    {
        int depth = 0;

        // if nodes available
        if (mNodes.size() > 0)
        {
            // get depth of last node and decrease by 1 (== target depth)
            depth = mNodes[mNodes.size() - 1]->GetDepth() - 1;

            // now go backwards and return index of last node on target depth
            for (int i = mNodes.size() - 1; i >= 0; --i)
            {
                if (mNodes[i]->GetDepth() == depth)
                {
                    return i;
                }
            }

            return 0;
        }
        else
        {
            return -1;
        }
    }


    EntangledForestTree::~EntangledForestTree()
    {
        int n = mNodes.size();
        for (int i = 0; i < n; ++i)
        {
            delete mNodes[i];
        }
    }
}