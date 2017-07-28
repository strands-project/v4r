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

#include <iostream>
#include <fstream>
#include <algorithm>
#include <random>

#include <boost/serialization/utility.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/range/algorithm/random_shuffle.hpp>
#include <boost/filesystem.hpp>

#include <v4r/semantic_segmentation/entangled_data.h>

using namespace std;
using namespace cv;
namespace fs = boost::filesystem;

namespace v4r
{
    bool EntangledForestData::PairwiseComparator(const std::pair<double, int> &l, const std::pair<double, int> r)
    {
        return l.first < r.first;
    }

    EntangledForestData::EntangledForestData()
    {
    }


    bool EntangledForestData::LoadTrainingData(string trainingDataDir, string idxfile, string labelNameFile)
    {
        vector<string> filenames;
        string filename;

        ifstream ifs(idxfile.c_str());
        while (ifs >> filename)
        {
            // only add image listed in index file if data for it exists
            boost::filesystem::path unarypath(trainingDataDir + "/unary/" + filename);
            if (boost::filesystem::exists(unarypath))
                filenames.push_back(filename);

            for (int i = 1; i < 1000; ++i)
            {
                stringstream ss;
                ss << filename << "_" << i;
                boost::filesystem::path augmentedpath(trainingDataDir + "/unary/" + ss.str());
                if (boost::filesystem::exists(augmentedpath))
                    filenames.push_back(ss.str());
                else
                    break;
            }
        }
        ifs.close();

        LOG_PLAIN("Loading unary features...");
        if (LoadUnaryFeaturesBinary(trainingDataDir + "/unary", filenames))
        {
            string pairwiseDir = trainingDataDir + "/pairwise";

            LOG_PLAIN("Loading pairwise features (euclid)...");
            if (LoadPairwiseFeaturesBinary(pairwiseDir, filenames, mPairwiseEdist, "euclid"))
            {
                LOG_PLAIN("Loading pairwise features (hangles)...");
                if (LoadPairwiseFeaturesBinary(pairwiseDir, filenames, mPairwiseHangle, "hangles"))
                {
                    LOG_PLAIN("Loading pairwise features (vangles)...");
                    if (LoadPairwiseFeaturesBinary(pairwiseDir, filenames, mPairwiseVangle, "vangles"))
                    {
                        LOG_PLAIN("Loading pairwise features (ptpl)...");
                        if (LoadPairwiseFeaturesBinary(pairwiseDir, filenames, mPairwisePtPldist, "ptpl"))
                        {
                            LOG_PLAIN("Loading pairwise features (iptpl)...");
                            if (LoadPairwiseFeaturesBinary(pairwiseDir, filenames, mPairwiseIPtPldist, "iptpl"))
                            {
                                LOG_PLAIN("Loading labels...");
                                if (LoadLabelsBinary(trainingDataDir + "/labels", filenames))
                                {
                                    LOG_PLAIN("Loading label names...");
                                    if (labelNameFile.empty() || LoadLabelNames(labelNameFile))
                                    {
                                        mClusterNodeIdxs.resize(mNrOfImages, std::vector<std::vector<int> >());

                                        for (int i = 0; i < mNrOfImages; ++i)
                                        {
                                            for (unsigned int c = 0; c < mNrOfClusters[i]; ++c)
                                            {
                                                mClusterNodeIdxs[i].push_back(std::vector<int>());
                                            }
                                        }

                                        LOG_INFO("DONE");
                                        return true;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        return false;
    }

    bool EntangledForestData::LoadTestData(string dataDir, string filename)
    {
        std::vector<string> filenames;
        filenames.push_back(filename);

        LOG_PLAIN("Loading unary features...");
        if (LoadUnaryFeaturesBinary(dataDir + "/unary", filenames))
        {
            string pairwiseDir = dataDir + "/pairwise";

            LOG_PLAIN("Loading pairwise features (euclid)...");
            if (LoadPairwiseFeaturesBinary(pairwiseDir, filenames, mPairwiseEdist, "euclid"))
            {
                LOG_PLAIN("Loading pairwise features (hangles)...");
                if (LoadPairwiseFeaturesBinary(pairwiseDir, filenames, mPairwiseHangle, "hangles"))
                {
                    LOG_PLAIN("Loading pairwise features (vangles)...");
                    if (LoadPairwiseFeaturesBinary(pairwiseDir, filenames, mPairwiseVangle, "vangles"))
                    {
                        LOG_PLAIN("Loading pairwise features (ptpl)...");
                        if (LoadPairwiseFeaturesBinary(pairwiseDir, filenames, mPairwisePtPldist, "ptpl"))
                        {
                            LOG_PLAIN("Loading pairwise features (iptpl)...");
                            if (LoadPairwiseFeaturesBinary(pairwiseDir, filenames, mPairwiseIPtPldist, "iptpl"))
                            {
                                // TODO: could be done more efficient, right now this is only
                                // copy paste from loadtrainingdatanew
                                // mNrOfImages = 0 for classification
                                mClusterNodeIdxs.resize(mNrOfImages, std::vector<std::vector<int> >());

                                for (int i = 0; i < mNrOfImages; ++i)
                                {
                                    for (unsigned int c = 0; c < mNrOfClusters[i]; ++c)
                                    {
                                        mClusterNodeIdxs[i].push_back(std::vector<int>());
                                    }
                                }

                                for (unsigned int j = 0; j < mNrOfClusters[0]; ++j)
                                {
                                    // ignore unlabeled clusters
                                    ClusterIdx c = {0, j};
                                    mClusters.push_back(c);
                                }

                                LOG_INFO("DONE");
                                return true;
                            }
                        }
                    }
                }
            }
        }

        return false;
    }

    bool EntangledForestData::LoadTestDataLive(std::vector<std::vector<double> > &unaryFeaturesLive,
                                std::vector<std::vector<std::pair<double, int> > > &pointPlaneDistancesLive,
                                std::vector<std::vector<std::pair<double, int> > > &inversePointPlaneDistancesLive,
                                std::vector<std::vector<std::pair<double, int> > > &verticalAngleDifferencesLive,
                                std::vector<std::vector<std::pair<double, int> > > &horizontalAngleDifferencesLive,
                                std::vector<std::vector<std::pair<double, int> > > &euclideanDistancesLive)
    {
        mUnaryFeatures.resize(1);
        mUnaryFeatures[0] = unaryFeaturesLive;
        mPairwisePtPldist.resize(1);
        mPairwisePtPldist[0] = pointPlaneDistancesLive;
        mPairwiseIPtPldist.resize(1);
        mPairwiseIPtPldist[0] = inversePointPlaneDistancesLive;
        mPairwiseHangle.resize(1);
        mPairwiseHangle[0] = horizontalAngleDifferencesLive;
        mPairwiseVangle.resize(1);
        mPairwiseVangle[0] = verticalAngleDifferencesLive;
        mPairwiseEdist.resize(1);
        mPairwiseEdist[0] = euclideanDistancesLive;

        mNrOfClusters.resize(1);
        mNrOfClusters[0] = unaryFeaturesLive.size();
        mTotalNrOfClusters = unaryFeaturesLive.size();
        mNrOfImages = 1;

        mClusterNodeIdxs.resize(1);
        mClusterNodeIdxs[0].clear();

        mClusters.clear();

        // we only have one image
        for (unsigned int c = 0; c < mNrOfClusters[0]; ++c)
        {
            mClusterNodeIdxs[0].push_back(std::vector<int>());
            ClusterIdx j = {0, c};
            mClusters.push_back(j);
        }

        return true;
    }

    bool EntangledForestData::LoadGroundtruth(string gtDir, string filename, std::vector<int> &gt)
    {
        string filestr = gtDir + "/" + filename + ".txt";

        if (!fs::exists(fs::path(filestr)))
        {
            LOG_ERROR("Labels for image " << filename << " do not exist!");
            return false;
        }

        gt.clear();

        int label;
        std::ifstream ifs(filestr.c_str());

        while (ifs >> label)
        {
            gt.push_back(label);
        }

        // check if each test data cluster has a gt label
        if (gt.size() != mNrOfClusters[0])
        {
            LOG_ERROR("Number of labels in image " << filename << " does not match number of clusters!");
            return false;
        }

        return true;
    }

    void EntangledForestData::SetLabelMap(std::map<int, int> &labelMapping)
    {
        this->mLabelMap = labelMapping;
        mNrOfLabels = labelMapping.size();
    }

    bool EntangledForestData::LoadPairwiseFeatures(std::string pairwiseDir, std::vector<string> &imagefiles,
                                    std::vector<std::vector<std::vector<std::pair<double, int> > > > &pairwiseFeatures,
                                    string featurefileending)
    {
        pairwiseFeatures.clear();
        pairwiseFeatures.reserve(imagefiles.size());

        int clusteridx = 0;

        for (unsigned int i = 0; i < imagefiles.size(); ++i)
        {
            string filename = pairwiseDir + "/" + imagefiles[i] + "_" + featurefileending + ".txt";

            if (!fs::exists(fs::path(filename)))
            {
                LOG_ERROR("Pairwise features (" << featurefileending << ") for image " << imagefiles[i]
                                                << " do not exist!");
                return false;
            }

            std::vector<std::vector<std::pair<double, int> > > currentImageFeatures(mNrOfClusters[clusteridx],
                                                                                    std::vector<std::pair<double, int> >(
                                                                                            mNrOfClusters[clusteridx] -
                                                                                            1));

            std::ifstream ifs(filename.c_str());
            std::string line1, line2;

            std::vector<double> distances(mNrOfClusters[clusteridx] - 1, 0.0);
            std::vector<int> indices(mNrOfClusters[clusteridx] - 1, 0);
            std::vector<std::pair<double, int> > currentLineFeatures(mNrOfClusters[clusteridx] - 1);

            int linecnt = 0;
            int cnt = 0;

            while (std::getline(ifs, line1))
            {
                // first read distances as doubles
                std::stringstream s1(line1);

                double distance;

                cnt = 0;
                while (s1 >> distance)
                {
                    distances[cnt++] = distance;
                }

                // then read indices as integers
                if (std::getline(ifs, line2))
                {
                    // first read distances as doubles
                    std::stringstream s2(line2);

                    int index;
                    cnt = 0;
                    while (s2 >> index)
                    {
                        indices[cnt++] = index;
                    }
                }
                else
                {
                    break;
                }

                if (distances.size() != indices.size())
                {
                    LOG_ERROR("Different number of pairwise distances and indices (" << featurefileending
                                                                                     << ") for image " << filename
                                                                                     << "!");
                    return false;
                }

                // check if distances to all other clusters are available
                if (distances.size() != mNrOfClusters[clusteridx] - 1)
                {
                    LOG_ERROR("Number of pairwise distances (" << featurefileending << ") for image " << filename
                                                               << " does not match number of unary features!");
                    return false;
                }

                for (unsigned int j = 0; j < distances.size(); ++j)
                {
                    currentLineFeatures[j] = std::pair<double, int>(distances[j], indices[j]);
                }

                currentImageFeatures[linecnt++] = currentLineFeatures;
            }

            if (currentImageFeatures.size() != mNrOfClusters[clusteridx])
            {
                LOG_ERROR("Number of pairwise features (" << featurefileending << ") for image " << filename
                                                          << " does not match number of unary features!");
                return false;
            }

            pairwiseFeatures.push_back(currentImageFeatures);
            clusteridx++;
        }

        return true;
    }

    bool EntangledForestData::LoadPairwiseFeaturesBinary(std::string pairwiseDir, std::vector<string> &imagefiles,
                                          std::vector<std::vector<std::vector<std::pair<double, int> > > > &pairwiseFeatures,
                                          string featurefileending)
    {
        pairwiseFeatures.clear();
        pairwiseFeatures.reserve(imagefiles.size());

        int clusteridx = 0;

        for (unsigned int i = 0; i < imagefiles.size(); ++i)
        {
            string filename = pairwiseDir + "/" + imagefiles[i] + "_" + featurefileending;

            if (!fs::exists(fs::path(filename)))
            {
                LOG_ERROR("Pairwise features (" << featurefileending << ") for image " << imagefiles[i]
                                                << " do not exist!");
                return false;
            }

            std::vector<std::vector<std::pair<double, int> > > currentImageFeatures; //(mNrOfClusters[clusteridx], std::vector<std::pair<double, int> >(mNrOfClusters[clusteridx]-1));

            std::ifstream ifs(filename.c_str(), std::ios::binary);
            boost::archive::binary_iarchive ia(ifs);
            ia >> currentImageFeatures;
            ifs.close();

            if (currentImageFeatures.size() != mNrOfClusters[clusteridx])
            {
                LOG_ERROR("Number of pairwise features (" << featurefileending << ") for image " << filename
                                                          << " does not match number of unary features!");
                return false;
            }

            pairwiseFeatures.push_back(currentImageFeatures);
            clusteridx++;
        }

        return true;
    }

    bool EntangledForestData::LoadUnaryFeatures(string unaryDir, std::vector<string> &imagefiles)
    {
        mUnaryFeatures.clear();
        mUnaryFeatures.reserve(imagefiles.size());

        mNrOfClusters.clear();
        mTotalNrOfClusters = 0;

        for (unsigned int i = 0; i < imagefiles.size(); ++i)
        {
            string filename = unaryDir + "/" + imagefiles[i] + ".txt";

            if (!fs::exists(fs::path(filename)))
            {
                LOG_ERROR("Unary features for image " << imagefiles[i] << " do not exist!");
                return false;
            }

            std::vector<std::vector<double> > currentImageFeatures;

            std::ifstream ifs(filename.c_str());
            std::string line;

            while (std::getline(ifs, line))
            {
                std::stringstream s(line);
                std::vector<double> values;

                double value;
                while (s >> value)
                {
                    values.push_back(value);
                }

                currentImageFeatures.push_back(values);
            }

            mUnaryFeatures.push_back(currentImageFeatures);
            mNrOfClusters.push_back(currentImageFeatures.size());
            mTotalNrOfClusters += currentImageFeatures.size();
        }

        mNrOfImages = mUnaryFeatures.size();

        return true;
    }

    bool EntangledForestData::LoadUnaryFeaturesBinary(string unaryDir, std::vector<string> &imagefiles)
    {
        mUnaryFeatures.clear();
        mUnaryFeatures.reserve(imagefiles.size());

        mNrOfClusters.clear();
        mTotalNrOfClusters = 0;

        for (unsigned int i = 0; i < imagefiles.size(); ++i)
        {
            string filename = unaryDir + "/" + imagefiles[i];

            if (!fs::exists(fs::path(filename)))
            {
                LOG_ERROR("Unary features for image " << imagefiles[i] << " do not exist!");
                return false;
            }

            std::vector<std::vector<double> > currentImageFeatures;

            std::ifstream ifs(filename.c_str(), std::ios::binary);
            boost::archive::binary_iarchive ia(ifs);
            ia >> currentImageFeatures;
            ifs.close();

            mUnaryFeatures.push_back(currentImageFeatures);
            mNrOfClusters.push_back(currentImageFeatures.size());
            mTotalNrOfClusters += currentImageFeatures.size();
        }

        mNrOfImages = mUnaryFeatures.size();

        return true;
    }

    bool EntangledForestData::LoadLabels(string labelDir, std::vector<string> &imagefiles)
    {
        mClusterLabels.reserve(imagefiles.size());
        mLabelMap.clear();

        int labeled = 0;
        int unlabeled = 0;

        for (unsigned int i = 0; i < imagefiles.size(); ++i)
        {
            string filename = labelDir + "/" + imagefiles[i] + ".txt";

            if (!fs::exists(fs::path(filename)))
            {
                LOG_ERROR("Labels for image " << imagefiles[i] << " do not exist!");
                return false;
            }

            std::vector<int> currentLabels;
            int label;

            std::ifstream ifs(filename.c_str());

            while (ifs >> label)
            {
                currentLabels.push_back(label);

                // ignore label 0 (unlabeled)
                if (label > 0)
                {
                    mLabelMap[label] = 0;
                    labeled++;
                }
                else
                {
                    unlabeled++;
                }
            }

            if (currentLabels.size() != mNrOfClusters[i])
            {
                LOG_ERROR("Number of labels in image " << filename << " does not match number of unary features!");
                return false;
            }

            mClusterLabels.push_back(currentLabels);
        }

        LOG_ERROR("LABELED:   " << labeled);
        LOG_ERROR("UNLABELED: " << unlabeled);
        mNrOfLabels = mLabelMap.size();

        // set mLabelMap indices
        int n = 0;
        for (std::map<int, int>::iterator it = mLabelMap.begin(); it != mLabelMap.end(); ++it, ++n)
        {
            it->second = n;
        }

        // now that we know the available labels, we can count how often they appear in each image
        mLabelsPerImage.resize(mClusterLabels.size(), std::vector<unsigned int>(mNrOfLabels, 0));
        mClustersPerLabel.resize(mNrOfLabels, 0);

        for (unsigned int i = 0; i < mClusterLabels.size(); ++i)
        {
            for (unsigned int j = 0; j < mClusterLabels[i].size(); ++j)
            {
                int l = mClusterLabels[i][j];
                if (l > 0)
                {
                    mLabelsPerImage[i][mLabelMap[l]]++;
                    mClustersPerLabel[mLabelMap[l]]++;
                }
            }
        }

        return true;
    }

    bool EntangledForestData::LoadLabelsBinary(string labelDir, std::vector<string> &imagefiles)
    {
        mClusterLabels.reserve(imagefiles.size());
        mLabelMap.clear();

        for (unsigned int i = 0; i < imagefiles.size(); ++i)
        {
            string filename = labelDir + "/" + imagefiles[i];

            if (!fs::exists(fs::path(filename)))
            {
                LOG_ERROR("Labels for image " << imagefiles[i] << " do not exist!");
                return false;
            }

            std::vector<int> currentLabels;
            std::ifstream ifs(filename.c_str(), std::ios::binary);
            boost::archive::binary_iarchive ia(ifs);
            ia >> currentLabels;
            ifs.close();

            if (currentLabels.size() != mNrOfClusters[i])
            {
                LOG_ERROR("Number of labels in image " << filename << " does not match number of unary features!");
                return false;
            }

            mClusterLabels.push_back(currentLabels);
        }


        int labeled = 0;
        int unlabeled = 0;

        for (unsigned int i = 0; i < mClusterLabels.size(); ++i)
        {
            for (unsigned int j = 0; j < mClusterLabels[i].size(); ++j)
            {
                int label = mClusterLabels[i][j];

                // ignore label 0 (unlabeled)
                if (label > 0)
                {
                    mLabelMap[label] = 0;
                    labeled++;
                }
                else
                {
                    unlabeled++;
                }
            }
        }

        LOG_ERROR("LABELED:   " << labeled);
        LOG_ERROR("UNLABELED: " << unlabeled);
        mNrOfLabels = mLabelMap.size();

        // set mLabelMap indices
        int n = 0;
        for (std::map<int, int>::iterator it = mLabelMap.begin(); it != mLabelMap.end(); ++it, ++n)
        {
            it->second = n;
        }

        // now that we know the available labels, we can count how often they appear in each image
        mLabelsPerImage.resize(mClusterLabels.size(), std::vector<unsigned int>(mNrOfLabels, 0));
        mClustersPerLabel.resize(mNrOfLabels, 0);

        for (unsigned int i = 0; i < mClusterLabels.size(); ++i)
        {
            for (unsigned int j = 0; j < mClusterLabels[i].size(); ++j)
            {
                int l = mClusterLabels[i][j];
                if (l > 0)
                {
                    mLabelsPerImage[i][mLabelMap[l]]++;
                    mClustersPerLabel[mLabelMap[l]]++;
                }
            }
        }

        return true;
    }

    bool EntangledForestData::LoadLabelNames(string labelNameFile)
    {
        mLabelNames.clear();
        std::string name;

        ifstream ifs(labelNameFile.c_str());

        if (ifs.fail())
        {
            LOG_ERROR("The label name file " << labelNameFile << " could not be opened!");
            return false;
        }

        while (getline(ifs, name))
        {
            mLabelNames.push_back(name);
        }
        ifs.close();

        if ((int)mLabelNames.size() < mNrOfLabels)
        {
            LOG_ERROR("Too few entries in label names file (only " << mLabelNames.size() << ", but data contains "
                                                                   << mNrOfLabels << " labels.");
            return false;
        }
        return true;
    }

    void EntangledForestData::GetBeginAndEndIterator(ClusterIdxItr &begin, ClusterIdxItr &end)
    {
        begin = mClusters.begin();
        end = mClusters.end();
    }


    bool EntangledForestData::SmallerThan(const pair<double, int> &l, const double &r)
    {
        return l.first < r;
    }

    bool EntangledForestData::LargerThan(const double &r, const pair<double, int> &l)
    {
        return r < l.first;
    }

    void EntangledForestData::GenerateBags(std::mt19937 *randomGenerator, double baggingRatio, int trees, bool tryUniformBags)
    {
        // how many images do we need?
        int nImages = floor(mNrOfImages * baggingRatio);

        vector<int> imageIndices;

        for (int i = 0; i < mNrOfImages; ++i)
        {
            imageIndices.push_back(i);
        }

        if (tryUniformBags)
        {
//        int factor = 100;
//        std::vector< std::vector<int> > tmp_bags(trees*factor);
//        vector<pair<double,int> > entropy_index(trees*factor);

//        for(int i=0; i < trees*factor; ++i)
//        {
//            vector<int> bag;
//            std::shuffle(std::begin(imageIndices), std::end(imageIndices), randomNumber);
//            bag.insert(bag.end(), imageIndices.begin(), imageIndices.begin()+nImages);
//            if(!CheckIfAllLabelsAvailable(bag))
//            {
//                i--;
//                continue;
//            }
//            tmp_bags[i] = bag;

//            // calculate entropy for current bag
//            vector<long long int> ppL(mNrOfLabels);
//            int nDataPoints = 0;

//            for(int i=0; i<bag.size(); ++i)
//            {
//                for(int j=0; j<mNrOfLabels; ++j)
//                {
//                    ppL[j] += mLabelsPerImage[bag[i]][j];
//                    nDataPoints += mLabelsPerImage[bag[i]][j];
//                }
//            }

//            double sumWeights(0.0f);
//            double entropy(0.0f);

//            vector<double> ppLnorm(ppL.size());
//            // normalize label histogram
//            for(int j=0; j < ppL.size(); ++j)
//            {
//                sumWeights += ppL[j];
//            }
//            for(int j=0; j < ppL.size(); ++j)
//            {
//                ppLnorm[j] = ((double)ppL[j]) / sumWeights;
//            }
//            for(int j=0; j < ppLnorm.size(); ++j)
//            {
//                entropy -= ppLnorm[j] * log2(ppLnorm[j] + std::numeric_limits<double>::min());
//            }
//            entropy_index[i].first = entropy;
//            entropy_index[i].second = i;
//        }

//        // sort entropies
//        std::sort(entropy_index.begin(), entropy_index.end(), pairwisecomparator);

//        // take ntrees mBags with highest entropy
//        for(int i=0; i < trees; ++i)
//        {
//            mBags.push_back(tmp_bags[entropy_index[i].second]);
//        }

            int factor = 100;
            bool ok = false;

            double maxRatio = 100;//-mNrOfLabels * log2(1.0/mNrOfLabels) * 1.0/mNrOfLabels;//3.5;
            double curBaggingRatio = baggingRatio;

            while (!ok)
            {
                std::map<int, std::vector<int> > tmp_bags;
                vector<pair<double, int> > ratio_index;

                for (int i = 0; i < trees * factor; ++i)
                {
                    vector<int> bag;
                    std::shuffle(std::begin(imageIndices), std::end(imageIndices), *randomGenerator);
                    bag.insert(bag.end(), imageIndices.begin(), imageIndices.begin() + nImages);
                    if (!CheckIfAllLabelsAvailable(bag))
                    {
//                    i--;
                        continue;
                    }
                    tmp_bags[i] = bag;

                    // calculate portion of largest label for each bag
                    vector<long long int> ppL(mNrOfLabels);
                    int nDataPoints = 0;
                    long long int maxppL = 0;
                    long long int minppL = std::numeric_limits<long long int>::max();

                    for (int j = 0; j < mNrOfLabels; ++j)
                    {
                        for (unsigned int b = 0; b < bag.size(); ++b)
                        {
                            ppL[j] += mLabelsPerImage[bag[b]][j];
                            nDataPoints += mLabelsPerImage[bag[b]][j];
                        }

                        if (ppL[j] > maxppL)
                            maxppL = ppL[j];
                        if (ppL[j] < minppL)
                            minppL = ppL[j];
                    }

//                double sumWeights(0.0f);
//                double entropy(0.0f);

//                vector<double> ppLnorm(ppL.size());
//                // normalize label histogram
//                for(int j=0; j < ppL.size(); ++j)
//                {
//                    sumWeights += ppL[j];
//                }
//                for(int j=0; j < ppL.size(); ++j)
//                {
//                    ppLnorm[j] = ((double)ppL[j]) / sumWeights;
//                }
//                for(int j=0; j < ppLnorm.size(); ++j)
//                {
//                    entropy -= ppLnorm[j] * log2(ppLnorm[j] + std::numeric_limits<double>::min());
//                }


                    double ratio = ((double) maxppL) / ((double) minppL);
                    ratio_index.push_back({ratio, i});
                }

                // sort ratios
                std::sort(ratio_index.begin(), ratio_index.end(), PairwiseComparator);

                // check lowest ratios
                if ((int)ratio_index.size() < trees)
                {
                    // limits too strict to find bag with all labels, weaken maxRatio and start over with initial bagging ratio
                    maxRatio *= 1.2;
                    curBaggingRatio = baggingRatio;

                    ok = false;
                    LOG_INFO("Not enough suitable mBags. Setting maxRatio to " << maxRatio);
                }
                else if (ratio_index[trees - 1].first > maxRatio)
                {
                    // failed to find high enough ratio, retry with lower bagging ratio
                    curBaggingRatio *= 0.8;

                    LOG_INFO("Ratio not achieved, setting bagging to " << curBaggingRatio);
                    if (curBaggingRatio < 0.1)
                    {
                        maxRatio *= 1.2;
                        curBaggingRatio = baggingRatio;
                        LOG_INFO("Baggint too low. Resetting to " << curBaggingRatio << " and setting maxRatio to "
                                                                  << maxRatio);
                    }

                    nImages = floor(mNrOfImages * curBaggingRatio);
                    ok = false;
                }
                else
                {
                    LOG_INFO("Bags ok with bagging " << curBaggingRatio << " and maxRatio " << maxRatio);
                    // take ntrees mBags with lowest ratios
                    for (int i = 0; i < trees; ++i)
                    {
                        mBags.push_back(tmp_bags[ratio_index[i].second]);
                    }

                    ok = true;
                }
            }
        }
        else
        {
            for (int i = 0; i < trees; ++i)
            {
                vector<int> bag;
                std::shuffle(std::begin(imageIndices), std::end(imageIndices), *randomGenerator);
                bag.insert(bag.end(), imageIndices.begin(), imageIndices.begin() + nImages);
                if (!CheckIfAllLabelsAvailable(bag))
                {
                    i--;
                    continue;
                }
                mBags.push_back(bag);
            }
        }
    }

    void EntangledForestData::LoadBag(int bagidx)
    {
        vector<int> bag = mBags[bagidx];
        vector<long long int> ppL(mNrOfLabels);
        int nDataPoints = 0;

        mClusters.clear();

        LOG_INFO("Using " << bag.size() << " images");

        for (unsigned int i = 0; i < bag.size(); ++i)
        {
            for (int j = 0; j < mNrOfLabels; ++j)
            {
                ppL[j] += mLabelsPerImage[bag[i]][j];
                nDataPoints += mLabelsPerImage[bag[i]][j];
            }

            for (unsigned int j = 0; j < mNrOfClusters[bag[i]]; ++j)
            {
                // ignore unlabeled clusters
                if (mClusterLabels[bag[i]][j] > 0)
                {
                    ClusterIdx c = {bag[i], j};
                    mClusters.push_back(c);
                }
            }
        }

        // calculate class weights for bag
        mClassWeights.clear();
        mNormFactor = 0.0f;

        LOG_INFO("Pts per Label / Class Weights:");
        for (int i = 0; i < mNrOfLabels; ++i)
        {
            mClassWeights.push_back(((double) ppL[i]) / ((double) nDataPoints) + std::numeric_limits<double>::min());
            mNormFactor += ((double) ppL[i] + 1) / mClassWeights[i];

            LOG_PLAIN(setw(13) << ppL[i] << " / " << setw(13) << mClassWeights[i]);
        }
    }


    double EntangledForestData::CheckIfAllLabelsAvailable(std::vector<int> imageIndices)
    {
        vector<bool> available(mNrOfLabels, false);

        bool ok = false;

        for (unsigned int i = 0; i < imageIndices.size() && !ok; ++i)
        {
            for (int j = 0; j < mNrOfLabels; ++j)
            {
                if (mLabelsPerImage[imageIndices[i]][j])
                {
                    available[j] = true;
                }
            }

            ok = true;
            for (int j = 0; j < mNrOfLabels; ++j)
            {
                if (!available[j])
                {
                    ok = false;
                    break;
                }
            }
        }

        return ok;
    }

    double EntangledForestData::CalculateEnergy(const std::vector<unsigned int> &unnormalizedHistogram, double &energy)
    {
        double normalizationTerm(0.0f);
        // EDIT DW Jul10
        double sum(0.0f);

        // parallelizing this is not worth it, too much overhead

        // normalize histogram
        for (unsigned int i = 0; i < unnormalizedHistogram.size(); ++i)
        {
            sum += (double) unnormalizedHistogram[i];
            normalizationTerm += ((double) unnormalizedHistogram[i]) / mClassWeights[i];
        }

        if (normalizationTerm < std::numeric_limits<double>::min())
            return std::numeric_limits<double>::lowest();

        energy = 0.0f;
        for (unsigned int i = 0; i < unnormalizedHistogram.size(); ++i)
        {
            energy -= ((double) unnormalizedHistogram[i]) *
                      log2(((double) unnormalizedHistogram[i]) / (mClassWeights[i] * normalizationTerm) +
                           std::numeric_limits<double>::min());
        }

        energy /= sum;
        return normalizationTerm;
    }

    void EntangledForestData::CalculateHistogram(const ClusterIdxItr dataBegin, const ClusterIdxItr dataEnd,
                                    std::vector<unsigned int> &hist)
    {
        hist.resize(mNrOfLabels, 0);

        ClusterIdxItr start = dataBegin;
        int npoints = std::distance(dataBegin, dataEnd);

#ifdef NDEBUG
#pragma omp parallel
#endif
        {
            // private histogram for each thread
            std::vector<unsigned int> hist_private(mNrOfLabels, 0);
            int label = 0;

#ifdef NDEBUG
#pragma omp for nowait
#endif
            for (int i = 0; i < npoints; ++i)
            {
                ClusterIdx idx = *(start + i);
                label = mClusterLabels[idx[0]][idx[1]];
                if (label > 0)   // ignore label 0 (=unlabeled)
                {
                    hist_private[mLabelMap[label]]++;
                }
            }

            // add private histogram of thread to total histogram
#ifdef NDEBUG
#pragma omp critical
#endif
            std::transform(hist.begin(), hist.end(), hist_private.begin(), hist.begin(), std::plus<unsigned int>());
        }
    }

    double EntangledForestData::CalculateEnergy(const ClusterIdxItr dataBegin, const ClusterIdxItr dataEnd, double &energy)
    {
        std::vector<unsigned int> hist;     // histogram for all labels
        CalculateHistogram(dataBegin, dataEnd, hist);
        return CalculateEnergy(hist, energy);
    }

    double EntangledForestData::CalculateEntropy(const std::vector<unsigned int> &unnormalizedHistogram, double &entropy)
    {
        double sumWeights(0.0f);
        entropy = 0.0f;

        std::vector<double> unnormalizedWeightedHistogram(unnormalizedHistogram.size());

        // normalize histogram
        for (unsigned int i = 0; i < unnormalizedHistogram.size(); ++i)
        {
            unnormalizedWeightedHistogram[i] = ((double) unnormalizedHistogram[i]) / mClassWeights[i];
            sumWeights += unnormalizedWeightedHistogram[i];
        }

        if (sumWeights < std::numeric_limits<double>::min())
            return std::numeric_limits<double>::lowest();

        double normHistWeights = 0.0f;

        for (unsigned int i = 0; i < unnormalizedHistogram.size(); ++i)
        {
            normHistWeights = unnormalizedWeightedHistogram[i] / sumWeights;
            entropy -= normHistWeights * log2(normHistWeights + std::numeric_limits<double>::min());
        }

        return sumWeights;
    }


    double EntangledForestData::CalculateEntropy(ClusterIdxItr dataBegin, ClusterIdxItr dataEnd, double &entropy)
    {
        std::vector<unsigned int> hist;             // histogram for all labels
        CalculateHistogram(dataBegin, dataEnd, hist);
        return CalculateEntropy(hist, entropy);
    }

    void EntangledForestData::CalculateLabelDistributions(ClusterIdxItr dataBegin, ClusterIdxItr dataEnd,
                                           std::vector<unsigned int> &absLabelDistribution,
                                           std::vector<double> &relLabelDistribution)
    {
        double sum(0.0f);

        CalculateAbsoluteLabelDistribution(dataBegin, dataEnd, absLabelDistribution);

        relLabelDistribution.resize(absLabelDistribution.size());

#ifdef LEAF_WEIGHTED
        // parallelization does not make sense for this small array, too much overhead
        for(unsigned int i=0; i < absLabelDistribution.size(); ++i)
        {
            sum += ((double)(absLabelDistribution[i])) / mClassWeights[i];
        }

        for(unsigned int i=0; i<relLabelDistribution.size(); ++i)
        {
            //relLabelDistribution[i] = ((double)(absLabelDistribution[i])+1.0f) / (sum+1.0f);
            relLabelDistribution[i] = ((double)(absLabelDistribution[i]) / mClassWeights[i]) / (sum);
        }
#else
        for (auto n : absLabelDistribution)
        { sum += n; }

        for (int i = 0; i < relLabelDistribution.size(); ++i)
        {
            //relLabelDistribution[i] = ((double)(absLabelDistribution[i])+1.0f) / (sum+1.0f);
            relLabelDistribution[i] = ((double) (absLabelDistribution[i])) / (sum);
        }
#endif
    }

    void EntangledForestData::AddTreesToClusterNodeIdx(int ntrees)
    {
        for (int i = 0; i < mNrOfImages; ++i)
        {
            for (unsigned int c = 0; c < mNrOfClusters[i]; ++c)
            {
                mClusterNodeIdxs[i][c].resize(ntrees, 0);
            }
        }
    }

    void EntangledForestData::CalculateRelativeLabelDistribution(ClusterIdxItr dataBegin, ClusterIdxItr dataEnd,
                                                  vector<double> &labelDistribution)
    {
        vector<unsigned int> absdist;
        CalculateLabelDistributions(dataBegin, dataEnd, absdist, labelDistribution);
    }


    void EntangledForestData::CalculateAbsoluteLabelDistribution(ClusterIdxItr dataBegin, ClusterIdxItr dataEnd,
                                                  vector<unsigned int> &labelDistribution)
    {
        // initialize final distribution array
        labelDistribution.resize(mNrOfLabels, 0);

        ClusterIdxItr start = dataBegin;
        int npoints = std::distance(dataBegin, dataEnd);

#ifdef NDEBUG
#pragma omp parallel
#endif
        {
            // initialize private distribution array for each thread
            std::vector<unsigned int> labelDistribution_private(mNrOfLabels, 0);
            int label(0);

#ifdef NDEBUG
#pragma omp for nowait
#endif
            for (int i = 0; i < npoints; ++i)
            {
                ClusterIdx pointidx = *(start + i);
                label = mClusterLabels[pointidx[0]][pointidx[1]];

                // don't count 0 (=unlabeled)
                if (label > 0)
                {
                    labelDistribution_private[mLabelMap[label]]++;
                }
            }

            // add distribution of thread to final result (element-wise summation)
#ifdef NDEBUG
#pragma omp critical
#endif
            std::transform(labelDistribution.begin(), labelDistribution.end(), labelDistribution_private.begin(),
                           labelDistribution.begin(), std::plus<unsigned int>());
        }
    }

    void EntangledForestData::SetClusterNodeIdx(int imageIdx, int clusterIdx, int treeIdx, int nodeIdx)
    {
        mClusterNodeIdxs[imageIdx][clusterIdx][treeIdx] = nodeIdx;
    }

    void EntangledForestData::SetClusterNodeIdx(ClusterIdx &datapoint, int treeIdx, int nodeIdx)
    {
        SetClusterNodeIdx(datapoint[0], datapoint[1], treeIdx, nodeIdx);
    }

    int EntangledForestData::SecondElement(const std::pair<double, int> &p)
    {
        return p.second;
    }

    void EntangledForestData::FilterClustersByGeometry(int imageIdx, int clusterIdx, bool horizontal, double minAngleDiff,
                                        double maxAngleDiff, double minPtPlaneDist, double maxPtPlaneDist,
                                        double maxEuclidDist, std::vector<int> &remainingClusters)
    {
        int i = imageIdx;
        int c = clusterIdx;

        std::vector<std::pair<double, int> >::iterator lb1, ub1;

        // first constraint
        if (horizontal)
        {
            lb1 = std::lower_bound(mPairwiseHangle[i][c].begin(), mPairwiseHangle[i][c].end(), minAngleDiff,
                                   SmallerThan);
            ub1 = std::upper_bound(mPairwiseHangle[i][c].begin(), mPairwiseHangle[i][c].end(), maxAngleDiff, LargerThan);
        }
        else
        {
            lb1 = std::lower_bound(mPairwiseVangle[i][c].begin(), mPairwiseVangle[i][c].end(), minAngleDiff,
                                   SmallerThan);
            ub1 = std::upper_bound(mPairwiseVangle[i][c].begin(), mPairwiseVangle[i][c].end(), maxAngleDiff, LargerThan);
        }

        // return if no cluster fulfills criteria
        if (lb1 == ub1)
        {
            return;
        }

        // second constraint
        auto lb2 = std::lower_bound(mPairwisePtPldist[i][c].begin(), mPairwisePtPldist[i][c].end(), minPtPlaneDist,
                                    SmallerThan);
        auto ub2 = std::upper_bound(mPairwisePtPldist[i][c].begin(), mPairwisePtPldist[i][c].end(), maxPtPlaneDist,
                                    LargerThan);

        // return if no cluster fulfills criteria
        if (lb2 == ub2)
        {
            return;
        }

        // third constraint
        auto ub3 = std::upper_bound(mPairwiseEdist[i][c].begin(), mPairwiseEdist[i][c].end(), maxEuclidDist, LargerThan);

        // return if no cluster fulfills criteria
        if (ub3 == mPairwiseEdist[i][c].begin())
        {
            return;
        }

        int max = (ub1 - lb1);
        std::vector<int> indices1;
        indices1.reserve(max);
        std::transform(lb1, ub1, std::back_inserter(indices1), SecondElement);
        std::sort(indices1.begin(), indices1.end());

        std::vector<int> indices2;
        indices2.reserve(ub2 - lb2);
        std::transform(lb2, ub2, std::back_inserter(indices2), SecondElement);
        std::sort(indices2.begin(), indices2.end());

        // allocate enough space for first intersection result
        max = std::max(max, (int) indices2.size());

        // intersect all received indices
        std::vector<int> intersection1;
        intersection1.reserve(max);
        set_intersection(indices1.begin(), indices1.end(), indices2.begin(), indices2.end(),
                         std::back_inserter(intersection1));

        // return if first intersection is already empty
        if (intersection1.size() == 0)
        {
            return;
        }

        std::vector<int> indices3;
        indices3.reserve(ub3 - mPairwiseEdist[i][c].begin());
        std::transform(mPairwiseEdist[i][c].begin(), ub3, std::back_inserter(indices3), SecondElement);
        std::sort(indices3.begin(), indices3.end());

        // allocate enough space for second intersection result
        remainingClusters.clear();
        remainingClusters.reserve(std::max(indices3.size(), intersection1.size()));

        set_intersection(intersection1.begin(), intersection1.end(), indices3.begin(), indices3.end(),
                         std::back_inserter(remainingClusters));
    }

    void EntangledForestData::FilterClustersByInversePtPl(int imageIdx, int clusterIdx, double minAngle, double maxAngle,
                                           double minIPtPlaneDist, double maxIPtPlaneDist, double maxEuclidDist,
                                           std::vector<int> &remainingClusters)
    {
        int i = imageIdx;
        int c = clusterIdx;

        int nclusters = mNrOfClusters[i];

        std::vector<std::pair<double, int> >::iterator lb1, ub1;

        // first constraint
        lb1 = std::lower_bound(mPairwiseIPtPldist[i][c].begin(), mPairwiseIPtPldist[i][c].end(), minIPtPlaneDist,
                               SmallerThan);
        ub1 = std::upper_bound(mPairwiseIPtPldist[i][c].begin(), mPairwiseIPtPldist[i][c].end(), maxIPtPlaneDist,
                               LargerThan);

        // return if no cluster fulfills criteria
        if (lb1 == ub1)
        {
            return;
        }

        // second constraint
        std::vector<int> indices2;
        indices2.reserve(nclusters);
        for (unsigned int n = 0; n < mNrOfClusters[i]; ++n)
        {
            if ((int)n != c)
            {
                double a = mUnaryFeatures[i][n][0];  // get mean cluster angle from unaries
                if (minAngle <= a && a <= maxAngle)
                {
                    indices2.push_back(n);
                }
            }
        }

        // return if no cluster fulfills criteria
        if (indices2.size() == 0)
        {
            return;
        }

        // third constraint
        auto ub3 = std::upper_bound(mPairwiseEdist[i][c].begin(), mPairwiseEdist[i][c].end(), maxEuclidDist, LargerThan);

        // return if no cluster fulfills criteria
        if (ub3 == mPairwiseEdist[i][c].begin())
        {
            return;
        }

        int max = (ub1 - lb1);
        std::vector<int> indices1;
        indices1.reserve(max);
        std::transform(lb1, ub1, std::back_inserter(indices1), SecondElement);
        std::sort(indices1.begin(), indices1.end());

        std::sort(indices2.begin(), indices2.end());

        // allocate enough space for first intersection result
        max = std::max(max, (int) indices2.size());

        // intersect all received indices
        std::vector<int> intersection1;
        intersection1.reserve(max);
        set_intersection(indices1.begin(), indices1.end(), indices2.begin(), indices2.end(),
                         std::back_inserter(intersection1));

        // return if first intersection is already empty
        if (intersection1.size() == 0)
        {
            return;
        }

        std::vector<int> indices3;
        indices3.reserve(ub3 - mPairwiseEdist[i][c].begin());
        std::transform(mPairwiseEdist[i][c].begin(), ub3, std::back_inserter(indices3), SecondElement);
        std::sort(indices3.begin(), indices3.end());

        remainingClusters.clear();
        remainingClusters.reserve(std::max(indices3.size(), intersection1.size()));

        set_intersection(intersection1.begin(), intersection1.end(), indices3.begin(), indices3.end(),
                         std::back_inserter(remainingClusters));
    }

    void EntangledForestData::UpdateClusterNodeIndices(int imageIdx, int treeIdx, std::vector<int> &clusterNodeIndices)
    {
        for (unsigned int i = 0; i < mNrOfClusters[imageIdx]; ++i)
        {
            mClusterNodeIdxs[imageIdx][i][treeIdx] = clusterNodeIndices[i];
        }
    }

    void EntangledForestData::UpdateClusterNodeIndicesPerTree(int treeIdx, std::vector<std::vector<int> > &clusterNodeIndices)
    {
        for (unsigned int i = 0; i < clusterNodeIndices.size(); ++i)
        {
            for (unsigned int c = 0; c < clusterNodeIndices[i].size(); ++c)
                mClusterNodeIdxs[i][c][treeIdx] = clusterNodeIndices[i][c];
        }
    }

    void EntangledForestData::GetClusterNodeIndices(int treeIdx, std::vector<int> &clusterNodeIndices)
    {
        clusterNodeIndices.resize(mNrOfClusters[0], 0);

        for (unsigned int i = 0; i < mNrOfClusters[0]; ++i)
        {
            clusterNodeIndices[i] = mClusterNodeIdxs[0][i][treeIdx];
        }
    }

    void EntangledForestData::GetClusterNodeIndicesPerTree(int treeIdx, std::vector<std::vector<int> > &clusterNodeIndices)
    {
        clusterNodeIndices.resize(mNrOfImages);

        for (int i = 0; i < mNrOfImages; ++i)
        {
            std::vector<int> imageClusterNodeIdx(mNrOfClusters[i], 0);

            for (unsigned int c = 0; c < mNrOfClusters[i]; ++c)
            {
                imageClusterNodeIdx[c] = mClusterNodeIdxs[i][c][treeIdx];
            }

            clusterNodeIndices[i] = imageClusterNodeIdx;
        }
    }


    void EntangledForestData::ResetClusterNodeIdxs()
    {
        for (unsigned int i = 0; i < mClusterNodeIdxs.size(); ++i)
        {
            for (unsigned int c = 0; c < mClusterNodeIdxs[i].size(); ++c)
            {
                for (unsigned int t = 0; t < mClusterNodeIdxs[i][c].size(); ++t)
                {
                    mClusterNodeIdxs[i][c][t] = -1;
                }
            }
        }
    }

    void EntangledForestData::ResetClusterNodeIdxs(int imageIdx)
    {
        for (unsigned int c = 0; c < mClusterNodeIdxs[imageIdx].size(); ++c)
        {
            for (unsigned int t = 0; t < mClusterNodeIdxs[imageIdx][c].size(); ++t)
            {
                mClusterNodeIdxs[imageIdx][c][t] = -1;
            }
        }
    }

}