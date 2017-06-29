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
#include <time.h>

#include <omp.h>

#include <v4r/semantic_segmentation/entangled_feature_extraction.h>


using namespace std;
using namespace boost::posix_time;

namespace v4r
{
    bool EntangledForestFeatureExtraction::PairwiseComparator(const std::pair<double, int> &l, const std::pair<double, int> r)
    {
        return l.first < r.first;
    }

    EntangledForestFeatureExtraction::EntangledForestFeatureExtraction()
    {
    }

    void EntangledForestFeatureExtraction::setInputCloud(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr input,
                                          pcl::PointCloud<pcl::Normal>::ConstPtr normals,
                                          pcl::PointCloud<pcl::PointXYZL>::ConstPtr labels, int nlabels = -1)
    {
        // TODO: Check if they are of equal size?
        mInputCloud = input;
        mNormalCloud = normals;
        mLabelCloud = labels;

        mNrOfSegments = nlabels;

        if (mNrOfSegments < 0)
        {
            // we have to count ourselves how many segments there are
            // we assume consecutive, 0-based labels!!!
            // TODO: maybe need to switch to 1-based? to have 0 as unlabeled?
            for (unsigned int i = 0; i < mLabelCloud->points.size(); ++i)
            {
                mNrOfSegments = std::max(mNrOfSegments, (int) mLabelCloud->at(i).label);
            }
            mNrOfSegments++;
        }
    }

    void EntangledForestFeatureExtraction::setCameraExtrinsics(double height, double pitch, double roll)
    {
        mCameraHeight = height;

        // compute rotation matrix w.r.t. ground plane
        Eigen::AngleAxisf rollAngle(roll, Eigen::Vector3f::UnitZ());
        Eigen::AngleAxisf yawAngle(0, Eigen::Vector3f::UnitY());
        Eigen::AngleAxisf pitchAngle(pitch, Eigen::Vector3f::UnitX());

        Eigen::Quaternion<float> q = rollAngle * yawAngle * pitchAngle;
        Eigen::Matrix3f rotMatrix = q.matrix();

        mTransformationMatrix = Eigen::Matrix4f::Zero();
        mTransformationMatrix.block(0, 0, 3, 3) = rotMatrix;
        mTransformationMatrix(3, 3) = 1.0;
    }

    void EntangledForestFeatureExtraction::setCameraExtrinsics(double height, Eigen::Matrix3f &rotationMatrix)
    {
        mTransformationMatrix = Eigen::Matrix4f::Zero();
        mTransformationMatrix.block(0, 0, 3, 3) = rotationMatrix;
        mTransformationMatrix(3, 3) = 1.0;
        mCameraHeight = height;
    }

    void EntangledForestFeatureExtraction::CalculatePairwiseFeatures(std::vector<Eigen::Vector3f> &centroids,
                                                      std::vector<Eigen::Vector3f> &normals,
                                                      std::vector<double> &verticalAngles)
    {
        mPairwiseEuclid.clear();
        mPairwiseEuclid.resize(mNrOfSegments, std::vector<std::pair<double, int> >(mNrOfSegments - 1));
        mPairwisePtPl.clear();
        mPairwisePtPl.resize(mNrOfSegments, std::vector<std::pair<double, int> >(mNrOfSegments - 1));
        mPairwiseIPtPl.clear();
        mPairwiseIPtPl.resize(mNrOfSegments, std::vector<std::pair<double, int> >(mNrOfSegments - 1));
        mPairwiseVAngle.clear();
        mPairwiseVAngle.resize(mNrOfSegments, std::vector<std::pair<double, int> >(mNrOfSegments - 1));
        mPairwiseHAngle.clear();
        mPairwiseHAngle.resize(mNrOfSegments, std::vector<std::pair<double, int> >(mNrOfSegments - 1));

        for (int segidx1 = 0; segidx1 < mNrOfSegments - 1; ++segidx1)
        {
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr seg1 = mSegments[segidx1];
            pcl::KdTreeFLANN <pcl::PointXYZRGB> tree;
            tree.setInputCloud(seg1);

            // for ptpl
            Eigen::Vector3f c1 = centroids[segidx1];
            Eigen::Vector3f n1 = normals[segidx1];

            // for angles
            double v1 = verticalAngles[segidx1];
            Eigen::Vector2d h1;
            h1 << n1[0], n1[2];

            for (int segidx2 = segidx1 + 1; segidx2 < mNrOfSegments; ++segidx2)
            {
                pcl::PointCloud<pcl::PointXYZRGB>::Ptr seg2 = mSegments[segidx2];

                // for ptpl
                Eigen::Vector3f c2 = centroids[segidx2];
                Eigen::Vector3f n2 = normals[segidx2];

                // for angles
                double v2 = verticalAngles[segidx2];
                Eigen::Vector2d h2;
                h2 << n2[0], n2[2];

                float minEDist = std::numeric_limits<float>::max();
                float minPtPlDist1 = std::numeric_limits<float>::max();
                float minPtPlDist2 = std::numeric_limits<float>::max();

                // TODO: This is going to be much slower now, since we compute distances of all POINTS and not
                // centroids of SVs any more!!!
                // calculate min distance between segment1 and segment2
                std::vector<int> eidx(1);
                std::vector<float> edist(1);

                for (unsigned int ptidx2 = 0; ptidx2 < seg2->points.size(); ++ptidx2)
                {
                    tree.nearestKSearch(seg2->at(ptidx2), 1, eidx, edist);

                    if (edist[0] < minEDist)
                    {
                        minEDist = edist[0];
                    }

                    // also calc. ptpl
                    Eigen::Vector3f pt2 = seg2->at(ptidx2).getVector3fMap();
                    float d = std::abs(n1.dot(pt2 - c1));

                    if (d < minPtPlDist1)
                    {
                        minPtPlDist1 = d;
                    }
                }

                minEDist = sqrt(minEDist);

                for (unsigned int ptidx1 = 0; ptidx1 < seg1->points.size(); ++ptidx1)
                {
                    // also calc. ptpl
                    Eigen::Vector3f pt1 = seg1->at(ptidx1).getVector3fMap();
                    float d = std::abs(n2.dot(pt1 - c2));

                    if (d < minPtPlDist2)
                    {
                        minPtPlDist2 = d;
                    }
                }

                // angles
                // to make sure argument for acos is inside defined range
                double hordiff = acos(std::max(-1.0, std::min(1.0, h2.dot(h1) / (h2.norm() * h1.norm()))));
                double verdiff = v2 - v1;

                mPairwiseEuclid[segidx1][segidx2 - 1].first = minEDist;
                mPairwiseEuclid[segidx1][segidx2 - 1].second = segidx2;
                mPairwiseEuclid[segidx2][segidx1].first = minEDist;
                mPairwiseEuclid[segidx2][segidx1].second = segidx1;

                mPairwisePtPl[segidx1][segidx2 - 1].first = minPtPlDist1;
                mPairwisePtPl[segidx1][segidx2 - 1].second = segidx2;
                mPairwisePtPl[segidx2][segidx1].first = minPtPlDist2;
                mPairwisePtPl[segidx2][segidx1].second = segidx1;

                mPairwiseIPtPl[segidx1][segidx2 - 1].first = minPtPlDist2;
                mPairwiseIPtPl[segidx1][segidx2 - 1].second = segidx2;
                mPairwiseIPtPl[segidx2][segidx1].first = minPtPlDist1;
                mPairwiseIPtPl[segidx2][segidx1].second = segidx1;

                mPairwiseHAngle[segidx1][segidx2 - 1].first = hordiff;
                mPairwiseHAngle[segidx1][segidx2 - 1].second = segidx2;
                mPairwiseHAngle[segidx2][segidx1].first = -hordiff;
                mPairwiseHAngle[segidx2][segidx1].second = segidx1;

                mPairwiseVAngle[segidx1][segidx2 - 1].first = verdiff;
                mPairwiseVAngle[segidx1][segidx2 - 1].second = segidx2;
                mPairwiseVAngle[segidx2][segidx1].first = -verdiff;
                mPairwiseVAngle[segidx2][segidx1].second = segidx1;
            }
        }

        for (int i = 0; i < mNrOfSegments; ++i)
        {
            std::sort(mPairwiseEuclid[i].begin(), mPairwiseEuclid[i].end(), PairwiseComparator);
            std::sort(mPairwisePtPl[i].begin(), mPairwisePtPl[i].end(), PairwiseComparator);
            std::sort(mPairwiseIPtPl[i].begin(), mPairwiseIPtPl[i].end(), PairwiseComparator);
            std::sort(mPairwiseHAngle[i].begin(), mPairwiseHAngle[i].end(), PairwiseComparator);
            std::sort(mPairwiseVAngle[i].begin(), mPairwiseVAngle[i].end(), PairwiseComparator);
        }
    }

    void EntangledForestFeatureExtraction::CalculatePairwiseFeaturesOctree(std::vector<Eigen::Vector3f> &centroids,
                                                            std::vector<Eigen::Vector3f> &normals,
                                                            std::vector<double> &verticalAngles)
    {
        mPairwiseEuclid.clear();
        mPairwiseEuclid.resize(mNrOfSegments, std::vector<std::pair<double, int> >(mNrOfSegments - 1));
        mPairwisePtPl.clear();
        mPairwisePtPl.resize(mNrOfSegments, std::vector<std::pair<double, int> >(mNrOfSegments - 1));
        mPairwiseIPtPl.clear();
        mPairwiseIPtPl.resize(mNrOfSegments, std::vector<std::pair<double, int> >(mNrOfSegments - 1));
        mPairwiseVAngle.clear();
        mPairwiseVAngle.resize(mNrOfSegments, std::vector<std::pair<double, int> >(mNrOfSegments - 1));
        mPairwiseHAngle.clear();
        mPairwiseHAngle.resize(mNrOfSegments, std::vector<std::pair<double, int> >(mNrOfSegments - 1));

        std::vector<pcl::octree::OctreePointCloudVoxelCentroid<pcl::PointXYZ>::AlignedPointTVector> vectorOfSeeds(
                mNrOfSegments);
        for (int s = 0; s < mNrOfSegments; ++s)
        {
            pcl::PointCloud<pcl::PointXYZ>::Ptr tmp(new pcl::PointCloud <pcl::PointXYZ>);
            pcl::copyPointCloud(*mSegments[s], *tmp);
            pcl::octree::OctreePointCloudVoxelCentroid <pcl::PointXYZ> octree(0.1);
            octree.setInputCloud(tmp);
            octree.addPointsFromInputCloud();
            octree.getVoxelCentroids(vectorOfSeeds[s]);
        }

        for (int segidx1 = 0; segidx1 < mNrOfSegments - 1; ++segidx1)
        {
//        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr seg1 = segments[segidx1];

            // for ptpl
            Eigen::Vector3f c1 = centroids[segidx1];
            Eigen::Vector3f n1 = normals[segidx1];

            // for angles
            double v1 = verticalAngles[segidx1];
            Eigen::Vector2d h1;
            h1 << n1[0], n1[2];

            for (int segidx2 = segidx1 + 1; segidx2 < mNrOfSegments; ++segidx2)
            {
//            pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr seg2 = segments[segidx2];

                // for ptpl
                Eigen::Vector3f c2 = centroids[segidx2];
                Eigen::Vector3f n2 = normals[segidx2];

                // for angles
                double v2 = verticalAngles[segidx2];
                Eigen::Vector2d h2;
                h2 << n2[0], n2[2];

                float minEDist = std::numeric_limits<float>::max();
                float minPtPlDist1 = std::numeric_limits<float>::max();
                float minPtPlDist2 = std::numeric_limits<float>::max();

                // TODO: This is going to be much slower now, since we compute distances of all POINTS and not
                // centroids of SVs any more!!!
                // calculate min distance between segment1 and segment2
                std::vector<int> eidx(1);
                std::vector<float> edist(1);

//            for(int ptidx2=0; ptidx2 < seg2->points.size(); ++ptidx2)
                for (unsigned int ptidx1 = 0; ptidx1 < vectorOfSeeds[segidx1].size(); ++ptidx1)
                {
                    Eigen::Vector3f pt1 = vectorOfSeeds[segidx1][ptidx1].getVector3fMap();

                    for (unsigned int ptidx2 = 0; ptidx2 < vectorOfSeeds[segidx2].size(); ++ptidx2)
                    {
                        Eigen::Vector3f pt2 = vectorOfSeeds[segidx2][ptidx2].getVector3fMap();
                        float dist = (pt1 - pt2).norm();

                        if (dist < minEDist)
                        {
                            minEDist = dist;
                        }

                        // also calc. ptpl
                        float d = std::abs(n1.dot(pt2 - c1));

                        if (d < minPtPlDist1)
                        {
                            minPtPlDist1 = d;
                        }
                    }

                    // also calc. ptpl
                    float d2 = std::abs(n2.dot(pt1 - c2));

                    if (d2 < minPtPlDist2)
                    {
                        minPtPlDist2 = d2;
                    }
                }



                // angles
                // to make sure argument for acos is inside defined range
                double hordiff = acos(std::max(-1.0, std::min(1.0, h2.dot(h1) / (h2.norm() * h1.norm()))));
                double verdiff = v2 - v1;

                mPairwiseEuclid[segidx1][segidx2 - 1].first = minEDist;
                mPairwiseEuclid[segidx1][segidx2 - 1].second = segidx2;
                mPairwiseEuclid[segidx2][segidx1].first = minEDist;
                mPairwiseEuclid[segidx2][segidx1].second = segidx1;

                mPairwisePtPl[segidx1][segidx2 - 1].first = minPtPlDist1;
                mPairwisePtPl[segidx1][segidx2 - 1].second = segidx2;
                mPairwisePtPl[segidx2][segidx1].first = minPtPlDist2;
                mPairwisePtPl[segidx2][segidx1].second = segidx1;

                mPairwiseIPtPl[segidx1][segidx2 - 1].first = minPtPlDist2;
                mPairwiseIPtPl[segidx1][segidx2 - 1].second = segidx2;
                mPairwiseIPtPl[segidx2][segidx1].first = minPtPlDist1;
                mPairwiseIPtPl[segidx2][segidx1].second = segidx1;

                mPairwiseHAngle[segidx1][segidx2 - 1].first = hordiff;
                mPairwiseHAngle[segidx1][segidx2 - 1].second = segidx2;
                mPairwiseHAngle[segidx2][segidx1].first = -hordiff;
                mPairwiseHAngle[segidx2][segidx1].second = segidx1;

                mPairwiseVAngle[segidx1][segidx2 - 1].first = verdiff;
                mPairwiseVAngle[segidx1][segidx2 - 1].second = segidx2;
                mPairwiseVAngle[segidx2][segidx1].first = -verdiff;
                mPairwiseVAngle[segidx2][segidx1].second = segidx1;
            }
        }

        for (int i = 0; i < mNrOfSegments; ++i)
        {
            std::sort(mPairwiseEuclid[i].begin(), mPairwiseEuclid[i].end(), PairwiseComparator);
            std::sort(mPairwisePtPl[i].begin(), mPairwisePtPl[i].end(), PairwiseComparator);
            std::sort(mPairwiseIPtPl[i].begin(), mPairwiseIPtPl[i].end(), PairwiseComparator);
            std::sort(mPairwiseHAngle[i].begin(), mPairwiseHAngle[i].end(), PairwiseComparator);
            std::sort(mPairwiseVAngle[i].begin(), mPairwiseVAngle[i].end(), PairwiseComparator);
        }
    }

    void EntangledForestFeatureExtraction::extract()
    {
        // first of all, rotate pointcloud such that groundplane is horizontal
        pcl::PointCloud <pcl::PointXYZRGB> rotated;
        pcl::transformPointCloud(*mInputCloud, rotated, mTransformationMatrix);

        // fill vector of segments with rotated pointcloud, each segment is a pointcloud on its own
        mSegments.clear();
        mSegmentNormals.clear();

        for (int i = 0; i < mNrOfSegments; ++i)
        {
            mSegments.push_back(pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud <pcl::PointXYZRGB>));
            mSegmentNormals.push_back(pcl::PointCloud<pcl::Normal>::Ptr(new pcl::PointCloud <pcl::Normal>));
        }

        for (unsigned int i = 0; i < mLabelCloud->points.size(); ++i)
        {
            int lbl = mLabelCloud->at(i).label;

            if (lbl >= 0 && lbl < mNrOfSegments)
            {
                mSegments[lbl]->push_back(rotated.at(i));

                pcl::Normal n = mNormalCloud->at(i);
                n.getNormalVector4fMap() = mTransformationMatrix * n.getNormalVector4fMap();
                mSegmentNormals[lbl]->push_back(n);
            }
        }

        // Define normal vector of the horizontal plane
        Eigen::Vector3f hor(0, -1, 0);
        EIGEN_ALIGN16
        Eigen::Matrix3f covarianceMatrix;
        EIGEN_ALIGN16
        Eigen::Vector3f eigenValues3D;
        EIGEN_ALIGN16
        Eigen::Matrix3f eigenVectors3D;
        EIGEN_ALIGN16
        Eigen::Vector2f eigenValues2D;
        EIGEN_ALIGN16
        Eigen::Matrix2f eigenVectors2D;
        Eigen::Vector4f xyzCentroid;

        mUnaries.assign(mNrOfSegments, std::vector<double>(18));

        // for point2plane distances
        std::vector<Eigen::Vector3f> centroids(mNrOfSegments);
        std::vector<Eigen::Vector3f> normals(mNrOfSegments);

        // for pairwise angle diff
        std::vector<double> verticalAngles(mNrOfSegments);

        for (int i = 0; i < mNrOfSegments; ++i)
        {
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr segment = mSegments[i];
            pcl::PointCloud<pcl::Normal>::Ptr seg_normals = mSegmentNormals[i];

            std::vector<double> &f = mUnaries[i];     // feature vector for current cluster
            int npoints = segment->points.size();

            // now calculate features for complete cluster
            float l, a, b;
            l = a = b = 0.0f;
            float minHeight = std::numeric_limits<float>::max();
            float maxHeight = -std::numeric_limits<float>::max();

            // POINTNESS, SURFACENESS, LINEARNESS, HEIGHT /////////
            pcl::computeMeanAndCovarianceMatrix(*segment, covarianceMatrix, xyzCentroid);
            pcl::eigen33(covarianceMatrix, eigenVectors3D, eigenValues3D);

            // get span in x-z plane
            Eigen::Matrix2f covarianceMatrix2D;
            covarianceMatrix2D << covarianceMatrix(0, 0), covarianceMatrix(0, 2), covarianceMatrix(2,
                                                                                                   0), covarianceMatrix(
                    2, 2);
            pcl::eigen22(covarianceMatrix2D, eigenVectors2D, eigenValues2D);
            // rotate about y axis
            Eigen::Matrix4f transformationMatrix2D;
            transformationMatrix2D << eigenVectors2D(0, 1), 0, eigenVectors2D(1, 1), 0, 0, 1, 0, 0, eigenVectors2D(0,
                                                                                                                   0), 0, eigenVectors2D(
                    1, 0), 0, 0, 0, 0, 1;
            pcl::PointCloud <pcl::PointXYZRGB> transformed;
            pcl::transformPointCloud(*segment, transformed, transformationMatrix2D);
            Eigen::Vector4f minpt, maxpt;
            pcl::getMinMax3D(transformed, minpt, maxpt);

            pcl::PointXYZ centroid;
            centroid.x = xyzCentroid[0];
            centroid.y = xyzCentroid[1];
            centroid.z = xyzCentroid[2];

            // for point to plane distance
            centroids[i] = xyzCentroid.head(3);
            normals[i] = eigenVectors3D.col(0);

            // ANGULAR DEVIATION, AVG LAB COLOR, STD OF LAB COLOR, MIN AND MAX HEIGHT

            // to calculate std deviation of color channels
            std::vector<std::vector<double> > labvalues(npoints, std::vector<double>(3, 0.0f));

            double angledevhorx = 0.0f;
            double angledevhory = 0.0f;
            double angledevx = 0.0f;
            double angledevy = 0.0f;

#pragma omp parallel for    \
    reduction(+:angledevhorx, angledevhory, angledevx, angledevy, l, a, b) \
    reduction(max:maxHeight) \
    reduction(min:minHeight)
            for (int j = 0; j < npoints; ++j)
            {
                pcl::PointXYZRGB &pt = segment->at(j);
                pcl::Normal &n = seg_normals->at(j);

                // angular deviation ////////////
                Eigen::Vector3f voxelnormal = n.getNormalVector3fMap();

                float dotphor = std::max(-1.0f, std::min(1.0f, voxelnormal.dot(
                        hor)));      // to get mean angle w.r.t. groundplane
                float devhor = acos(dotphor);
                angledevhorx += cos(devhor);
                angledevhory += sin(devhor);

                float dotp = std::max(-1.0f, std::min(1.0f, voxelnormal.dot(normals[i])));  // to get angular std. dev.
                float dev = acos(dotp);
                angledevx += cos(dev);
                angledevy += sin(dev);
                /////////////////////////////////

                // min and max height ///////////
                float h = mCameraHeight - pt.y;

                if (h > maxHeight)
                    maxHeight = h;
                if (h < minHeight)
                    minHeight = h;
                /////////////////////////////////

                // mean lab color and std deviation
                Eigen::Vector3f rgb;
                rgb << pt.r / 255.0f, pt.g / 255.0f, pt.b / 255.0f;
                Eigen::Vector3f lab = RGB2Lab(rgb);
                l += lab[0];
                a += lab[1];
                b += lab[2];

                labvalues[j][0] = lab[0];
                labvalues[j][1] = lab[1];
                labvalues[j][2] = lab[2];
                /////////////////////////////////
            }

            angledevhorx /= npoints;
            angledevhory /= npoints;

            angledevx /= npoints;
            angledevy /= npoints;

            double meanangle = atan2(angledevhory, angledevhorx);
            f[0] = meanangle;                                 // average angle between normal and normal of ground plane

            // for pairwise angle diff
            verticalAngles[i] = meanangle;

            double meananglevectorlength = sqrt(angledevx * angledevx + angledevy * angledevy);
            f[1] = sqrt(-2 * log(meananglevectorlength));       // deviation of normals from average normal

            float avgL = l / npoints;
            float avga = a / npoints;
            float avgb = b / npoints;

            f[2] = avgL;                               // average L value
            f[3] = avga;                               // average a value
            f[4] = avgb;                               // average b value

            // STD LAB DEVIATION ////////////////
            float sl = 0.0f, sa = 0.0f, sb = 0.0f;
            for (int j = 0; j < npoints; ++j)
            {
                sl += (labvalues[j][0] - avgL) * (labvalues[j][0] - avgL);
                sa += (labvalues[j][1] - avga) * (labvalues[j][1] - avga);
                sb += (labvalues[j][2] - avgb) * (labvalues[j][2] - avgb);
            }

            f[5] = sl / (npoints - 1);                        // L std dev
            f[6] = sa / (npoints - 1);                        // a std dev
            f[7] = sb / (npoints - 1);                        // b std dev
            /////////////////////////////////////

            f[8] = minHeight;                                 // minimum cluster height
            f[9] = maxHeight;                                 // maximum cluster height

            // new bounding box features
            Eigen::Vector4f bbox = maxpt - minpt;
            f[10] = bbox[0];                           // bbox width
            f[11] = bbox[1];                           // bbox height
            f[12] = bbox[2];                           // bbox depth

            f[13] = bbox[0] * bbox[1];                   // vertical plane area (e.g. wall)
            f[14] = bbox[0] * bbox[2];                   // horizontal plane area (e.g. table)

            f[15] = bbox[1] / bbox[0];                 // vertical elongation
            f[16] = bbox[2] / bbox[0];                 // horizontal elongation
            f[17] = bbox[1] / bbox[2];                 // "volumeness" how thick is the cluster

            // END OF FEATURE CALCULATION ////////////////////////////////////////////////

            mUnaries[i] = f;
        }

        CalculatePairwiseFeaturesOctree(centroids, normals, verticalAngles);
    }

    void EntangledForestFeatureExtraction::extractFromReconstruction(bool adjustHeights)
    {
        // fill vector of segments with rotated pointcloud, each segment is a pointcloud on its own
        mSegments.clear();
        mSegmentNormals.clear();

        for (int i = 0; i < mNrOfSegments; ++i)
        {
            mSegments.push_back(pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud <pcl::PointXYZRGB>));
            mSegmentNormals.push_back(pcl::PointCloud<pcl::Normal>::Ptr(new pcl::PointCloud <pcl::Normal>));
        }

        for (unsigned int i = 0; i < mLabelCloud->points.size(); ++i)
        {
            int lbl = mLabelCloud->at(i).label;

            if (lbl >= 0 && lbl < mNrOfSegments)
            {
                mSegments[lbl]->push_back(mInputCloud->at(i));
                mSegmentNormals[lbl]->push_back(mNormalCloud->at(i));
            }
        }

        // Define normal vector of the horizontal plane
        Eigen::Vector3f hor(0, 0, 1);
        EIGEN_ALIGN16
        Eigen::Matrix3f covarianceMatrix;
        EIGEN_ALIGN16
        Eigen::Vector3f eigenValues3D;
        EIGEN_ALIGN16
        Eigen::Matrix3f eigenVectors3D;
        EIGEN_ALIGN16
        Eigen::Vector2f eigenValues2D;
        EIGEN_ALIGN16
        Eigen::Matrix2f eigenVectors2D;
        Eigen::Vector4f xyzCentroid;

        mUnaries.assign(mNrOfSegments, std::vector<double>(18));

        // for point2plane distances
        std::vector<Eigen::Vector3f> centroids(mNrOfSegments);
        std::vector<Eigen::Vector3f> normals(mNrOfSegments);

        // for pairwise angle diff
        std::vector<double> verticalAngles(mNrOfSegments);

        float overallMinHeight = std::numeric_limits<float>::max();

        for (int i = 0; i < mNrOfSegments; ++i)
        {
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr segment = mSegments[i];
            pcl::PointCloud<pcl::Normal>::Ptr seg_normals = mSegmentNormals[i];

            std::vector<double> &f = mUnaries[i];     // feature vector for current cluster
            int npoints = segment->points.size();

            // now calculate features for complete cluster
            float l, a, b;
            l = a = b = 0.0f;
            float minHeight = std::numeric_limits<float>::max();
            float maxHeight = -std::numeric_limits<float>::max();

            // POINTNESS, SURFACENESS, LINEARNESS, HEIGHT /////////
            pcl::computeMeanAndCovarianceMatrix(*segment, covarianceMatrix, xyzCentroid);
            pcl::eigen33(covarianceMatrix, eigenVectors3D, eigenValues3D);

            // get span in x-y plane
            Eigen::Matrix2f covarianceMatrix2D;
            covarianceMatrix2D << covarianceMatrix.block(0, 0, 2, 2);
            pcl::eigen22(covarianceMatrix2D, eigenVectors2D, eigenValues2D);

            // rotate about z axis
            Eigen::Matrix4f transformationMatrix2D = Eigen::Matrix4f::Identity(4, 4);
            transformationMatrix2D.block(0, 0, 2, 2) << eigenVectors2D;

            pcl::PointCloud <pcl::PointXYZRGB> transformed;
            pcl::transformPointCloud(*segment, transformed, transformationMatrix2D);
            Eigen::Vector4f minpt, maxpt;
            pcl::getMinMax3D(transformed, minpt, maxpt);

            pcl::PointXYZ centroid;
            centroid.x = xyzCentroid[0];
            centroid.y = xyzCentroid[1];
            centroid.z = xyzCentroid[2];

            // for point to plane distance
            centroids[i] = xyzCentroid.head(3);
            normals[i] = eigenVectors3D.col(0);

            // ANGULAR DEVIATION, AVG LAB COLOR, STD OF LAB COLOR, MIN AND MAX HEIGHT

            // to calculate std deviation of color channels
            std::vector<std::vector<double> > labvalues(npoints, std::vector<double>(3, 0.0f));

            double angledevhorx = 0.0f;
            double angledevhory = 0.0f;
            double angledevx = 0.0f;
            double angledevy = 0.0f;

#pragma omp parallel for    \
    reduction(+:angledevhorx, angledevhory, angledevx, angledevy, l, a, b) \
    reduction(max:maxHeight) \
    reduction(min:minHeight)
            for (int j = 0; j < npoints; ++j)
            {
                pcl::PointXYZRGB &pt = segment->at(j);
                pcl::Normal &n = seg_normals->at(j);

                // angular deviation ////////////
                Eigen::Vector3f voxelnormal = n.getNormalVector3fMap();

                float dotphor = std::max(-1.0f, std::min(1.0f, voxelnormal.dot(
                        hor)));      // to get mean angle w.r.t. groundplane
                float devhor = acos(dotphor);
                angledevhorx += cos(devhor);
                angledevhory += sin(devhor);

                float dotp = std::max(-1.0f, std::min(1.0f, voxelnormal.dot(normals[i])));  // to get angular std. dev.
                float dev = acos(dotp);
                angledevx += cos(dev);
                angledevy += sin(dev);
                /////////////////////////////////

                // min and max height ///////////
                float h = pt.z;

                if (h > maxHeight)
                    maxHeight = h;
                if (h < minHeight)
                    minHeight = h;
                /////////////////////////////////

                // mean lab color and std deviation
                Eigen::Vector3f rgb;
                rgb << pt.r / 255.0f, pt.g / 255.0f, pt.b / 255.0f;
                Eigen::Vector3f lab = RGB2Lab(rgb);
                l += lab[0];
                a += lab[1];
                b += lab[2];

                labvalues[j][0] = lab[0];
                labvalues[j][1] = lab[1];
                labvalues[j][2] = lab[2];
                /////////////////////////////////
            }

            overallMinHeight = min(minHeight, overallMinHeight);

            angledevhorx /= npoints;
            angledevhory /= npoints;

            angledevx /= npoints;
            angledevy /= npoints;

            double meanangle = atan2(angledevhory, angledevhorx);
            f[0] = meanangle;                                 // average angle between normal and normal of ground plane

            // for pairwise angle diff
            verticalAngles[i] = meanangle;

            double meananglevectorlength = sqrt(angledevx * angledevx + angledevy * angledevy);
            f[1] = sqrt(-2 * log(meananglevectorlength));       // deviation of normals from average normal

            float avgL = l / npoints;
            float avga = a / npoints;
            float avgb = b / npoints;

            f[2] = avgL;                               // average L value
            f[3] = avga;                               // average a value
            f[4] = avgb;                               // average b value

            // STD LAB DEVIATION ////////////////
            float sl = 0.0f, sa = 0.0f, sb = 0.0f;
            for (int j = 0; j < npoints; ++j)
            {
                sl += (labvalues[j][0] - avgL) * (labvalues[j][0] - avgL);
                sa += (labvalues[j][1] - avga) * (labvalues[j][1] - avga);
                sb += (labvalues[j][2] - avgb) * (labvalues[j][2] - avgb);
            }

            f[5] = sl / (npoints - 1);                        // L std dev
            f[6] = sa / (npoints - 1);                        // a std dev
            f[7] = sb / (npoints - 1);                        // b std dev
            /////////////////////////////////////

            f[8] = minHeight;                                 // minimum cluster height
            f[9] = maxHeight;                                 // maximum cluster height

            // new bounding box features
            Eigen::Vector4f bbox = maxpt - minpt;
            f[10] = bbox[1];                           // bbox width
            f[11] = bbox[2];                           // bbox height
            f[12] = bbox[0];                           // bbox depth

            f[13] = bbox[1] * bbox[2];                   // vertical plane area (e.g. wall)
            f[14] = bbox[1] * bbox[0];                   // horizontal plane area (e.g. table)

            f[15] = bbox[2] / bbox[1];                 // vertical elongation
            f[16] = bbox[0] / bbox[1];                 // horizontal elongation
            f[17] = bbox[2] / bbox[0];                 // "volumeness" how thick is the cluster

            // END OF FEATURE CALCULATION ////////////////////////////////////////////////

            mUnaries[i] = f;
        }

        if (adjustHeights)
        {
            // update height features (only required if reconstruction has not been aligned w/ ground plane
            for (int i = 0; i < mNrOfSegments; ++i)
            {
                mUnaries[i][8] -= overallMinHeight;
                mUnaries[i][9] -= overallMinHeight;
            }
        }

        CalculatePairwiseFeaturesOctree(centroids, normals, verticalAngles);
    }

    void EntangledForestFeatureExtraction::getFeatures(std::vector<std::vector<double> > &unaries,
                                        std::vector<std::vector<std::pair<double, int> > > &ptpl,
                                        std::vector<std::vector<std::pair<double, int> > > &iptpl,
                                        std::vector<std::vector<std::pair<double, int> > > &verAngle,
                                        std::vector<std::vector<std::pair<double, int> > > &horAngle,
                                        std::vector<std::vector<std::pair<double, int> > > &euclid)
    {
        unaries = this->mUnaries;
        ptpl = this->mPairwisePtPl;
        iptpl = this->mPairwiseIPtPl;
        verAngle = this->mPairwiseVAngle;
        horAngle = this->mPairwiseHAngle;
        euclid = this->mPairwiseEuclid;
    }

    void EntangledForestFeatureExtraction::prepareClassification(EntangledForestData *d)
    {
        d->LoadTestDataLive(mUnaries, mPairwisePtPl, mPairwiseIPtPl, mPairwiseVAngle, mPairwiseHAngle, mPairwiseEuclid);
    }
}