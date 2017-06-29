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

#include <pcl/features/integral_image_normal.h>
#include <pcl/pcl_config.h>

#include <v4r/semantic_segmentation/supervoxel_segmentation.h>

using namespace pcl;

namespace v4r
{
    template<typename PointInT>
    bool SupervoxelSegmentation<PointInT>::PairwiseComparator(const std::pair<double, int> &l, const std::pair<double, int> r)
    { return l.first < r.first; }

    template<typename PointInT>
    SupervoxelSegmentation<PointInT>::SupervoxelSegmentation()
    {
        mImageWidth = 0;
        mImageHeight = 0;

        mSVVoxelResolution = 0.01;
        mSVSeedResolution = 0.1;
        mSVColorImportance = 0.6;
        mSVSpatialImportance = 0.1;
        mSVNormalImportance = 1.0;

        mMergeThreshold = 0.06;
        mMergeColorWeight = 0.6;
        mMergeNormalWeight = 0.7;
        mMergeCurvatureWeight = 1.0;
        mMergePtPlWeight = 1.0;

        mColorTransformation << 0.412453, 0.357580, 0.180423,
                0.212671, 0.715160, 0.072169,
                0.019334, 0.119193, 0.950227;
    }

    template<typename PointInT>
    SupervoxelSegmentation<PointInT>::SupervoxelSegmentation(float sv_voxelResolution, float sv_seedResolution,
                                                             float sv_colorWeight, float sv_normalWeight,
                                                             float sv_spatialWeight, float merge_threshold,
                                                             float merge_colorWeight, float merge_normalWeight,
                                                             float merge_curvatureWeight, float merge_pointPlaneWeight,
                                                             bool useCIE94distance, bool useSingleCameraTransform)
    {
        mImageWidth = 0;
        mImageHeight = 0;

        mSVVoxelResolution = sv_voxelResolution;
        mSVSeedResolution = sv_seedResolution;
        mSVColorImportance = sv_colorWeight;
        mSVSpatialImportance = sv_spatialWeight;
        mSVNormalImportance = sv_normalWeight;

        mMergeThreshold = merge_threshold;
        mMergeColorWeight = merge_colorWeight;
        mMergeNormalWeight = merge_normalWeight;
        mMergeCurvatureWeight = merge_curvatureWeight;
        mMergePtPlWeight = merge_pointPlaneWeight;

        mUseCIE94 = useCIE94distance;

        this->mUseSingleCameraTransform = useSingleCameraTransform;

        mColorTransformation << 0.412453, 0.357580, 0.180423,
                0.212671, 0.715160, 0.072169,
                0.019334, 0.119193, 0.950227;
    }


    template<typename PointInT>
    int SupervoxelSegmentation<PointInT>::RunSegmentation(typename pcl::PointCloud<PointInT>::ConstPtr input,
                                                          pcl::PointCloud<pcl::Normal>::ConstPtr normals)
    {
        RunSupervoxelClustering(input, normals);
        mMergedClusters = MergeSupervoxels();

        return mMergedClusters;
    }

    template<typename PointInT>
    void SupervoxelSegmentation<PointInT>::GetClusterIDArray(cv::Mat &clusterIDs)
    {
        clusterIDs.create(mImageHeight, mImageWidth, CV_32F);

        for (int y = 0; y < mImageHeight; ++y)
        {
            for (int x = 0; x < mImageWidth; ++x)
            {
                int svlbl = mSVLabeledCloud->at(x, y).label;

                // return -1 if point is not assigned to any cluster
                clusterIDs.at<float>(y, x) = (svlbl > 0) ? (float) mSVToCluster[svlbl] : -1;
            }
        }
    }

    template<typename PointInT>
    void SupervoxelSegmentation<PointInT>::CalculatePairwiseAngleDistances(std::vector<Eigen::Vector3f> &normalVectors,
                                                                           std::vector<double> &verticalAngles,
                                                                           std::vector<std::vector<std::pair<double, int> > > &sortedVerAngleDifferences,
                                                                           std::vector<std::vector<std::pair<double, int> > > &sortedHorAngleDifferences)
    {
        sortedVerAngleDifferences.resize(mMergedClusters, std::vector<std::pair<double, int> >(mMergedClusters - 1));
        sortedHorAngleDifferences.resize(mMergedClusters, std::vector<std::pair<double, int> >(mMergedClusters - 1));

        for (int i = 0; i < mMergedClusters - 1; ++i)
        {
            double v1 = verticalAngles[i];
            Eigen::Vector2d n1;
            n1 << normalVectors[i][0], normalVectors[i][2];

            for (int j = i + 1; j < mMergedClusters; ++j)
            {
                double v2 = verticalAngles[j];
                Eigen::Vector2d n2;
                n2 << normalVectors[j][0], normalVectors[j][2];

                // to make sure argument for acos is inside defined range
                double hordiff = acos(std::max(-1.0, std::min(1.0, n2.dot(n1) / (n2.norm() * n1.norm()))));
                double verdiff = v2 - v1;

                sortedHorAngleDifferences[i][j - 1].first = hordiff;
                sortedHorAngleDifferences[i][j - 1].second = j;
                sortedHorAngleDifferences[j][i].first = -hordiff;
                sortedHorAngleDifferences[j][i].second = i;

                sortedVerAngleDifferences[i][j - 1].first = verdiff;
                sortedVerAngleDifferences[i][j - 1].second = j;
                sortedVerAngleDifferences[j][i].first = -verdiff;
                sortedVerAngleDifferences[j][i].second = i;
            }
        }

        for (int i = 0; i < mMergedClusters; ++i)
        {
            std::sort(sortedHorAngleDifferences[i].begin(), sortedHorAngleDifferences[i].end(), PairwiseComparator);
            std::sort(sortedVerAngleDifferences[i].begin(), sortedVerAngleDifferences[i].end(), PairwiseComparator);
        }
    }

    template<typename PointInT>
    void SupervoxelSegmentation<PointInT>::CalculatePoint2PlaneDistances(std::vector<Eigen::Vector3f> &centroids,
                                                                         std::vector<Eigen::Vector3f> &normals,
                                                                         std::vector<std::vector<std::pair<double, int> > > &sortedPointPlaneDistances,
                                                                         std::vector<std::vector<std::pair<double, int> > > &sortedInversePointPlaneDistances)
    {
        sortedPointPlaneDistances.resize(mMergedClusters, std::vector<std::pair<double, int> >(mMergedClusters - 1));
        sortedInversePointPlaneDistances.resize(mMergedClusters,
                                                std::vector<std::pair<double, int> >(mMergedClusters - 1));

        for (int i = 0; i < mMergedClusters - 1; ++i)
        {
            Eigen::Vector3f c1 = centroids[i];
            Eigen::Vector3f n1 = normals[i];

            for (int j = i + 1; j < mMergedClusters; ++j)
            {
                Eigen::Vector3f c2 = centroids[j];
                Eigen::Vector3f n2 = normals[j];

                float mindist1 = std::numeric_limits<float>::max();

                // loop through all SVs of cluster 2 and get closest pt2pl distance
                for (int s = 0; s < mClusterToSv[j].size(); ++s)
                {
                    Eigen::Vector3f svcentroid = mSVClusters[(unsigned int) (mClusterToSv[j][s])]->centroid_.getVector3fMap();
                    float d = n1.dot(svcentroid - c1);

                    if (std::abs(d) < std::abs(mindist1))
                    {
                        mindist1 = d;
                    }
                }

                float mindist2 = std::numeric_limits<float>::max();

                // loop through all SVs of cluster 1 and get closest pt2pl distance
                for (int s = 0; s < mClusterToSv[i].size(); ++s)
                {
                    Eigen::Vector3f svcentroid = mSVClusters[(unsigned int) (mClusterToSv[i][s])]->centroid_.getVector3fMap();
                    float d = n2.dot(svcentroid - c2);

                    if (std::abs(d) < std::abs(mindist2))
                    {
                        mindist2 = d;
                    }
                }

                sortedPointPlaneDistances[i][j - 1].first = mindist1;
                sortedPointPlaneDistances[i][j - 1].second = j;
                sortedPointPlaneDistances[j][i].first = mindist2;
                sortedPointPlaneDistances[j][i].second = i;

                sortedInversePointPlaneDistances[i][j - 1].first = mindist2;
                sortedInversePointPlaneDistances[i][j - 1].second = j;
                sortedInversePointPlaneDistances[j][i].first = mindist1;
                sortedInversePointPlaneDistances[j][i].second = i;
            }
        }

        for (int i = 0; i < mMergedClusters; ++i)
        {
            std::sort(sortedPointPlaneDistances[i].begin(), sortedPointPlaneDistances[i].end(), PairwiseComparator);
            std::sort(sortedInversePointPlaneDistances[i].begin(), sortedInversePointPlaneDistances[i].end(),
                      PairwiseComparator);
        }
    }

    template<typename PointInT>
    void SupervoxelSegmentation<PointInT>::CalculateFeatures(float height, float roll, float pitch,
                                                             std::vector<std::vector<double> > &features,
                                                             std::vector<std::vector<std::pair<double, int> > > &sortedEuclideanDistances,
                                                             std::vector<std::vector<std::pair<double, int> > > &sortedPointPlaneDistances,
                                                             std::vector<std::vector<std::pair<double, int> > > &sortedInversePointPlaneDistances,
                                                             std::vector<std::vector<std::pair<double, int> > > &sortedVerAngleDifferences,
                                                             std::vector<std::vector<std::pair<double, int> > > &sortedHorAngleDifferences)
    {
        // bugfix?
        pitch = -pitch;
        //roll = -roll;

        Eigen::Matrix3f p;
        p << 1, 0, 0,
                0, cos(pitch), -sin(pitch),
                0, sin(pitch), cos(pitch);

        Eigen::Matrix3f r;
        r << cos(roll), -sin(roll), 0,
                sin(roll), cos(roll), 0,
                0, 0, 1;

        Eigen::Matrix3f R = p * r;
        Eigen::Vector3f T;
        T << 0, -height, 0;

        Eigen::Matrix4f transformationMatrix;

        transformationMatrix << R(0, 0), R(0, 1), R(0, 2), T(0),
                R(1, 0), R(1, 1), R(1, 2), T(1),
                R(2, 0), R(2, 1), R(2, 2), T(2),
                0, 0, 0, 1;


        Eigen::Matrix3f rotationMatrix = R;

        // Define normal vector of the horizontal plane
        Eigen::Vector3f hor(0, -1, 0);
        EIGEN_ALIGN16 Eigen::Matrix3f covarianceMatrix;
        EIGEN_ALIGN16 Eigen::Vector3f eigenValues;
        EIGEN_ALIGN16 Eigen::Matrix3f eigenVectors;
        EIGEN_ALIGN16 Eigen::Vector2f eigenValues2D;
        EIGEN_ALIGN16 Eigen::Matrix2f eigenVectors2D;
        Eigen::Vector4f xyzCentroid;

        features.clear();

        // for point2plane distances
        std::vector<Eigen::Vector3f> centroids(mMergedClusters);
        std::vector<Eigen::Vector3f> normals(mMergedClusters);
        // for pairwise angle diff
        std::vector<double> verticalAngles(mMergedClusters);
        ////////////////////////////

        for (int i = 0; i < mMergedClusters; ++i)
        {
            std::vector<double> f(14, 0.0f);      // feature vector for current cluster
            PointCloud<PointInT> pc;
            PointCloud<Normal> pn;

            // create pointcloud of complete cluster
            for (int j = 0; j < mClusterToSv[i].size(); ++j)
            {
                pc += *(mSVClusters[mClusterToSv[i][j]]->voxels_);
                pn += *(mSVClusters[mClusterToSv[i][j]]->normals_);
            }

            int npoints = pc.points.size();

            // now calculate features for complete cluster
            float l, a, b;
            l = a = b = 0.0f;
            float minHeight = __FLT_MAX__;
            float maxHeight = -__FLT_MAX__;

            // POINTNESS, SURFACENESS, LINEARNESS, HEIGHT /////////
            pcl::computeMeanAndCovarianceMatrix(pc, covarianceMatrix, xyzCentroid);
            pcl::eigen33(covarianceMatrix, eigenVectors, eigenValues);

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
            pcl::PointCloud<PointInT> transformed;
            pcl::transformPointCloud(pc, transformed, transformationMatrix2D);
            Eigen::Vector4f minpt, maxpt;
            pcl::getMinMax3D(transformed, minpt, maxpt);


            PointXYZ centroid;
            centroid.x = xyzCentroid[0];
            centroid.y = xyzCentroid[1];
            centroid.z = xyzCentroid[2];

            // for point to plane distance
            centroids[i] = xyzCentroid.head(3);
            normals[i] = eigenVectors.col(0);

            // simple geometric features
            f[0] = eigenValues[0];                   // compactness
            f[1] = eigenValues[2] - eigenValues[1];    // elongation
            f[2] = eigenValues[1] - eigenValues[0];    // planarity

            // average height (inverted y-coordinate of supervoxel centroid, transformed to world coordinates)
            Eigen::Vector4f pWH = transformationMatrix * xyzCentroid;
            f[3] = -pWH(1);                          // avg. height

            // AVERAGE NORMAL //////////
            Eigen::Vector3f svnormal = eigenVectors.col(0); // eigenvector with smalles eigenvalue = normal

            // rotate by camera angle
            Eigen::Vector3f normalVectorW = rotationMatrix * svnormal;
            // Flip normal such that it points towards camera
            flipNormalTowardsViewpoint(centroid, 0, -height, 0, normalVectorW);
            float dotp1 = std::max(-1.0f, std::min(1.0f, normalVectorW.dot(hor)));
            f[4] = acos(dotp1 / normalVectorW.norm());  // average normal

            // for pairwise angle diff
            verticalAngles[i] = f[4];
            ////////////////////////////

            // ANGULAR DEVIATION, AVG LAB COLOR, STD OF LAB COLOR, MIN AND MAX HEIGHT

            // to calculate std deviation of color channels
            std::vector<std::vector<double> > labvalues(npoints);
            for (int j = 0; j < npoints; ++j)
                labvalues[j].resize(3, 0.0f);

            double angledevx = 0.0f;
            double angledevy = 0.0f;

            for (int j = 0; j < npoints; ++j)
            {
                // angular deviation ////////////
                Eigen::Vector3f voxelnormal;
                voxelnormal << pn.points[j].normal_x, pn.points[j].normal_y, pn.points[j].normal_z;

//            if(voxelnormal.dot(xyzCentroid.head(3)) > 0)
//                voxelnormal = -voxelnormal;

                float dotp = std::max(-1.0f, std::min(1.0f, voxelnormal.dot(svnormal)));
                float dev = acos(dotp);
                angledevx += cos(dev);
                angledevy += sin(dev);
                /////////////////////////////////

                // min and max height ///////////
                Eigen::Vector4f pointKinect;
                pointKinect << pc.points[j].x, pc.points[j].y, pc.points[j].z, 1;
                Eigen::Vector4f pointWorld = transformationMatrix * pointKinect;
                float h = -pointWorld(1);

                if (h > maxHeight)
                    maxHeight = h;
                if (h < minHeight)
                    minHeight = h;
                /////////////////////////////////

                // mean lab color and std deviation
                Eigen::Vector3f rgb;
                PointInT pt = pc.points[j];
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

            //double ht = maxpt[1] - minpt[1];
            //double ht2 = maxHeight - minHeight;

            angledevx /= npoints;
            angledevy /= npoints;

            double meananglevectorlength = sqrt(angledevx * angledevx + angledevy * angledevy);
            f[5] = sqrt(-2 * log(meananglevectorlength));       // normal deviation

            f[6] = minHeight;                                 // minimum cluster height
            f[7] = maxHeight;                                 // maximum cluster height

            float avgL = l / npoints;
            float avga = a / npoints;
            float avgb = b / npoints;

            f[8] = avgL;                               // average L value
            f[9] = avga;                               // average a value
            f[10] = avgb;                              // average b value

            // STD LAB DEVIATION ////////////////
            float sl = 0.0f, sa = 0.0f, sb = 0.0f;
            for (int j = 0; j < npoints; ++j)
            {
                sl += (labvalues[j][0] - avgL) * (labvalues[j][0] - avgL);
                sa += (labvalues[j][1] - avga) * (labvalues[j][1] - avga);
                sb += (labvalues[j][2] - avgb) * (labvalues[j][2] - avgb);
            }

            f[11] = sl / (npoints - 1);                        // L std dev
            f[12] = sa / (npoints - 1);                        // a std dev
            f[13] = sb / (npoints - 1);                        // b std dev
            /////////////////////////////////////
            // END OF FEATURE CALCULATION ////////////////////////////////////////////////

            features.push_back(f);
        }

        GetEuclideanClusterDistance(sortedEuclideanDistances);
        CalculatePoint2PlaneDistances(centroids, normals, sortedPointPlaneDistances, sortedInversePointPlaneDistances);
        CalculatePairwiseAngleDistances(normals, verticalAngles, sortedVerAngleDifferences, sortedHorAngleDifferences);
    }


    template<typename PointInT>
    void SupervoxelSegmentation<PointInT>::CalculateFeatures2(float height, std::vector<std::vector<double> > &features,
                                                              std::vector<std::vector<std::pair<double, int> > > &sortedEuclideanDistances,
                                                              std::vector<std::vector<std::pair<double, int> > > &sortedPointPlaneDistances,
                                                              std::vector<std::vector<std::pair<double, int> > > &sortedInversePointPlaneDistances,
                                                              std::vector<std::vector<std::pair<double, int> > > &sortedVerAngleDifferences,
                                                              std::vector<std::vector<std::pair<double, int> > > &sortedHorAngleDifferences)
    {
        // Define normal vector of the horizontal plane
        Eigen::Vector3f hor(0, -1, 0);
        EIGEN_ALIGN16 Eigen::Matrix3f covarianceMatrix;
        EIGEN_ALIGN16 Eigen::Vector3f eigenValues3D;
        EIGEN_ALIGN16 Eigen::Matrix3f eigenVectors3D;
        EIGEN_ALIGN16 Eigen::Vector2f eigenValues2D;
        EIGEN_ALIGN16 Eigen::Matrix2f eigenVectors2D;
        Eigen::Vector4f xyzCentroid;

        features.assign(mMergedClusters, std::vector<double>(18));

        // for point2plane distances
        std::vector<Eigen::Vector3f> centroids(mMergedClusters);
        std::vector<Eigen::Vector3f> normals(mMergedClusters);
        // for pairwise angle diff
        std::vector<double> verticalAngles(mMergedClusters);
        ////////////////////////////

        for (int i = 0; i < mMergedClusters; ++i)
        {
            std::vector<double> &f = features[i]; //(18, 0.0f);      // feature vector for current cluster
            PointCloud<PointInT> p;
            PointCloud<Normal> pn;

            // create pointcloud of complete cluster
            for (int j = 0; j < mClusterToSv[i].size(); ++j)
            {
                p += *(mSVClusters[mClusterToSv[i][j]]->voxels_);
                pn += *(mSVClusters[mClusterToSv[i][j]]->normals_);
            }

            int npoints = p.points.size();

            // now calculate features for complete cluster
            float l, a, b;
            l = a = b = 0.0f;
            float minHeight = __FLT_MAX__;
            float maxHeight = -__FLT_MAX__;

            // POINTNESS, SURFACENESS, LINEARNESS, HEIGHT /////////
            pcl::computeMeanAndCovarianceMatrix(p, covarianceMatrix, xyzCentroid);
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
            pcl::PointCloud<PointInT> transformed;
            pcl::transformPointCloud(p, transformed, transformationMatrix2D);
            Eigen::Vector4f minpt, maxpt;
            pcl::getMinMax3D(transformed, minpt, maxpt);

            PointXYZ centroid;
            centroid.x = xyzCentroid[0];
            centroid.y = xyzCentroid[1];
            centroid.z = xyzCentroid[2];

            // for point to plane distance
            centroids[i] = xyzCentroid.head(3);
            normals[i] = eigenVectors3D.col(0);

            // simple geometric features
//        f[0] = eigenValues3D[0];                   // compactness
//        f[1] = eigenValues3D[2]-eigenValues3D[1];    // elongation
//        f[2] = eigenValues3D[1]-eigenValues3D[0];    // planarity

            // average height (inverted y-coordinate of supervoxel centroid, transformed to world coordinates)
//        f[3] = height-xyzCentroid(1);                          // avg. height

            // AVERAGE NORMAL //////////
//        Eigen::Vector3f svnormal = eigenVectors.col(0); // eigenvector with smalles eigenvalue = normal

//        flipNormalTowardsViewpoint(centroid, 0, 0, 0, svnormal);
//        float dotp1 = std::max(-1.0f, std::min(1.0f, svnormal.dot(hor)));
//        f[4] = acos(dotp1 / svnormal.norm());  // average normal
            ////////////////////////////

            // ANGULAR DEVIATION, AVG LAB COLOR, STD OF LAB COLOR, MIN AND MAX HEIGHT

            // to calculate std deviation of color channels
            std::vector<std::vector<double> > labvalues(npoints, std::vector<double>(3, 0.0f));

            double angledevhorx = 0.0f;
            double angledevhory = 0.0f;
            double angledevx = 0.0f;
            double angledevy = 0.0f;

            for (int j = 0; j < npoints; ++j)
            {
                // angular deviation ////////////
                Eigen::Vector3f voxelnormal;
                voxelnormal << pn.points[j].normal_x, pn.points[j].normal_y, pn.points[j].normal_z;

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
                float h = height - p.points[j].y;

                if (h > maxHeight)
                    maxHeight = h;
                if (h < minHeight)
                    minHeight = h;
                /////////////////////////////////

                // mean lab color and std deviation
                Eigen::Vector3f rgb;
                PointInT pt = p.points[j];
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

//        f[8] = maxHeight - minHeight;

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

            features[i] = f;
        }
        GetEuclideanClusterDistance(sortedEuclideanDistances);
        CalculatePoint2PlaneDistances(centroids, normals, sortedPointPlaneDistances, sortedInversePointPlaneDistances);
        CalculatePairwiseAngleDistances(normals, verticalAngles, sortedVerAngleDifferences, sortedHorAngleDifferences);
    }


    template<typename PointInT>
    void SupervoxelSegmentation<PointInT>::GetLabels(PointCloud<PointXYZL>::Ptr labelCloud, cv::Mat clusterIDArray,
                                                     std::vector<int> &clusterLabels)
    {
        clusterLabels.clear();

        // find out max label nr to initialize histograms
        int maxLbl = -1;
        for (int i = 0; i < labelCloud->points.size(); ++i)
        {
            int lbl = labelCloud->at(i).label;
            if (lbl > maxLbl)
                maxLbl = lbl;
        }

        // initialize histograms
        std::vector<std::vector<int> > labelHist;
        for (int i = 0; i < mMergedClusters; ++i)
        {
            labelHist.push_back(std::vector<int>(maxLbl, 0));
        }

        // sweep through image and create histogram for each label
        for (int y = 0; y < clusterIDArray.rows; ++y)
        {
            for (int x = 0; x < clusterIDArray.cols; ++x)
            {
                int lbl = labelCloud->at(x, y).label;
                float clusterID = clusterIDArray.at<float>(y, x);

                if (clusterID >= 0 && lbl > 0)
                {
                    labelHist[(int) clusterID][lbl - 1]++;
                }
            }
        }

        // go through histograms and assign largest label to each cluster, if not too ambiguous
        for (int i = 0; i < mMergedClusters; ++i)
        {
            // find 2 most frequent labels
            std::vector<int> hist = labelHist[i];
            std::sort(hist.begin(), hist.end());

            int first = hist[hist.size() - 1];
            int second = hist[hist.size() - 2];

            int maxlbl = 0;

            // 2nd most label has less than 50% of points of first label -> good cluster
            if (first > 0 && ((double) second) / ((double) first) < 0.5)
            {
                maxlbl = std::distance(labelHist[i].begin(),
                                       std::max_element(labelHist[i].begin(), labelHist[i].end())) + 1;
            }

            clusterLabels.push_back(maxlbl);
        }
    }

    template<typename PointInT>
    void SupervoxelSegmentation<PointInT>::GetLabelsForClusters(PointCloud<PointXYZL>::Ptr labelCloud,
                                                                std::vector<int> &clusterLabels)
    {
        clusterLabels.clear();

        // find out max label nr to initialize histograms
        int maxLbl = -1;
        for (int i = 0; i < labelCloud->points.size(); ++i)
        {
            int lbl = labelCloud->at(i).label;
            if (lbl > maxLbl)
                maxLbl = lbl;
        }

        // initialize histograms
        std::vector<std::vector<int> > labelHist;
        for (int i = 0; i < mMergedClusters; ++i)
        {
            labelHist.push_back(std::vector<int>(maxLbl + 1, 0));     // +1 to also include "unlabeled"
        }

        // sweep through image and create histogram for each label
        for (int p = 0; p < labelCloud->points.size(); ++p)
        {
            unsigned int svlbl = mSVLabeledCloud->at(p).label;

            if (mSVToCluster.find(svlbl) != mSVToCluster.end())
            {
                unsigned int clusterID = mSVToCluster[svlbl];
                unsigned int lbl = labelCloud->at(p).label;

                if (clusterID >= 0) // && lbl > 0)       // TODO: Not necessary any more because unsigned!!!
                {
                    labelHist[clusterID][lbl]++;    // also add unlabeled to the histogram
                }
            }
        }

        // go through histograms and assign largest label to each cluster, if not too ambiguous
        for (int i = 0; i < mMergedClusters; ++i)
        {
            // find 2 most frequent labels
            std::vector<int> hist = labelHist[i];
            std::sort(hist.begin(), hist.end());

            int first = hist[hist.size() - 1];
            int second = hist[hist.size() - 2];

            int maxlbl = 0;

            // 2nd most label has less than 20% of points of first label -> good cluster
            if (first > 0 && ((double) second) / ((double) first) < 0.5)//0.2)
            {
                maxlbl = std::distance(labelHist[i].begin(),
                                       std::max_element(labelHist[i].begin(), labelHist[i].end())); // + 1;
            }

            clusterLabels.push_back(maxlbl);
        }
    }

    template<typename PointInT>
    void SupervoxelSegmentation<PointInT>::RunSupervoxelClustering(typename pcl::PointCloud<PointInT>::ConstPtr input,
                                                                   pcl::PointCloud<pcl::Normal>::ConstPtr normals)
    {
        // set up supervoxel clustering

        //function only available from 1.8.0 on
#if PCL_VERSION_COMPARE(<, 1, 8, 0)
        pcl::SupervoxelClustering<PointInT> super (mSVVoxelResolution, mSVSeedResolution, mUseSingleCameraTransform);
#else
        pcl::SupervoxelClustering<PointInT> super(mSVVoxelResolution, mSVSeedResolution);
        super.setUseSingleCameraTransform(mUseSingleCameraTransform);
#endif

        super.setInputCloud(input);
        super.setColorImportance(mSVColorImportance);
        super.setSpatialImportance(mSVSpatialImportance);
        super.setNormalImportance(mSVNormalImportance);

        if (normals)
        {
            super.setNormalCloud(normals);
        }

        mSVClusters.clear();

        // Run supervoxel clustering
        super.extract(mSVClusters);
        super.refineSupervoxels(1, mSVClusters);

        mSVLabeledCloud = super.getLabeledCloud();
        mSVLabeledVoxelCloud = super.getLabeledVoxelCloud();
        mSVColoredCloud = super.getColoredCloud();
        mSVColoredVoxelCloud = super.getColoredVoxelCloud();
        mSVVoxelCentroidCloud = super.getVoxelCentroidCloud();
        super.getSupervoxelAdjacency(mSVAdjacency);

        mImageWidth = input->width;
        mImageHeight = input->height;
    }

    template<typename PointInT>
    bool SupervoxelSegmentation<PointInT>::sortDistances(std::vector<double> i, std::vector<double> j)
    { return (i[0] < j[0]); }

    template<typename PointInT>
    int SupervoxelSegmentation<PointInT>::MergeSupervoxels()
    {
        // Save cluster pair distances as [distance, index1, index2]
        std::vector<std::vector<double> > distances;

        // calc distances between adjacent supervoxels
        for (std::multimap<uint32_t, uint32_t>::iterator label_itr = mSVAdjacency.begin();
             label_itr != mSVAdjacency.end(); label_itr++)
        {
            // get labels of adjacent sv's
            uint32_t supervoxel_label = label_itr->first;
            uint32_t neighbor_label = label_itr->second;

            // pair has already been considered
            if (neighbor_label < supervoxel_label)
                continue;

            //Now get the supervoxels corresponding to the labels
            typename pcl::Supervoxel<PointInT>::Ptr supervoxel = mSVClusters.at(supervoxel_label);
            typename pcl::Supervoxel<PointInT>::Ptr neighbor_supervoxel = mSVClusters.at(neighbor_label);

            // and calculate the distance between them
            double d = CalculateDistance(supervoxel, neighbor_supervoxel);
//        bool tomerge = Merge2Clusters(supervoxel,neighbor_supervoxel );
            std::vector<double> x(3);
            x[0] = d; // tomerge ? 0.0 : std::numeric_limits<double>::max();//d;
            x[1] = supervoxel_label;
            x[2] = neighbor_label;

            distances.push_back(x);
        }

        // sort by distance
        std::sort(distances.begin(), distances.end(), sortDistances);

        // cluster ids assigned
        unsigned int cid = 0;

        mSVToCluster.clear();

        // merge closest supervoxels until thresh is reached
        for (std::vector<std::vector<double> >::iterator it = distances.begin(); it != distances.end(); ++it)
        {
            if ((*it)[0] > mMergeThreshold)
                break;

            unsigned int sv1 = (*it)[1];
            unsigned int sv2 = (*it)[2];

            unsigned int cid1 = 0;
            unsigned int cid2 = 0;

            if (mSVToCluster.find(sv1) != mSVToCluster.end())
                cid1 = mSVToCluster[sv1];
            if (mSVToCluster.find(sv2) != mSVToCluster.end())
                cid2 = mSVToCluster[sv2];

            // both SV are not assigned to a cluster yet -> create new one consisting of both SV
            if (cid1 == 0 && cid2 == 0)
            {
                cid++;
                mSVToCluster[sv1] = cid;
                mSVToCluster[sv2] = cid;
            }
                // SV 1 does not have a cluster yet -> assign to SV 1 to cluster of SV 2
            else if (cid1 == 0)
            {
                mSVToCluster[sv1] = cid2;
            }
                // SV 2 does not have a cluster yet -> assign to SV 2 to cluster of SV 1
            else if (cid2 == 0)
            {
                mSVToCluster[sv2] = cid1;
            }
                // both SV already have a cluster -> smaller cluster number wins and is assigned to SV2
            else if (cid1 < cid2)
            {
                // all other SV of cluster of SV2 have to change cluster ID as well!
                ChangeClusterLabel(cid2, cid1);
            }
                // both SV already have a cluster -> smaller cluster number wins and is assigned to SV1
            else if (cid1 > cid2)
            {
                // all other SV of cluster of SV1 have to change cluster ID as well!
                ChangeClusterLabel(cid1, cid2);
            }
        }

        // assign free cluster numbers to all SV which have not been merged with other SV
        for (typename std::map<uint32_t, typename pcl::Supervoxel<PointInT>::Ptr>::iterator it = mSVClusters.begin();
             it != mSVClusters.end(); ++it)
        {
            int npoints = (*it).second->voxels_->points.size();

            if (mSVToCluster.find(it->first) == mSVToCluster.end() && npoints > 3)
                mSVToCluster[it->first] = ++cid;
        }

        // create subsequent cluster numbers starting with 0
        std::map<unsigned int, unsigned int> clustermap;
        unsigned int uniquecnt = 0;
//    mClusterToSv.clear();

//    for(int i=0; i < mSVToCluster.size(); ++i)
//    {
//        mClusterToSv.push_back(std::vector<int>());
//    }

        for (std::map<unsigned int, unsigned int>::iterator it = mSVToCluster.begin(); it != mSVToCluster.end(); ++it)
        {
            if (clustermap.find(it->second) == clustermap.end())
            {
                clustermap[it->second] = uniquecnt++;
            }
        }

        mClusterToSv.resize(uniquecnt);
        for (std::map<unsigned int, unsigned int>::iterator it = mSVToCluster.begin(); it != mSVToCluster.end(); ++it)
        {
            it->second = clustermap[it->second];
            mClusterToSv[it->second].push_back(it->first);
        }

        return uniquecnt;
    }

    template<typename PointInT>
    void SupervoxelSegmentation<PointInT>::GetEuclideanClusterDistance(
            std::vector<std::vector<std::pair<double, int> > > &sortedEuclideanDistances)
    {
        sortedEuclideanDistances.resize(mMergedClusters, std::vector<std::pair<double, int> >(mMergedClusters - 1));

        for (int cluster1 = 0; cluster1 < mClusterToSv.size() - 1; ++cluster1)
        {
            for (int cluster2 = cluster1 + 1; cluster2 < mClusterToSv.size(); ++cluster2)
            {
                float minDist = std::numeric_limits<float>::max();

                // calculate min distance between cluster1 and cluster2
                for (int svidx1 = 0; svidx1 < mClusterToSv[cluster1].size(); ++svidx1)
                {
                    for (int svidx2 = 0; svidx2 < mClusterToSv[cluster2].size(); ++svidx2)
                    {
                        float dist = (
                                mSVClusters[(unsigned int) (mClusterToSv[cluster1][svidx1])]->centroid_.getVector3fMap() -
                                mSVClusters[(unsigned int) (mClusterToSv[cluster2][svidx2])]->centroid_.getVector3fMap()).norm();

                        if (dist < minDist)
                        {
                            minDist = dist;
                        }
                    }
                }

                sortedEuclideanDistances[cluster1][cluster2 - 1].first = minDist;
                sortedEuclideanDistances[cluster1][cluster2 - 1].second = cluster2;
                sortedEuclideanDistances[cluster2][cluster1].first = minDist;
                sortedEuclideanDistances[cluster2][cluster1].second = cluster1;
            }
        }

        for (int i = 0; i < mMergedClusters; ++i)
        {
            std::sort(sortedEuclideanDistances[i].begin(), sortedEuclideanDistances[i].end(), PairwiseComparator);
        }
    }

    template<typename PointInT>
    void SupervoxelSegmentation<PointInT>::GetClusterAdjacency(std::vector<std::vector<int> > &adjacencies)
    {
        adjacencies.resize(mClusterToSv.size());

        for (std::map<unsigned int, unsigned int>::iterator it = mSVToCluster.begin(); it != mSVToCluster.end(); ++it)
        {
            unsigned int currentSv = it->first;
            unsigned int currentCluster = it->second;

            for (std::multimap<uint32_t, uint32_t>::iterator svit = mSVAdjacency.lower_bound(currentSv);
                 svit != mSVAdjacency.upper_bound(currentSv); ++svit)
            {
                unsigned int adjacentsv = svit->second;
                unsigned int adjacentCluster = mSVToCluster[adjacentsv];

                if (adjacentCluster != currentCluster &&
                    std::find(adjacencies[currentCluster].begin(), adjacencies[currentCluster].end(),
                              adjacentCluster) == adjacencies[currentCluster].end())
                {
                    adjacencies[currentCluster].push_back(adjacentCluster);
                }
            }
        }
    }

    template<typename PointInT>
    double SupervoxelSegmentation<PointInT>::CalculateDistance(typename pcl::Supervoxel<PointInT>::Ptr svA,
                                                               typename pcl::Supervoxel<PointInT>::Ptr svB)
    {
        Eigen::Vector3f rgbA, rgbB, labA, labB;

        rgbA << ((float) svA->centroid_.r) / 255.0, ((float) svA->centroid_.g) / 255.0, ((float) svA->centroid_.b) /
                                                                                        255.0;
        rgbB << ((float) svB->centroid_.r) / 255.0, ((float) svB->centroid_.g) / 255.0, ((float) svB->centroid_.b) /
                                                                                        255.0;

        labA = RGB2Lab(rgbA);
        labB = RGB2Lab(rgbB);

        double color = (mUseCIE94 ? CalculateCIE94Distance(labA, labB) : (labA - labB).norm()) / 100.0;

        Eigen::Vector3f nA = svA->normal_.getNormalVector3fMap();
        Eigen::Vector3f nB = svB->normal_.getNormalVector3fMap();
        Eigen::Vector3f cA = svA->centroid_.getVector3fMap();
        Eigen::Vector3f cB = svB->centroid_.getVector3fMap();

        nA /= nA.norm();
        nB /= nB.norm();

        double sum = mMergeColorWeight * color;

        if (nA.allFinite() && nB.allFinite())
        {
            double ptpl1 = nA.dot(cB - cA);
            double ptpl2 = nB.dot(cA - cB);
            double ptpl = std::max(std::abs(ptpl1), std::abs(ptpl2));

            double normal = 1 - std::abs(nA.dot(nB));
            sum += mMergeNormalWeight * normal + mMergePtPlWeight * ptpl;
        }

        return sum;
    }

    template<typename PointInT>
    bool SupervoxelSegmentation<PointInT>::Merge2Clusters(typename pcl::Supervoxel<PointInT>::Ptr svA,
                                                          typename pcl::Supervoxel<PointInT>::Ptr svB)
    {
        Eigen::Vector3f rgbA, rgbB, labA, labB;

        rgbA << ((float) svA->centroid_.r) / 255.0, ((float) svA->centroid_.g) / 255.0, ((float) svA->centroid_.b) /
                                                                                        255.0;
        rgbB << ((float) svB->centroid_.r) / 255.0, ((float) svB->centroid_.g) / 255.0, ((float) svB->centroid_.b) /
                                                                                        255.0;

        labA = RGB2Lab(rgbA);
        labB = RGB2Lab(rgbB);

        double ptpl1 = svA->normal_.getNormalVector3fMap().dot(
                svB->centroid_.getVector3fMap() - svA->centroid_.getVector3fMap());
        double ptpl2 = svB->normal_.getNormalVector3fMap().dot(
                svA->centroid_.getVector3fMap() - svB->centroid_.getVector3fMap());
        double ptpl = std::max(std::abs(ptpl1), std::abs(ptpl2));

        double color = (mUseCIE94 ? CalculateCIE94Distance(labA, labB) : (labA - labB).norm());
        double normal = acos(std::abs(svA->normal_.getNormalVector3fMap().dot(svB->normal_.getNormalVector3fMap())));
        // double curvature = std::abs(svA->normal_.curvature - svB->normal_.curvature);

        return (color < mMergeColorWeight && normal < mMergeNormalWeight && ptpl < mMergePtPlWeight);
    }

    template<typename PointInT>
    void SupervoxelSegmentation<PointInT>::ChangeClusterLabel(unsigned int oldLbl, unsigned int newLbl)
    {
        for (std::map<unsigned int, unsigned int>::iterator it = mSVToCluster.begin(); it != mSVToCluster.end(); ++it)
        {
            if (it->second == oldLbl)
                it->second = newLbl;
        }
    }

    template<typename PointInT>
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr SupervoxelSegmentation<PointInT>::getColoredClusterPointCloud()
    {
        // create random colors
        Eigen::MatrixXi colors(mMergedClusters, 3);

        /* initialize random seed: */
        srand(time(NULL));

        for (int i = 0; i < mMergedClusters; ++i)
        {
            /* generate random number between 0 and 255: */
            int r = rand() % 256;
            int g = rand() % 256;
            int b = rand() % 256;

            colors.row(i) << r, g, b;
        }

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr p = boost::make_shared<pcl::PointCloud<pcl::PointXYZRGB> >();

        for (int i = 0; i < mSVLabeledCloud->points.size(); ++i)
        {
            pcl::PointXYZL ptl = mSVLabeledCloud->at(i);

            unsigned int svlbl = ptl.label;
            if (svlbl > 0)
            {
                unsigned int clbl = mSVToCluster[svlbl];

                pcl::PointXYZRGB ptc;
                ptc.x = ptl.x;
                ptc.y = ptl.y;
                ptc.z = ptl.z;

                ptc.r = colors(clbl, 0);
                ptc.g = colors(clbl, 1);
                ptc.b = colors(clbl, 2);
                p->push_back(ptc);
            }
        }

        p->height = 1;
        p->width = p->points.size();
        p->is_dense = false;
        return p;
    }

    template<typename PointInT>
    pcl::PointCloud<pcl::PointXYZL>::Ptr SupervoxelSegmentation<PointInT>::GetClusterIDPointcloud()
    {
        pcl::PointCloud<pcl::PointXYZL>::Ptr p = boost::make_shared<pcl::PointCloud<pcl::PointXYZL> >();

        for (int i = 0; i < mSVLabeledCloud->points.size(); ++i)
        {
            pcl::PointXYZL ptsv = mSVLabeledCloud->at(i);

            unsigned int svlbl = ptsv.label;
            unsigned int clbl = std::numeric_limits<unsigned int>::max();

            if (svlbl > 0 && mSVToCluster.find(svlbl) != mSVToCluster.end())
            {
                clbl = mSVToCluster[svlbl];
            }
            pcl::PointXYZL ptl;
            ptl.x = ptsv.x;
            ptl.y = ptsv.y;
            ptl.z = ptsv.z;

            ptl.label = clbl;
            p->push_back(ptl);
        }

        p->height = 1;
        p->width = p->points.size();
        p->is_dense = false;
        return p;
    }

    template<typename PointInT>
    void SupervoxelSegmentation<PointInT>::GetVoxelizedResults(typename pcl::PointCloud<PointInT>::Ptr voxels,
                                                               pcl::PointCloud<pcl::Normal>::Ptr normals,
                                                               pcl::PointCloud<pcl::PointXYZL>::Ptr clusterIDs)
    {
        voxels->clear();
        normals->clear();
        clusterIDs->clear();

        for (typename std::map<uint32_t, typename pcl::Supervoxel<PointInT>::Ptr>::iterator sv_itr = mSVClusters.begin();
             sv_itr != mSVClusters.end(); ++sv_itr)
        {
            *voxels += *(sv_itr->second->voxels_);
            *normals += *(sv_itr->second->normals_);

            uint32_t sv_lbl = sv_itr->first;
            uint32_t cluster_lbl = std::numeric_limits<unsigned int>::max();

            if (sv_lbl > 0 && mSVToCluster.find(sv_lbl) != mSVToCluster.end())
            {
                cluster_lbl = mSVToCluster[sv_lbl];
            }

            for (int i = 0; i < sv_itr->second->voxels_->size(); ++i)
            {
                pcl::PointXYZL p;
                PointInT &v = sv_itr->second->voxels_->at(i);

                p.x = v.x;
                p.y = v.y;
                p.z = v.z;
                p.label = cluster_lbl;
                clusterIDs->push_back(p);
            }
        }

        voxels->height = 1;
        voxels->width = voxels->size();
        voxels->is_dense = false;
        normals->height = 1;
        normals->width = normals->size();
        normals->is_dense = false;
        clusterIDs->height = 1;
        clusterIDs->width = clusterIDs->size();
        clusterIDs->is_dense = false;
    }

    template<typename PointInT>
    void SupervoxelSegmentation<PointInT>::drawSupervoxelAdjacency(pcl::visualization::PCLVisualizer *visualizer,
                                                                   int &viewport)
    {
        visualizer->removeAllShapes(viewport);

        std::multimap<uint32_t, uint32_t>::iterator label_itr = mSVAdjacency.begin();
        for (; label_itr != mSVAdjacency.end();)
        {
            //First get the label
            uint32_t supervoxel_label = label_itr->first;

            //Now get the supervoxel corresponding to the label
            typename pcl::Supervoxel<PointInT>::Ptr supervoxel = mSVClusters.at(supervoxel_label);

            //Now we need to iterate through the adjacent supervoxels and make a point cloud of them
            pcl::PointCloud<pcl::PointXYZRGBA> adjacent_supervoxel_centers;

            std::multimap<uint32_t, uint32_t>::iterator adjacent_itr = mSVAdjacency.equal_range(supervoxel_label).first;
            for (; adjacent_itr != mSVAdjacency.equal_range(supervoxel_label).second; ++adjacent_itr)
            {
                typename pcl::Supervoxel<PointInT>::Ptr neighbor_supervoxel = mSVClusters.at(adjacent_itr->second);
                adjacent_supervoxel_centers.push_back(neighbor_supervoxel->centroid_);
            }
            //Now we make a name for this polygon
            std::stringstream ss;
            ss << "supervoxel_" << supervoxel_label;
            //This function is shown below, but is beyond the scope of this tutorial - basically it just generates a "star" polygon mesh from the points given
            addSupervoxelConnectionsToViewer(supervoxel->centroid_, adjacent_supervoxel_centers, ss.str(), visualizer,
                                             viewport);
            //Move iterator forward to next label
            label_itr = mSVAdjacency.upper_bound(supervoxel_label);
        }
    }

    template<typename PointInT>
    void SupervoxelSegmentation<PointInT>::addSupervoxelConnectionsToViewer(pcl::PointXYZRGBA &supervoxel_center,
                                                                            pcl::PointCloud<pcl::PointXYZRGBA> &adjacent_supervoxel_centers,
                                                                            std::string supervoxel_name,
                                                                            pcl::visualization::PCLVisualizer *visualizer,
                                                                            int &viewport)
    {

        vtkSmartPointer <vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
        vtkSmartPointer <vtkCellArray> cells = vtkSmartPointer<vtkCellArray>::New();
        vtkSmartPointer <vtkPolyLine> polyLine = vtkSmartPointer<vtkPolyLine>::New();

        //Iterate through all adjacent points, and add a center point to adjacent point pair
        pcl::PointCloud<pcl::PointXYZRGBA>::iterator adjacent_itr = adjacent_supervoxel_centers.begin();
        for (; adjacent_itr != adjacent_supervoxel_centers.end(); ++adjacent_itr)
        {
            points->InsertNextPoint(supervoxel_center.data);
            points->InsertNextPoint(adjacent_itr->data);
        }
        // Create a polydata to store everything in
        vtkSmartPointer <vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();
        // Add the points to the dataset
        polyData->SetPoints(points);
        polyLine->GetPointIds()->SetNumberOfIds(points->GetNumberOfPoints());
        for (unsigned int i = 0; i < points->GetNumberOfPoints(); i++)
            polyLine->GetPointIds()->SetId(i, i);
        cells->InsertNextCell(polyLine);
        // Add the lines to the dataset
        polyData->SetLines(cells);

        visualizer->addModelFromPolyData(polyData, supervoxel_name, viewport);

    }

    template<typename PointInT>
    pcl::PointCloud<pcl::PointXYZL>::Ptr SupervoxelSegmentation<PointInT>::getSVlabeledCloud()
    {
        return mSVLabeledCloud;
    }

    template<typename PointInT>
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr SupervoxelSegmentation<PointInT>::getSVColoredCloud()
    {
        return mSVColoredCloud;
    }

#define PCL_INSTANTIATE_SupervoxelSegmentation(T) template class PCL_EXPORTS SupervoxelSegmentation<T>;

}