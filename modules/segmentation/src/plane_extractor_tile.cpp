#include <v4r/segmentation/plane_utils.h>
#include <v4r/segmentation/plane_extractor_tile.h>

#include <pcl/impl/instantiate.hpp>
#include <glog/logging.h>
#include <iostream>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <glm/glm.hpp>

//#define DEBUG_IMAGES
//#define DEBUG_TEXT
//#define DEBUG_TIMINGS

namespace v4r
{
bool isInlier(const Eigen::Vector3f &point, const Eigen::Vector4f &normal, const Eigen::Vector4f &plane, float cosThreshold, float distThreshold, bool doNormalTest = true);
bool isInPlane(const Eigen::Vector4f &plane1, const Eigen::Vector3f &centerPlane2, float distThreshold);
bool isParallel(const Eigen::Vector4f &plane1, const Eigen::Vector4f &plane2, float cosThreshold);

bool
isInlier(const Eigen::Vector3f &point, const Eigen::Vector4f &normal, const Eigen::Vector4f &plane, float cosThreshold, float distThreshold, bool doNormalTest)
{
    float distance = fabs(dist2plane(point, plane));
    if(distance<distThreshold)
    {
        if(doNormalTest)
        {
            float cosAlpha = normal.head(3).dot(plane.head(3))/ (plane.head(3).norm() * normal.head(3).norm());
            return (cosAlpha>cosThreshold);
        }
        else
            return true;
    }

    return false;
}

bool
isInPlane(const Eigen::Vector4f &plane1, const Eigen::Vector3f &centerPlane2, float distThreshold)
{
    float distance = fabs(dist2plane(centerPlane2, plane1)); // std::abs(1.0f-plane1.dot(centerPlane2))/plane1.norm();//DEBUG!!!!!!!!why does this get zero??????
    return (distance<distThreshold);
}

bool
isParallel(const Eigen::Vector4f &plane1, const Eigen::Vector4f &plane2, float cosThreshold)
{
    return (CosAngleBetweenPlanes(plane1, plane2) >cosThreshold);//maybe use plane angle for this
}

template <typename PointT>
cv::Mat
PlaneExtractorTile<PointT>::getDebugImage(bool doNormalTest)
{
    int lastPlaneId=0;
    Eigen::Vector4f plane(0,0,0,0);//=planeList[planeId].plane;
//    cv::Mat debug2(segmentation.rows,segmentation.cols,CV_32SC1);
    for(int i=0;i<rowsOfPatches;i++)
    {
        for(int j=0;j<colsOfPatches;j++)
        {
            float distThreshold;
            float cosThreshold;
            if(param_.useVariableThresholds_)
            {
                //read it from the buffer
                const Eigen::Vector4f &thresholds = thresholdsBuffer[i][j];
                distThreshold = thresholds[0];
                cosThreshold = thresholds[1];
            }
            else
            {
                distThreshold=param_.maxInlierBlockDist_;
                cosThreshold=minCosBlockAngle;
            }

            int planeId=patchIds.at<int>(i,j);
            if(planeId)
            {
                if(planeId!=lastPlaneId)
                {
                    plane = planeList[planeId].plane;
                    lastPlaneId=planeId;
                }

                //Mark the pixel in the segmentation map for the already existing patches
                if( planes[i][j].nrInliers > minAbsBlockInlier )
                {
                    for(int k=0;k<param_.patchDim_;k++)
                    {
                        for(int l=0;l<param_.patchDim_;l++)
                        {
                            //mark the points in debug:
                            //honestly we should still check if the point is an inlier
                            const Eigen::Vector4f &normal = normal_cloud_->at(j*param_.patchDim_+l, i*param_.patchDim_+k).getNormalVector4fMap(); //normals.at<Eigen::Vector4f>(i*param_.patchDim_+k, j*param_.patchDim_+l);
                            const Eigen::Vector3f &point = cloud_->at(j*param_.patchDim_+l+1, i*param_.patchDim_+k+1).getVector3fMap(); //points.at<Eigen::Vector4f>(i*param_.patchDim_+k+1, j*param_.patchDim_+l+1)

                            if( isInlier(point, normal, plane, cosThreshold, distThreshold, doNormalTest) )
                                segmentation.at<int>(i*param_.patchDim_+k,j*param_.patchDim_+l)=planeId;
                        }
                    }
                }
            }
        }
    }

    return generateColorCodedTextureDebug();
}

template<typename PointT>
cv::Mat
PlaneExtractorTile<PointT>::generateColorCodedTexture() const
{
    cv::Mat colorMap(1,64*48,CV_8UC3); ///TODO Is this is a bug (why fixed???)
    colorMap.at<cv::Vec3b>(0)=cv::Vec3b(0,0,0);
    colorMap.at<cv::Vec3b>(1)=cv::Vec3b(0,0,200);
    colorMap.at<cv::Vec3b>(2)=cv::Vec3b(0,200,0);
    colorMap.at<cv::Vec3b>(3)=cv::Vec3b(200,0,0);
    colorMap.at<cv::Vec3b>(4)=cv::Vec3b(0,200,200);
    colorMap.at<cv::Vec3b>(5)=cv::Vec3b(250,0,0);
    colorMap.at<cv::Vec3b>(6)=cv::Vec3b(200,200,200);
    colorMap.at<cv::Vec3b>(7)=cv::Vec3b(0,0,100);
    colorMap.at<cv::Vec3b>(8)=cv::Vec3b(0,100,0);
    colorMap.at<cv::Vec3b>(9)=cv::Vec3b(100,0,0);
    colorMap.at<cv::Vec3b>(10)=cv::Vec3b(0,100,100);
    colorMap.at<cv::Vec3b>(11)=cv::Vec3b(100,100,0);
    colorMap.at<cv::Vec3b>(12)=cv::Vec3b(100,100,100);

    for (int n=13;n<colorMap.cols;n++)
        colorMap.at<cv::Vec3b>(n)=cv::Vec3b(n/10*50,((n%10)/5)*50,(n%5)*50);

    cv::Mat coloredImage (cloud_->height, cloud_->width, CV_8UC3);
    for(size_t i=0; i<cloud_->height; i++)
    {
        for(size_t j=0; j<cloud_->width; j++)
            coloredImage.at<cv::Vec3b>(i,j)=colorMap.at<cv::Vec3b>(0,segmentation.at<int>(i,j));
    }
    return coloredImage;
}

template<typename PointT>
cv::Mat
PlaneExtractorTile<PointT>::generateColorCodedTextureDebug() const
{
    cv::Mat colorMap(1, 64*48, CV_8UC3); ///TODO Is this is a bug (why fixed???)
    colorMap.at<cv::Vec3b>(0)=cv::Vec3b(0,0,0);
    colorMap.at<cv::Vec3b>(1)=cv::Vec3b(0,0,200);
    colorMap.at<cv::Vec3b>(2)=cv::Vec3b(0,200,0);
    colorMap.at<cv::Vec3b>(3)=cv::Vec3b(200,0,0);
    colorMap.at<cv::Vec3b>(4)=cv::Vec3b(0,200,200);
    colorMap.at<cv::Vec3b>(5)=cv::Vec3b(250,0,0);
    colorMap.at<cv::Vec3b>(6)=cv::Vec3b(200,200,200);
    colorMap.at<cv::Vec3b>(7)=cv::Vec3b(0,0,100);
    colorMap.at<cv::Vec3b>(8)=cv::Vec3b(0,100,0);
    colorMap.at<cv::Vec3b>(9)=cv::Vec3b(100,0,0);
    colorMap.at<cv::Vec3b>(10)=cv::Vec3b(0,100,100);
    colorMap.at<cv::Vec3b>(11)=cv::Vec3b(100,100,0);
    colorMap.at<cv::Vec3b>(12)=cv::Vec3b(100,100,100);

    for (int n=13;n<colorMap.cols;n++)
        colorMap.at<cv::Vec3b>(n)=cv::Vec3b(n/10*50,((n%10)/5)*50,(n%5)*50);

    //TODO: take cols and rows from the segmentation Mat

    cv::Mat coloredImage (cloud_->height, cloud_->width, CV_8UC3);
    for(size_t i=0; i<cloud_->height; i++)
    {
        for(size_t j=0; j<cloud_->width; j++)
        {
            if( segmentation.at<int>(i,j)>0 )
                coloredImage.at<cv::Vec3b>(i,j)=colorMap.at<cv::Vec3b>(0, segmentation.at<int>(i,j) );
            else
                coloredImage.at<cv::Vec3b>(i,j)=debug.at<cv::Vec3b>(i,j);
        }
    }
    return coloredImage;
}


template<typename PointT>
Eigen::Vector4f
PlaneExtractorTile<PointT>::calcPlaneFromMatrix(const PlaneExtractorTile<PointT>::PlaneMatrix &m) const
{
    Eigen::Matrix3d mat;
    mat <<  m.xx,m.xy,m.xz,
            m.xy,m.yy,m.yz,
            m.xz,m.yz,m.zz;

    //hopefully this is fast!
    const Eigen::Vector3d plane = mat.ldlt().solve(m.sum);//what do i know?

    Eigen::Vector4f plane4 (plane[0], plane[1], plane[2], -1.f);

    // flip normals always towards viewpoint
    if( Eigen::Vector3f::UnitZ().dot(plane4.head(3)) > 0 )
        plane4 = -plane4;

    return plane4;
}

template<typename PointT>
void
PlaneExtractorTile<PointT>::replace(int from, int to, int maxIndex)
{
    int *pointer=(int*)patchIds.data;
    for(int i=0;i<maxIndex;i++)
    {
        if(*pointer==from)
            *pointer=to;
        pointer++;
    }
}

template<typename PointT>
int
PlaneExtractorTile<PointT>::allocateMemory()
{
    minAbsBlockInlier = (float)( param_.patchDim_*param_.patchDim_ ) * param_.minBlockInlierRatio_;
    colsOfPatches = cloud_->width / param_.patchDim_;
    rowsOfPatches = cloud_->height / param_.patchDim_;

    int nrMatrices=colsOfPatches*rowsOfPatches;
    matrices.resize(nrMatrices);
    planeMatrices.resize(nrMatrices);

    planeList.clear();
    planeList.resize(rowsOfPatches*colsOfPatches+1); //TODO: eliminate this memory leak

    //Initializing matrices:
    for(int i=0;i<nrMatrices;i++)
    {
        matrices[i].sum=Eigen::Vector3d(0,0,0);
        matrices[i].xx=0;
        matrices[i].xy=0;
        matrices[i].xz=0;
        matrices[i].yy=0;
        matrices[i].yz=0;
        matrices[i].zz=0;
        matrices[i].nrPoints=0;
        planeMatrices[i]=matrices[i];
    }

    segmentation = cv::Mat(cloud_->height,cloud_->width, CV_32SC1);
    segmentation.setTo(cv::Scalar(0));


    if(param_.doZTest_)
    {
        zBuffer = cv::Mat(cloud_->height, cloud_->width, CV_32FC1);
        zBuffer.setTo( std::numeric_limits<float>::max() );
    }

    if(param_.useVariableThresholds_)
        thresholdsBuffer = std::vector<std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f> > >
                (rowsOfPatches, std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f> >(colsOfPatches) );


//    if(normals.cols!=cloud_->width || normals.rows!=cloud_->height)
    {
        planes = std::vector<std::vector<PlaneSegment> >(rowsOfPatches, std::vector<PlaneSegment>(colsOfPatches) );
        centerPoints = std::vector<std::vector<Eigen::Vector3f> >
                (rowsOfPatches, std::vector<Eigen::Vector3f >(colsOfPatches) );
        debug.create(cloud_->height, cloud_->width, CV_8UC3);
        debug.setTo( cv::Vec3b(0,0,0) );
        patchIds.create(rowsOfPatches,colsOfPatches,CV_32SC1);
    }

    return 1;
}

template<typename PointT>
void
PlaneExtractorTile<PointT>::calculatePlaneSegments(bool doNormalTest)
{
    //create the blockwise plane description
    for(int i=0;i<rowsOfPatches;i++)
    {
        for(int j=0;j<colsOfPatches;j++)
        {
            PlaneMatrix pm={};
            for(int m=0;m<param_.patchDim_;m++)
            {
                for(int n=0;n<param_.patchDim_;n++)
                {
                    const PointT &p = cloud_->at(j*param_.patchDim_+n, i*param_.patchDim_+m);//points.at<Eigen::Vector4f>(i*param_.patchDim_+m+1,j*param_.patchDim_+n+1);
                    if( pcl::isFinite(p) )
                    {
                        pm.sum+=p.getVector3fMap().template cast<double>();
                        pm.xx+=p.x * p.x;
                        pm.xy+=p.x * p.y;
                        pm.xz+=p.x * p.z;
                        pm.yy+=p.y * p.y;
                        pm.yz+=p.y * p.z;
                        pm.zz+=p.z * p.z;
                        pm.nrPoints++;
                    }
                }
            }
            int index = i*colsOfPatches+j;
            matrices[index] = pm;
        }
    }


    //calculate all plane segments
    for(int i=0;i<rowsOfPatches;i++)
    {
        for(int j=0;j<colsOfPatches;j++)
        {
            int index=j+i*colsOfPatches;
            const PlaneMatrix &m = matrices[index];
            const Eigen::Vector3f msum = m.sum.template cast<float>();
            centerPoints[i][j] = msum / m.nrPoints;
            float cosThreshold = minCosAngle;
            float distThreshold = param_.maxInlierDist_;
            if ( param_.useVariableThresholds_ && m.nrPoints>0 )
            {
                float z = centerPoints[i][j][2];
                Eigen::Vector4f thresholds;
                thresholds[0] = maxInlierBlockDistFunc(z);
                thresholds[1] = minCosBlockAngleFunc(z);
                distThreshold = maxInlierDistFunc(z);
                thresholds[2] = distThreshold;
                cosThreshold = minCosAngleFunc(z);
                thresholds[3] = cosThreshold;
                thresholdsBuffer[i][j] = thresholds;
            }
            else
                thresholdsBuffer[i][j] = Eigen::Vector4f(-1.f,-1.f,-1.f,-1.f);

            if(m.nrPoints > minAbsBlockInlier)
            {
                const Eigen::Vector4f plane=calcPlaneFromMatrix(m);//what do i know?
               //invert matrix and create plane estimation
                planes[i][j].plane = plane;
                planes[i][j].nrInliers=m.nrPoints;

                //Calculate Thresholds here:
                //TODO!!!!!!

                Eigen::Vector4f N;
                for(int k=0;k<param_.patchDim_;k++)
                {
                    for(int l=0;l<param_.patchDim_;l++)
                    {
                        int u=l+j*param_.patchDim_;
                        int v=k+i*param_.patchDim_;
                        const Eigen::Vector3f &p = cloud_->at(u,v).getVector3fMap(); //points.at<Eigen::Vector4f>(v+1,u+1);

                        if(doNormalTest)
                            N = normal_cloud_->at(u,v).getNormalVector4fMap(); //normals.at<Eigen::Vector4f>(v,u);

                        //TODO: remove this isInlier.... or at least store the norm for this so it does not have to be recalculated for every pixel
                        if(isInlier(p, N, plane, cosThreshold,distThreshold,doNormalTest))
                        { //distance < inlierDistance
                            //mark the inlying points somehow
                            debug.at<cv::Vec3b>(v,u)=cv::Vec3b(255,0,0);
                            segmentation.at<int>(v,u)=-1;//mark every valid element in the segmentation
                        }
                        else
                            planes[i][j].nrInliers--;
                    }
                }
                if(planes[i][j].nrInliers>minAbsBlockInlier)
                {
                    for(int k=0;k<param_.patchDim_;k++)
                    {
                        for(int l=0;l<param_.patchDim_;l++)
                        {
                            int u=l+j*param_.patchDim_;
                            int v=k+i*param_.patchDim_;
                            const Eigen::Vector3f &p = cloud_->at(u,v).getVector3fMap(); //points.at<Eigen::Vector4f>(v+1,u+1);

                            if(doNormalTest)
                                N=normal_cloud_->at(u,v).getNormalVector4fMap();// normals.at<Eigen::Vector4f>(v,u);

                            //TODO: remove this isInlier.... or at least store the norm for this so it does not have to be recalculated for every pixel
                            if( isInlier(p, N, plane, cosThreshold, distThreshold, doNormalTest) )
                            { //distance < inlierDistance
                                //mark the inlying points somehow
                                debug.at<cv::Vec3b>(v,u)=cv::Vec3b(0,255,0);
                            }

                        }
                    }
                }

            }
            else
            {
                planes[i][j].plane = NAN * Eigen::Vector4f::Ones();
                planes[i][j].nrInliers=0;
            }
        }
    }
}

template<typename PointT>
void
PlaneExtractorTile<PointT>::rawPatchClustering()
{
    //TODO: make this code more compact somehow (and readable)
    for(int i=0;i<rowsOfPatches;i++)
    {
        Eigen::Vector4f lastPatch(0,0,0,0);
        //int lastId=0;
        int currentId=0;
        for(int j=0;j<colsOfPatches;j++)
        {
            int index=j+i*colsOfPatches;
            const PlaneMatrix &currentPlaneMatrix = matrices[index];
            const PlaneSegment &currentPlaneSeg = planes[i][j];//TODO:planes should be renamed to patches
            const Eigen::Vector4f &currentPatch = currentPlaneSeg.plane;//( currentPlaneSeg.x, currentPlaneSeg.y, currentPlaneSeg.z, currentPlaneSeg.d);
            const Eigen::Vector3f &currentCenter = centerPoints[i][j];
            //test if plane is valid
            bool gotSet=false;

            if(currentPlaneSeg.nrInliers>minAbsBlockInlier)
            {
                float distThreshold;
                float cosThreshold;
                if(param_.useVariableThresholds_)
                {
                    //read it from the buffer
                    const Eigen::Vector4f &thresholds = thresholdsBuffer[i][j];
                    distThreshold=thresholds[0];
                    cosThreshold=thresholds[1];
                }
                else
                {
                    distThreshold=param_.maxInlierBlockDist_;
                    cosThreshold=minCosBlockAngle;
                }
                bool alreadyTested=false;
                //test if lastID is valid
                if(currentId!=0)
                {
                    //test if the new Plane element fits to the existing one
                    const Eigen::Vector4f &otherPlane = calcPlaneFromMatrix(planeMatrices[currentId]);

                    if(     isInPlane(lastPatch,currentCenter,distThreshold) &&
                            isParallel(currentPatch,otherPlane,cosThreshold))
                    {
                        gotSet = true;
                        patchIds.at<int>(i,j) = currentId;

                        planeMatrices[currentId] += currentPlaneMatrix;
                        //if this here is the case, it is not necessary to test the top and top left plane segment
                        //except for the first row(column?)... there the top element should be tested
                        //Note: this does not bring much of an speed advantage (none)
                        //alreadyTested=true;
                    }
                    else
                    {//debug attempt
                        currentId=0;//SIMON DEBUG ATTEMPT
                    }
                }

                lastPatch=currentPatch;
                //test if one of the 3 upper elements is already segmented and connect planes if necessary
                if(i>0)
                {
                    Eigen::Vector4f newPlane(0,0,0,0);
                    //Eigen::Vector4f newPlane(0,0,0,0);
                    if(j>0 && !alreadyTested)
                    {
                        //do upper left
                        int newId=patchIds.at<int>(i-1,j-1);// it is only testing for blocks from the past(so the threshold is already checked
                        if(newId)
                        {
                            const PlaneSegment &testPlaneSeg = planes[i-1][j-1];
                            const Eigen::Vector4f &newPatch = testPlaneSeg.plane;

                            newPlane = calcPlaneFromMatrix(planeMatrices[newId]);
                            if(     isInPlane(newPatch,currentCenter,distThreshold) &&
                                    isParallel(currentPatch,newPlane,cosThreshold))
                            {
                                gotSet=true;
                                //test if this is the right time to connect
                                if(!currentId)
                                {//if the current patch does not have an ID yet just add this one:
                                    currentId=newId;
                                    patchIds.at<int>(i,j)=currentId;
                                    planeMatrices[currentId]+=currentPlaneMatrix;
                                }
                                else
                                {
                                    if(currentId!=newId)
                                    {
                                        int nrCurrent=planeMatrices[currentId].nrPoints;
                                        int nrNew=planeMatrices[newId].nrPoints;

                                        if(nrCurrent>nrNew)
                                        {//replace the one with fewer elements:
                                            replace(newId,currentId,j+i*colsOfPatches);
                                            planeMatrices[currentId]+=planeMatrices[newId];
                                            planeMatrices[newId].nrPoints=0;
                                        }
                                        else
                                        {
                                            replace(currentId,newId,j+i*colsOfPatches);
                                            planeMatrices[newId]+=planeMatrices[currentId];
                                            planeMatrices[currentId].nrPoints=0;
                                            currentId=newId;

                                        }
                                        patchIds.at<int>(i,j)=currentId;
                                    }
                                }
                            }
                        }
                    }

                    //upper
                    if(!alreadyTested || j==0)
                    {//forget alreadyTested. it does not make sense
                        int newId = patchIds.at<int>(i-1,j);
                        if(newId)
                        {
                            const PlaneSegment &testPlaneSeg = planes[i-1][j];
                            const Eigen::Vector4f &newPatch = testPlaneSeg.plane; //(currentPlaneSeg.x,currentPlaneSeg.y,currentPlaneSeg.z,currentPlaneSeg.d);

                            newPlane = calcPlaneFromMatrix(planeMatrices[newId]);
                            if(     isInPlane(newPatch,currentCenter,distThreshold) &&
                                    isParallel(currentPatch,newPlane,cosThreshold))
                            {
                                gotSet=true;
                                //test if this is the right time to connect
                                if(!currentId)
                                {//if the current patch does not have an ID yet just add this one:
                                    currentId=newId;
                                    patchIds.at<int>(i,j)=currentId;
                                    planeMatrices[currentId]+=currentPlaneMatrix;
                                }
                                else
                                {
                                    if(currentId!=newId)
                                    {
                                        int nrCurrent=planeMatrices[currentId].nrPoints;
                                        int nrNew=planeMatrices[newId].nrPoints;

                                        if(nrCurrent>nrNew)
                                        {//replace the one with fewer elements:
                                            replace(newId,currentId,j+i*colsOfPatches);

                                            planeMatrices[currentId] += planeMatrices[newId];
                                            planeMatrices[newId].nrPoints = 0;
                                        }
                                        else
                                        {
                                            replace(currentId,newId,j+i*colsOfPatches);
                                            planeMatrices[newId] += planeMatrices[currentId];
                                            planeMatrices[currentId].nrPoints = 0;
                                            currentId=newId;
                                        }
                                        patchIds.at<int>(i,j)=currentId;
                                    }
                                }
                            }
                        }
                    }

                    //upper right
                    if(j+1<colsOfPatches)
                    {
                        int newId=patchIds.at<int>(i-1,j+1);
                        if(newId)
                        {
                            const PlaneSegment &testtPlaneSeg = planes[i-1][j+1];
                            const Eigen::Vector4f &newPatch = testtPlaneSeg.plane; //(currentPlaneSeg.x,currentPlaneSeg.y,currentPlaneSeg.z,currentPlaneSeg.d);

                            newPlane = calcPlaneFromMatrix(planeMatrices[newId]);
                            if(     isInPlane(newPatch,currentCenter,distThreshold) &&
                                    isParallel(currentPatch,newPlane,cosThreshold))
                            {
                                gotSet=true;
                                //test if this is the right time to connect
                                if(!currentId)
                                {//if the current patch does not have an ID yet just add this one:
                                    currentId=newId;
                                    patchIds.at<int>(i,j)=currentId;
                                    planeMatrices[currentId]+=currentPlaneMatrix;
                                }
                                else
                                {
                                    if(currentId!=newId){
                                        int nrCurrent=planeMatrices[currentId].nrPoints;
                                        int nrNew=planeMatrices[newId].nrPoints;

                                        if(nrCurrent>nrNew)
                                        {//replace the one with fewer elements:
                                            replace(newId,currentId,j+i*colsOfPatches);
                                            planeMatrices[currentId]+=planeMatrices[newId];
                                            planeMatrices[newId].nrPoints=0;
                                        }
                                        else
                                        {
                                            replace(currentId,newId,j+i*colsOfPatches);
                                            planeMatrices[newId]+=planeMatrices[currentId];
                                            planeMatrices[currentId].nrPoints=0;
                                            currentId=newId;
                                        }
                                        //howOften++;
                                        patchIds.at<int>(i,j)=currentId;
                                    }
                                }
                            }
                        }

                    }

                }

                if(!gotSet)
                    currentId=0;

                //in case the current ID could not be connected to an already existing plane:
                //create a new one
                if(currentId==0)
                {
                    //create a new id
                    currentId=++maxId;
                    patchIds.at<int>(i,j)=currentId;
                    planeMatrices[currentId]=currentPlaneMatrix;
                }

            }
            else//if the current patch does not contain enought members
                currentId=0;

            //do some debug output
            if(false)
            {
                VLOG(1) << "DEBUG: currentId " << currentId;
                cv::imshow("current", getDebugImage( param_.pointwiseNormalCheck_) );
                cv::waitKey(1);
            }
        }
    }
}

template<typename PointT>
void
PlaneExtractorTile<PointT>::postProcessing1Direction(const int offsets[][2], bool doNormalTest,bool reverse,bool zTest)
{
    //TODO: add a distance buffer to store which patch is the better fit for a segment
    for(int i=reverse ? segmentation.rows-1 : 0 ;reverse ? i>0 : i<segmentation.rows; reverse ? i-- : i++)
    {
        int oldId=0;
        Eigen::Vector4f oldPlane;
        int oldPlaneTTL=0;
        float cosThreshold=minCosAngle;
        float distThreshold=param_.maxInlierDist_;
        for(int j=reverse ? segmentation.cols-1 : 0 ;reverse ? j>0 : j<segmentation.cols; reverse ? j-- : j++)
        {
            int currentId = segmentation.at<int>(i,j);
            if(currentId>0)
            {
                // First step is to check for the surrounding patches if there are elements of the current segment ID
                if(currentId!=oldId || !oldPlaneTTL)
                {
                    oldId=currentId;
                    bool found=false;
                    const int surrounding[9][2]={{0,0},{-1,-1},{-1,0},{-1,1},{0,1},{1,1},{1,0},{1,-1},{0,-1}};
                    //search the plane in one of the surrounding patches
                    for(int k=0;k<9;k++)
                    {
                        int _i=i/param_.patchDim_+surrounding[k][0];
                        int _j=j/param_.patchDim_+surrounding[k][0];
                        if(_i>=0 && _j>=0 && _i<rowsOfPatches && j<colsOfPatches)
                        {
                            //TODO: do the readout for the fitting thresholds here!!!!
                            ///TODOOOOOOOOO!!!!
                            if(oldId==patchIds.at<int>(_i,_j))
                            {
                                //read according patch segment
                                const PlaneSegment &p = planes[_i][_j];
                                oldPlane = p.plane; //Eigen::Vector4f(p.x,p.y,p.z,p.d);
                                found=true;

                                //is this really the best place for reading out the thresholds?
                                if(param_.useVariableThresholds_)
                                {
                                    const Eigen::Vector4f &thresholds = thresholdsBuffer[_i][_j];
                                    distThreshold = thresholds[2];
                                    cosThreshold = thresholds[3];
                                }

                                break;
                            }
                        }
                    }
                    if(!found)
                    {
                        //if no fitting patch is found. we take the plane with this id
                        const Plane &p = resultingPlanes[oldId-1];
                        oldPlane = p.plane;

                        //the thresholds are used from the fitting patch if no other threshold is found
                        //DON'T KNOW WHY THIS DOES NOT WORK!!!
                        //TODO: FIND THIS BUG!!!!!
                        ///BUG:
                        if(param_.useVariableThresholds_)
                        {
                            const Eigen::Vector4f &thresholds = thresholdsBuffer[i/param_.patchDim_][j/param_.patchDim_];
                            distThreshold = thresholds[2];
                            cosThreshold = thresholds[3];
                        }

                    }
                    oldPlaneTTL=param_.patchDim_;
                }
                else
                    oldPlaneTTL--;

                //const int offsets[4][2]={{0,1},{1,-1},{1,0},{1,1}};
                if(oldId)
                {
                    for(int k=0;k<4;k++)
                    {
                        int _i=i+offsets[k][0];
                        int _j=j+offsets[k][1];
                        if(_i>=0 && _j>=0 && _i<segmentation.rows && _j<segmentation.cols){
                            int otherId=segmentation.at<int>(_i,_j);

                            if(otherId<=0 || zTest)
                            {//only do this if pixel is not yet set
                                //test if the pixel is inside of oldPlane and set the pixel accordingly
                                const Eigen::Vector3f &otherPoint = cloud_->at(_j,_i).getVector3fMap();// points.at<Eigen::Vector4f>(_i+1,_j+1);
                                float newDist = fabs(dist2plane(otherPoint, oldPlane));
                                float oldDist=0;
                                if(zTest)
                                    oldDist = zBuffer.at<float>(_i,_j);

                                if(newDist<oldDist || !zTest)
                                {
                                    Eigen::Vector4f otherNormal;
                                    if(doNormalTest)
                                        otherNormal = normal_cloud_->at(_j,_i).getNormalVector4fMap();// normals.at<Eigen::Vector4f>(_i,_j);

                                    if( isInlier(otherPoint, otherNormal, oldPlane, cosThreshold, distThreshold, doNormalTest))
                                    {
                                        segmentation.at<int>(_i,_j)=oldId;
                                        if(zTest)
                                            zBuffer.at<float>(_i,_j)=newDist;
                                    }
                                }
                            }
                        }
                    }
                    if(false)
                    {
                        VLOG(1) << "DEBUG: currentId " << currentId;
                        cv::imshow("current",generateColorCodedTexture());
                        cv::waitKey(1);
                    }
                }
            }
        }
    }
    //cv::imshow("debug2",debug2);
    //cv::waitKey();
}

template<typename PointT>
void
PlaneExtractorTile<PointT>::postProcessing(bool doNormalTest, bool zTest)
{
    //TODO: find out why this does not deliver the same results as the code below
    const int offsets1[4][2]={{0,1},{1,-1},{1,0},{1,1}};
    postProcessing1Direction(offsets1, doNormalTest,false,zTest);
    const int offsets2[4][2]={{0,-1},{-1,-1},{-1,0},{-1,1}};
    postProcessing1Direction(offsets2, doNormalTest,true,zTest);
}


template<typename PointT>
PlaneExtractorTile<PointT>::PlaneExtractorTile(const PlaneExtractorTileParameter &p) :
    param_(p)
{
    maxAngle=M_PI/180.0f*10.0f;//10° max angle(maybe make it dependant on distance)
    //minCosAngle=cos(maxAngle);

    maxBlockAngle=M_PI/180.0f*10.0f;
    minCosBlockAngle=cos(maxBlockAngle);

    //define the functions for thresholds:

    maxInlierBlockDistFunc=[] (float z) -> float
    {
        //return 0.005f;//5mm
        float zmin=0.2f;
        float zmax=5.0f;
        float thresholdmin=0.005f;//5mm
        float thresholdmax=0.07f;//5cm
        return thresholdmin+(z-zmin)*(z-zmin)/(zmax*zmax)*(thresholdmax-thresholdmin);
    };

    minCosBlockAngleFunc=[] (float z) -> float
    {
//        float maxBlockAngle_local=M_PI/180.0f*10.0f;
        //return cos(maxBlockAngle_local);
        float zmin=0.2f;//distance measurement starts at 0.4m
        float zmax=5.0f;
        float alphamin = 10.0f;
        float alphamax = 60.0f;//40
        float maxAngle_local = M_PI/180.0f * std::min(alphamin+(z-zmin)*(z-zmin)/(zmax*zmax)*(alphamax-alphamin),90.0f);
        return cos(maxAngle_local);
    };

    minCosAngleFunc = [] (float z) -> float {
        float zmin=0.2f;//distance measurement starts at 0.4m
        float zmax=5.0f;
        float alphamin=40.0f;
        float alphamax=90.0f;
        float maxAngle_local=M_PI/180.0f*std::min(alphamin+(z-zmin)*(z-zmin)/(zmax*zmax)*(alphamax-alphamin),90.0f);
        return cos(maxAngle_local);
    };


    maxInlierDistFunc = [] (float z) -> float {
        //return 0.01f;
        float zmin=0.2f;
        float zmax=5.0f;
        float thresholdmin=0.005f;//5mm
        float thresholdmax=0.05f;//10cm
        return thresholdmin+(z-zmin)*(z-zmin)/(zmax*zmax)*(thresholdmax-thresholdmin);
    };


    //TODOOOOO
    //maxInlierDistFunc;
            //maxInlierBockDistFunc=
            //minCosAngleFunc
            //minCosBlockAngleFunc

}

template<typename PointT>
void
PlaneExtractorTile<PointT>::compute()
{
    if(!allocateMemory())
        return; //no input data

    //calculating a patchwise plane description
    calculatePlaneSegments(param_.pointwiseNormalCheck_);

#ifdef DEBUG_IMAGES
    cv::imshow("debug",debug);
#endif

    //setting the list of planes to zero
    //TODO: this list of planes is not needed anyway
    /*for(int i=0;i<heightBlocks*widthBlocks+1;i++){
        planeList[i].plane=Eigen::Vector3f(0,0,0);
        planeList[i].nrElements=0;
    }*/
    //cv::Mat patchIds(heightBlocks,widthBlocks,CV_32SC1);//TODO: init this later
    patchIds.setTo(cv::Scalar(0));//test if this is necessary (but i think so)
    maxId=0;//everytime a new id is created, it is done by this number

    rawPatchClustering();

    //create vector of planes
    resultingPlanes.clear();//clear the vector of resulting planes
    all_planes_.clear();
    plane_inliers_.clear();

    std::vector<int> newPlaneIds (maxId+1);
    int newId=1;

    for(int i=0;i<=maxId;i++)
    {
        const PlaneMatrix &pm = planeMatrices[i];
        if(pm.nrPoints > param_.minNrPatches_ * param_.patchDim_*param_.patchDim_)
        {
            newPlaneIds[i]=newId++;
            Plane pl;
            const Eigen::Vector4f &p = calcPlaneFromMatrix(pm);
            pl.plane = p;
            pl.nrElements = planeMatrices[i].nrPoints;
            resultingPlanes.push_back(pl);
            all_planes_.push_back( p );
            std::vector<int> plane_inliers_tmp;// (pl.nrElements, 0);
            plane_inliers_.push_back( plane_inliers_tmp );
        }
        else
            newPlaneIds[i]=0;
    }

    for(int i=0;i<rowsOfPatches;i++)
    {
        for(int j=0;j<colsOfPatches;j++)
        {
            int planeId=patchIds.at<int>(i,j);
            int newPlaneId=newPlaneIds[planeId];
            patchIds.at<int>(i,j)=newPlaneId;

            const PlaneSegment &p = planes[i][j];
            const Eigen::Vector4f &plane = p.plane; //Eigen::Vector4f(p.x,p.y,p.z,p.d);
            if(newPlaneId)
            {
                //Mark the pixel in the segmentation map for the already existing patches
                if(planes[i][j].nrInliers > minAbsBlockInlier)
                {
                    for(int k=0; k<param_.patchDim_; k++)
                    {
                        for(int l=0; l<param_.patchDim_; l++)
                        {
                            if(segmentation.at<int>(i*param_.patchDim_+k, j*param_.patchDim_+l)==-1)
                            {
                                //we already touched these points and found out if they are in there.(they are marked with -1)
                                //debug.at<cv::Vec3b>(i*blockDim+k,j*blockDim+l)=cv::Vec3b(0,255,0);
                                segmentation.at<int>(i*param_.patchDim_+k, j*param_.patchDim_+l) = newPlaneId;
                                const Eigen::Vector3f &point = cloud_->at(j*param_.patchDim_+l, i*param_.patchDim_+k).getVector3fMap();
                                float distance = fabs(dist2plane(point,plane));
                                if(param_.doZTest_)
                                {
                                    //setting the zBuffer to zero effectively sets these patches to be fixed
                                    zBuffer.at<float>(i*param_.patchDim_+k,j*param_.patchDim_+l) = distance*0.0f;
                                }
                            }
                        }
                    }
                }
                else
                {
                    for(int k=0;k<param_.patchDim_;k++)
                    {
                        for(int l=0;l<param_.patchDim_;l++)
                        {
                            if(segmentation.at<int>(i*param_.patchDim_+k,j*param_.patchDim_+l)==-1)
                                segmentation.at<int>(i*param_.patchDim_+k,j*param_.patchDim_+l)=0; //segmentation is set to zero at every other points
                        }
                    }
                }

            }
        }
    }

    //postProcessing(newId);
/*
    std::cout << "number of resulting planes" << resultingPlanes.size() << std::endl;
    cv::imshow("beforePost",generateColorCodedTexture());
    for(int i=0;i<resultingPlanes.size();i++){
        Plane pl=resultingPlanes[i];
        Eigen::Vector4f p(pl.plane[0],pl.plane[1],pl.plane[2],0);
        cv::imshow("planeDebug",generateDebugTextureForPlane(p,i+1));
        cv::waitKey();
    }*/
#ifdef DEBUG_IMAGES
    cv::imshow("beforePost",generateColorCodedTexture());
#endif
    postProcessing(param_.pointwiseNormalCheck_, param_.doZTest_);

    for(size_t v=0; v<cloud_->height; v++)
    {
        for(size_t u=0; u<cloud_->width; u++)
        {
            int label = segmentation.at<int>(v,u);

            if(label > 0)
                plane_inliers_[label-1].push_back( v*cloud_->width + u);
        }
    }

#ifdef DEBUG_IMAGES
    cv::imshow("afterPost",generateColorCodedTexture());
    cv::waitKey();
#endif

}


#define PCL_INSTANTIATE_PlaneExtractorTile(T) template class V4R_EXPORTS PlaneExtractorTile<T>;
PCL_INSTANTIATE(PlaneExtractorTile, PCL_XYZ_POINT_TYPES )

}
