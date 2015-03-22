#include "ms_utils.h"
#include "v4r/on_nurbs/fitting_surface_depth_im.h"
#include <pcl/common/pca.h>
#include "v4r/on_nurbs/fitting_surface_depth.h"
#include <pcl/visualization/pcl_visualizer.h>

#undef NDEBUG

template<typename PointT>
float v4rOCTopDownSegmenter::MergeCandidate<PointT>::computePlaneErrorPoint(typename pcl::PointCloud<PointT>::Ptr & cloud,
                                                                            std::vector<int> & indices,
                                                                            Eigen::Vector3f & nn, float d, Eigen::Vector3f & point_on_plane,
                                                                            std::vector<Line> & los_)
{
    float error = 0.f;
    size_t incr = 1;

    for(size_t i=0; i < indices.size(); i+=incr)
    {
        //const Eigen::Vector3f & p = cloud->at(r1_->indices_[i]).getVector3fMap();
        //Eigen::Vector3f intersect = intersectLineWithPlane(los_[r1_->indices_[i]], nn, d, point_on_plane);
        //error += std::abs(cloud->points[r1_->indices_[i]].z - intersect[2]);

        //quicker way...
        register int idx = indices[i];

        /*float z_intersect = intersectLineWithPlaneZValue(los_[idx], nn, d, point_on_plane);
        //error += std::abs(cloud->points[idx].z - z_intersect);
        error += (cloud->points[idx].z - z_intersect) * (cloud->points[idx].z - z_intersect);*/

        Eigen::Vector3f intersect = intersectLineWithPlane(los_[idx], nn, d, point_on_plane);
        error += (intersect - cloud->points[idx].getVector3fMap()).squaredNorm();

    }

    return error;
}

template<typename PointT>
float v4rOCTopDownSegmenter::MergeCandidate<PointT>::computePointToPlaneErrorPoint(typename pcl::PointCloud<PointT>::Ptr & cloud,
                                                                                   std::vector<int> & indices,
                                                                                   Eigen::Vector3f & nn, float d, Eigen::Vector3f & point_on_plane)
{
    float error = 0.f;
    size_t incr = 1;

    for(size_t i=0; i < indices.size(); i+=incr)
    {
        Eigen::Vector3f p = cloud->at(indices[i]).getVector3fMap();
        Eigen::Vector3f v = p - point_on_plane;
        Eigen::Vector3f proj = p - v.dot(nn) * nn;
        error += (proj - p).norm();
        //error += std::abs(v.dot(nn));
    }

    return error;
}

template<typename PointT>
float v4rOCTopDownSegmenter::MergeCandidate<PointT>::computeColorErrorPoint(typename pcl::PointCloud<PointT>::Ptr & cloud,
                                                                            std::vector<Eigen::Vector3d> & lab_values,
                                                                            std::vector<int> & indices)
{
    float error = 0.f;
    size_t incr = 1;

    for(size_t i=0; i < indices.size(); i+=incr)
    {
        /*unsigned char r = cloud->at(indices[i]).r;
        unsigned char g = cloud->at(indices[i]).g;
        unsigned char b = cloud->at(indices[i]).b;

        Eigen::Vector3d color = Eigen::Vector3d(r,g,b);*/
        error += (lab_values[indices[i]] - color_mean_).squaredNorm();
    }

    return error;
}

template<typename PointT>
float v4rOCTopDownSegmenter::MergeCandidate<PointT>::computeColorError
(typename pcl::PointCloud<PointT>::Ptr & cloud,
 std::vector<Eigen::Vector3d> & lab_values)
{
    int n1, n2, n;
    n1 = (int)r1_->indices_.size();
    n2 = (int)r2_->indices_.size();
    n = n1 + n2;

    Eigen::Vector3d incremental_mean = (n1 * r1_->color_mean_ + n2 * r2_->color_mean_) / (n);

    color_mean_ = incremental_mean;

    float error = 0.f;
    //what is the color being made when merging these two regions?
    error = computeColorErrorPoint(cloud, lab_values, r1_->indices_);
    error += computeColorErrorPoint(cloud, lab_values, r2_->indices_);

    return error;
}

template<typename PointT>
float v4rOCTopDownSegmenter::MergeCandidate<PointT>::computePlaneError
(typename pcl::PointCloud<PointT>::Ptr &cloud, std::vector<Line> & los_)
{

    /*std::vector<int> indices;
    indices.reserve( r1_->indices_.size() + r2_->indices_.size() );
    indices.insert( indices.end(), r2_->indices_.begin(), r2_->indices_.end() );
    indices.insert( indices.end(), r1_->indices_.begin(), r1_->indices_.end() );
    EIGEN_ALIGN16 Eigen::Matrix3d covariance_matrix;
    Eigen::Vector4d xyz_centroid;
    pcl::computeMeanAndCovarianceMatrix (*cloud, indices, covariance_matrix, xyz_centroid);*/

    int n1, n2, n;
    n1 = (int)r1_->indices_.size();
    n2 = (int)r2_->indices_.size();
    n = n1 + n2;

    Eigen::Vector3d incremental_mean = (n1 * r1_->mean_ + n2 * r2_->mean_) / (n);

    Eigen::Matrix3d incremental_cov;
    Eigen::Matrix<double, 1, 9, Eigen::RowMajor> accu;
    accu = static_cast<double> (n1) / static_cast<double> (n) * r1_->accu_ +
            static_cast<double> (n2) / static_cast<double> (n) * r2_->accu_;

    incremental_cov.coeffRef (0) = accu [0] - accu [6] * accu [6];
    incremental_cov.coeffRef (1) = accu [1] - accu [6] * accu [7];
    incremental_cov.coeffRef (2) = accu [2] - accu [6] * accu [8];
    incremental_cov.coeffRef (4) = accu [3] - accu [7] * accu [7];
    incremental_cov.coeffRef (5) = accu [4] - accu [7] * accu [8];
    incremental_cov.coeffRef (8) = accu [5] - accu [8] * accu [8];
    incremental_cov.coeffRef (3) = incremental_cov.coeff (1);
    incremental_cov.coeffRef (6) = incremental_cov.coeff (2);
    incremental_cov.coeffRef (7) = incremental_cov.coeff (5);

    mean_ = incremental_mean;
    covariance_ = incremental_cov;
    accu_ = accu;

    EIGEN_ALIGN16 Eigen::Vector3d eigen_values;
    EIGEN_ALIGN16 Eigen::Matrix3d eigenVectors;

    /*std::cout << n1 << " " << n2 << std::endl;
    std::cout << covariance_ << std::endl;*/

    pcl::eigen33 (covariance_, eigenVectors, eigen_values);
    //pcl::eigen33 (covariance_matrix, eigenVectors, eigen_values);

    //the error is the distance in the z-axis from the point to the plane
    //point-to-plane distance (projection)

    Eigen::Vector3f nn;
    nn[0] = eigenVectors (0,0);
    nn[1] = eigenVectors (1,0);
    nn[2] = eigenVectors (2,0);

    Eigen::Vector3f point_on_plane = incremental_mean.cast<float>();
    float d = nn.dot(point_on_plane) * -1.f;

    float error = 0.f;
    /*error = computePlaneErrorPoint(cloud, r1_->indices_, nn, d, point_on_plane, los_);
    error += computePlaneErrorPoint(cloud, r2_->indices_, nn, d, point_on_plane, los_);*/

    error = computePointToPlaneErrorPoint(cloud, r1_->indices_, nn, d, point_on_plane);
    error += computePointToPlaneErrorPoint(cloud, r2_->indices_, nn, d, point_on_plane);

    plane_model_defined_ = true;
    planar_model_ = std::make_pair(nn, d);

    smoothness_ = 0.f;

    return error;
}

template<typename PointT>
float v4rOCTopDownSegmenter::MergeCandidate<PointT>::computePointToPlaneError
(typename pcl::PointCloud<PointT>::Ptr &cloud)
{

    int n1, n2, n;
    n1 = (int)r1_->indices_.size();
    n2 = (int)r2_->indices_.size();
    n = n1 + n2;

    Eigen::Vector3d incremental_mean = (n1 * r1_->mean_ + n2 * r2_->mean_) / (n);

    Eigen::Matrix3d incremental_cov;
    Eigen::Matrix<double, 1, 9, Eigen::RowMajor> accu;
    accu = static_cast<double> (n1) / static_cast<double> (n) * r1_->accu_ +
            static_cast<double> (n2) / static_cast<double> (n) * r2_->accu_;

    incremental_cov.coeffRef (0) = accu [0] - accu [6] * accu [6];
    incremental_cov.coeffRef (1) = accu [1] - accu [6] * accu [7];
    incremental_cov.coeffRef (2) = accu [2] - accu [6] * accu [8];
    incremental_cov.coeffRef (4) = accu [3] - accu [7] * accu [7];
    incremental_cov.coeffRef (5) = accu [4] - accu [7] * accu [8];
    incremental_cov.coeffRef (8) = accu [5] - accu [8] * accu [8];
    incremental_cov.coeffRef (3) = incremental_cov.coeff (1);
    incremental_cov.coeffRef (6) = incremental_cov.coeff (2);
    incremental_cov.coeffRef (7) = incremental_cov.coeff (5);

    mean_ = incremental_mean;
    covariance_ = incremental_cov;
    accu_ = accu;

    EIGEN_ALIGN16 Eigen::Vector3d eigen_values;
    EIGEN_ALIGN16 Eigen::Matrix3d eigenVectors;

    pcl::eigen33 (covariance_, eigenVectors, eigen_values);
    //point-to-plane distance (projection)

    Eigen::Vector3f nn;
    nn[0] = eigenVectors (0,0);
    nn[1] = eigenVectors (1,0);
    nn[2] = eigenVectors (2,0);

    Eigen::Vector3f point_on_plane = incremental_mean.cast<float>();
    float d = nn.dot(point_on_plane) * -1.f;

    float error = 0.f;
    error = computePointToPlaneErrorPoint(cloud, r1_->indices_, nn, d, point_on_plane);
    error += computePointToPlaneErrorPoint(cloud, r2_->indices_, nn, d, point_on_plane);

    plane_model_defined_ = true;
    planar_model_ = std::make_pair(nn, d);

    smoothness_ = 0.f;

    return error;
}

template<typename PointT>
float v4rOCTopDownSegmenter::MergeCandidate<PointT>::computeBsplineError
(const Eigen::VectorXd& cloud_z, int cloud_width, int order, int CPx, int CPy)
{
    throw std::runtime_error("[v4rOCTopDownSegmenter::MergeCandidate<PointT>::computeBsplineError] Use computeBsplineErrorUnorganized instead.");

    // merge indices
    std::vector<int> indices;
    indices.reserve(r1_->indices_.size()+r2_->indices_.size());
    indices.insert(indices.end(), r1_->indices_.begin(), r1_->indices_.end());
    indices.insert(indices.end(), r2_->indices_.begin(), r2_->indices_.end());

    // fit B-spline
    pcl::on_nurbs::FittingSurfaceDepthIM::ROI roi(indices, cloud_width); // defines domain of B-spline

    pcl::on_nurbs::FittingSurfaceDepthIM fit;
    fit.initSurface(order, CPx, CPy, roi);  // initialize B-spline (b = vec of control points)
    fit.initSolver(indices,cloud_width);    // initialize solver (K*b = z); (z = depth values)
    // i.e. fill K with basis functions and compute QR-decomposition
    fit.solve(cloud_z, indices);            // solves (K*b = z) b = R^-1 * Q^T * z

    // copy B-spline model
    bspline_model_ = fit.getSurface();

    // compute point-wise error
    Eigen::VectorXd e = fit.GetError(cloud_z, indices);  // get error: returns (K*b-z)

    //Todo: Tom => sum of curvatures over the domain?

    double curvature_sqr(0.0);
    double ptd[6];
    for(int i=0; i < (static_cast<int>(indices.size())); i++)
    {
        int u = i % cloud_width;
        int v = i / cloud_width;
        bspline_model_.Evaluate(u,v,2,1,ptd);
        double c = (ptd[3]+ptd[5])*0.5;
        curvature_sqr += (c*c);
    }

    smoothness_ = curvature_sqr;
    //smoothness_ = sum of curvatures

    return e.squaredNorm();
}

template<typename PointT>
float v4rOCTopDownSegmenter::MergeCandidate<PointT>::computeBsplineErrorUnorganized
(typename pcl::PointCloud<PointT>::Ptr & cloud, int order, int CPx, int CPy)
{
    // merge indices
    std::vector<int> indices;
    indices.reserve(r1_->indices_.size()+r2_->indices_.size());
    indices.insert(indices.end(), r1_->indices_.begin(), r1_->indices_.end());
    indices.insert(indices.end(), r2_->indices_.begin(), r2_->indices_.end());

    pcl::IndicesPtr ind;
    ind.reset(new std::vector<int>(indices));

    pca_bspline_.reset(new pcl::PCA<PointT>());
    //pcl::PCA<PointT> basis;
    pca_bspline_->setInputCloud(cloud);
    pca_bspline_->setIndices(ind);

    // #################### PLANE PROJECTION #########################
    typename pcl::PointCloud<PointT>::Ptr proj_cloud(new pcl::PointCloud<PointT>);
    pca_bspline_->project(pcl::PointCloud<PointT>(*cloud,*ind), *proj_cloud);

    Eigen::Vector4f proj_min, proj_max, proj_del;
    pcl::getMinMax3D(*proj_cloud, proj_min, proj_max);
    proj_del = proj_max - proj_min;

    Eigen::MatrixXd points(proj_cloud->size(),3);
    for(size_t j=0; j< proj_cloud->size(); j++)
    {
        PointT& p = proj_cloud->at(j);
        points(j,0) = p.x;
        points(j,1) = p.y;
        points(j,2) = p.z;
    }

    pcl::on_nurbs::FittingSurfaceDepth::ROI roi(proj_min(0),proj_min(1),proj_del(0),proj_del(1));
    pcl::on_nurbs::FittingSurfaceDepth fit(order,CPx,CPy,roi,points);

    // convert from depth-nurbs to 3d nurbs
    pcl::on_nurbs::IncreaseDimension(fit.getSurface(), bspline_model_, 3);

    // compute x and y coordinates of controlpoints using Greville abcissae
    double gx[bspline_model_.CVCount(0)];
    double gy[bspline_model_.CVCount(1)];
    bspline_model_.GetGrevilleAbcissae(0, gx);
    bspline_model_.GetGrevilleAbcissae(1, gy);

    // assign Greville abcissae (required for curvature computation)
    for(int j=0; j<bspline_model_.CVCount(1); j++)
    {
      for(int i=0; i<bspline_model_.CVCount(0); i++)
      {
        ON_3dPoint cp;
        bspline_model_.GetCV(i,j,cp);

        cp.z = cp.x;  // assign depth value (from 1d depth nurbs, hence cp.x)
        cp.x = gx[i]; // assign Greville abcissae
        cp.y = gy[j]; // assign Greville abcissae

        bspline_model_.SetCV(i,j,cp);
      }
    }


    typename pcl::PointCloud<PointT>::Ptr reconstruction(new pcl::PointCloud<PointT>);

    int nder = 2; // number of derivatives
    int nvals = bspline_model_.Dimension()*(nder+1)*(nder+2)/2;
    double P[nvals];
    Eigen::Vector3d n, xu, xv, xuu, xvv, xuv;
    Eigen::Matrix2d II;

    // evaluate error and curvature for each point
    double error(0);
    smoothness_ = 0;
    for (std::size_t i = 0; i < points.rows(); i++)
    {
      bspline_model_.Evaluate (points(i,0), points(i,1), nder, bspline_model_.Dimension(), P);

      // positions
      PointT p1, p2;
      p1.x = P[0];    p1.y = P[1];    p1.z = P[2];
      pca_bspline_->reconstruct(p1,p2);
      reconstruction->push_back(p2);

      // 1st derivatives (for normals)
      xu(0) = P[3];    xu(1) = P[4];    xu(2) = P[5];
      xv(0) = P[6];    xv(1) = P[7];    xv(2) = P[8];

      n = xu.cross(xv);
      n.normalize();

      // 2nd derivatives (for curvature)
      xuu(0) = P[9];     xuu(1) = P[10];    xuu(2) = P[11];
      xuv(0) = P[12];    xuv(1) = P[13];    xuv(2) = P[14];
      xvv(0) = P[15];    xvv(1) = P[16];    xvv(2) = P[17];

      // fundamental form
      II(0,0) = n.dot(xuu);   // principal curvature along u
      II(0,1) = n.dot(xuv);
      II(1,0) = II(0,1);
      II(1,1) = n.dot(xvv);   // principal curvature along v

//      float mean = 0.5*( II(0,0)+II(1,1) ); // mean curvature
//      float gauss = II(0,0)*II(1,1) - II(0,1)*II(1,0); // gauss curvature

      smoothness_ += sqrt(II(0,0)*II(0,0) + II(1,1)*II(1,1)); // norm of principal curvatures

      error += (cloud->points[indices[i]].getVector3fMap() - p2.getVector3fMap()).norm();
    }

//    double z;
//    double error = 0;
//    for (std::size_t i = 0; i < proj_cloud->size(); i++)
//    {
//        bspline_model_.Evaluate (points(i,0), points(i,1), 0, 1, &z);

//        PointT p1, p2;
//        p1.x = points(i,0);
//        p1.y = points(i,1);
//        p1.z = z;

//        pca_bspline_reconstruct(p1,p2);
//        reconstruction->push_back(p2);

//        /*Eigen::Vector3f p = cloud->at(indices[i]).getVector3fMap();
//        Eigen::Vector3f v = p - p2.getVector3fMap();
//        error += std::pow(v.dot(e2), 2);*/
//        error += (cloud->points[indices[i]].getVector3fMap() - p2.getVector3fMap()).norm();

//    }

    /*
     * void EigenSpace_2_World(pcl::PCA<Point>& basis, TomGine::tgModel& mesh)
{
  Eigen::Vector3f e2 = basis.getEigenVectors().col(2);

  Point p1, p2;
  for(size_t i=0; i<mesh.m_vertices.size(); i++)
  {
    TomGine::tgVertex& v = mesh.m_vertices[i];
    p1.x = v.pos.x;
    p1.y = v.pos.y;
    p1.z = v.pos.z;

    basis.reconstruct(p1,p2);
    v.pos.x = p2.x;
    v.pos.y = p2.y;
    v.pos.z = p2.z;
    v.normal.x = e2(0);
    v.normal.y = e2(1);
    v.normal.z = e2(2);
  }
}

*/

    /*pcl::visualization::PCLVisualizer vis("compute B-spline");
    vis.addPointCloud<PointT>(cloud);
    pcl::visualization::PointCloudColorHandlerCustom<PointT> handler(reconstruction, 255,0,0);
    vis.addPointCloud<PointT>(reconstruction, handler, "proj");
    vis.spin();*/

    return error;

    //bspline_model_.EvaluatePoint()

    /*pcl::visualization::PCLVisualizer vis("compute B-spline");
    vis.addPointCloud<PointT>(cloud);
    pcl::visualization::PointCloudColorHandlerCustom<PointT> handler(proj_cloud, 255,0,0);
    vis.addPointCloud<PointT>(proj_cloud, handler, "proj");
    vis.spin();*/

    // trim mesh
    /*mesh.Clear();
    TomGine::tgShapeCreator::CreatePlaneXY(mesh,
                                           proj_min(0),proj_min(1),0,
                                           proj_del(0),proj_del(1),
                                           width>>2, height>>2);
    objectmodeling::Triangulation::trimmTgModel(b_curve, mesh);
    double z;
    for (std::size_t i = 0; i < mesh.m_vertices.size (); i++)
    {
      TomGine::tgVertex &v = mesh.m_vertices[i];
      d_surf.Evaluate (v.pos.x, v.pos.y, 0, 1, &z);
      v.pos.z = z;
    }*/

    return 0.f;
}

//rewrite function: do it efficiently... maybe based on the neighbours
//problem with neighbours is that we need to update the neighborhood of Neighbours(r2) to neighbour r1
//THE FUNCTION DEPENDS ON THE NUMBER OF INITIAL REGIONS... WHICH IS BAD...

template<typename PointT>
void v4rOCTopDownSegmenter::MergeCandidate<PointT>::merge
(std::vector< std::vector<int> > & adjacent,
 std::vector<MergeCandidate<PointT> > & candidates,
 std::vector< std::pair<int, int> > & recompute_candidates)
{
    int i_, j_;
    i_ = r1_->id_;
    j_ = r2_->id_;

    assert(i_ > j_);

    //be aware that this merge might have regions in common (add only once)
    std::set<int> unique_neighbours;
    for(int j=0; j < j_; j++)
    {
        if(adjacent[j_][j] < 0)
            continue;

        if(candidates[adjacent[j_][j]].isValid())
        {
            unique_neighbours.insert(j);
            //candidates[adjacent[j_][j]].valid_ = false;
        }
    }

    for(int j=0; j < i_; j++)
    {
        if(adjacent[i_][j] < 0)
            continue;

        if(candidates[adjacent[i_][j]].isValid())
        {
            unique_neighbours.insert(j);
            candidates[adjacent[i_][j]].valid_ = false;
            //adjacent[i_][j] = -1;
        }
    }

    for(int i=(i_ + 1); i < (int)adjacent.size(); i++)
    {
        if(adjacent[i][i_] < 0)
            continue;

        if(candidates[adjacent[i][i_]].isValid())
        {
            unique_neighbours.insert(i);
            candidates[adjacent[i][i_]].valid_ = false;
            //adjacent[i][i_] = -1;
        }
    }

    for(int i=(j_ + 1); i < (int)adjacent.size(); i++)
    {
        if(adjacent[i][j_] < 0)
            continue;

        if(candidates[adjacent[i][j_]].isValid())
        {
            unique_neighbours.insert(i);
            //candidates[adjacent[i][j_]].valid_ = false;
        }
    }

    recompute_candidates.resize(unique_neighbours.size());
    std::set<int>::iterator it;
    int i=0;
    for(it=unique_neighbours.begin(); it != unique_neighbours.end(); ++it, ++i)
    {
        assert(r1_->id_ != *it);
        recompute_candidates[i] = (std::make_pair<int, int>(std::max(r1_->id_, *it),
                                                            std::min(r1_->id_, *it)) );
    }

    //invalidate region2
    r2_->valid_ = false;

    //assign to r1 the mean and covariances of the merge
    if(current_model_type_ == PLANAR_MODEL_TYPE_)
    {
        r1_->mean_ = mean_;
        r1_->covariance_ = covariance_;
        r1_->accu_ = accu_;
        r1_->planar_error_ = planar_error_;
        r1_->planar_model_ = planar_model_;
        r1_->plane_model_defined_ = true;
        r1_->smoothness_ = smoothness_;
    }
    else if(current_model_type_ == BSPLINE_MODEL_TYPE_3x3 || current_model_type_ == BSPLINE_MODEL_TYPE_5x5)
    {
        r1_->bspline_error_ = bspline_error_;
        r1_->bspline_model_ = bspline_model_;
        r1_->bspline_model_defined_ = true;
        r1_->pca_bspline_ = pca_bspline_;
        r1_->current_model_type_ = current_model_type_;
        r1_->smoothness_ = smoothness_;
    }

    //merge indices from the second one to the first
    r1_->indices_.insert(r1_->indices_.end(), r2_->indices_.begin(), r2_->indices_.end());
    r1_->color_error_ = color_error_;
}

template<typename PointT>
float v4rOCTopDownSegmenter::Region<PointT>::computePlaneErrorPoint(typename pcl::PointCloud<PointT>::Ptr & cloud,
                                                                    std::vector<int> & indices,
                                                                    Eigen::Vector3f & nn, float d, Eigen::Vector3f & point_on_plane,
                                                                    std::vector<Line> & los_)
{
    float error = 0.f;
    size_t incr = 1;

    for(size_t i=0; i < indices.size(); i+=incr)
    {
        //const Eigen::Vector3f & p = cloud->at(r1_->indices_[i]).getVector3fMap();
        //Eigen::Vector3f intersect = intersectLineWithPlane(los_[r1_->indices_[i]], nn, d, point_on_plane);
        //error += std::abs(cloud->points[r1_->indices_[i]].z - intersect[2]);

        //quicker way...
        register int idx = indices[i];

        /*float z_intersect = intersectLineWithPlaneZValue(los_[idx], nn, d, point_on_plane);
        //error += std::abs(cloud->points[idx].z - z_intersect);
        error += (cloud->points[idx].z - z_intersect) * (cloud->points[idx].z - z_intersect);*/

        Eigen::Vector3f intersect = intersectLineWithPlane(los_[idx], nn, d, point_on_plane);
        error += (intersect - cloud->points[idx].getVector3fMap()).squaredNorm();

    }

    return error;
}

template<typename PointT>
float v4rOCTopDownSegmenter::Region<PointT>::computeColorErrorPoint(typename pcl::PointCloud<PointT>::Ptr & cloud,
                                                                    std::vector<Eigen::Vector3d> & lab_values,
                                                                    std::vector<int> & indices)
{
    float error = 0.f;
    size_t incr = 1;

    for(size_t i=0; i < indices.size(); i+=incr)
    {
        /*unsigned char r = cloud->at(indices[i]).r;
        unsigned char g = cloud->at(indices[i]).g;
        unsigned char b = cloud->at(indices[i]).b;

        Eigen::Vector3d color = Eigen::Vector3d(r,g,b);*/
        error += (lab_values[indices[i]] - color_mean_).squaredNorm();
    }

    return error;
}

template<typename PointT>
float v4rOCTopDownSegmenter::Region<PointT>::computeColorError
(typename pcl::PointCloud<PointT>::Ptr & cloud,
 std::vector<Eigen::Vector3d> & lab_values)
{
    return computeColorErrorPoint(cloud, lab_values, indices_);
}

template<typename PointT>
float v4rOCTopDownSegmenter::Region<PointT>::computePlaneError
(typename pcl::PointCloud<PointT>::Ptr &cloud, std::vector<Line> & los_)
{
    EIGEN_ALIGN16 Eigen::Vector3d eigen_values;
    EIGEN_ALIGN16 Eigen::Matrix3d eigenVectors;

    pcl::eigen33 (covariance_, eigenVectors, eigen_values);
    //pcl::eigen33 (covariance_matrix, eigenVectors, eigen_values);

    //the error is the distance in the z-axis from the point to the plane
    //point-to-plane distance (projection)

    Eigen::Vector3f nn;
    nn[0] = eigenVectors (0,0);
    nn[1] = eigenVectors (1,0);
    nn[2] = eigenVectors (2,0);

    Eigen::Vector3f point_on_plane = mean_.cast<float>();
    float d = nn.dot(point_on_plane) * -1.f;

    float error = 0.f;
    error = computePlaneErrorPoint(cloud, indices_, nn, d, point_on_plane, los_);

    plane_model_defined_ = true;
    planar_model_ = std::make_pair(nn, d);

    smoothness_ = 0.f;

    return error;
}

template<typename PointT>
float v4rOCTopDownSegmenter::Region<PointT>::computePointToPlaneError
(typename pcl::PointCloud<PointT>::Ptr &cloud)
{
    EIGEN_ALIGN16 Eigen::Vector3d eigen_values;
    EIGEN_ALIGN16 Eigen::Matrix3d eigenVectors;

    pcl::eigen33 (covariance_, eigenVectors, eigen_values);

    //point-to-plane distance (projection)

    Eigen::Vector3f nn;
    nn[0] = eigenVectors (0,0);
    nn[1] = eigenVectors (1,0);
    nn[2] = eigenVectors (2,0);

    Eigen::Vector3f point_on_plane = mean_.cast<float>();
    float d = nn.dot(point_on_plane) * -1.f;

    float error = 0.f;
    for(size_t i=0; i < indices_.size(); i++)
    {
        Eigen::Vector3f p = cloud->at(indices_[i]).getVector3fMap();
        Eigen::Vector3f v = p - point_on_plane;
        error += std::abs(v.dot(nn));
        //error += std::pow(v.dot(nn), 2);
    }

    plane_model_defined_ = true;
    planar_model_ = std::make_pair(nn, d);

    smoothness_ = 0.f;

    return error;
}

template class v4rOCTopDownSegmenter::Region<pcl::PointXYZRGB>;
template class v4rOCTopDownSegmenter::MergeCandidate<pcl::PointXYZRGB>;

