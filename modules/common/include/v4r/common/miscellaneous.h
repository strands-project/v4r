/******************************************************************************
 * Copyright (c) 2015 Thomas Faeulhammer
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

/**
*
*      @author Thomas Faeulhammer (faeulhammer@acin.tuwien.ac.at)
*      @date July, 2015
*      @brief some commonly used functions
*/

#ifndef V4R_COMMON_MISCELLANEOUS_H_
#define V4R_COMMON_MISCELLANEOUS_H_

#include <boost/dynamic_bitset.hpp>
#include <pcl/common/common.h>
#include <pcl/kdtree/flann.h>
#include <pcl/octree/octree.h>
#include <pcl/octree/octree_pointcloud_pointvector.h>
#include <pcl/octree/impl/octree_iterator.hpp>
#include <v4r/core/macros.h>
#include <omp.h>

namespace v4r
{

V4R_EXPORTS void transformNormals(const pcl::PointCloud<pcl::Normal> & normals_cloud,
                             pcl::PointCloud<pcl::Normal> & normals_aligned,
                             const Eigen::Matrix4f & transform);

V4R_EXPORTS void transformNormals(const pcl::PointCloud<pcl::Normal> & normals_cloud,
                             pcl::PointCloud<pcl::Normal> & normals_aligned,
                             const std::vector<int> & indices,
                             const Eigen::Matrix4f & transform);

V4R_EXPORTS inline void transformNormal(const Eigen::Vector3f & nt,
                            Eigen::Vector3f & normal_out,
                            const Eigen::Matrix4f & transform)
{
    normal_out[0] = static_cast<float> (transform (0, 0) * nt[0] + transform (0, 1) * nt[1] + transform (0, 2) * nt[2]);
    normal_out[1] = static_cast<float> (transform (1, 0) * nt[0] + transform (1, 1) * nt[1] + transform (1, 2) * nt[2]);
    normal_out[2] = static_cast<float> (transform (2, 0) * nt[0] + transform (2, 1) * nt[1] + transform (2, 2) * nt[2]);
}

/**
 * @brief Returns homogenous 4x4 transformation matrix for given rotation (quaternion) and translation components
 * @param q rotation represented as quaternion
 * @param trans homogenous translation
 * @return tf 4x4 homogeneous transformation matrix
 *
 */
V4R_EXPORTS inline Eigen::Matrix4f
RotTrans2Mat4f(const Eigen::Quaternionf &q, const Eigen::Vector4f &trans)
{
    Eigen::Matrix4f tf = Eigen::Matrix4f::Identity();;
    tf.block<3,3>(0,0) = q.normalized().toRotationMatrix();
    tf.block<4,1>(0,3) = trans;
    tf(3,3) = 1.f;
    return tf;
}


/**
 * @brief Returns homogenous 4x4 transformation matrix for given rotation (quaternion) and translation components
 * @param q rotation represented as quaternion
 * @param trans translation
 * @return tf 4x4 homogeneous transformation matrix
 *
 */
V4R_EXPORTS inline Eigen::Matrix4f
RotTrans2Mat4f(const Eigen::Quaternionf &q, const Eigen::Vector3f &trans)
{
    Eigen::Matrix4f tf = Eigen::Matrix4f::Identity();
    tf.block<3,3>(0,0) = q.normalized().toRotationMatrix();
    tf.block<3,1>(0,3) = trans;
    return tf;
}

/**
 * @brief Returns rotation (quaternion) and translation components from a homogenous 4x4 transformation matrix
 * @param tf 4x4 homogeneous transformation matrix
 * @param q rotation represented as quaternion
 * @param trans homogenous translation
 */
inline void
V4R_EXPORTS Mat4f2RotTrans(const Eigen::Matrix4f &tf, Eigen::Quaternionf &q, Eigen::Vector4f &trans)
{
    Eigen::Matrix3f rotation = tf.block<3,3>(0,0);
    q = rotation;
    trans = tf.block<4,1>(0,3);
}

V4R_EXPORTS
void voxelGridWithOctree(pcl::PointCloud<pcl::PointXYZRGB>::Ptr & cloud,
                                pcl::PointCloud<pcl::PointXYZRGB> & voxel_grided,
                                float resolution);


/**
 * @brief returns point indices from a point cloud which are closest to search points
 * @param full_input_cloud
 * @param search_points
 * @param indices
 * @param resolution (optional)
 */
template<typename PointInT>
V4R_EXPORTS void
getIndicesFromCloud(const typename pcl::PointCloud<PointInT>::ConstPtr & full_input_cloud,
                    const typename pcl::PointCloud<PointInT>::ConstPtr & search_points,
                    std::vector<int> & indices,
                    float resolution = 0.005f);

/**
 * @brief returns point indices from a point cloud which are closest to search points
 * @param full_input_cloud
 * @param search_points
 * @param indices
 * @param resolution (optional)
 */
template<typename PointT, typename Type>
V4R_EXPORTS void
getIndicesFromCloud(const typename pcl::PointCloud<PointT>::ConstPtr & full_input_cloud,
                    const typename pcl::PointCloud<PointT> & search_pts,
                    typename std::vector<Type> & indices,
                    float resolution = 0.005f);

DEPRECATED(template<typename DistType>
V4R_EXPORTS void convertToFLANN ( const std::vector<std::vector<float> > &data, boost::shared_ptr< typename flann::Index<DistType> > &flann_index));

DEPRECATED(template<typename DistType>
V4R_EXPORTS void nearestKSearch ( typename boost::shared_ptr< flann::Index<DistType> > &index, std::vector<float> descr, int k, flann::Matrix<int> &indices,
                                                  flann::Matrix<float> &distances ));

V4R_EXPORTS inline std::vector<size_t>
convertVecInt2VecSizet(const std::vector<int> &input)
{
    std::vector<size_t> v_size_t;
    v_size_t.resize(input.size());
    for (size_t i=0; i<input.size(); i++)
    {
        if(input[i] < 0)
            std::cerr << "Casting a negative integer to unsigned type size_t!" << std::endl;

        v_size_t[i] = static_cast<size_t>( input[i] );
    }
    return v_size_t;
}

V4R_EXPORTS inline std::vector<int>
convertVecSizet2VecInt(const std::vector<size_t> &input)
{
    std::vector<int> v_int;
    v_int.resize(input.size());
    for (size_t i=0; i<input.size(); i++)
    {
        if ( input[i] > static_cast<size_t>(std::numeric_limits<int>::max()) )
            std::cerr << "Casting an unsigned type size_t with a value larger than limits of integer!" << std::endl;

        v_int[i] = static_cast<int>( input[i] );
    }
    return v_int;
}

V4R_EXPORTS inline pcl::PointIndices
convertVecSizet2PCLIndices(const std::vector<size_t> &input)
{
    pcl::PointIndices pind;
    pind.indices.resize(input.size());
    for (size_t i=0; i<input.size(); i++)
    {
        if ( input[i] > static_cast<size_t>(std::numeric_limits<int>::max()) )
            std::cerr << "Casting an unsigned type size_t with a value larger than limits of integer!" << std::endl;

        pind.indices[i] = static_cast<int>( input[i] );
    }
    return pind;
}

V4R_EXPORTS inline std::vector<size_t>
convertPCLIndices2VecSizet(const pcl::PointIndices &input)
{
    std::vector<size_t> v_size_t;
    v_size_t.resize(input.indices.size());
    for (size_t i=0; i<input.indices.size(); i++)
    {
        if(input.indices[i] < 0)
            std::cerr << "Casting a negative integer to unsigned type size_t!" << std::endl;

        v_size_t[i] = static_cast<size_t>( input.indices[i] );
    }
    return v_size_t;
}

V4R_EXPORTS inline boost::dynamic_bitset<>
createMaskFromIndices(const std::vector<size_t> &indices, size_t image_size)
{
    boost::dynamic_bitset<> mask (image_size, 0);

    for (size_t obj_pt_id = 0; obj_pt_id < indices.size(); obj_pt_id++)
        mask.set(indices[obj_pt_id]);

    return mask;
}

/**
 * @brief createMaskFromIndices creates a boolean mask of all indices set
 * @param indices
 * @param image_size
 * @return
 */
V4R_EXPORTS inline boost::dynamic_bitset<>
createMaskFromIndices(const std::vector<int> &indices, size_t image_size)
{
    boost::dynamic_bitset<> mask (image_size, 0);

    for (size_t obj_pt_id = 0; obj_pt_id < indices.size(); obj_pt_id++)
        mask.set(indices[obj_pt_id]);

    return mask;
}


template<typename T>
V4R_EXPORTS std::vector<T>
createIndicesFromMask(const boost::dynamic_bitset<> &mask, bool invert=false)
{
    std::vector<T> out (mask.size());

    size_t kept=0;
    for(size_t i=0; i<mask.size(); i++)
    {
        if( ( mask[i] && !invert ) || ( !mask[i] && invert ))
        {
            out[kept] = static_cast<T>(i);
            kept++;
        }
    }
    out.resize(kept);
    return out;
}

/**
 * @brief computeMaskFromImageMap
 * @param image_map map indicating which pixel belong to which point of the point cloud.
 * @param nr_points number of points
 * @return bitmask indicating which points are represented in the image map
 */
V4R_EXPORTS boost::dynamic_bitset<>
computeMaskFromIndexMap( const Eigen::MatrixXi &image_map, size_t nr_points );

/**
  * @brief: Increments a boolean vector by 1 (LSB at the end)
  * @param v Input vector
  * @param inc_v Incremented output vector
  * @return overflow (true if overflow)
  */
V4R_EXPORTS bool
incrementVector(const std::vector<bool> &v, std::vector<bool> &inc_v);


/**
  * @brief: extracts elements from a vector indicated by some indices
  * @param[in] Input vector
  * @param[in] indices to extract
  * @return extracted elements
  */
template<typename T>
inline V4R_EXPORTS typename std::vector<T>
filterVector(const std::vector<T> &in, const std::vector<int> &indices)
{
    std::vector<T> out;
    out.reserve ( indices.size() );
    for(size_t i = 0; i < indices.size(); i++)
        out.push_back( in[ indices[i] ] );
    return out;
}

/**
 * @brief checks if value is in the range between min and max
 * @param[in] value to check
 * @param[in] min range
 * @param[in] max range
 * @return true if within range
 */
template<class T>
bool is_in_range(T value, T min, T max)
{
    return (value >= min) && (value <= max);
}

/**
 * @brief sorts a vector and returns sorted indices
 */
template <typename T>
std::vector<size_t> sort_indexes(const std::vector<T> &v) {

  // initialize original index locations
  std::vector<size_t> idx(v.size());
  for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;

  // sort indexes based on comparing values in v
  std::sort(idx.begin(), idx.end(),
       [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

  return idx;
}

/**
 * @brief computePointCloudProperties computes centroid and elongations along principal compenents for a point cloud
 * @param[in] cloud input cloud
 * @param centroid computed centroid of cloud
 * @param elongationsXYZ computes elongations along first, second and third principal component
 * @param eigenBasis matrix that aligns point cloud with eigenvectors
 * @param indices region of interest (if empty, whole point cloud will be processed)
 */
template<typename PointT>
V4R_EXPORTS
void
computePointCloudProperties(const pcl::PointCloud<PointT> &cloud, Eigen::Vector4f &centroid, Eigen::Vector3f &elongationsXYZ,  Eigen::Matrix4f &eigenBasis, const std::vector<int> &indices = std::vector<int>());


V4R_EXPORTS inline void
removeRow(Eigen::MatrixXd& matrix, unsigned int rowToRemove)
{
    unsigned int numRows = matrix.rows()-1;
    unsigned int numCols = matrix.cols();

    if( rowToRemove < numRows )
        matrix.block(rowToRemove,0,numRows-rowToRemove,numCols) = matrix.block(rowToRemove+1,0,numRows-rowToRemove,numCols);

    matrix.conservativeResize(numRows,numCols);
}

V4R_EXPORTS inline void
removeColumn(Eigen::MatrixXd& matrix, unsigned int colToRemove)
{
    unsigned int numRows = matrix.rows();
    unsigned int numCols = matrix.cols()-1;

    if( colToRemove < numCols )
        matrix.block(0,colToRemove,numRows,numCols-colToRemove) = matrix.block(0,colToRemove+1,numRows,numCols-colToRemove);

    matrix.conservativeResize(numRows,numCols);
}

V4R_EXPORTS inline void
removeRow(Eigen::MatrixXf& matrix, int rowToRemove)
{
    int numRows = matrix.rows()-1;
    int numCols = matrix.cols();

    if( rowToRemove < numRows )
        matrix.block(rowToRemove,0,numRows-rowToRemove,numCols) = matrix.block(rowToRemove+1,0,numRows-rowToRemove,numCols);

    matrix.conservativeResize(numRows,numCols);
}

V4R_EXPORTS inline void
removeColumn(Eigen::MatrixXf& matrix, int colToRemove)
{
    int numRows = matrix.rows();
    int numCols = matrix.cols()-1;

    if( colToRemove < numCols )
        matrix.block(0,colToRemove,numRows,numCols-colToRemove) = matrix.block(0,colToRemove+1,numRows,numCols-colToRemove);

    matrix.conservativeResize(numRows,numCols);
}

/**
 * @brief runningAverage computes incrementally the average of a vector
 * @param old_average
 * @param old_size the number of contributing vectors before updating
 * @param increment the new vector being added
 * @return
 */
inline Eigen::VectorXf
V4R_EXPORTS runningAverage (const Eigen::VectorXf &old_average, size_t old_size, const Eigen::VectorXf &increment) {    // update average point
    double w = old_size / double(old_size + 1);
    Eigen::VectorXf newAvg = old_average  * w + increment / double(old_size + 1);
    return newAvg;
}

/**
 * @brief computeRotationMatrixTwoAlignVectors Calculate Rotation Matrix to align Vector src to Vector target in 3d
 * @param src
 * @param target
 * @return 3x3 rotation matrix
 */
Eigen::Matrix3f
V4R_EXPORTS
computeRotationMatrixToAlignVectors(const Eigen::Vector3f &src, const Eigen::Vector3f &target);

template<typename PointT>
V4R_EXPORTS float computeMeshResolution (const typename pcl::PointCloud<PointT>::ConstPtr & input);

}


#endif
