/******************************************************************************
 * Copyright (c) 2015 Simon Schreiberhuber
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
*      @author Simon Schreiberhuber (schreiberhuber@acin.tuwien.ac.at)
*      @date November, 2015
*/


#ifndef __V4R_DEPTHMAP_RENDERER__
#define __V4R_DEPTHMAP_RENDERER__


#include <GL/glew.h>

#include <opencv2/opencv.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <eigen3/Eigen/Eigen>
#include <v4r/core/macros.h>
#include "dmRenderObject.h"



//This is for testing the offscreen rendering context:

namespace v4r{
const size_t maxMeshSize=1000000; //this will lead to a big ssbo^^ (16mb)
/**
 * @brief The Renderer class
 * renders a depth map from a model (every model you can load via assimp)
 * Altough technically not part of the same problem this class can give points
 * along a sphere. (Good for generating views to an object)
 */
class V4R_EXPORTS DepthmapRenderer{
private:

    static bool glfwRunning;

    //hide the default constructor
    DepthmapRenderer();

    //pointer to the current model
    DepthmapRendererModel* model;

    //Shader for rendering all that stuff
    GLuint shaderProgram;
    GLuint projectionUniform;
    GLuint poseUniform;
    GLuint viewportResUniform;
    GLuint posAttribute;

    //Used to give each triangle its own id
    GLuint atomicCounterBuffer;

    //For each triangle the ssbo stores the number of pixel it would have if it weren't occluded
    //and also the surface area of the triangle
    GLuint SSBO;

    //framebuffer to render the model into
    GLuint FBO;
    GLuint zBuffer;
    GLuint depthTex;
    //stores the triangle id for each pixel
    GLuint indexTex;
    GLuint colorTex;

    //the obligatory VAO
    GLuint VAO;

    //Buffer for geometry
    GLuint VBO;
    GLuint IBO;

    //camera intrinsics:
    Eigen::Vector4f fxycxy;
    Eigen::Vector2i res;

    //Stores the camera pose:
    Eigen::Matrix4f pose;

    //this here is to create points as part of a sphere
    //The next two i stole from thomas mörwald
    int search_midpoint(int &index_start, int &index_end, size_t &n_vertices, int &edge_walk,
                       std::vector<int> &midpoint, std::vector<int> &start, std::vector<int> &end, std::vector<float> &vertices);
    void subdivide(size_t &n_vertices, size_t &n_edges, size_t &n_faces, std::vector<float> &vertices,
                   std::vector<int> &faces);

public:
    /**
     * @brief DepthmapRenderer
     * @param resx the resolution has to be fixed at the beginning of the program
     * @param resy
     */
    DepthmapRenderer(int resx,int resy);

    ~DepthmapRenderer();


    /**
     * @brief createSphere
     * @param r radius
     * @param subdivisions there are 12 points by subdividing you add a lot more to them
     * @return vector of poses around a sphere
     */
    std::vector<Eigen::Vector3f> createSphere(float r, size_t subdivisions);

    /**
     * @brief setIntrinsics
     * @param fx focal length
     * @param fy
     * @param cx center of projection
     * @param cy
     */
    void setIntrinsics(float fx,float fy,float cx,float cy);

    /**
     * @brief setModel
     * @param model
     */
    void setModel(DepthmapRendererModel* _model);

    /**
     * @brief getPoseLookingToCenterFrom
     * @param position
     * @return
     */
    Eigen::Matrix4f getPoseLookingToCenterFrom(Eigen::Vector3f position);

    /**
     * @brief setCamPose
     * @param pose
     * A 4x4 Matrix giving the pose
     */
    void setCamPose(Eigen::Matrix4f _pose);

    /**
     * @brief renderDepthmap
     * @param visibleSurfaceArea: Returns an estimate of how much of the models surface area
     *        is visible.
     * @param color: if the geometry contains color information this cv::Mat will contain
     *        a color image after calling this method. (otherwise it will be plain black)
     * @return a depthmap
     */
    cv::Mat renderDepthmap(float &visibleSurfaceArea, cv::Mat &color) const;


    /**
     * @brief renderPointcloud
     * @param visibleSurfaceArea: Returns an estimate of how much of the models surface area
     *        is visible.
     * @return
     */
    pcl::PointCloud<pcl::PointXYZ> renderPointcloud(float &visibleSurfaceArea) const;

    /**
     * @brief renderPointcloudColor
     * @param visibleSurfaceArea: Returns an estimate of how much of the models surface area
     *        is visible.
     * @return
     */
    pcl::PointCloud<pcl::PointXYZRGB> renderPointcloudColor(float &visibleSurfaceArea) const;
};
}


#endif /* defined(__DEPTHMAP_RENDERER__) */
