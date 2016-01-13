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


#ifndef __DM_RENDERER_OBJECT__
#define __DM_RENDERER_OBJECT__


#include <GL/glew.h>
#include <GL/gl.h>


#include <eigen3/Eigen/Eigen>
#include <v4r/core/macros.h>



namespace v4r{



class V4R_EXPORTS DepthmapRendererModel{
private:
    friend class DepthmapRenderer;

    struct Vertex;

    Vertex *vertices;
    unsigned int *indices;
    int vertexCount;
    unsigned int indexCount;



    float scale;
    Eigen::Vector3f offset;
    bool color;
    bool geometry;

    /**
     * @brief loadToGPU
     *        Uploads geometry data to GPU and returns the OpenGL handles for Vertex and Index Buffers
     * @param VBO
     * @param IBO
     */
    void loadToGPU(GLuint &VBO,GLuint &IBO);

    /**
     * @brief getIndexCount
     * @return
     */
    unsigned int getIndexCount();



public:

    /**
     * @brief DepthmapRendererModel loads the geometry data and creates the necessary opengl ressources
     * @param file filename of the geometry file
     */
    DepthmapRendererModel(const std::string &file);


    ~DepthmapRendererModel();







    /**
     * @brief getScale
     * @return the models get fitted into a unity sphere.... Scale and Ofset gives the shift and offset of the object
     * Note: the translation was applied first... so to transform the pointcloud back you have to
     * apply the scale first and and translate afterwards
     */
    float getScale();



    /**
     * @brief hasColor
     * @return
     * true if the loaded mesh contains color information (suitable to generate a colored pointcloud)
     */
    bool hasColor();


    /**
     * @brief hasGeometry
     * @return
     * true if the geometry file loaded correctly and contains geometry.
     */
    bool hasGeometry();

    /**
     * @brief getOffset
     * @return the models get fitted into a unity sphere.... Scale and Offset gives the shift and offset of the object
     */
    Eigen::Vector3f getOffset();

};

}

#endif /* defined(__DM_RENDERER_OBJECT__) */
