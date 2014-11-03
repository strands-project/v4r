/*
 * Software License Agreement (GNU General Public License)
 *
 *  Copyright (c) 2014, Simon Schreiberhuber
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * @author simon.schreiberhuber
 *
 */
#ifndef _MESH_H_
#define _MESH_H_

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include "v4r/TomGine5/Scene.h"
#include "v4r/TomGine5/GLSLProgram.h"


namespace tg{
class Mesh : public SceneObject{
private:
    Assimp::Importer m_importer;
    const aiScene* m_scene;
    unsigned int m_faceCount;


    GLSLProgram m_program;
    GLuint m_MVPUniform;//uniforms
    GLuint m_projUniform;
    GLuint m_modelViewUniform;
    GLuint m_normalUniform;


    GLuint m_VAO;
    GLuint m_posVBO;
    GLuint m_normalVBO;
    GLuint m_colorVBO;
    GLuint m_indexVBO;

    glm::vec3 m_center;
public:
    Mesh(std::string filePath);
    ~Mesh();

    void initInContext();
    void removedWhileInContext();

    void draw(Scene* scene);
    glm::vec3 getCenter();
};

}
#endif
