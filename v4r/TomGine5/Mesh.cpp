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
#include "Mesh.h"
#include <stdio.h>
#include <iostream>
#include <vector>

#include <assimp/scene.h>
#include <assimp/Importer.hpp>
#include <iostream>
#include <fstream>

using namespace tg;

Mesh::Mesh()
{
  m_mesh = new aiMesh();
}

Mesh::Mesh(std::string filePath)
{
  Assimp::Importer importer;
  const aiScene* scene = importer.ReadFile(filePath,0);

  if(!scene)
  {
    printf("[Mesh::Mesh] %s",importer.GetErrorString());
    throw std::runtime_error("[Mesh::constructor] Error loading file.");
  }

  if(!scene->HasMeshes())
  {
    printf("[Mesh::Mesh] Error, file '%s' does not contain a mesh.", filePath.c_str());
    throw std::runtime_error("[Mesh::Mesh] Error, file does not contain a mesh.");
  }

  m_mesh = new aiMesh();
  (*m_mesh) = (*scene->mMeshes[0]);

  //  if(scene->HasMeshes())
  //  {
  //    for(int i=0;i<(int)scene->mNumMeshes;i++)
  //    {
  //      aiMesh* mesh = scene->mMeshes[i];
  //      printf("[Mesh::Mesh] faces %d, vertices %d, bones %d, animMeshes %d \n",mesh->mNumFaces,mesh->mNumVertices,mesh->mNumBones,mesh->mNumAnimMeshes);
  //    }
  //  }

}

Mesh::~Mesh()
{
  delete m_mesh;
}

void Mesh::initInContext(Scene *scene)
{
  m_shader = scene->GetShaderDiffuse();
  m_shader->use();

  glGenVertexArrays(1,&m_VAO);
  glBindVertexArray(m_VAO);

  // init mesh data
  if(m_mesh->HasPositions())
  {
    glGenBuffers(1,&m_posVBO);
    glBindBuffer(GL_ARRAY_BUFFER,m_posVBO);
    glBufferData(GL_ARRAY_BUFFER,sizeof(glm::vec3)*m_mesh->mNumVertices,m_mesh->mVertices,GL_STATIC_DRAW);
    m_center=glm::vec3(0);
    for(unsigned int i=0;i<m_mesh->mNumVertices;i++)
      m_center += glm::vec3(m_mesh->mVertices[i].x,m_mesh->mVertices[i].y,m_mesh->mVertices[i].z);
    m_center =m_center/(float)m_mesh->mNumVertices;
    tg::GLUtils::checkForOpenGLError("[Mesh::initInContext] positions");
  }
  if(m_mesh->HasNormals())
  {
    glGenBuffers(1,&m_normalVBO);
    glBindBuffer(GL_ARRAY_BUFFER,m_normalVBO);
    glBufferData(GL_ARRAY_BUFFER,sizeof(glm::vec3)*m_mesh->mNumVertices,m_mesh->mNormals,GL_STATIC_DRAW);
    tg::GLUtils::checkForOpenGLError("[Mesh::initInContext] normals");
  }
  if(m_mesh->HasVertexColors(0))
  {
    glGenBuffers(1,&m_colorVBO);
    glBindBuffer(GL_ARRAY_BUFFER,m_colorVBO);
    glBufferData(GL_ARRAY_BUFFER,sizeof(glm::vec4)*m_mesh->mNumVertices,m_mesh->mColors[0],GL_STATIC_DRAW);
    tg::GLUtils::checkForOpenGLError("[Mesh::initInContext] colors");
  }
  else
  {
    std::vector<glm::vec4> color(m_mesh->mNumVertices, glm::vec4(1.0));
    glGenBuffers(1,&m_colorVBO);
    glBindBuffer(GL_ARRAY_BUFFER,m_colorVBO);
    glBufferData(GL_ARRAY_BUFFER,sizeof(glm::vec4)*m_mesh->mNumVertices,&color[0],GL_STATIC_DRAW);
    tg::GLUtils::checkForOpenGLError("[Mesh::initInContext] colors");
  }
  if(m_mesh->HasFaces())
  {
    glGenBuffers(1,&m_IBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,m_IBO);

    std::vector<GLuint> indexBufferData;
    m_faceCount=0;
    for(unsigned int i=0;i<m_mesh->mNumFaces;i++)
    {
      const aiFace& face = m_mesh->mFaces[i];
      if(face.mNumIndices==3)
      {
        indexBufferData.push_back(face.mIndices[0]);
        indexBufferData.push_back(face.mIndices[1]);
        indexBufferData.push_back(face.mIndices[2]);
        m_faceCount++;
        //printf("%d,%d,%d\n",indexBufferData[i*3+0],indexBufferData[i*3+1],indexBufferData[i*3+2]);
      }
    }
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,sizeof(GLuint)*indexBufferData.size(),&indexBufferData[0],GL_STATIC_DRAW);
    tg::GLUtils::checkForOpenGLError("[Mesh::initInContext] colors");
  }



  GLuint posLoc =   m_shader->getAttribLocation("VertexPosition");
  GLuint normLoc =  m_shader->getAttribLocation("VertexNormal");
  GLuint colorLoc = m_shader->getAttribLocation("ColorDiffuse");

  glBindBuffer(GL_ARRAY_BUFFER,m_posVBO);
  glEnableVertexAttribArray(posLoc);
  glVertexAttribPointer(posLoc,3,GL_FLOAT,GL_FALSE,0,NULL);

  glBindBuffer(GL_ARRAY_BUFFER,m_normalVBO);
  glEnableVertexAttribArray(normLoc);
  glVertexAttribPointer(normLoc,3,GL_FLOAT,GL_FALSE,0,NULL);

  glBindBuffer(GL_ARRAY_BUFFER,m_colorVBO);
  glEnableVertexAttribArray(colorLoc);
  glVertexAttribPointer(colorLoc,4,GL_FLOAT,GL_FALSE,0,NULL);

  glBindVertexArray(0);
}

void Mesh::removedWhileInContext()
{

}

void Mesh::draw(Scene *scene)
{
  const glm::mat4& cam_modelview = scene->getCam()->getModelView();
  const glm::mat4& cam_projection = scene->getCam()->getProjectionMatrix();

  poseMutex.lock();
  glm::mat4 modelview = cam_modelview * m_pose;
  poseMutex.unlock();

  glm::mat3 normalmatrix(modelview);
  glm::mat4 MVP = cam_projection * modelview;

  m_shader->use();
  m_shader->setUniform("ModelViewMatrix", modelview);
  m_shader->setUniform("NormalMatrix", normalmatrix);
  m_shader->setUniform("ProjectionMatrix", cam_projection);
  m_shader->setUniform("MVP", MVP);

  glm::vec4 lightpos = glm::vec4(0.0,0.0,0.0,1.0);  // light-source attached to camera
  //  glm::vec4 lightpos = cam_modelview * glm::vec4(0.0,0.0,0.0,1.0);  // light-source in world coordinates
  glm::vec3 lightdiff(1.0,1.0,1.0);
  m_shader->setUniform("LightPosition", lightpos);
  m_shader->setUniform("LightDiffuse", lightdiff);

  glBindVertexArray(m_VAO);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,m_IBO);

  //now comes the indexed drawing:
  glDrawElements(GL_TRIANGLES,3*m_faceCount,GL_UNSIGNED_INT,0);
}

glm::vec3 Mesh::getCenter()
{
  return m_center;
}

void Mesh::setPose(const glm::mat4& pose)
{
  poseMutex.lock();
  m_pose = pose;
  poseMutex.unlock();
}

void Mesh::ExportAssimpMesh(std::string filename)
{
  printf("[Mesh::ExportAssimpMesh] Error, function not implemented.\n");
//  aiScene scene;
//  scene.mRootNode = new aiNode();

//  scene.mMeshes = new aiMesh*[1];
//  scene.mNumMeshes = 1;
//  scene.mMeshes[0] = m_mesh;

//  scene.mRootNode->mMeshes = new unsigned[1];
//  scene.mRootNode->mMeshes[0] = 0;
//  scene.mRootNode->mNumMeshes = 1;

//  Assimp::Exporter exporter;
//  exporter.Export(&scene, ".ply", filename);
}

void Mesh::ExportToPLY(const aiMesh *mesh, std::string filename)
{
  std::ofstream file;
  file.open(filename.c_str());

  if(!file.is_open())
    throw std::runtime_error("[Mesh::ExportToPLY] Error cannot open file.");

  file << "ply" << std::endl;
  file << "format ascii 1.0" << std::endl;
  file << "comment Created by TomGine 5" << std::endl;
  file << "element vertex " << mesh->mNumVertices << std::endl;
  file << "property float x" << std::endl;
  file << "property float y" << std::endl;
  file << "property float z" << std::endl;
  file << "element face " << mesh->mNumFaces << std::endl;
  file << "property list uchar uint vertex_indices" << std::endl;
  file << "end_header" << std::endl;

  for(unsigned i=0; i<mesh->mNumVertices; i++)
  {
    const aiVector3D& v = mesh->mVertices[i];
    file <<  v.x << " " << v.y << " " << v.z << std::endl;
  }

  for(unsigned i=0; i<mesh->mNumFaces; i++)
  {
    const aiFace& face = mesh->mFaces[i];
    file << face.mNumIndices;
    for(unsigned j=0; j<face.mNumIndices; j++)
      file << " " << face.mIndices[j];
    file << std::endl;
  }

  file.close();
}
