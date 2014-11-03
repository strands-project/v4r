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


using namespace tg;


Mesh::Mesh(std::string filePath){

    m_scene = m_importer.ReadFile(filePath,0);

    if(!m_scene){
        fprintf(stderr,"[Mesh::constructor] %s",m_importer.GetErrorString());
        return;
    }

    if(m_scene->HasAnimations()){
        printf("wooohooo!!! animatiions\n");
    }
    if(m_scene->HasCameras()){
        printf("wohoooo! cameras\n");
    }
    if(m_scene->HasLights()){
        printf("lights!!\n");
    }
    if(m_scene->HasMaterials()){
        printf("materials:%d!\n",m_scene->mNumMaterials);
        for(int i=0;i<(int)m_scene->mNumMaterials;i++){
            aiMaterial* material = m_scene->mMaterials[i];

            aiString name;
            material->Get(AI_MATKEY_NAME,name);

            //material->

            printf("material%d name:%s\n",i,name.C_Str());
        }
    }
    if(m_scene->HasMeshes()){
        printf("meshes:%d!!\n",m_scene->mNumMeshes);
        for(int i=0;i<(int)m_scene->mNumMeshes;i++){
            aiMesh* mesh = m_scene->mMeshes[i];
            printf("faces %d, vertices %d, bones %d, animMeshes %d \n",mesh->mNumFaces,mesh->mNumVertices,mesh->mNumBones,mesh->mNumAnimMeshes);
        }

    }
    if(m_scene->HasTextures()){
        printf("textures!!!\n");
    }
    std::cout << std::endl;
}
Mesh::~Mesh(){

}

void Mesh::initInContext()
{
    glGenVertexArrays(1,&m_VAO);
    glBindVertexArray(m_VAO);


    if(m_scene->HasMeshes()){
        if(m_scene->mNumMeshes==1){
            aiMesh* mesh = m_scene->mMeshes[0];
            if(mesh->HasPositions()){
                printf("positions\n");
                glGenBuffers(1,&m_posVBO);
                glBindBuffer(GL_ARRAY_BUFFER,m_posVBO);
                glBufferData(GL_ARRAY_BUFFER,sizeof(glm::vec3)*mesh->mNumVertices,mesh->mVertices,GL_STATIC_DRAW);
                 m_center=glm::vec3(0);
                for(unsigned int i=0;i<mesh->mNumVertices;i++){
                    //printf("Vertex:%f,%f,%f \n",mesh->mVertices[i].x,mesh->mVertices[i].y,mesh->mVertices[i].z);
                     m_center += glm::vec3(mesh->mVertices[i].x,mesh->mVertices[i].y,mesh->mVertices[i].z);
                }
                m_center =m_center/(float)mesh->mNumVertices;
            }
            if(mesh->HasNormals()){
                printf("normals\n");
                glGenBuffers(1,&m_normalVBO);
                glBindBuffer(GL_ARRAY_BUFFER,m_normalVBO);
                glBufferData(GL_ARRAY_BUFFER,sizeof(glm::vec3)*mesh->mNumVertices,mesh->mNormals,GL_STATIC_DRAW);
            }
            if(mesh->HasVertexColors(0)){
                printf("vertex colors\n");

                glGenBuffers(1,&m_colorVBO);
                glBindBuffer(GL_ARRAY_BUFFER,m_colorVBO);
                glBufferData(GL_ARRAY_BUFFER,sizeof(glm::vec4)*mesh->mNumVertices,mesh->mColors[0],GL_STATIC_DRAW);

                for(unsigned int i=0;i<mesh->mNumVertices;i++){
                    //printf("colors:%f,%f,%f,%f \n",mesh->mColors[0].,mesh->mColors[i].y,mesh->mColors[i].z,mesh->mColors);

                }
            }
            /*for(int i=0;i<8;i++){
                if(mesh->HasVertexColors(i)){
                    printf("vertex colors%d\n",i);
                }
            }*/
            if(mesh->HasFaces()){
                printf("Mesh has faces\n");
                glGenBuffers(1,&m_indexVBO);
                glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,m_indexVBO);

                GLuint indexBufferData[mesh->mNumFaces*3];
                m_faceCount=mesh->mNumFaces;
                for(unsigned int i=0;i<mesh->mNumFaces;i++){
                    if(mesh->mFaces[i].mNumIndices==3){
                        indexBufferData[i*3+0]=mesh->mFaces[i].mIndices[0];
                        indexBufferData[i*3+1]=mesh->mFaces[i].mIndices[1];
                        indexBufferData[i*3+2]=mesh->mFaces[i].mIndices[2];
                        //printf("%d,%d,%d\n",indexBufferData[i*3+0],indexBufferData[i*3+1],indexBufferData[i*3+2]);
                    }

                }
                glBufferData(GL_ELEMENT_ARRAY_BUFFER,sizeof(GLuint)*3*mesh->mNumFaces,indexBufferData,GL_STATIC_DRAW);
            }

        }else{
            fprintf(stderr,"[Mesh::initInContext] More than one mesh is not supportet (yet)! \n");
            return;
        }
        for(int i=0;i<(int)m_scene->mNumMeshes;i++){


        }
    }



    m_program.compileShader(std::string(TOMGINE_5_SHADER) + "mesh.fsh");
    m_program.bindFragDataLocation(0,"fragColor");

    m_program.compileShader(std::string(TOMGINE_5_SHADER) + "mesh.vsh");


    m_program.link();
    m_program.bindAttribLocation(0,"pos");

    glBindBuffer(GL_ARRAY_BUFFER,m_posVBO);
    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,0,NULL);
    glEnableVertexAttribArray(0);
    //glVertexAttribBinding(0,0);

    m_program.bindAttribLocation(1,"color");
    glBindBuffer(GL_ARRAY_BUFFER,m_colorVBO);
    glVertexAttribPointer(1,4,GL_FLOAT,GL_FALSE,0,NULL);
    glEnableVertexAttribArray(1);
    /*GLuint location= glGetAttribLocation(m_program.getHandle(),"color");
    glBindBuffer(GL_ARRAY_BUFFER,m_colorVBO);


    glEnableVertexAttribArray(location);
    tg::GLUtils::checkForOpenGLError("oaga föla dazwischn");
    glVertexAttribPointer(location,4,GL_FLOAT,GL_FALSE,0,NULL);
    //glVertexAttribBinding(1,1);

    tg::GLUtils::checkForOpenGLError("oaga föla danoch");*/

    m_MVPUniform = m_program.getUniformLocation("MVP");


    /*m_blur.link();
    m_blur.bindAttribLocation(0,"postion");
    glBindBuffer(GL_ARRAY_BUFFER,m_quadVBO);
    glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,0,NULL);
    glEnableVertexAttribArray(0);
    glVertexAttribDivisor(0,0);

    m_blur.use();
    m_blur.setUniform("tex",0);
    m_vBlurSub = glGetSubroutineIndex(m_blur.getHandle(),GL_FRAGMENT_SHADER,"blurv");
    m_hBlurSub = glGetSubroutineIndex(m_blur.getHandle(),GL_FRAGMENT_SHADER,"blurh");*/
    glBindVertexArray(0);
}

void Mesh::removedWhileInContext()
{

}

void Mesh::draw(Scene *scene)
{
    glBindVertexArray(m_VAO);
    m_program.use();
    if(scene->getCam())
        scene->getCam()->applyMat(-1,-1,-1,m_MVPUniform);

    //now comes the indexed drawing:
    glDrawElements(GL_TRIANGLES,3*m_faceCount,GL_UNSIGNED_INT,0);




}

glm::vec3 Mesh::getCenter()
{
    return m_center;
}
