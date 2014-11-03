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
#include "PointCloudProjected.h"
#include <stdio.h>

using namespace tg;

PointCloudProjected::PointCloudProjected()
{
  initialized=false;
  this->depthTex=0;
  this->depthChanged=false;

  this->rgbTex=0;
  this->rgbChanged=false;
  this->f=glm::vec2(600);
  this->uv0=glm::vec2(320,240);
  this->depthScale=1;
  this->pointSize=2;
  this->meanDepth = 0.0;
}

PointCloudProjected::PointCloudProjected(cv::Mat depth, cv::Mat rgb, glm::vec2 f, glm::vec2 uv0, float depthScale)
{
  initialized=false;
  this->programDotted=0;
  this->depthData=depth;
  this->depthTex=0;
  this->depthChanged=true;

  this->rgbData=rgb;
  this->rgbTex=0;
  this->rgbChanged=true;
  this->f=f;
  this->uv0=uv0;
  this->depthScale = depthScale;
  this->pointSize=2;
}

PointCloudProjected::~PointCloudProjected()
{
  if(depthTex!=0)
  {
    delete depthTex;
    depthTex=0;
  }
  if(rgbTex!=0)
  {
    delete rgbTex;
    rgbTex=0;
  }
  if(programDotted!=0)
  {
    delete programDotted;
    programDotted=0;
  }

  //TODO: this is just a bugfix...... DIRTY
  //rgbMutex.unlock();
  //depthMutex.unlock();
}

void PointCloudProjected::initInContext()
{
  printf("[PointCloudProjected::initInContext] initCloudInContext\n");
  fflush(stdout);
  //load shader
  this->programDotted=new GLSLProgram();

  programDotted->compileShader(std::string(TOMGINE_5_SHADER) + "projPointCloudDotted.fsh");
  programDotted->bindFragDataLocation(0,"fragColor");

  programDotted->compileShader(std::string(TOMGINE_5_SHADER) + "projPointCloudDotted.vsh");

  programDotted->link();
  programDotted->bindAttribLocation(0,"postion");

  /*uniform float depthScale;
    uniform float pointSize;
    uniform vec2 uv0;
    uniform vec2 f;
    uniform mat4 mvp;

    uniform sampler2D depthTex;
    uniform sampler2D rgbTex;*/
  //do the uniforms we already know:
  programDotted->use();
  programDotted->setUniform("depthScale",depthScale);
  programDotted->setUniform("pointSize",pointSize);
  programDotted->setUniform("uv0",uv0);
  programDotted->setUniform("f",f);

  programDotted->setUniform("depthTex",0);
  programDotted->setUniform("rgbTex",1);

  mvpUniform=programDotted->getUniformLocation("mvp");
  tg::GLUtils::checkForOpenGLError("[PointCloudProjected::initInContext]");

  initialized=true;
}

void PointCloudProjected::removedWhileInContext()
{
  //delete shader if created:
  if(programDotted){
    delete programDotted;
    programDotted=0;
  }
  //delete textures if created:
  if(depthTex){
    delete depthTex;
    depthTex=0;
  }
  if(rgbTex){
    delete rgbTex;
    rgbTex=0;
  }
}

void PointCloudProjected::draw(Scene *scene)
{
  if(initialized)
  {
    if(depthChanged)
    {
      depthMutex.lock();
      if(!depthTex)
      {
        depthTex=new GLTexture2D(depthData);
        depthTex->setFilter(GL_NEAREST,GL_NEAREST);
        glm::ivec2 r(depthData.rows, depthData.cols);

        //create VBO Data:
        std::vector<glm::vec2> vboData(r.x*r.y);
//        glm::vec2* vboData=new glm::vec2[depthData.cols*depthData.rows];

        printf("[PointCloudProjected::draw] %d %d  %d\n", r.x, r.y, r.x*r.y);

        int i=0;
        meanDepth = 0.0;
        for(int m=0; m<r.y; m++){
          for(int n=0; n<r.x; n++){

            //vboData[i]=glm::vec2((float)m/(float)depthData.rows,(float)n/(float)depthData.cols);
            vboData[i]=glm::vec2((float)n,(float)m);

            i++;
          }
        }
        meanDepth /= (i+1);

        //upload vbo data:
        glGenBuffers(1,&VBO);
        glBindBuffer(GL_ARRAY_BUFFER,VBO);
        glBufferData(GL_ARRAY_BUFFER,vboData.size()*sizeof(glm::vec2),&vboData[0],GL_STATIC_DRAW);

        // create ibo data
        polyCount=6*(r.x-1)*(r.y-1);
        gridIndexData.resize(polyCount);
        for(int m=0; m<(r.y-1); m++){
          for(int n=0; n<(r.x-1); n++){
            //create indices:
            gridIndexData[(m*(r.x-1)+n)*6+0] =  m    *(r.x-1) + n;
            gridIndexData[(m*(r.x-1)+n)*6+1] =  m    *(r.x-1) + n+1;
            gridIndexData[(m*(r.x-1)+n)*6+2] = (m+1) *(r.x-1) + n;

            gridIndexData[(m*(r.x-1)+n)*6+3] =  m    *(r.x-1) + n+1;
            gridIndexData[(m*(r.x-1)+n)*6+4] = (m+1) *(r.x-1) + n+1;
            gridIndexData[(m*(r.x-1)+n)*6+5] = (m+1) *(r.x-1) + n;
          }
        }

        GLuint imax(-UINT_MAX), imin(UINT_MAX);
        for(size_t i=0; i<gridIndexData.size(); i++)
        {
          GLuint& idx = gridIndexData[i];
          if(idx>imax)
            imax=idx;
          if(idx<imin)
            imin=idx;
        }

        printf("[PointCloudProjected::draw] max: %d  min: %d\n", imax, imin);


        //upload ibo data:
        glGenBuffers(1,&IBO);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,IBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,gridIndexData.size()*sizeof(GLuint),&gridIndexData[0],GL_STATIC_DRAW);

        glGenVertexArrays(1,&VAO);
        glBindVertexArray(VAO);

        //bind data to shader:
        programDotted->use();
        GLuint posLoc = programDotted->getAttribLocation("pos");
        glVertexAttribPointer(posLoc,2,GL_FLOAT,GL_FALSE,0,NULL);
        glEnableVertexAttribArray(posLoc);
        tg::GLUtils::checkForOpenGLError("[PointCloudProjected::draw]");
        glBindVertexArray(0);

      }else
      {
        depthTex->updateTexture(depthData);
      }
      depthChanged=false;
      depthMutex.unlock();
    } // if(depthChanged)

    if(rgbChanged)
    {
      rgbMutex.lock();
      if(!rgbTex){
        rgbTex=new GLTexture2D(rgbData);
        //rgbTex->imshow("rgbData");
        //cv::Mat rgbdatafter= rgbTex->getData();
        //cv::imshow("test rgbdata",rgbData);
        //cv::imshow("test rgbdatafter",rgbdatafter);
      }else{
        rgbTex->updateTexture(rgbData);

      }
      rgbChanged=false;
      rgbMutex.unlock();
    } // if(rgbChanged)

    //bind texture vbo and shader:
    //render:
    tg::GLUtils::checkForOpenGLError("[PointCloudProjected::draw]");
    glBindVertexArray(VAO);
    glActiveTexture(GL_TEXTURE0);
    depthTex->bind();
    glActiveTexture(GL_TEXTURE1);
    rgbTex->bind();
    programDotted->use();

    tg::GLUtils::checkForOpenGLError("[PointCloudProjected::draw]");
    scene->getCam()->applyMat(-1,-1,-1,mvpUniform);
    tg::GLUtils::checkForOpenGLError("[PointCloudProjected::draw]");
    glDrawArrays(GL_POINTS,0,depthData.rows*depthData.cols);
//    glDrawElements(GL_TRIANGLES,polyCount,GL_UNSIGNED_INT,&gridIndexData[0]);//GL_TRIANGLES

    tg::GLUtils::checkForOpenGLError("[PointCloudProjected::draw]");

  } // if(initialized)
}

void PointCloudProjected::updateDepthData(cv::Mat data)
{
  double mind(DBL_MAX), maxd(-DBL_MAX);
  for(int j=0; j<data.rows; j++)
  {
    for(int i=0; i<data.cols; i++)
    {
      double d = data.at<double>(i,j);
      if(d<mind)
        mind=d;
      if(d>maxd)
        maxd=d;
    }
  }
  printf("PointCloudProjected::updateDepthData depth: %f %f\n", mind, maxd);


  depthMutex.lock();
  this->depthData=data.clone();
  this->depthChanged=true;
  depthMutex.unlock();

}

void PointCloudProjected::updateRGBData(cv::Mat data)
{
  rgbMutex.lock();
  this->rgbData=data.clone();
  this->rgbChanged=true;
  rgbMutex.unlock();

}

void PointCloudProjected::setModelMat(glm::mat4 pos)
{
  this->modelMat = pos;
}
