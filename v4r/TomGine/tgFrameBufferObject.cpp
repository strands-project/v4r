/*
 * Software License Agreement (GNU General Public License)
 *
 *  Copyright (c) 2011, Thomas Mörwald
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
 * @author thomas.moerwald
 *
 */
#include "tgFrameBufferObject.h"
#include "tgError.h"
#include <opencv2/highgui/highgui.hpp>
#include <stdexcept>

using namespace TomGine;

tgFrameBufferObject::tgFrameBufferObject(unsigned w, unsigned h, GLint colorInternal, GLint depthInternal)
{

  m_width = w;
  m_height = h;

  texColor.Bind();
  texColor.Load(NULL, m_width, m_height, colorInternal, GL_RGBA, GL_UNSIGNED_BYTE);
  tgCheckError("[main] fbo_tex");

  texDepth.Bind();
  texDepth.Load(NULL, m_width, m_height, depthInternal, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE);
  tgCheckError("[main] fbo_depth_tex");

  glGenFramebuffers(1, &m_fbo_id);
  Bind();

  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texColor.GetTextureID(), 0);
  tgCheckError("[tgFrameBufferObject::tgFrameBufferObject] attach color texture");

  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, texDepth.GetTextureID(), 0);
  tgCheckError("[tgFrameBufferObject::tgFrameBufferObject] attach depth texture");

  Unbind();

  if (tgCheckFBError(GL_FRAMEBUFFER, "[tgFrameBufferObject::tgFrameBufferObject]") != GL_FRAMEBUFFER_COMPLETE
      || tgCheckError("[tgFrameBufferObject::tgFrameBufferObject]") != GL_NO_ERROR)
  {
    std::string errmsg =
        std::string("[tgFrameBufferObject::tgFrameBufferObject] Error generating frame buffer objects");
    throw std::runtime_error(errmsg.c_str());
  }
  glDisable(GL_TEXTURE_2D);
}
tgFrameBufferObject::~tgFrameBufferObject()
{
  if (glIsFramebuffer(m_fbo_id))
    glDeleteFramebuffers(1, &m_fbo_id);
}

void tgFrameBufferObject::Clear()
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void tgFrameBufferObject::Bind()
{
  glBindFramebuffer(GL_FRAMEBUFFER, m_fbo_id);
  tgCheckError("[tgFrameBufferObject::Bind]");
}

void tgFrameBufferObject::Unbind()
{
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void tgFrameBufferObject::SaveColor(const char* filename)
{
  texColor.Bind();
  cv::Mat img(m_height, m_width, CV_8UC3);
  glGetTexImage(GL_TEXTURE_2D, 0, GL_BGR, GL_UNSIGNED_BYTE, img.data);
  glDisable(GL_TEXTURE_2D);
  tgCheckError("[tgFrameBufferObject::SaveColor]");
  cv::imwrite(filename, img);
}

void tgFrameBufferObject::SaveDepth(const char* filename)
{
  texDepth.Bind();
  cv::Mat img(m_height, m_width, CV_8U);
  glGetTexImage(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, img.data);
  glDisable(GL_TEXTURE_2D);
  tgCheckError("[tgFrameBufferObject::SaveDepth]");
  cv::imwrite(filename, img);
}
