/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2013, Thomas Mörwald
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Thomas Mörwald nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * @file tgScene.h
 * @author Thomas Mörwald
 * @date May 2013
 * @version 0.1
 * @brief aiScene rendering functions.
 */

#ifndef _TG_SCENE_H_
#define _TG_SCENE_H_

#include <assimp/cimport.h>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <GL/gl.h>
#include <v4r/core/macros.h>

namespace TomGine {

class V4R_EXPORTS tgScene
{
private:
  aiVector3D scene_min, scene_max, scene_center;

public:
  const aiScene* m_scene;
  GLuint m_displaylist;

  bool m_scale, m_center, m_bb;

public:
  tgScene();

  void RenderRecursive(const aiNode* nd) const;
  void Draw();

  static void color4_to_float4(const aiColor4D *c, float f[4]);
  static void set_float4(float f[4], float a, float b, float c, float d);

  void ApplyMaterial(const aiMaterial* material) const;

  void Scale(bool scale=true);
  void Center(bool center=true);
  void GetBoundingBox(aiVector3D& min, aiVector3D& max, aiVector3D& center);


private:
  void GetBoundingBox(const aiNode* nd, aiVector3D* min, aiVector3D* max, aiMatrix4x4* trafo);

};

}

#endif
