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

#include <v4r/tomgine/tgScene.h>
#include <stdio.h>

using namespace TomGine;

tgScene::tgScene() :
  m_displaylist(0), m_scale(false), m_center(false), m_bb(false)
{
}

void tgScene::RenderRecursive(const aiNode* nd) const
{
  aiMatrix4x4 m = nd->mTransformation;

  // update transform
  aiTransposeMatrix4(&m);
  glPushMatrix();
  glMultMatrixf((float*) &m);

  // draw all meshes assigned to this node
  for (unsigned n = 0; n < nd->mNumMeshes; ++n)
  {
    const aiMesh* mesh = m_scene->mMeshes[nd->mMeshes[n]];
    ApplyMaterial(m_scene->mMaterials[mesh->mMaterialIndex]);

    if (mesh->mNormals == NULL)
      glDisable( GL_LIGHTING);
    else
      glEnable(GL_LIGHTING);

    for (unsigned t = 0; t < mesh->mNumFaces; ++t)
    {
      const aiFace* face = &mesh->mFaces[t];
      GLenum face_mode;

      switch (face->mNumIndices) {
      case 1:
        face_mode = GL_POINTS;
        break;
      case 2:
        face_mode = GL_LINES;
        break;
      case 3:
        face_mode = GL_TRIANGLES;
        break;
      default:
        face_mode = GL_POLYGON;
        break;
      }

      glBegin(face_mode);
      for (unsigned i = 0; i < face->mNumIndices; i++)
      {
        int index = face->mIndices[i];
        if (mesh->mColors[0] != NULL)
          glColor4fv((GLfloat*) &mesh->mColors[0][index]);
        if (mesh->mNormals != NULL)
          glNormal3fv(&mesh->mNormals[index].x);
        glVertex3fv(&mesh->mVertices[index].x);
      }
      glEnd();
    }

  }

  // draw all children
  for (unsigned n = 0; n < nd->mNumChildren; ++n)
    RenderRecursive(nd->mChildren[n]);

  glPopMatrix();
}

void tgScene::Draw()
{
  if (m_scale) // scale the whole asset to fit into our view frustum
  {
    float tmp = scene_max.x - scene_min.x;
    tmp = std::max(scene_max.y - scene_min.y, tmp);
    tmp = std::max(scene_max.z - scene_min.z, tmp);
    tmp = 2.f / tmp;
    glScalef(tmp, tmp, tmp);
  }

  if (m_center) // center the model
    glTranslatef(-scene_center.x, -scene_center.y, -scene_center.z);

  if (m_displaylist == 0)
  {
    m_displaylist = glGenLists(1);
    glNewList(m_displaylist, GL_COMPILE);
    RenderRecursive(m_scene->mRootNode);
    glEndList();
  }

  glCallList( m_displaylist);
}

void tgScene::color4_to_float4(const aiColor4D *c, float f[4])
{
  f[0] = c->r;
  f[1] = c->g;
  f[2] = c->b;
  f[3] = c->a;
}

// ----------------------------------------------------------------------------
void tgScene::set_float4(float f[4], float a, float b, float c, float d)
{
  f[0] = a;
  f[1] = b;
  f[2] = c;
  f[3] = d;
}

void tgScene::ApplyMaterial(const aiMaterial* material) const
{
  float c[4];

  GLenum fill_mode;
  int ret1, ret2;
  aiColor4D diffuse;
  aiColor4D specular;
  aiColor4D ambient;
  aiColor4D emission;
  float shininess, strength;
  int two_sided;
  int wireframe;
  unsigned int max;

  set_float4(c, 0.8f, 0.8f, 0.8f, 1.0f);
  if (AI_SUCCESS == aiGetMaterialColor(material, AI_MATKEY_COLOR_DIFFUSE, &diffuse))
    color4_to_float4(&diffuse, c);
  glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, c);

  set_float4(c, 0.0f, 0.0f, 0.0f, 1.0f);
  if (AI_SUCCESS == aiGetMaterialColor(material, AI_MATKEY_COLOR_SPECULAR, &specular))
    color4_to_float4(&specular, c);
  glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, c);

  set_float4(c, 0.2f, 0.2f, 0.2f, 1.0f);
  if (AI_SUCCESS == aiGetMaterialColor(material, AI_MATKEY_COLOR_AMBIENT, &ambient))
    color4_to_float4(&ambient, c);
  glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, c);

  set_float4(c, 0.0f, 0.0f, 0.0f, 1.0f);
  if (AI_SUCCESS == aiGetMaterialColor(material, AI_MATKEY_COLOR_EMISSIVE, &emission))
    color4_to_float4(&emission, c);
  glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, c);

  max = 1;
  ret1 = aiGetMaterialFloatArray(material, AI_MATKEY_SHININESS, &shininess, &max);
  if (ret1 == AI_SUCCESS)
  {
    max = 1;
    ret2 = aiGetMaterialFloatArray(material, AI_MATKEY_SHININESS_STRENGTH, &strength, &max);
    if (ret2 == AI_SUCCESS)
      glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, shininess * strength);
    else
      glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, shininess);
  } else
  {
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 0.0f);
    set_float4(c, 0.0f, 0.0f, 0.0f, 0.0f);
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, c);
  }

  max = 1;
  if (AI_SUCCESS == aiGetMaterialIntegerArray(material, AI_MATKEY_ENABLE_WIREFRAME, &wireframe, &max))
    fill_mode = wireframe ? GL_LINE : GL_FILL;
  else
    fill_mode = GL_FILL;
  glPolygonMode(GL_FRONT_AND_BACK, fill_mode);

  max = 1;
  if ((AI_SUCCESS == aiGetMaterialIntegerArray(material, AI_MATKEY_TWOSIDED, &two_sided, &max)) && two_sided)
    glDisable( GL_CULL_FACE);
  else
    glEnable(GL_CULL_FACE);
}

void tgScene::Scale(bool scale)
{
  if (!m_bb && scale)
    GetBoundingBox(scene_min, scene_max, scene_center);

  m_scale = scale;
}

void tgScene::Center(bool center)
{
  if (!m_bb && center)
    GetBoundingBox(scene_min, scene_max, scene_center);

  m_center = center;
}

void tgScene::GetBoundingBox(aiVector3D& min, aiVector3D& max, aiVector3D& center)
{
  aiMatrix4x4 trafo;
  aiIdentityMatrix4(&trafo);

  min.x = min.y = min.z = 1e10f;
  max.x = max.y = max.z = -1e10f;
  GetBoundingBox(m_scene->mRootNode, &min, &max, &trafo);

  center.x = (min.x + max.x) / 2.0f;
  center.y = (min.y + max.y) / 2.0f;
  center.z = (min.z + max.z) / 2.0f;

  m_bb = true;
}

// --------------------------------------------------------------
// Private
// --------------------------------------------------------------

void tgScene::GetBoundingBox(const aiNode* nd, aiVector3D* min, aiVector3D* max, aiMatrix4x4* trafo)
{
  aiMatrix4x4 prev;
  unsigned int n = 0, t;

  prev = *trafo;
  aiMultiplyMatrix4(trafo, &nd->mTransformation);

  for (; n < nd->mNumMeshes; ++n)
  {
    const aiMesh* mesh = m_scene->mMeshes[nd->mMeshes[n]];
    for (t = 0; t < mesh->mNumVertices; ++t)
    {

      aiVector3D tmp = mesh->mVertices[t];
      aiTransformVecByMatrix4(&tmp, trafo);

      min->x = std::min(min->x, tmp.x);
      min->y = std::min(min->y, tmp.y);
      min->z = std::min(min->z, tmp.z);

      max->x = std::max(max->x, tmp.x);
      max->y = std::max(max->y, tmp.y);
      max->z = std::max(max->z, tmp.z);
    }
  }

  for (n = 0; n < nd->mNumChildren; ++n)
    GetBoundingBox(nd->mChildren[n], min, max, trafo);

  *trafo = prev;
}
