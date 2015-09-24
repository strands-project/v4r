#include <v4r/tomgine/tgTextureModelAI.h>
#include <assimp/cimport.h>
#include <assimp/IOSystem.hpp>
#include <assimp/Exporter.hpp>
#include <assimp/postprocess.h>
#include <stdexcept>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <boost/filesystem.hpp>

#include <v4r/tomgine/tgTomGine.h>

using namespace TomGine;

tgTextureModelAI::tgTextureModelAI(const aiMesh &mesh, const aiMaterial &material, const std::string &file_path)
{

  // Vertices
  for (unsigned i = 0; i < mesh.mNumVertices; i++)
  {
    TomGine::tgVertex v;
    if (mesh.HasNormals())
    {
      v.normal.x = mesh.mNormals[i].x;
      v.normal.y = mesh.mNormals[i].y;
      v.normal.z = mesh.mNormals[i].z;
    }

    if (mesh.HasTextureCoords(0))
    {
      v.texCoord.x = mesh.mTextureCoords[0][i].x;
      v.texCoord.y = mesh.mTextureCoords[0][i].y;
    }

    //    if (mesh.HasVertexColors(0))
    //    {
    //      v.color[0] = static_cast<unsigned char> (255.0 * mesh.mColors[0][i].r);
    //      v.color[1] = static_cast<unsigned char> (255.0 * mesh.mColors[0][i].g);
    //      v.color[2] = static_cast<unsigned char> (255.0 * mesh.mColors[0][i].b);
    //      v.color[3] = static_cast<unsigned char> (255.0 * mesh.mColors[0][i].a);
    //    }

    v.pos.x = mesh.mVertices[i].x;
    v.pos.y = mesh.mVertices[i].y;
    v.pos.z = mesh.mVertices[i].z;

    m_vertices.push_back(v);
  }

  // Faces
  for (unsigned i = 0; i < mesh.mNumFaces; i++)
  {
    TomGine::tgFace f;

    aiFace &af = mesh.mFaces[i];
    if(af.mNumIndices<3 || af.mNumIndices>4)
      continue;

    for (unsigned j = 0; j < af.mNumIndices; j++)
      f.v.push_back(af.mIndices[j]);

    m_faces.push_back(f);
  }

  ComputeBoundingSphere();

  // Material
  CopyFromMaterial(material);

//  aiMaterial::CopyPropertyList(&m_materialAI, &material);
//  unsigned index = 0; //material.GetTextureCount(aiTextureType_DIFFUSE) - 1;
//  aiString path;

  unsigned num_tex = material.GetTextureCount(aiTextureType_DIFFUSE);
  std::string tex_file;

  if(num_tex>0)
  {
    aiString path;
    material.GetTexture(aiTextureType_DIFFUSE, num_tex-1, &path);
    tex_file = file_path + std::string(path.data);

    cv::Mat3b img = cv::imread(tex_file);
    cv::flip(img,img,0);

    if(img.empty())
    {
      printf("[tgTextureModelAI::tgTextureModelAI] Warning, could not load texture '%s'\n", tex_file.c_str());
      printf("[tgTextureModelAI::tgTextureModelAI] file_path: '%s'\n", file_path.c_str());
    }
    else
    {
      // resize such that it is power of 2
      cv::Size r( 2<<ilogb(img.rows), 2<<ilogb(img.cols) );
      m_tex_cv.resize(1);
      cv::resize(img, m_tex_cv[0], r);

//      printf("texture: %d %d | p: %d %d | s: %d %d\n", img.rows, img.cols, pr, pc, r.height, r.width);

      m_face_tex_id.assign(m_faces.size(),0);
    }

  }

//  printf("[tgTextureModelAI::tgTextureModelAI] vertices: %lu, faces: %lu, texture[%d]: '%s'\n", m_vertices.size(),
//      m_faces.size(), num_tex, tex_file.c_str());

}

void tgTextureModelAI::copy(aiMesh* mesh, aiMaterial* material)
{
  // vertices
  mesh->mNumVertices = m_vertices.size();
  mesh->mVertices = new aiVector3D[m_vertices.size()];
  mesh->mNormals = new aiVector3D[m_vertices.size()];
  mesh->mTextureCoords[0] = new aiVector3D[ m_vertices.size() ];
  mesh->mNumUVComponents[0] = 2;

  for(size_t i=0; i<m_vertices.size(); i++)
  {
    const TomGine::tgVertex& v = m_vertices[i];
    mesh->mVertices[i] = aiVector3D(v.pos.x,v.pos.y,v.pos.z);
    mesh->mNormals[i] = aiVector3D(v.normal.x, v.normal.y, v.normal.z);
    mesh->mTextureCoords[0][i] = aiVector3D( v.texCoord.x, v.texCoord.y, 0 );
  }

  mesh->mFaces = new aiFace[ m_faces.size() ];
  mesh->mNumFaces = m_faces.size();
  for(size_t i=0; i<m_faces.size(); i++)
  {
    aiFace& face = mesh->mFaces[i];
    const TomGine::tgFace& tgf = m_faces[i];

    face.mNumIndices = tgf.v.size();
    face.mIndices = new unsigned int[face.mNumIndices];

    for(size_t j=0; j<tgf.v.size(); j++)
      face.mIndices[j] = tgf.v[j];
  }

  // material
  if(!m_tex_name.empty())
  {
    aiString tex_name(m_tex_name.c_str());
    material->AddProperty(&tex_name, AI_MATKEY_TEXTURE(aiTextureType_DIFFUSE, 0));
  }

}

tgTextureModelAI& tgTextureModelAI::operator=(const aiMesh &mesh)
{

  return *this;
}

//void tgTextureModelAI::Draw()
//{
//  ApplyMaterial();

//  for (size_t i = 0; i < m_faces.size(); ++i)
//  {
//    const tgFace &face = m_faces[i];
//    GLenum face_mode;

//    switch (face.v.size()) {
//    case 1:
//      face_mode = GL_POINTS;
//      break;
//    case 2:
//      face_mode = GL_LINES;
//      break;
//    case 3:
//      face_mode = GL_TRIANGLES;
//      break;
//    default:
//      face_mode = GL_POLYGON;
//      break;
//    }

//    glBegin(face_mode);

//    for (size_t j = 0; j < face.v.size(); j++)
//    {
//      unsigned index = face.v[j];
//      //      glColor4ubv(m_vertices[index].color);
//      glNormal3fv(m_vertices[index].normal);
//      glVertex3fv(m_vertices[index].pos);
//    }

//    glEnd();
//  }
//}

void color4_to_vec4(const aiColor4D *c, TomGine::vec4& v)
{
  v.x = c->r;
  v.y = c->g;
  v.z = c->b;
  v.w = c->a;
}

void color4_to_float4(const aiColor4D *c, float f[4])
{
  f[0] = c->r;
  f[1] = c->g;
  f[2] = c->b;
  f[3] = c->a;
}

void aiMatrix4x4_to_mat4(const aiMatrix4x4& aiM, TomGine::mat4& tgM)
{
  tgM.data[0] = aiM.a1; tgM.data[1] = aiM.a2; tgM.data[2] = aiM.a3; tgM.data[3] = aiM.a4;
  tgM.data[4] = aiM.b1; tgM.data[5] = aiM.b2; tgM.data[6] = aiM.b3; tgM.data[7] = aiM.b4;
  tgM.data[8] = aiM.c1; tgM.data[9] = aiM.c2; tgM.data[10] = aiM.c3; tgM.data[11] = aiM.c4;
  tgM.data[12] = aiM.d1; tgM.data[13] = aiM.d2; tgM.data[14] = aiM.d3; tgM.data[15] = aiM.d4;

  tgM = tgM.transpose();

//  tgM.data[0] = aiM.a1; tgM.data[1] = aiM.b1; tgM.data[2] = aiM.c1; tgM.data[3] = aiM.d1;
//  tgM.data[4] = aiM.a2; tgM.data[5] = aiM.b2; tgM.data[6] = aiM.c2; tgM.data[7] = aiM.d2;
//  tgM.data[8] = aiM.a3; tgM.data[9] = aiM.b3; tgM.data[10] = aiM.c3; tgM.data[11] = aiM.d3;
//  tgM.data[12] = aiM.a4; tgM.data[13] = aiM.b4; tgM.data[14] = aiM.c4; tgM.data[15] = aiM.d4;
}

// ----------------------------------------------------------------------------
void set_float4(float f[4], float a, float b, float c, float d)
{
  f[0] = a;
  f[1] = b;
  f[2] = c;
  f[3] = d;
}

void tgTextureModelAI::CopyFromMaterial(const aiMaterial &material)
{
  int ret1, ret2;
  aiColor4D diffuse;
  aiColor4D specular;
  aiColor4D ambient;
  aiColor4D emission;
  float shininess, strength;
  unsigned int max;

  float opacity;
  max = 1;
  if(AI_SUCCESS == aiGetMaterialFloatArray(&material, AI_MATKEY_OPACITY, &opacity, &max))
  {
//     printf("[tgTextureModelAI::CopyFromMaterial] opacity: %f\n", opacity);
  }

  m_material.diffuse = TomGine::vec4(0.8f, 0.8f, 0.8f, 1.0f);
  if (AI_SUCCESS == aiGetMaterialColor(&material, AI_MATKEY_COLOR_DIFFUSE, &diffuse))
  {
    color4_to_vec4(&diffuse, m_material.diffuse);
    color4_to_vec4(&diffuse, m_material.color);
//    printf("[tgTextureModelAI::CopyFromMaterial] color: %f %f %f %f\n", diffuse.r, diffuse.g, diffuse.b, diffuse.a);
  }

  m_material.specular = TomGine::vec4(0.0f, 0.0f, 0.0f, 1.0f);
  if (AI_SUCCESS == aiGetMaterialColor(&material, AI_MATKEY_COLOR_SPECULAR, &specular))
  {
    color4_to_vec4(&specular, m_material.specular);
//    printf("[tgTextureModelAI::CopyFromMaterial] specular: %f %f %f %f\n", specular.r, specular.g, specular.b, specular.a);
  }

  m_material.ambient = TomGine::vec4(0.2f, 0.2f, 0.2f, 1.0f);
  if (AI_SUCCESS == aiGetMaterialColor(&material, AI_MATKEY_COLOR_AMBIENT, &ambient))
  {
    color4_to_vec4(&ambient, m_material.ambient);
//    printf("[tgTextureModelAI::CopyFromMaterial] ambient: %f %f %f %f\n", ambient.r, ambient.g, ambient.b, ambient.a);
  }

  m_material.emission = TomGine::vec4(0.0f, 0.0f, 0.0f, 1.0f);
  if (AI_SUCCESS == aiGetMaterialColor(&material, AI_MATKEY_COLOR_EMISSIVE, &emission))
  {
    color4_to_vec4(&emission, m_material.emission);
//    printf("[tgTextureModelAI::CopyFromMaterial] emission: %f %f %f %f\n", emission.r, emission.g, emission.b, emission.a);
  }

  max = 1;
  ret1 = aiGetMaterialFloatArray(&material, AI_MATKEY_SHININESS, &shininess, &max);
  if (ret1 == AI_SUCCESS)
  {
//    printf("[tgTextureModelAI::CopyFromMaterial] shininess: %f\n", shininess);
    max = 1;
    ret2 = aiGetMaterialFloatArray(&material, AI_MATKEY_SHININESS_STRENGTH, &strength, &max);
    if (ret2 == AI_SUCCESS)
    {
      m_material.shininess = shininess * strength;
//      printf("[tgTextureModelAI::CopyFromMaterial] strength: %f\n", strength);
    }else
      m_material.shininess = shininess;
  } else
  {
    m_material.shininess = 0.0f;
    m_material.specular = TomGine::vec4(0,0,0,0);
  }
}

//void tgTextureModelAI::ApplyMaterial()
//{
//  float c[4];

//  GLenum fill_mode;
//  int ret1, ret2;
//  aiColor4D diffuse;
//  aiColor4D specular;
//  aiColor4D ambient;diffuse
//  aiColor4D emission;
//  float shininess, strength;
//  int two_sided;
//  int wireframe;
//  unsigned int max;

//  set_float4(c, 0.8f, 0.8f, 0.8f, 1.0f);
//  if (AI_SUCCESS == aiGetMaterialColor(&m_materialAI, AI_MATKEY_COLOR_DIFFUSE, &diffuse))
//    color4_to_float4(&diffuse, c);
//  glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, c);

//  set_float4(c, 0.0f, 0.0f, 0.0f, 1.0f);
//  if (AI_SUCCESS == aiGetMaterialColor(&m_materialAI, AI_MATKEY_COLOR_SPECULAR, &specular))
//    color4_to_float4(&specular, c);
//  glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, c);

//  set_float4(c, 0.2f, 0.2f, 0.2f, 1.0f);
//  if (AI_SUCCESS == aiGetMaterialColor(&m_materialAI, AI_MATKEY_COLOR_AMBIENT, &ambient))
//    color4_to_float4(&ambient, c);
//  glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, c);

//  set_float4(c, 0.0f, 0.0f, 0.0f, 1.0f);
//  if (AI_SUCCESS == aiGetMaterialColor(&m_materialAI, AI_MATKEY_COLOR_EMISSIVE, &emission))
//    color4_to_float4(&emission, c);
//  glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, c);

//  max = 1;
//  ret1 = aiGetMaterialFloatArray(&m_materialAI, AI_MATKEY_SHININESS, &shininess, &max);
//  if (ret1 == AI_SUCCESS)
//  {
//    max = 1;
//    ret2 = aiGetMaterialFloatArray(&m_materialAI, AI_MATKEY_SHININESS_STRENGTH, &strength, &max);
//    if (ret2 == AI_SUCCESS)
//      glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, shininess * strength);
//    else
//      glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, shininess);
//  } else
//  {
//    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 0.0f);
//    set_float4(c, 0.0f, 0.0f, 0.0f, 0.0f);
//    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, c);
//  }

//  max = 1;
//  if (AI_SUCCESS == aiGetMaterialIntegerArray(&m_materialAI, AI_MATKEY_ENABLE_WIREFRAME, &wireframe, &max))
//    fill_mode = wireframe ? GL_LINE : GL_FILL;
//  else
//    fill_mode = GL_FILL;
//  glPolygonMode(GL_FRONT_AND_BACK, fill_mode);

//  max = 1;
//  if ((AI_SUCCESS == aiGetMaterialIntegerArray(&m_materialAI, AI_MATKEY_TWOSIDED, &two_sided, &max)) && two_sided)
//    glDisable(GL_CULL_FACE);
//  else
//    glEnable(GL_CULL_FACE);
//}

static void cut_file_name(std::string full_file_name, std::string& file_name, std::string& path)
{
  size_t c_slash_idx = full_file_name.find_last_of("/\\");
  size_t c_dot_idx = full_file_name.find_last_of(".");
  if(c_slash_idx == std::string::npos)
    c_slash_idx = 0;
  else
    c_slash_idx++;
  if(c_dot_idx == std::string::npos || c_dot_idx < c_slash_idx)
    c_dot_idx = full_file_name.size();
  file_name = full_file_name.substr(c_slash_idx,c_dot_idx-c_slash_idx);
  path = full_file_name.substr(0, c_slash_idx);
  if(c_slash_idx==0)
    path= "./";
}

void TomGine::LoadNode(const aiNode* node, const aiScene* scene, const TomGine::mat4& T,
                       std::vector<tgTextureModelAI*>& models, std::string& file_path)
{
  for(unsigned i=0; i<node->mNumChildren; i++)
    LoadNode(node->mChildren[i], scene, T, models, file_path);

  for(unsigned i=0; i<node->mNumMeshes; i++)
  {
    aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];

    tgTextureModelAI *m = new tgTextureModelAI(*mesh, *scene->mMaterials[mesh->mMaterialIndex], file_path);


    TomGine::mat4 p;
    aiMatrix4x4_to_mat4(node->mTransformation, p);

    m->m_pose.SetPose(T*p);

    models.push_back(m);
  }
}

std::vector<tgTextureModelAI*> TomGine::LoadScene(std::string file, bool scale, bool center)
{
  std::string file_name;
  std::string file_path;
  cut_file_name(file, file_name, file_path);

  const aiScene* scene = aiImportFile(file.c_str(), aiProcessPreset_TargetRealtime_MaxQuality);

  if (!scene)
    throw std::runtime_error("[TomGine::LoadScene] Error: failed to load model file (tgTextureModelAI.cpp)");

  std::vector<tgTextureModelAI*> models;

  TomGine::vec3 c(0,0,0);
  float radius(0.0f);

//  for(unsigned i = 0; i < scene->mNumMeshes; i++)
//  {
//    printf("[TomGine::LoadScene] mNumMaterials: %d %d\n", scene->mNumMaterials, scene->mMeshes[i]->mMaterialIndex);

//    tgTextureModelAI *m = new tgTextureModelAI(*scene->mMeshes[i],
//        *scene->mMaterials[scene->mMeshes[i]->mMaterialIndex], file_path);
//    models.push_back(m);

//    c += m->m_bs.center;
//  }

  TomGine::mat4 T;
  T.identity();
  LoadNode(scene->mRootNode, scene, T, models, file_path);

  for(size_t i=0; i<models.size(); i++)
  {
    tgTextureModelAI* m = models[i];
    c += m->m_bs.center;
  }
  c /= models.size();

  for(size_t i=0; i<models.size(); i++)
  {
    tgTextureModelAI *m = models[i];
    float cm = (m->m_bs.center-c).length();
    radius = std::max(radius, cm+m->m_bs.radius);
  }

  for(size_t i=0; i<models.size(); i++)
  {
    tgTextureModelAI* m = models[i];

//    m->m_pose.Translate(-c);

//    mat4 p = m->m_pose.GetMat4();
//    mat4 s;
//    s.identity();
//    s *= (1.0/radius);
//    s[15] = 1.0;
//    p = s*p;

//    printf("[TomGine::LoadNode] matrix:\n");
//    p.print();
//    m->m_pose.SetPose(p);
//    m->m_pose.GetMat4().print();


    for(size_t j=0; j<m->m_vertices.size(); j++)
    {
      if(center)
        m->m_vertices[j].pos -= c;
      if(scale)
        m->m_vertices[j].pos /= radius;
    }
  }

  return models;
}

// format: "collada", "obj"
void TomGine::SaveScene(std::string file, std::string format, std::vector<tgTextureModelAI*>& meshes)
{
  if(meshes.empty())
  {
    printf("[TomGineAI::SaveScene] Warning, meshes empty.\n");
    return;
  }

  aiScene scene;

  scene.mRootNode = new aiNode();

  scene.mNumMaterials = meshes.size();
  scene.mMaterials = new aiMaterial*[scene.mNumMaterials];

  scene.mNumMeshes = meshes.size();
  scene.mMeshes = new aiMesh*[scene.mNumMeshes];

  scene.mRootNode->mNumMeshes = meshes.size();
  scene.mRootNode->mMeshes = new unsigned int[scene.mRootNode->mNumMeshes];

  for(size_t i=0; i<meshes.size(); i++)
  {
    scene.mMaterials[i] = new aiMaterial();
    scene.mMeshes[i] = new aiMesh();
    scene.mMeshes[i]->mMaterialIndex = i;
    scene.mRootNode->mMeshes[i] = i;

    meshes[i]->copy(scene.mMeshes[i], scene.mMaterials[i]);

    if(!meshes[i]->m_tex_cv.empty() &&
       !meshes[i]->m_tex_cv[0].empty() &&
       !meshes[i]->m_tex_name.empty())
    {
      std::string file_name, file_path;
      TomGine::cut_file_name(file, file_name, file_path);

      std::string tex_name, tex_path;
      TomGine::cut_file_name(meshes[i]->m_tex_name, tex_name, tex_path);
      tex_path = file_path + tex_path;

      boost::filesystem::path boost_tex_path(tex_path.c_str());
      if(!boost::filesystem::exists(boost_tex_path))
        boost::filesystem::create_directories(boost_tex_path);

      cv::Mat img;
      cv::flip(meshes[i]->m_tex_cv[0],img,0);
      cv::imwrite(file_path + meshes[i]->m_tex_name, img);
    }
  }

  Assimp::Exporter mAiExporter;
  mAiExporter.Export(&scene, format, file);

  printf("[TomGine::SaveScene] Exported %lu meshes to '%s'\n", meshes.size(), file.c_str());

}

// ----------------------------------------------------------------------------
void TomGine::get_bounding_box_for_node(const aiScene* scene, const aiNode* nd, aiVector3D* _min, aiVector3D* _max,
                                        aiMatrix4x4* trafo)
{
  aiMatrix4x4 prev;
  unsigned int n = 0, t;

  prev = *trafo;
  aiMultiplyMatrix4(trafo, &nd->mTransformation);

  for (; n < nd->mNumMeshes; ++n)
  {
    const aiMesh* mesh = scene->mMeshes[nd->mMeshes[n]];
    for (t = 0; t < mesh->mNumVertices; ++t)
    {

      aiVector3D tmp = mesh->mVertices[t];
      aiTransformVecByMatrix4(&tmp, trafo);

      _min->x = std::min(_min->x, tmp.x);
      _min->y = std::min(_min->y, tmp.y);
      _min->z = std::min(_min->z, tmp.z);

      _max->x = std::max(_max->x, tmp.x);
      _max->y = std::max(_max->y, tmp.y);
      _max->z = std::max(_max->z, tmp.z);
    }
  }

  for (n = 0; n < nd->mNumChildren; ++n)
  {
    get_bounding_box_for_node(scene, nd->mChildren[n], _min, _max, trafo);
  }
  *trafo = prev;
}

void TomGine::get_bounding_box(const aiScene* scene, aiVector3D* _min, aiVector3D* _max)
{
  aiMatrix4x4 trafo;
  aiIdentityMatrix4(&trafo);

  _min->x = _min->y = _min->z = 1e10f;
  _max->x = _max->y = _max->z = -1e10f;
  get_bounding_box_for_node(scene, scene->mRootNode, _min, _max, &trafo);
}
