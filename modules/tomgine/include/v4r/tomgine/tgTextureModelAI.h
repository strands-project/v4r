#ifndef _TG_TEXTURE_MODEL_AI_H_
#define _TG_TEXTURE_MODEL_AI_H_

#include <v4r/tomgine/tgTextureModel.h>
#include <v4r/core/macros.h>

#include <assimp/scene.h>


namespace TomGine {

class V4R_EXPORTS tgTextureModelAI: public tgTextureModel
{
public:
  std::string m_tex_name;

public:
  tgTextureModelAI() : tgTextureModel(){}
  tgTextureModelAI(const aiMesh &mesh, const aiMaterial &material, const std::string& file_path);

  tgTextureModelAI& operator=(const aiMesh &mesh);

//  void Draw();

  void CopyFromMaterial(const aiMaterial &material);

  void copy(aiMesh *mesh, aiMaterial *material);

//  void ApplyMaterial();

};

void LoadNode(const aiNode* node, const aiScene* scene, const mat4 &T,
              std::vector<tgTextureModelAI*>& models,
              std::string& file_path);

std::vector<tgTextureModelAI*> LoadScene(std::string file, bool scale = true, bool center = true);

void SaveScene(std::string file, std::string format, std::vector<tgTextureModelAI*>& meshes);

void get_bounding_box_for_node(const aiScene* scene, const aiNode* nd, aiVector3D* min, aiVector3D* max,
                               aiMatrix4x4* trafo);

void get_bounding_box(const aiScene* scene, aiVector3D* min, aiVector3D* max);

}

#endif
