#ifndef TG_POINT_CLOUD_RENDERING_H
#define TG_POINT_CLOUD_RENDERING_H

#include <v4r/tomgine/GLWindow.h>
#include <v4r/tomgine/tgTextureModelAI.h>

#include <v4r/core/macros.h>

namespace TomGine{

class V4R_EXPORTS PointCloudRendering
{
private:
  TomGine::GLWindow m_context;
  std::vector<tgTextureModelAI*> m_models;

  std::vector<cv::Mat4f> m_pointclouds;

public:
  PointCloudRendering(std::string file, bool scale=true, bool center=true);
  ~PointCloudRendering();

  // flag world_coords defines if point cloud values are in the world coordinate
  // system of the object, or if false in the camera coordinate system of the rendered view
  void Generate(TomGine::tgCamera cam, bool world_coords);


  // const
  const std::vector<tgTextureModelAI*>& GetModels() const { return m_models; }

  const size_t GetModelCount() const { return m_models.size(); }

  const tgTextureModelAI& GetModel(size_t id) const { return (*m_models[id]); }

  const cv::Mat4f& GetPointCloud(size_t id) const { return m_pointclouds[id]; }


};

} // namespace TomGine

#endif // POINTCLOUDRENDERING_H
