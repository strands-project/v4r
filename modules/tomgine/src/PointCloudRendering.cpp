#include <v4r/tomgine/PointCloudRendering.h>
#include <v4r/tomgine/tgFrameBufferObject.h>

using namespace TomGine;

PointCloudRendering::PointCloudRendering(std::string file, bool scale, bool center)
{
  m_models = LoadScene(file, scale, center);

  m_context.Activate();
  glDepthFunc(GL_LEQUAL);
  glEnable(GL_DEPTH_TEST);
  glDisable(GL_BLEND);

  for(size_t i=0; i<m_models.size(); i++)
  {
    tgTextureModelAI* m = m_models[i];
    m->m_tex_env_mode.assign(m->m_tex_cv.size(), GL_REPLACE);
    m->m_coloring = TomGine::tgModel::FULL_COLORING_NO_LIGHTING;
  }
}

PointCloudRendering::~PointCloudRendering()
{
  for (size_t i = 0; i < m_models.size(); i++)
    delete m_models[i];
}

void PointCloudRendering::Generate(TomGine::tgCamera cam, bool world_coords)
{
  m_context.Activate();
  TomGine::tgFrameBufferObject fbo(cam.GetWidth(), cam.GetHeight());
  fbo.Bind();

  cam.Activate();

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glDisable(GL_TEXTURE_2D);
  for(size_t i=0; i<m_models.size(); i++)
    m_models[i]->DrawFaces();

  m_pointclouds.push_back(cv::Mat4f(cam.GetHeight(),cam.GetWidth()));
  fbo.GetPointCloud(m_pointclouds.back(), world_coords);

  fbo.Unbind();
}
