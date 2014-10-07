#include "output/tgRenderer.h"

#include <pcl/common/common.h>
#include <pcl/common/transforms.h>

namespace object_modeller
{
namespace output
{

/*
void TomGineRenderer::renderImage(unsigned char *data, int width, int height)
{
    win->Clear();
    win->SetImage(data, width, height);
    win->Update();
}
*/

output::Renderer::Event TomGineRenderer::waitForEvent()
{
    std::list<TomGine::Event> events;

    while (true)
    {
        win->GetEventQueue(events);

        if (events.size() > 0)
        {
            TomGine::Event e = events.front();
            if (e.type == TomGine::TMGL_Press)
            {
                if (e.input == TomGine::TMGL_Space)
                    return output::Renderer::STEP;

                if (e.input == TomGine::TMGL_c)
                    return output::Renderer::CONTINUE;
            }
        }
    }

}

void TomGineRenderer::update()
{
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*pointclouds_rgb[0][0], centroid);
    win->SetRotationCenter(centroid[0], centroid[1], centroid[2]);

    win->Clear();

    for (int i=0;i<pointclouds_rgb[0].size();i++)
    {
        win->AddPointCloudPCL(*pointclouds_rgb[0][i]);
    }

    win->AddLabel2D("name", 10, 10, 580);

    win->Update();

    /*
    if (step)
    {
        std::cout << "Wait for key press" << std::endl;

        win->AddLabel2D("Press SPACE to continue", 10, 10, 10);
        win->WaitForEvent(TomGine::TMGL_Press, TomGine::TMGL_Space);
    }
    */
}

/*
boost::shared_ptr<TomGine::tgModel> TomGineRenderer::convert (pcl::PolygonMesh &mesh)
{
    boost::shared_ptr<TomGine::tgModel> model(new TomGine::tgModel());
  std::string type = pcl::getFieldsList(mesh.cloud);

  if (type == "x y z")
  {
      std::cout << "convert to xyz" << std::endl;
      pcl::PointCloud<pcl::PointXYZ> cloud;
      pcl::fromPCLPointCloud2 (mesh.cloud, cloud);

      for (size_t i = 0; i < cloud.size (); i++)
      {
        pcl::PointXYZ &pt = cloud.at (i);
        TomGine::tgVertex v;
        v.pos = TomGine::vec3 (pt.x, pt.y, pt.z);
        //    v.normal = vec3(pt.normal_x, pt.normal_y, pt.normal_z);
        model->m_vertices.push_back (v);
      }
  } else {
      std::cout << "convert to xyz color" << std::endl;
      pcl::PointCloud<pcl::PointXYZRGBNormal> cloud;
      pcl::fromPCLPointCloud2 (mesh.cloud, cloud);

      for (size_t i = 0; i < cloud.size (); i++)
      {
        pcl::PointXYZRGBNormal &pt = cloud.at (i);
        TomGine::tgVertex v;
        v.pos = TomGine::vec3 (pt.x, pt.y, pt.z);
        v.color[0] = pt.r;
        v.color[1] = pt.g;
        v.color[2] = pt.b;
        //    v.normal = vec3(pt.normal_x, pt.normal_y, pt.normal_z);
        model->m_vertices.push_back (v);
      }
  }

  for (size_t i = 0; i < mesh.polygons.size (); i++)
  {
    pcl::Vertices &triangle = mesh.polygons[i];
    TomGine::tgFace face;
    for (size_t j = 0; j < triangle.vertices.size (); j++)
    {
      face.v.push_back (triangle.vertices[j]);
    }
    model->m_faces.push_back (face);
  }

  model->ComputeNormals ();
  return model;
}


void TomGineRenderer::renderTexturedMesh(TexturedMesh mesh, std::string text, bool step)
{
    std::cout << "rendering textured mesh" << std::endl;

    boost::shared_ptr<TomGine::tgModel> model = convert(*(mesh.mesh));
    boost::shared_ptr<TomGine::tgTextureModel> texturedModel(new TomGine::tgTextureModel(*model));


    std::cout << "assign textures" << std::endl;

    for (int i=0;i<mesh.textures.size();i++)
    {
        texturedModel->m_tex_cv.push_back(mesh.textures[i]);
    }

    for (int i=0;i<mesh.textureIndex.size();i++)
    {
        texturedModel->m_face_tex_id.push_back(mesh.textureIndex[i]);
    }

    for (int i=0;i<mesh.textureCoordinates.size();i++)
    {
        texturedModel->m_vertices[i].texCoord = TomGine::vec2(mesh.textureCoordinates[i][0], mesh.textureCoordinates[i][1]);
    }

    std::cout << "sync" << std::endl;

    texturedModel->Sync();


    std::cout << "render" << std::endl;

    win->Clear();

    win->AddModel(*texturedModel);

    win->AddLabel2D(text, 10, 10, 580);

    win->Update();

    if (step)
    {
        win->AddLabel2D("Press SPACE to continue", 10, 10, 10);
        win->WaitForEvent(TomGine::TMGL_Press, TomGine::TMGL_Space);
    }
}

void TomGineRenderer::renderMesh(pcl::PolygonMesh::Ptr mesh, std::string text, bool step)
{
    //Eigen::Vector4f centroid;
    //pcl::compute3DCentroid(*pointClouds[0], centroid);
    //win->SetRotationCenter(centroid[0], centroid[1], centroid[2]);

    win->Clear();

    boost::shared_ptr<TomGine::tgModel> model = convert(*mesh);
    TomGine::tgTextureModel m;

    win->AddModel(*model);

    win->AddLabel2D(text, 10, 10, 580);

    win->Update();

    if (step)
    {
        win->AddLabel2D("Press SPACE to continue", 10, 10, 10);
        win->WaitForEvent(TomGine::TMGL_Press, TomGine::TMGL_Space);
    }
}
*/

}
}
