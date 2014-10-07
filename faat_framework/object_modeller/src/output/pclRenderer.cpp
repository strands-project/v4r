#include "output/pclRenderer.h"

#include <pcl/common/common.h>
#include <pcl/common/transforms.h>

#include <pcl/filters/crop_box.h>

#include <vtkOpenGLHardwareSupport.h>
#include <vtkOpenGLRenderWindow.h>
#include <vtkImageData.h>
#include <vtkImageImport.h>

namespace object_modeller
{
namespace output
{

bool AdvancedPclVisualizer::addTextureMesh (TexturedMesh::Ptr mesh, const std::string &id, int viewport)
{
    std::cout << "converting textured mesh" << std::endl;

    TextureMesh m;
    m.cloud = mesh->mesh->cloud;
    m.header = mesh->mesh->header;

    //m.tex_materials.reserve(mesh->textures.size());
    m.tex_coordinates.resize(1);
    m.tex_polygons.resize(1);

    TexMaterial mat;
    mat.data = mesh->textures[0];
    m.tex_materials.push_back(mat);

    std::cout << "created materials" << std::endl;

    for (unsigned int i=0;i<mesh->mesh->polygons.size();i++)
    {
        //int textureIndex = mesh->textureIndex[i];
        int textureIndex = 0;

        pcl::Vertices v = mesh->mesh->polygons[i];
        m.tex_polygons.at(textureIndex).push_back(v);

        /*
        for (int j=0;j<v.vertices.size();j++)
        {
            //std::cout << "get coordinates for " << v.vertices[j] << std::endl;
            Eigen::Vector2f coords = mesh->textureCoordinates[v.vertices[j]];

            //m.tex_coordinates.at(textureIndex).push_back(coords);
        }
        */
    }

    for (unsigned int i=0;i<mesh->textureCoordinates.size();i++)
    {
        Eigen::Vector2f coords = mesh->textureCoordinates[i];
        m.tex_coordinates.at(0).push_back(coords);
    }

    return addTextureMesh(m, id, viewport);
}

int AdvancedPclVisualizer::textureFromTexMaterial (const TexMaterial& tex_mat, vtkTexture* vtk_tex) const
{
    vtkSmartPointer<vtkImageImport> importer = vtkSmartPointer<vtkImageImport>::New();

    int width = tex_mat.data.cols;
    int height = tex_mat.data.rows;

    /*
    cv::Mat3b texture(width, height, CV_8UC3);

    for (int i=0;i<width;i++)
    {
        for (int j=0;j<height;j++)
        {
            texture.at<cv::Vec3b>(i, j).val[0] = i % 255; //image->at(i, j).b;
            texture.at<cv::Vec3b>(i, j).val[1] = 0; //image->at(i, j).g;
            texture.at<cv::Vec3b>(i, j).val[2] = 0; //image->at(i, j).r;
        }
    }
    */

    importer->SetDataSpacing( 1, 1, 1 );
    importer->SetDataOrigin( 0, 0, 0 );
    importer->SetWholeExtent(   0, width-1, 0, height-1, 0, 0 );
    importer->SetDataExtentToWholeExtent();
    importer->SetDataScalarTypeToUnsignedChar();
    importer->SetNumberOfScalarComponents( 3 );
    importer->SetImportVoidPointer( tex_mat.data.data );
    importer->Update();

    vtk_tex->SetInputConnection(importer->GetOutputPort());
    /*
    vtkSmartPointer<vtkJPEGReader> jpeg_reader = vtkSmartPointer<vtkJPEGReader>::New ();
    jpeg_reader->SetFileName (full_path.string ().c_str ());
    jpeg_reader->Update ();
    vtk_tex->SetInputConnection (jpeg_reader->GetOutputPort ());
    */

  return (0);
}

bool AdvancedPclVisualizer::addTextureMesh (const TextureMesh &mesh, const std::string &id, int viewport)
{
    std::cout << "rendering textured mesh" << std::endl;

  pcl::visualization::CloudActorMap::iterator am_it = getCloudActorMap()->find (id);
  if (am_it != getCloudActorMap()->end ())
  {
    PCL_ERROR ("[PCLVisualizer::addTextureMesh] A shape with id <%s> already exists!"
               " Please choose a different id and retry.\n",
               id.c_str ());
    return (false);
  }
  // no texture materials --> exit
  if (mesh.tex_materials.size () == 0)
  {
    PCL_ERROR("[PCLVisualizer::addTextureMesh] No textures found!\n");
    return (false);
  }
  // polygons are mapped to texture materials
  if (mesh.tex_materials.size () != mesh.tex_polygons.size ())
  {
    PCL_ERROR("[PCLVisualizer::addTextureMesh] Materials number %lu differs from polygons number %lu!\n",
              mesh.tex_materials.size (), mesh.tex_polygons.size ());
    return (false);
  }
  // each texture material should have its coordinates set
  if (mesh.tex_materials.size () != mesh.tex_coordinates.size ())
  {
    PCL_ERROR("[PCLVisualizer::addTextureMesh] Coordinates number %lu differs from materials number %lu!\n",
              mesh.tex_coordinates.size (), mesh.tex_materials.size ());
    return (false);
  }
  // total number of vertices
  std::size_t nb_vertices = 0;
  for (std::size_t i = 0; i < mesh.tex_polygons.size (); ++i)
    nb_vertices+= mesh.tex_polygons[i].size ();
  // no vertices --> exit
  if (nb_vertices == 0)
  {
    PCL_ERROR("[PCLVisualizer::addTextureMesh] No vertices found!\n");
    return (false);
  }
  // total number of coordinates
  std::size_t nb_coordinates = 0;
  for (std::size_t i = 0; i < mesh.tex_coordinates.size (); ++i)
    nb_coordinates+= mesh.tex_coordinates[i].size ();
  // no texture coordinates --> exit
  if (nb_coordinates == 0)
  {
    PCL_ERROR("[PCLVisualizer::addTextureMesh] No textures coordinates found!\n");
    return (false);
  }

  // Create points from mesh.cloud
  vtkSmartPointer<vtkPoints> poly_points = vtkSmartPointer<vtkPoints>::New ();
  vtkSmartPointer<vtkUnsignedCharArray> colors = vtkSmartPointer<vtkUnsignedCharArray>::New ();
  bool has_color = false;
  vtkSmartPointer<vtkMatrix4x4> transformation = vtkSmartPointer<vtkMatrix4x4>::New ();
  if ((pcl::getFieldIndex(mesh.cloud, "rgba") != -1) ||
      (pcl::getFieldIndex(mesh.cloud, "rgb") != -1))
  {
    pcl::PointCloud<pcl::PointXYZRGB> cloud;
    pcl::fromPCLPointCloud2(mesh.cloud, cloud);
    if (cloud.points.size () == 0)
    {
      PCL_ERROR("[PCLVisualizer::addTextureMesh] Cloud is empty!\n");
      return (false);
    }
    convertToVtkMatrix (cloud.sensor_origin_, cloud.sensor_orientation_, transformation);
    has_color = true;
    colors->SetNumberOfComponents (3);
    colors->SetName ("Colors");

    poly_points->SetNumberOfPoints (cloud.size ());
    for (std::size_t i = 0; i < cloud.points.size (); ++i)
    {
      const pcl::PointXYZRGB &p = cloud.points[i];
      poly_points->InsertPoint (i, p.x, p.y, p.z);
      const unsigned char color[3] = {p.r, p.g, p.b};
      colors->InsertNextTupleValue(color);
    }
  }
  else
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::fromPCLPointCloud2 (mesh.cloud, *cloud);
    // no points --> exit
    if (cloud->points.size () == 0)
    {
      PCL_ERROR("[PCLVisualizer::addTextureMesh] Cloud is empty!\n");
      return (false);
    }
    convertToVtkMatrix (cloud->sensor_origin_, cloud->sensor_orientation_, transformation);
    poly_points->SetNumberOfPoints (cloud->points.size ());
    for (std::size_t i = 0; i < cloud->points.size (); ++i)
    {
      const pcl::PointXYZ &p = cloud->points[i];
      poly_points->InsertPoint (i, p.x, p.y, p.z);
    }
  }

  //create polys from polyMesh.tex_polygons
  vtkSmartPointer<vtkCellArray> polys = vtkSmartPointer<vtkCellArray>::New ();
  for (std::size_t i = 0; i < mesh.tex_polygons.size (); i++)
  {
    for (std::size_t j = 0; j < mesh.tex_polygons[i].size (); j++)
    {
      std::size_t n_points = mesh.tex_polygons[i][j].vertices.size ();
      polys->InsertNextCell (int (n_points));
      for (std::size_t k = 0; k < n_points; k++)
        polys->InsertCellPoint (mesh.tex_polygons[i][j].vertices[k]);
    }
  }
  vtkSmartPointer<vtkPolyData> polydata = vtkSmartPointer<vtkPolyData>::New();
  polydata->SetPolys (polys);
  polydata->SetPoints (poly_points);
  if (has_color)
    polydata->GetPointData()->SetScalars(colors);

  vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New ();
#if VTK_MAJOR_VERSION < 6
    mapper->SetInput (polydata);
#else
    mapper->SetInputData (polydata);
#endif

  vtkSmartPointer<vtkLODActor> actor = vtkSmartPointer<vtkLODActor>::New ();
  vtkOpenGLHardwareSupport* hardware = vtkOpenGLRenderWindow::SafeDownCast (getRenderWindow())->GetHardwareSupport ();
  bool supported = hardware->GetSupportsMultiTexturing ();
  // Check if hardware support multi texture
  std::size_t texture_units (hardware->GetNumberOfFixedTextureUnits ());
  if ((mesh.tex_materials.size () > 1) && supported && (texture_units > 1))
  {
    if (texture_units < mesh.tex_materials.size ())
      PCL_WARN ("[PCLVisualizer::addTextureMesh] GPU texture units %d < mesh textures %d!\n",
                texture_units, mesh.tex_materials.size ());
    // Load textures
    std::size_t last_tex_id = std::min (mesh.tex_materials.size (), texture_units);
    int tu = vtkProperty::VTK_TEXTURE_UNIT_0;
    std::size_t tex_id = 0;
    while (tex_id < last_tex_id)
    {
      vtkSmartPointer<vtkTexture> texture = vtkSmartPointer<vtkTexture>::New ();
      if (textureFromTexMaterial (mesh.tex_materials[tex_id], texture))
      {
        PCL_WARN ("[PCLVisualizer::addTextureMesh] Failed to load texture %s, skipping!\n",
                  mesh.tex_materials[tex_id].tex_name.c_str ());
        continue;
      }
      // the first texture is in REPLACE mode others are in ADD mode
      if (tex_id == 0)
        texture->SetBlendingMode(vtkTexture::VTK_TEXTURE_BLENDING_MODE_REPLACE);
      else
        texture->SetBlendingMode(vtkTexture::VTK_TEXTURE_BLENDING_MODE_ADD);
      // add a texture coordinates array per texture
      vtkSmartPointer<vtkFloatArray> coordinates = vtkSmartPointer<vtkFloatArray>::New ();
      coordinates->SetNumberOfComponents (2);
      std::stringstream ss; ss << "TCoords" << tex_id;
      std::string this_coordinates_name = ss.str ();
      coordinates->SetName (this_coordinates_name.c_str ());

      for (std::size_t t = 0 ; t < mesh.tex_coordinates.size (); ++t)
        if (t == tex_id)
          for (std::size_t tc = 0; tc < mesh.tex_coordinates[t].size (); ++tc)
            coordinates->InsertNextTuple2 (mesh.tex_coordinates[t][tc][0],
                                           mesh.tex_coordinates[t][tc][1]);
        else
          for (std::size_t tc = 0; tc < mesh.tex_coordinates[t].size (); ++tc)
            coordinates->InsertNextTuple2 (-1.0, -1.0);

      mapper->MapDataArrayToMultiTextureAttribute(tu,
                                                  this_coordinates_name.c_str (),
                                                  vtkDataObject::FIELD_ASSOCIATION_POINTS);
      polydata->GetPointData ()->AddArray (coordinates);
      actor->GetProperty ()->SetTexture(tu, texture);
      ++tex_id;
      ++tu;
    }
  } // end of multi texturing
  else
  {
    if (!supported || texture_units < 2)
      PCL_WARN ("[PCLVisualizer::addTextureMesh] Your GPU doesn't support multi texturing. "
                "Will use first one only!\n");

    vtkSmartPointer<vtkTexture> texture = vtkSmartPointer<vtkTexture>::New ();
    // fill vtkTexture from pcl::TexMaterial structure
    if (textureFromTexMaterial (mesh.tex_materials[0], texture))
      PCL_WARN ("[PCLVisualizer::addTextureMesh] Failed to create vtkTexture from %s!\n",
                mesh.tex_materials[0].tex_name.c_str ());

    // set texture coordinates
    vtkSmartPointer<vtkFloatArray> coordinates = vtkSmartPointer<vtkFloatArray>::New ();
    coordinates->SetNumberOfComponents (2);
    coordinates->SetNumberOfTuples (mesh.tex_coordinates[0].size ());
    for (std::size_t tc = 0; tc < mesh.tex_coordinates[0].size (); ++tc)
    {
      const Eigen::Vector2f &uv = mesh.tex_coordinates[0][tc];
      coordinates->SetTuple2 (tc, uv[0], uv[1]);
    }
    coordinates->SetName ("TCoords");
    polydata->GetPointData ()->SetTCoords(coordinates);
    // apply texture
    actor->SetTexture (texture);
  } // end of one texture

  // set mapper
  actor->SetMapper (mapper);
  addActorToRenderer2 (actor, viewport);

  // Save the pointer/ID pair to the global actor map
  (*getCloudActorMap())[id].actor = actor;

  // Save the viewpoint transformation matrix to the global actor map
  (*getCloudActorMap())[id].viewpoint_transformation_ = transformation;

  std::cout << "finished rendering" << std::endl;

  return (true);
}

void AdvancedPclVisualizer::addActorToRenderer2 (const vtkSmartPointer<vtkProp> &actor, int viewport)
{
  // Add it to all renderers
  getRendererCollection()->InitTraversal ();
  vtkRenderer* renderer = NULL;
  int i = 0;
  while ((renderer = getRendererCollection()->GetNextItem ()) != NULL)
  {
    // Should we add the actor to all renderers?
    if (viewport == 0)
    {
      renderer->AddActor (actor);
    }
    else if (viewport == i)               // add the actor only to the specified viewport
    {
      renderer->AddActor (actor);
    }
    ++i;
  }
}

PclRenderer::PclRenderer(bool createWindow)
{
    std::cout << "create visualizer" << std::endl;
    int argc = 0;
    interactor = new Interactor();
    vis.reset(new AdvancedPclVisualizer(argc, NULL, "", interactor, createWindow));

    roi = NULL;
}

void Interactor::OnLeftButtonUp()
{
    if (roi != NULL && roi->isPointSelected())
    {
        roi->deselect();
    }
    else
    {
        PCLVisualizerInteractorStyle::OnLeftButtonUp();
    }
}

void Interactor::OnLeftButtonDown()
{
    if (roi != NULL)
    {
        int x = this->GetInteractor()->GetEventPosition()[0];
        int y = this->GetInteractor()->GetEventPosition()[1];

        roi->selectPoint(renderer->getCamera(), x, y);

        if (!roi->isPointSelected())
        {
            PCLVisualizerInteractorStyle::OnLeftButtonDown();
        }
    }
    else
    {
        PCLVisualizerInteractorStyle::OnLeftButtonDown();
    }
}

void Interactor::OnMouseMove()
{
    if (roi == NULL || !roi->isPointSelected())
    {
        if (roi != NULL)
        {
            int x = this->GetInteractor()->GetEventPosition()[0];
            int y = this->GetInteractor()->GetEventPosition()[1];

            roi->selectPoint(renderer->getCamera(), x, y);

            if (roi->isPointSelected())
            {
                roi->highlightSelection();
                roi->deselect();
                renderer->updateRoi();
                GetInteractor()->Render();
            } else {
                if (roi->removeHighlight())
                {
                    renderer->updateRoi();
                    GetInteractor()->Render();
                }
            }
        }

        PCLVisualizerInteractorStyle::OnMouseMove();
    }
    else
    {
        int x = this->GetInteractor()->GetEventPosition()[0];
        int y = this->GetInteractor()->GetEventPosition()[1];

        //std::cout << "move point" << x << " - " << y << std::endl;

        roi->handleMouseMove(renderer->getCamera(), x, y);
        renderer->updateRoi();

        //GetInteractor()->GetRenderWindow()->Render();
        GetInteractor()->Render();

        Superclass::OnMouseMove ();
    }
}

void PclRenderer::updateRoi()
{
    vis->updatePointCloud(roi->getCloud(), "roicloud");

    Eigen::Vector3f d = *(roi->getDimension());
    Eigen::Vector3f t = *(roi->getTranslation());
    Eigen::Quaternionf r = *(roi->getRotation());

    vis->removeShape("roi");
    vis->addCube(t, r, d[0], d[1], d[2], "roi");

    updateRoiRotCylinder("rotX", Eigen::Vector3f(0.0, 0.0, 1.0), Eigen::Vector3f(0.3, 0.0, 0.0));
    updateRoiRotCylinder("rotY", Eigen::Vector3f(0.0, 1.0, 0.0), Eigen::Vector3f(0.0, 0.0, 0.3));
    updateRoiRotCylinder("rotZ", Eigen::Vector3f(1.0, 0.0, 0.0), Eigen::Vector3f(0.0, 0.3, 0.0));
}

void PclRenderer::updateRoiRotCylinder(std::string name, Eigen::Vector3f normal, Eigen::Vector3f color)
{
    vis->removeShape(name);
    Eigen::Vector3f t = *(roi->getTranslation());
    Eigen::Vector3f dir = roi->getRotation()->toRotationMatrix() * normal;
    dir.normalize();
    dir *= 0.001;

    pcl::ModelCoefficients cylinder_coeff;
    cylinder_coeff.values.resize (7); // We need 7 values
    cylinder_coeff.values[0] = t.x ();
    cylinder_coeff.values[1] = t.y ();
    cylinder_coeff.values[2] = t.z ();
    cylinder_coeff.values[3] = dir.x ();
    cylinder_coeff.values[4] = dir.y ();
    cylinder_coeff.values[5] = dir.z ();
    cylinder_coeff.values[6] = 0.2f;
    vis->addCylinder(cylinder_coeff, name);
    vis->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, color[0], color[1], color[2], name);
}

void PclRenderer::enableRoiMode(Eigen::Vector3f *dim, Eigen::Vector3f *translation, Eigen::Quaternionf *rotation)
{
    std::cout << "enabling roi mode" << std::endl;
    roi = new Roi(dim, translation, rotation);

    interactor->setRenderer(this);
    interactor->setRoi(roi);

    //vis->addPointCloud(roi->getCloud(), "roicloud");
    //updateRoi();

    //vis->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5.0, "roicloud");
}

void PclRenderer::disableRoiMode()
{
    Renderer::disableRoiMode();

    interactor->setRoi(NULL);
}

void PclRenderer::update()
{
    std::cout << "update renderer" << std::endl;
    int v;
    vis->createViewPort(0,0,1,1,v);
    vis->removeAllPointClouds();
    vis->removeAllShapes();

    int objectCounter = 0;

    std::cout << "render clouds" << std::endl;
    for (int i=0;i<pointclouds_rgb[getActiveSequenceId()].size();i++)
    {
        if (getActiveObjectId() == -1 || getActiveObjectId() == objectCounter)
        {
            pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler_rgb (pointclouds_rgb[getActiveSequenceId()][i]);
            std::stringstream name;
            name << "text" << i;
            vis->addPointCloud<pcl::PointXYZRGB> (pointclouds_rgb[getActiveSequenceId()][i], handler_rgb, name.str(), v);
        }

        objectCounter++;
    }

    std::cout << "render clouds" << std::endl;
    for (int i=0;i<pointclouds_rgbnormal[getActiveSequenceId()].size();i++)
    {
        if (getActiveObjectId() == -1 || getActiveObjectId() == objectCounter)
        {
            pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal> handler_rgb (pointclouds_rgbnormal[getActiveSequenceId()][i]);
            std::stringstream name;
            name << "text" << i;
            vis->addPointCloud<pcl::PointXYZRGBNormal> (pointclouds_rgbnormal[getActiveSequenceId()][i], handler_rgb, name.str(), v);
        }

        objectCounter++;
    }

    std::cout << "render textured meshes" << std::endl;
    for (int i=0;i<textured_meshs[getActiveSequenceId()].size();i++)
    {
        if (getActiveObjectId() == -1 || getActiveObjectId() == objectCounter)
        {
            std::stringstream name;
            name << "text" << i;
            TexturedMesh::Ptr mesh = textured_meshs[getActiveSequenceId()][i];
            vis->addTextureMesh(mesh, name.str(), v);

/*
            pcl::PointCloud<pcl::PointXYZ>::Ptr meshCloud(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::fromPCLPointCloud2 (mesh->mesh->cloud, *meshCloud);

            pcl::Vertices vertices = mesh->mesh->polygons[1000];
            pcl::PointXYZ p1 = meshCloud->points[vertices.vertices[0]];
            pcl::PointXYZ p2 = meshCloud->points[vertices.vertices[1]];
            pcl::PointXYZ p3 = meshCloud->points[vertices.vertices[2]];
            pcl::PointXYZ origin(0.0f, 0.0f, 0.0f);

            Eigen::Vector3f v1 = Eigen::Vector3f( p1.getArray3fMap() );
            Eigen::Vector3f v2 = Eigen::Vector3f( p2.getArray3fMap() );
            Eigen::Vector3f v3 = Eigen::Vector3f( p3.getArray3fMap() );

            v2 = v1 - v2;
            v3 = v1 - v3;
            v2.normalize();
            v3.normalize();

            Eigen::Vector3f result = v2.cross(v3);
            result.normalize();
            pcl::PointXYZ normal_target(origin.x + result.x(), origin.y + result.y(), origin.z + result.z());

            Eigen::Vector3f cam(p1.x, p1.y, p1.z);
            cam.normalize();
            double angle = result.dot(cam);

            std::cout << "angle: " << angle << std::endl;
            std::cout << "thresh: " << (M_PI / 6.0f) << std::endl;

            vis->addLine(origin, p1, "l1");
            vis->addLine(p1, normal_target);*/
        }

        objectCounter++;
    }

    std::cout << "render meshes" << std::endl;
    for (int i=0;i<meshs[getActiveSequenceId()].size();i++)
    {
        if (getActiveObjectId() == -1 || getActiveObjectId() == objectCounter)
        {
            std::stringstream name;
            name << "text" << i;
            pcl::PolygonMesh::Ptr mesh = meshs[getActiveSequenceId()][i];
            vis->addPolygonMesh(*(mesh), name.str(), v);
        }

        objectCounter++;
    }

    //vis->addCoordinateSystem();

    if (roi != NULL)
    {
        vis->addPointCloud(roi->getCloud(), "roicloud");
        updateRoi();
        vis->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5.0, "roicloud");
    }

    //trigger(EventManager::UPDATE_COMPLETE);
}

/*
void PclRenderer::renderTexturedMesh(TexturedMesh mesh, std::string text, bool step)
{
    renderMesh(mesh.mesh, text, step);
}

void PclRenderer::renderMesh(pcl::PolygonMesh::Ptr mesh, std::string text, bool step)
{
    int v;
    vis->createViewPort(0,0,1,1,v);
    vis->removeAllPointClouds();
    vis->removeAllShapes();

    vis->addPolygonMesh(*mesh, text, v);

    if (step)
    {
        vis->spin();
    }
    else
    {
        vis->spinOnce();
    }
}
*/

}
}
