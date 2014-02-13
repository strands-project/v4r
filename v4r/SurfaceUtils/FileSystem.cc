/**
 *  Copyright (C) 2012  
 *    Andreas Richtsfeld, Johann Prankl, Thomas Mörwald
 *    Automation and Control Institute
 *    Vienna University of Technology
 *    Gusshausstraße 25-29
 *    1170 Vienn, Austria
 *    ari(at)acin.tuwien.ac.at
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
 *  along with this program.  If not, see http://www.gnu.org/licenses/
 */

#include <set>
#include "FileSystem.hh"
#include "Utils.hh"
#include <pcl/sample_consensus/model_types.h>

using namespace surface;

template<typename T>
  void
  FileSystem::write_vector (const std::vector<T> &vec, FILE* pFile)
  {
    unsigned s = vec.size ();
    fwrite (&s, sizeof(unsigned), 1, pFile);
    for (unsigned i = 0; i < vec.size (); i++)
      fwrite (&vec[i], sizeof(T), 1, pFile);
  }

template<typename T>
  void
  FileSystem::write_set (const std::set<T> &se, FILE* pFile)
  {
    unsigned s = se.size ();
    fwrite (&s, sizeof(unsigned), 1, pFile);
    for (typename std::set<T>::iterator i = se.begin (); i != se.end (); i++)
      fwrite (&(*i), sizeof(T), 1, pFile);
  }

void
FileSystem::write_eigen_vector2d (const std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d> > &vec,
                                  FILE* pFile)
{
  unsigned s = vec.size ();
  fwrite (&s, sizeof(unsigned), 1, pFile);
  for (unsigned i = 0; i < vec.size (); i++)
    fwrite (&vec[i], sizeof(Eigen::Vector2d), 1, pFile);
}

void
FileSystem::write_eigen_vector3d (const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > &vec,
                                  FILE* pFile)
{
  unsigned s = vec.size ();
  fwrite (&s, sizeof(unsigned), 1, pFile);
  for (unsigned i = 0; i < vec.size (); i++)
    fwrite (&vec[i], sizeof(Eigen::Vector3d), 1, pFile);
}

void
FileSystem::write_eigen_vector3f (const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > &vec,
                                  FILE* pFile)
{
  unsigned s = vec.size ();
  fwrite (&s, sizeof(unsigned), 1, pFile);
  for (unsigned i = 0; i < vec.size (); i++)
    fwrite (&vec[i], sizeof(Eigen::Vector3f), 1, pFile);
}

void
FileSystem::write_nurbs_curve_vector (const std::vector<ON_NurbsCurve> &vec, FILE* pFile)
{
  unsigned s = vec.size ();
  fwrite (&s, sizeof(unsigned), 1, pFile);
  for (unsigned i = 0; i < vec.size (); i++)
    write_nurbs_curve (vec[i], pFile);
}

void
FileSystem::write_nurbs_surface (const ON_NurbsSurface &surf, FILE* pFile)
{
  fwrite (&surf.m_dim, sizeof(int), 1, pFile);
  fwrite (&surf.m_is_rat, sizeof(int), 1, pFile);
  fwrite (&surf.m_order[0], sizeof(int), 2, pFile);
  fwrite (&surf.m_cv_count[0], sizeof(int), 2, pFile);
  fwrite (&surf.m_cv_stride[0], sizeof(int), 2, pFile);

  for (int i = 0; i < surf.KnotCount (0); i++)
    fwrite (&surf.m_knot[0][i], sizeof(double), 1, pFile);

  for (int i = 0; i < surf.KnotCount (1); i++)
    fwrite (&surf.m_knot[1][i], sizeof(double), 1, pFile);

  int cv_capacity = surf.m_cv_count[0] * surf.m_cv_count[1] * surf.m_dim;
  for (int i = 0; i < cv_capacity; i++)
    fwrite (&surf.m_cv[i], sizeof(double), 1, pFile);
}

void
FileSystem::write_nurbs_curve (const ON_NurbsCurve &curve, FILE* pFile)
{
  fwrite (&curve.m_dim, sizeof(int), 1, pFile);
  fwrite (&curve.m_is_rat, sizeof(int), 1, pFile);
  fwrite (&curve.m_order, sizeof(int), 1, pFile);
  fwrite (&curve.m_cv_count, sizeof(int), 1, pFile);
  fwrite (&curve.m_cv_stride, sizeof(int), 1, pFile);

  for (int i = 0; i < curve.KnotCount (); i++)
    fwrite (&curve.m_knot[i], sizeof(double), 1, pFile);

  int cv_capacity = curve.m_cv_count * curve.m_dim;
  for (int i = 0; i < cv_capacity; i++)
    fwrite (&curve.m_cv[i], sizeof(double), 1, pFile);
}

void
FileSystem::write_tgModel (const TomGine::tgModel &model, FILE* pFile)
{
  write_vector (model.m_vertices, pFile);

  unsigned s = model.m_faces.size ();
  fwrite (&s, sizeof(unsigned), 1, pFile);

  for (unsigned i = 0; i < s; i++)
    write_vector (model.m_faces[i].v, pFile);
}

void
FileSystem::write_surface_model (const SurfaceModel &model, FILE *pFile)
{
  // save single variables
  fwrite (&model.idx, sizeof(int), 1, pFile);
  fwrite (&model.type, sizeof(int), 1, pFile);
//   fwrite (&model.level, sizeof(int), 1, pFile);
  fwrite (&model.label, sizeof(int), 1, pFile);
//   fwrite (&model.used, sizeof(bool), 1, pFile);
  fwrite (&model.valid, sizeof(bool), 1, pFile);
  fwrite (&model.selected, sizeof(bool), 1, pFile);
  fwrite (&model.savings, sizeof(double), 1, pFile);
  fwrite (&model.pose, sizeof(Eigen::Matrix4d), 1, pFile);

  // save vectors
  write_vector (model.coeffs, pFile);
  write_vector (model.indices, pFile);
  //   write_vector (model.contour, pFile);                       /// TODO
  write_vector (model.error, pFile);
  write_vector (model.probs, pFile);
  write_set (model.neighbors2D, pFile);
  write_set (model.neighbors3D, pFile);
  write_eigen_vector2d (model.nurbs_params, pFile);
  write_eigen_vector3d (model.normals, pFile);

  // save nurbs
  write_nurbs_surface (model.nurbs, pFile);
  write_nurbs_curve_vector (model.curves_image, pFile);
  write_nurbs_curve_vector (model.curves_param, pFile);

  // save mesh
  write_tgModel (model.mesh, pFile);

}

void
FileSystem::write_view (const View &view, FILE *pFile)
{
  fwrite (&view.width, sizeof(unsigned), 1, pFile);
  fwrite (&view.height, sizeof(unsigned), 1, pFile);
  fwrite (&view.intrinsic, sizeof(Eigen::Matrix3d), 1, pFile);
  fwrite (&view.extrinsic, sizeof(Eigen::Matrix4d), 1, pFile);

  unsigned s = view.surfaces.size ();
  fwrite (&s, sizeof(unsigned), 1, pFile);
  for (unsigned i = 0; i < s; i++)
  {
    write_surface_model (*(view.surfaces[i]), pFile);
  }

  printf ("view.corners.size: %ul\n", view.corners.size ());

  write_vector<Edgel> (view.edgels, pFile);
  write_vector<Corner> (view.corners, pFile);
  s = view.edges.size ();
  fwrite (&s, sizeof(unsigned), 1, pFile);
  for (unsigned i = 0; i < s; i++)
  {
    const Edge &e = view.edges[i];
    fwrite (&e.surface[0], sizeof(int), 2, pFile);
    fwrite (&e.corner[0], sizeof(int), 2, pFile);
    write_vector<int> (e.edgels, pFile);

    printf ("view.edges[%d] s: %d %d  c: %d %d\n", i, e.surface[0], e.surface[1], e.corner[0], e.corner[1]);
  }
  printf ("[FileSystem::write_view] %d edges ...\n", s);
}

template<typename T>
  size_t
  FileSystem::read_vector (std::vector<T> &vec, FILE* pFile)
  {
    size_t result (0);
    unsigned s;
    result += fread (&s, sizeof(unsigned), 1, pFile) * sizeof(unsigned);
    vec.assign (s, T ());
    for (unsigned i = 0; i < s; i++)
      result += fread (&vec[i], sizeof(T), 1, pFile) * sizeof(T);

    return result;
  }

template<typename T>
  size_t
  FileSystem::read_set (std::set<T> &se, FILE* pFile)
  {
    size_t result (0);
    unsigned s;
    result += fread (&s, sizeof(unsigned), 1, pFile) * sizeof(unsigned);
    for (unsigned i = 0; i < s; i++)
    {
      T t;
      result += fread (&t, sizeof(T), 1, pFile) * sizeof(T);
      se.insert(t);
    }

    return result;
  }

size_t
FileSystem::read_eigen_vector2d (std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d> > &vec,
                                 FILE* pFile)
{
  size_t result (0);
  unsigned s;
  result += fread (&s, sizeof(unsigned), 1, pFile) * sizeof(unsigned);
  vec.assign (s, Eigen::Vector2d (0.0, 0.0));
  for (unsigned i = 0; i < s; i++)
    result += fread (&vec[i], sizeof(Eigen::Vector2d), 1, pFile) * sizeof(Eigen::Vector2d);

  return result;
}

size_t
FileSystem::read_eigen_vector3d (std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > &vec,
                                 FILE* pFile)
{
  size_t result (0);
  unsigned s;
  result += fread (&s, sizeof(unsigned), 1, pFile) * sizeof(unsigned);
  vec.assign (s, Eigen::Vector3d (0.0, 0.0, 0.0));
  for (unsigned i = 0; i < s; i++)
    result += fread (&vec[i], sizeof(Eigen::Vector3d), 1, pFile) * sizeof(Eigen::Vector3d);

  return result;
}

size_t
FileSystem::read_eigen_vector3f (std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > &vec,
                                 FILE* pFile)
{
  size_t result (0);
  unsigned s;
  result += fread (&s, sizeof(unsigned), 1, pFile) * sizeof(unsigned);
  vec.assign (s, Eigen::Vector3f (0.0, 0.0, 0.0));
  for (unsigned i = 0; i < s; i++)
    result += fread (&vec[i], sizeof(Eigen::Vector3f), 1, pFile) * sizeof(Eigen::Vector3f);

  return result;
}

size_t
FileSystem::read_nurbs_curve_vector (std::vector<ON_NurbsCurve> &vec, FILE* pFile)
{
  size_t result (0);
  unsigned s;
  result += fread (&s, sizeof(unsigned), 1, pFile) * sizeof(unsigned);
  vec.assign (s, ON_NurbsCurve ());
  for (unsigned i = 0; i < s; i++)
    result += read_nurbs_curve (vec[i], pFile);

  return result;
}

size_t
FileSystem::read_nurbs_surface (ON_NurbsSurface &surf, FILE* pFile)
{
  size_t result (0);
  result += fread (&surf.m_dim, sizeof(int), 1, pFile) * sizeof(int);
  result += fread (&surf.m_is_rat, sizeof(int), 1, pFile) * sizeof(int);
  result += fread (&surf.m_order[0], sizeof(int), 2, pFile) * sizeof(int);
  result += fread (&surf.m_cv_count[0], sizeof(int), 2, pFile) * sizeof(int);
  result += fread (&surf.m_cv_stride[0], sizeof(int), 2, pFile) * sizeof(int);

  surf.ReserveKnotCapacity (0, surf.KnotCount (0));
  for (int i = 0; i < surf.KnotCount (0); i++)
    result += fread (&surf.m_knot[0][i], sizeof(double), 1, pFile) * sizeof(double);

  surf.ReserveKnotCapacity (1, surf.KnotCount (1));
  for (int i = 0; i < surf.KnotCount (1); i++)
    result += fread (&surf.m_knot[1][i], sizeof(double), 1, pFile) * sizeof(double);

  int cv_capacity = surf.m_cv_count[0] * surf.m_cv_count[1] * surf.m_dim;
  surf.ReserveCVCapacity (cv_capacity);
  for (int i = 0; i < cv_capacity; i++)
    result += fread (&surf.m_cv[i], sizeof(double), 1, pFile) * sizeof(double);

  return result;
}

size_t
FileSystem::read_nurbs_curve (ON_NurbsCurve &curve, FILE* pFile)
{
  size_t result (0);
  result += fread (&curve.m_dim, sizeof(int), 1, pFile) * sizeof(int);
  result += fread (&curve.m_is_rat, sizeof(int), 1, pFile) * sizeof(int);
  result += fread (&curve.m_order, sizeof(int), 1, pFile) * sizeof(int);
  result += fread (&curve.m_cv_count, sizeof(int), 1, pFile) * sizeof(int);
  result += fread (&curve.m_cv_stride, sizeof(int), 1, pFile) * sizeof(int);

  curve.ReserveKnotCapacity (curve.KnotCount ());
  for (int i = 0; i < curve.KnotCount (); i++)
    result += fread (&curve.m_knot[i], sizeof(double), 1, pFile) * sizeof(double);

  int cv_capacity = curve.m_cv_count * curve.m_dim;
  curve.ReserveCVCapacity (cv_capacity);
  for (int i = 0; i < cv_capacity; i++)
    result += fread (&curve.m_cv[i], sizeof(double), 1, pFile) * sizeof(double);

  return result;
}

size_t
FileSystem::read_tgModel (TomGine::tgModel &model, FILE* pFile)
{
  size_t result (0);
  result += read_vector (model.m_vertices, pFile);

  unsigned s;
  result += fread (&s, sizeof(unsigned), 1, pFile) * sizeof(unsigned);
  model.m_faces.assign (s, TomGine::tgFace ());

  for (unsigned i = 0; i < s; i++)
    result += read_vector (model.m_faces[i].v, pFile);

  return result;
}

size_t
FileSystem::read_surface_model (SurfaceModel &model, FILE *pFile)
{
  size_t result (0);

  // read single variables
  result += fread (&model.idx, sizeof(int), 1, pFile) * sizeof(int);
  result += fread (&model.type, sizeof(int), 1, pFile) * sizeof(int);
//   result += fread (&model.level, sizeof(int), 1, pFile) * sizeof(int);
  result += fread (&model.label, sizeof(int), 1, pFile) * sizeof(int);
//   result += fread (&model.used, sizeof(bool), 1, pFile) * sizeof(bool);
  result += fread (&model.valid, sizeof(bool), 1, pFile) * sizeof(bool);
  result += fread (&model.selected, sizeof(bool), 1, pFile) * sizeof(bool);
  result += fread (&model.savings, sizeof(double), 1, pFile) * sizeof(double);
  result += fread (&model.pose, sizeof(Eigen::Matrix4d), 1, pFile) * sizeof(Eigen::Matrix4d);

  // read vectors
  result += read_vector (model.coeffs, pFile);
  result += read_vector (model.indices, pFile);
  //  result += read_vector (model.contour, pFile);                         /// TODO Problem: contour is now vector< vector<int> > contours!
  result += read_vector (model.error, pFile);
  result += read_vector (model.probs, pFile);
  result += read_set (model.neighbors2D, pFile);
  result += read_set (model.neighbors3D, pFile);
  result += read_eigen_vector2d (model.nurbs_params, pFile);
  result += read_eigen_vector3d (model.normals, pFile);

  // read nurbs
  result += read_nurbs_surface (model.nurbs, pFile);
  result += read_nurbs_curve_vector (model.curves_image, pFile);
  result += read_nurbs_curve_vector (model.curves_param, pFile);

  // read mesh
  result += read_tgModel (model.mesh, pFile);

  return result;
}

size_t
FileSystem::read_view (View &view, FILE* pFile)
{
  size_t result (0);

  result += fread (&view.width, sizeof(unsigned), 1, pFile) * sizeof(unsigned);
  result += fread (&view.height, sizeof(unsigned), 1, pFile) * sizeof(unsigned);
  result += fread (&view.intrinsic, sizeof(Eigen::Matrix3d), 1, pFile) * sizeof(Eigen::Matrix3d);
  result += fread (&view.extrinsic, sizeof(Eigen::Matrix4d), 1, pFile) * sizeof(Eigen::Matrix4d);

  unsigned s;
  result += fread (&s, sizeof(unsigned), 1, pFile) * sizeof(unsigned);
  view.surfaces.resize (s);
  for (unsigned i = 0; i < s; i++)
  {
    view.surfaces[i].reset (new SurfaceModel ());
    result += read_surface_model (*(view.surfaces[i]), pFile);
  }

  result += read_vector<Edgel> (view.edgels, pFile);
  result += read_vector<Corner> (view.corners, pFile);
  result += fread (&s, sizeof(unsigned), 1, pFile) * sizeof(unsigned);
  view.edges.resize (s);
  for (unsigned i = 0; i < s; i++)
  {
    Edge &e = view.edges[i];
    result += fread (&e.surface[0], sizeof(int), 2, pFile) * sizeof(int);
    result += fread (&e.corner[0], sizeof(int), 2, pFile) * sizeof(int);
    result += read_vector<int> (e.edgels, pFile);
  }

  return result;
}

bool
FileSystem::Save (const std::vector<SurfaceModel> &models, const std::string &filename)
{
  FILE * pFile;
  pFile = fopen (filename.c_str (), "wb");
  if (pFile == NULL)
  {
    printf ("[surface::FileSystem::Save] Error: cannot open file for writing: '%s'\n", filename.c_str ());
    return false;
  }
  if (models.empty ())
  {
    printf ("[surface::FileSystem::Save] Warning: no data to save: '%s'\n", filename.c_str ());
    return false;
  }

  unsigned s = models.size ();
  fwrite (&s, sizeof(unsigned), 1, pFile);
  for (unsigned i = 0; i < s; i++)
  {
    write_surface_model (models[i], pFile);
  }

  fclose (pFile);

  return true;
}

bool
FileSystem::Load (std::vector<SurfaceModel> &models, const std::string &filename)
{
  FILE * pFile;
  pFile = fopen (filename.c_str (), "rb");
  if (pFile == NULL)
  {
    printf ("[surface::FileSystem::Load] Error: cannot open file for reading: '%s'\n", filename.c_str ());
    return false;
  }
  long lSize;
  fseek (pFile, 0, SEEK_END);
  lSize = ftell (pFile);
  rewind (pFile);
  long result (0);

  unsigned s;
  result += fread (&s, sizeof(unsigned), 1, pFile) * sizeof(unsigned);
  models.resize (s);
  for (unsigned i = 0; i < s; i++)
  {
    result += read_surface_model (models[i], pFile);
  }

  if (result != lSize)
  {
    printf ("[surface::FileSystem::Load(std::vector<SurfaceModel> &, const std::string &)] %ld %ld\n", result, lSize);
    throw std::runtime_error ("[surface::FileSystem::Load] Reading error (memory size does not match) ");
  }

  fclose (pFile);

  return true;
}

bool
FileSystem::Save (const std::vector<SurfaceModel::Ptr> &models, const std::string &filename)
{
  FILE * pFile;
  pFile = fopen (filename.c_str (), "wb");
  if (pFile == NULL)
  {
    printf ("[surface::FileSystem::Save] Error: cannot open file for writing: '%s'\n", filename.c_str ());
    return false;
  }
  if (models.empty ())
  {
    printf ("[surface::FileSystem::Save] Warning: no data to save: '%s'\n", filename.c_str ());
    return false;
  }

  unsigned s = models.size ();
  fwrite (&s, sizeof(unsigned), 1, pFile);
  for (unsigned i = 0; i < s; i++)
  {
    write_surface_model (*models[i], pFile);
  }

  fclose (pFile);

  return true;
}

bool
FileSystem::Load (std::vector<SurfaceModel::Ptr> &models, const std::string &filename)
{
  FILE * pFile;
  pFile = fopen (filename.c_str (), "rb");
  if (pFile == NULL)
  {
    printf ("[surface::FileSystem::Load] Error: cannot open file for reading: '%s'\n", filename.c_str ());
    return false;
  }
  long lSize;
  fseek (pFile, 0, SEEK_END);
  lSize = ftell (pFile);
  rewind (pFile);
  long result (0);

  unsigned s;
  result += fread (&s, sizeof(unsigned), 1, pFile) * sizeof(unsigned);
  models.resize (s);
  for (unsigned i = 0; i < s; i++)
  {
    models[i].reset (new SurfaceModel);
    result += read_surface_model (*models[i], pFile);
  }

  if (result != lSize)
  {
    printf ("[surface::FileSystem::Load(std::vector<SurfaceModel::Ptr> &, const std::string &)] %ld %ld\n", result,
            lSize);
    throw std::runtime_error (
                              "[surface::FileSystem::Load(std::vector<SurfaceModel::Ptr> &, const std::string &)] Reading error (memory size does not match) ");
  }

  fclose (pFile);

  return true;
}

bool
FileSystem::Save (const View &view, const std::string &filename)
{
  FILE * pFile;
  pFile = fopen (filename.c_str (), "wb");
  if (pFile == NULL)
  {
    printf (
            "[surface::FileSystem::Save(const View&, const std::string &)] Error: cannot open file for writing: '%s'\n",
            filename.c_str ());
    return false;
  }
  if (view.surfaces.empty ())
  {
    printf ("[surface::FileSystem::Save(const View&, const std::string &)] Warning: no data to save: '%s'\n",
            filename.c_str ());
    return false;
  }

  write_view (view, pFile);

  fclose (pFile);

  return true;
}
bool
FileSystem::Load (View &view, const std::string &filename)
{
  FILE * pFile;
  pFile = fopen (filename.c_str (), "rb");
  if (pFile == NULL)
  {
    printf ("[surface::FileSystem::Load(View &, const std::string &)] Error: cannot open file for reading: '%s'\n",
            filename.c_str ());
    return false;
  }
  long lSize;
  fseek (pFile, 0, SEEK_END);
  lSize = ftell (pFile);
  rewind (pFile);
  long result (0);

  result += read_view (view, pFile);

  if (result != lSize)
  {
    printf ("[surface::FileSystem::Load(View &, const std::string &)] %ld %ld\n", result, lSize);
    throw std::runtime_error (
                              "[surface::FileSystem::Load(View &, const std::string &)] Reading error (memory size does not match) ");
  }

  fclose (pFile);

  return true;
}

bool
FileSystem::SaveBrep (const std::vector<surface::SurfaceModel::Ptr> &surfaces, const std::string &filename)
{
  ON_Brep brep;

  Utils::convertSurfaces2Brep (surfaces, brep);

  FILE* fp = ON::OpenFile (filename.c_str (), "wb");
  ON_BinaryFile archive (ON::write3dm, fp);
  brep.Write (archive);
  ON::CloseFile (fp);

  return true;
}

bool
FileSystem::LoadBrep (std::vector<surface::SurfaceModel::Ptr> &surfaces, const std::string &filename)
{
  ON_Brep brep;

  if (!LoadBrep (brep, filename))
    return false;

  Utils::convertBrep2Surfaces (brep, surfaces);

  return true;
}

bool
FileSystem::SaveBrep (const ON_Brep &brep, const std::string &filename)
{

  FILE* fp = ON::OpenFile (filename.c_str (), "wb");

  if (fp == NULL)
  {
    printf ("[FileSystem::SaveBrep] ERROR, can not open file '%s'.\n", filename.c_str ());
    return false;
  }

  ON_BinaryFile archive (ON::write3dm, fp);

  if (!brep.Write (archive))
  {
    printf ("[FileSystem::SaveBrep] ERROR, can write file '%s'.\n", filename.c_str ());
    return false;
  }

  ON::CloseFile (fp);

  return true;
}

bool
FileSystem::LoadBrep (ON_Brep &brep, const std::string &filename)
{
  FILE* fp = ON::OpenFile (filename.c_str (), "rb");

  if (fp == NULL)
  {
    printf ("[FileSystem::LoadBrep] ERROR, can not open file '%s'.\n", filename.c_str ());
    return false;
  }

  ON_BinaryFile file (ON::read3dm, fp);

  if (!brep.Read (file))
  {
    printf ("[FileSystem::LoadBrep] ERROR, can read file '%s'.\n", filename.c_str ());
    return false;
  }

  ON::CloseFile (fp);

  return true;
}

// -------------------- Save File Sequence -------------------- //
SaveFileSequence::SaveFileSequence (Parameter p) :
  param (p)
{
  initialized = false;
  intrinsic = Eigen::Matrix3d::Zero ();
  extrinsic = Eigen::Matrix4d::Zero ();
}

SaveFileSequence::~SaveFileSequence ()
{
}

void
SaveFileSequence::InitFileSequence (std::string _filename, int _start, int _end)
{
  param.filename = _filename;
  param.start = _start;
  param.end = _end;
  current = param.start;
  initialized = true;
}

void
SaveFileSequence::SetCameraParameters (Eigen::Matrix3d &_intrinsic, Eigen::Matrix4d &_extrinsic)
{
  intrinsic = _intrinsic;
  extrinsic = _extrinsic;
}

void
SaveFileSequence::SaveNextView (const View &view)
{
  if (current > param.end)
    std::printf ("[SaveFileSequence::SaveNextView] Warning: End of sequence reached. Continue sequence.\n");

  char next_file[1024] = "";
  std::sprintf (next_file, param.filename.c_str (), current);

  FileSystem::Save (view, next_file);
  current++;

  printf ("corners: %ul\n", view.corners.size ());
  printf ("edges: %ul\n", view.edges.size ());
  printf ("edgels: %ul\n", view.edgels.size ());

  printf ("saved to: '%s'\n", next_file);
}

// TODO: remove this hacky function
void
SaveFileSequence::SaveNextView (const std::vector<SurfaceModel::Ptr> &surfaces)
{
  if (current > param.end)
    std::printf ("[SaveFileSequence::SaveNextView] Warning: End of sequence reached. Continue sequence.\n");

  char next_file[1024] = "";
  std::sprintf (next_file, param.filename.c_str (), current);

  View view;
  view.width = param.width;
  view.height = param.height;
  view.intrinsic = intrinsic;
  view.extrinsic = extrinsic;
  view.surfaces = surfaces;
  FileSystem::Save (view, next_file);
  current++;
}

// -------------------- Load File Sequence -------------------- //
LoadFileSequence::LoadFileSequence (Parameter p) :
  param (p)
{
}

LoadFileSequence::~LoadFileSequence ()
{
}

void
LoadFileSequence::InitFileSequence (std::string _filename, int _start, int _end)
{
  param.filename = _filename;
  param.start = _start;
  param.end = _end;
  current = param.start;
}

void
LoadFileSequence::LoadNextView (View &view)
{
  if (current > param.end)
    std::printf ("[LoadFileSequence::LoadNextView] Warning: End of sequence reached. Continue sequencing.\n");

  char next_file[1024] = "";
  std::sprintf (next_file, param.filename.c_str (), current);

  printf ("[LoadFileSequence::LoadNextView] Load %s!\n", next_file);

  FileSystem::Load (view, next_file);
  current++;
}

