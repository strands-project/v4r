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

#ifndef SURFACE_FILESYSTEM_HH
#define SURFACE_FILESYSTEM_HH

#include <cstdio>
#include <stdexcept>

#include "SurfaceModel.hpp"

#include "v4r/TomGine/tgModel.h"

namespace surface {

class FileSystem
{
private:

  template<typename T>
  static void write_vector(const std::vector<T> &vec, FILE* pFile);
  template<typename T>
  static void write_set (const std::set<T> &se, FILE* pFile);
  static void write_eigen_vector2d(const std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d> > &vec,
      FILE* pFile);
  static void write_eigen_vector3d(const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > &vec,
      FILE* pFile);
  static void write_eigen_vector3f(const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > &vec,
      FILE* pFile);
  static void write_nurbs_curve_vector (const std::vector<ON_NurbsCurve> &vec, FILE* pFile);
  static void write_nurbs_surface(const ON_NurbsSurface &surf, FILE* pFile);
  static void write_nurbs_curve(const ON_NurbsCurve &curve, FILE* pFile);
  static void write_tgModel(const TomGine::tgModel &model, FILE* pFile);
  static void write_surface_model(const SurfaceModel &model, FILE *pFile);
  static void write_view(const View &view, FILE *pFile);

  template<typename T>
  static size_t read_vector(std::vector<T> &vec, FILE* pFile);
  template<typename T>
  static size_t read_set (std::set<T> &se, FILE* pFile);
  static size_t read_eigen_vector2d(std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d> > &vec,
      FILE* pFile);
  static size_t read_eigen_vector3d(std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > &vec,
      FILE* pFile);
  static size_t read_eigen_vector3f(std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > &vec,
      FILE* pFile);
  static size_t read_nurbs_curve_vector (std::vector<ON_NurbsCurve> &vec, FILE* pFile);
  static size_t read_nurbs_surface(ON_NurbsSurface &surf, FILE* pFile);
  static size_t read_nurbs_curve(ON_NurbsCurve &curve, FILE* pFile);
  static size_t read_tgModel(TomGine::tgModel &model, FILE* pFile);
  static size_t read_surface_model(SurfaceModel &model, FILE *pFile);
  static size_t read_view(View &view, FILE* pFile);

public:
  static bool Save(const std::vector<SurfaceModel> &models, const std::string &filename);
  static bool Load(std::vector<SurfaceModel> &models, const std::string &filename);

  static bool Save(const std::vector<SurfaceModel::Ptr> &models, const std::string &filename);
  static bool Load(std::vector<SurfaceModel::Ptr> &models, const std::string &filename);

  static bool Save(const View &view, const std::string &filename);
  static bool Load(View &view, const std::string &filename);

  static bool SaveBrep(const std::vector<surface::SurfaceModel::Ptr> &surfaces, const std::string &filename);
  static bool LoadBrep(std::vector<surface::SurfaceModel::Ptr> &surfaces, const std::string &filename);

  static bool SaveBrep(const ON_Brep &brep, const std::string &filename);
  static bool LoadBrep(ON_Brep &brep, const std::string &filename);

};


class SaveFileSequence
{

public:

  class Parameter
  {
  public:
    std::string filename;
    unsigned start;
    unsigned end;
    unsigned width;
    unsigned height;

    Parameter(std::string _filename = "/media/Daten/OD-IROS/results/results%1d.sfv",
              unsigned _start = 0,
              unsigned _end = 0,
              unsigned _width = 640,
              unsigned _height = 480)
    : filename(_filename), start(_start), end(_end), width(_width), height(_height) {}
  };

private:
  bool initialized;
  unsigned current;

public:

  Parameter param;

  Eigen::Matrix3d intrinsic;
  Eigen::Matrix4d extrinsic;

  SaveFileSequence(Parameter p = Parameter());
  ~SaveFileSequence();

  /** Initialize saving of a sequence **/
  void InitFileSequence(std::string filename, int start, int end);

  /** Set intrinsic and extrinsic parameters **/
  void SetCameraParameters(Eigen::Matrix3d &_intrinsic, Eigen::Matrix4d &_extrinsic);

  /** Save next view **/
  void SaveNextView(const View &view);
  void SaveNextView(const std::vector<SurfaceModel::Ptr> &surfaces);
};


class LoadFileSequence
{
public:
  class Parameter
  {
  public:
    std::string filename;
    unsigned start;
    unsigned end;

    Parameter(std::string _filename = "/media/Daten/OD-IROS/results/results%1d.sfv",
              unsigned _start = 0,
              unsigned _end = 0)
    : filename(_filename), start(_start), end(_end) {}
  };

private:
  unsigned current;
  Parameter param;

public:

  LoadFileSequence(Parameter p = Parameter());
  ~LoadFileSequence();

  /** Initialize saving of a sequence **/
  void InitFileSequence(std::string filename, int start, int end);

  /** Load next view **/
  void LoadNextView(View &view);
};


}

#endif
