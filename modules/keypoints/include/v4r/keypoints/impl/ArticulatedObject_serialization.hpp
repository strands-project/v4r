/**
 * $Id$
 * 
 * Software License Agreement (GNU General Public License)
 *
 *  Copyright (C) 2015:
 *
 *    Johann Prankl, prankl@acin.tuwien.ac.at
 *    Aitor Aldoma, aldoma@acin.tuwien.ac.at
 *
 *      Automation and Control Institute
 *      Vienna University of Technology
 *      Gusshausstraße 25-29
 *      1170 Vienn, Austria
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
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * @author Johann Prankl, Aitor Aldoma
 *
 */

#ifndef ARTICULATED_OBJECT_BOOST_SERIALIZATION
#define ARTICULATED_OBJECT_BOOST_SERIALIZATION

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/base_object.hpp>
#include <v4r/keypoints/impl/pair_serialization.hpp>
#include <v4r/keypoints/impl/eigen_boost_serialization.hpp>
#include <v4r/keypoints/impl/opencv_serialization.hpp>
#include <v4r/keypoints/impl/triple_serialization.hpp>
#include <v4r/keypoints/impl/pair_serialization.hpp>
#include <v4r/keypoints/impl/Object.hpp>
#include <v4r/keypoints/ArticulatedObject.h>
#include <v4r/keypoints/PartMotion6D.h>
#include <v4r/keypoints/PartRotation1D.h>
 

/** ObjectView serialization **/
namespace boost{namespace serialization{

  template<class Archive>
  void serialize(Archive & ar, v4r::GlobalPoint &pt, const unsigned int version)
  {
    ar & pt.cnt;
    ar & pt.pt;
    ar & pt.n;
  }

}}

/** ObjectView serialization **/
namespace boost{namespace serialization{

  template<class Archive>
  void serialize(Archive & ar, v4r::ObjectView &view, const unsigned int version)
  {
    ar & view.idx;
    ar & view.camera_id;
    ar & view.center;
    ar & view.image;
    ar & view.descs;
    ar & view.keys;
    ar & view.points;
    ar & view.viewrays;
    ar & view.cam_points;
    ar & view.projs;
    ar & view.part_indices;
  }

}}


/** ObjectView::Ptr serialization **/
BOOST_SERIALIZATION_SPLIT_FREE(::v4r::ObjectView::Ptr)
namespace boost{namespace serialization{

  template<class Archive>
  void save(Archive & ar, const ::v4r::ObjectView::Ptr& view, const unsigned int version)
  {
    ar & *view;
  }

  template <class Archive>
  void load(Archive & ar, ::v4r::ObjectView::Ptr& view, const unsigned int version)
  {
    view.reset(new v4r::ObjectView(0));
    ar & *view;
  }

}}


/** Object serialization **/
BOOST_SERIALIZATION_SPLIT_FREE(::v4r::Object)
namespace boost{namespace serialization{

  template <class Archive>
  void save(Archive & ar, const v4r::Object &o, const unsigned int version)
  {
    ar & o.id;
    ar & o.cameras;
    ar & o.camera_parameter;
    ar & o.views;
    ar & o.points;
    ar & o.cb_centers;
    ar & o.cb_entries;
  }

  template <class Archive>
  void load(Archive & ar, v4r::Object &o, const unsigned int version)
  {
    ar & o.id;
    ar & o.cameras;
    ar & o.camera_parameter;
    ar & o.views;
    ar & o.points;
    ar & o.cb_centers;
    ar & o.cb_entries;

    for (unsigned i=0; i<o.views.size(); i++)
      o.views[i]->object = &o;
  }


}}

/** Part serialization **/
namespace boost{namespace serialization{

  template <class Archive>
  void serialize(Archive & ar, v4r::Part &p, const unsigned int version)
  {
    ar & p.type;
    ar & p.idx;
    ar & p.is_hyp;
    ar & p.pose;
    ar & p.features;
    ar & p.projs;
    ar & p.subparts;
  }

}}

/** PartRotation1D serialization **/
namespace boost{namespace serialization{

  template <class Archive>
  void serialize(Archive & ar, v4r::PartRotation1D &p, const unsigned int version)
  {
    ar & boost::serialization::base_object<v4r::Part> ( p );
    ar & p.angle;
    ar & p.rt;
  }

}}

/** PartMotion6D serialization **/
namespace boost{namespace serialization{

  template <class Archive>
  void serialize(Archive & ar, v4r::PartMotion6D &p, const unsigned int version)
  {
    ar & boost::serialization::base_object<v4r::Part> ( p );
    ar & p.rt;
  }

}}

/** Part::Ptr serialization **/
BOOST_SERIALIZATION_SPLIT_FREE(::v4r::Part::Ptr)
namespace boost{namespace serialization{

  template<class Archive>
  void save(Archive & ar, const ::v4r::Part::Ptr& p, const unsigned int version)
  {
    ar & p->type;

    switch (p->type)
    {
      case v4r::Part::STATIC:
      {
        ar & *p;
        break;
      }
      case v4r::Part::ROTATION_1D:
      {
        ar & *v4r::SmartPtr<v4r::PartRotation1D>(p);
        break;
      }
      case v4r::Part::MOTION_6D:
      {
        ar & *v4r::SmartPtr<v4r::PartMotion6D>(p);
        break;
      }
      default:
        std::cout<<"serialization::v4r::Part::Ptr not implemented!"<<std::endl;
        break;
    }
  }

  template <class Archive>
  void load(Archive & ar, ::v4r::Part::Ptr& p, const unsigned int version)
  {
    v4r::Part::Type type;

    ar & type;

    switch (type)
    {
      case v4r::Part::STATIC:
      {
        p.reset(new v4r::Part());
        ar & *p;
        break;
      }
      case v4r::Part::ROTATION_1D:
      {
        p.reset( new v4r::PartRotation1D() );
        ar & *v4r::SmartPtr<v4r::PartRotation1D>(p);
        break;
      }
      case v4r::Part::MOTION_6D:
      {
        p.reset( new v4r::PartMotion6D() );
        ar & *v4r::SmartPtr<v4r::PartMotion6D>(p);
        break;
      }
       default:
        std::cout<<"serialization::v4r::Part::Ptr not implemented!"<<std::endl;
        break;
    }

  }

}}


/** ArticulatedObject serialization **/
BOOST_SERIALIZATION_SPLIT_FREE(::v4r::ArticulatedObject::Ptr)
namespace boost{namespace serialization{

  template<class Archive>
  void save(Archive & ar, const ::v4r::ArticulatedObject::Ptr &o, const unsigned int version)
  {
    ar & o->version;
    ar & boost::serialization::base_object<v4r::Object> ( *o );
    ar & boost::serialization::base_object<v4r::PartMotion6D> ( *o );
    ar & o->part_parameter;

    int num_parts = o->parts.size();
    ar & num_parts;

    for (unsigned i=1; i<o->parts.size(); i++)  // the first part is the object itself
    {
      ar & o->parts[i];
    }
  }

  template <class Archive>
  void load(Archive & ar, ::v4r::ArticulatedObject::Ptr& o, const unsigned int version)
  {
    o.reset(new v4r::ArticulatedObject());

    ar & o->version;
    ar & boost::serialization::base_object<v4r::Object> ( *o );
    ar & boost::serialization::base_object<v4r::PartMotion6D> ( *o );
    ar & o->part_parameter;

    int num_parts;
    ar & num_parts;

    if (num_parts > 0)
    {
      v4r::Part::Ptr p;
      o->parts.resize(num_parts);
      o->parts[0] = o;  // the first part is the object itself

      for (unsigned i=1; i<num_parts; i++)
      {
        ar & o->parts[i];
      }
    }
  }

}}

#endif
