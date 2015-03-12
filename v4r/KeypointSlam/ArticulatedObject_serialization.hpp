/**
 * $Id$
 *
 * Copyright (c) 2014, Johann Prankl, All rights reserved.
 * @author Johann Prankl (prankl@acin.tuwien.ac.at)
 */

#ifndef ARTICULATED_OBJECT_BOOST_SERIALIZATION
#define ARTICULATED_OBJECT_BOOST_SERIALIZATION

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/base_object.hpp>
#include "v4r/KeypointTools/pair_serialization.hpp"
#include "v4r/KeypointTools/eigen_boost_serialization.hpp"
#include "v4r/KeypointTools/opencv_serialization.hpp"
#include "v4r/KeypointTools/triple_serialization.hpp"
#include "v4r/KeypointTools/pair_serialization.hpp"
#include "v4r/KeypointSlam/Object.hpp"
#include "ArticulatedObject.hh"
#include "PartMotion6D.hh"
#include "PartRotation1D.hh"
 

/** ObjectView serialization **/
namespace boost{namespace serialization{

  template<class Archive>
  void serialize(Archive & ar, kp::GlobalPoint &pt, const unsigned int version)
  {
    ar & pt.cnt;
    ar & pt.pt;
    ar & pt.n;
  }

}}

/** ObjectView serialization **/
namespace boost{namespace serialization{

  template<class Archive>
  void serialize(Archive & ar, kp::ObjectView &view, const unsigned int version)
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
BOOST_SERIALIZATION_SPLIT_FREE(::kp::ObjectView::Ptr)
namespace boost{namespace serialization{

  template<class Archive>
  void save(Archive & ar, const ::kp::ObjectView::Ptr& view, const unsigned int version)
  {
    ar & *view;
  }

  template <class Archive>
  void load(Archive & ar, ::kp::ObjectView::Ptr& view, const unsigned int version)
  {
    view.reset(new kp::ObjectView(0));
    ar & *view;
  }

}}


/** Object serialization **/
BOOST_SERIALIZATION_SPLIT_FREE(::kp::Object)
namespace boost{namespace serialization{

  template <class Archive>
  void save(Archive & ar, const kp::Object &o, const unsigned int version)
  {
    ar & o.id;
    ar & o.cameras;
    ar & o.camera_parameter;
    ar & o.views;
    ar & o.points;
  }

  template <class Archive>
  void load(Archive & ar, kp::Object &o, const unsigned int version)
  {
    ar & o.id;
    ar & o.cameras;
    ar & o.camera_parameter;
    ar & o.views;
    ar & o.points;

    for (unsigned i=0; i<o.views.size(); i++)
      o.views[i]->object = &o;
  }


}}

/** Part serialization **/
namespace boost{namespace serialization{

  template <class Archive>
  void serialize(Archive & ar, kp::Part &p, const unsigned int version)
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
  void serialize(Archive & ar, kp::PartRotation1D &p, const unsigned int version)
  {
    ar & boost::serialization::base_object<kp::Part> ( p );
    ar & p.angle;
    ar & p.rt;
  }

}}

/** PartMotion6D serialization **/
namespace boost{namespace serialization{

  template <class Archive>
  void serialize(Archive & ar, kp::PartMotion6D &p, const unsigned int version)
  {
    ar & boost::serialization::base_object<kp::Part> ( p );
    ar & p.rt;
  }

}}

/** Part::Ptr serialization **/
BOOST_SERIALIZATION_SPLIT_FREE(::kp::Part::Ptr)
namespace boost{namespace serialization{

  template<class Archive>
  void save(Archive & ar, const ::kp::Part::Ptr& p, const unsigned int version)
  {
    ar & p->type;

    switch (p->type)
    {
      case kp::Part::STATIC:
      {
        ar & *p;
        break;
      }
      case kp::Part::ROTATION_1D:
      {
        ar & *kp::SmartPtr<kp::PartRotation1D>(p);
        break;
      }
      case kp::Part::MOTION_6D:
      {
        ar & *kp::SmartPtr<kp::PartMotion6D>(p); 
        break;
      }
      default:
        std::cout<<"serialization::kp::Part::Ptr not implemented!"<<std::endl;
        break;
    }
  }

  template <class Archive>
  void load(Archive & ar, ::kp::Part::Ptr& p, const unsigned int version)
  {
    kp::Part::Type type;

    ar & type;

    switch (type)
    {
      case kp::Part::STATIC:
      {
        p.reset(new kp::Part());
        ar & *p;
        break;
      }
      case kp::Part::ROTATION_1D:
      {
        p.reset( new kp::PartRotation1D() );
        ar & *kp::SmartPtr<kp::PartRotation1D>(p);
        break;
      }
      case kp::Part::MOTION_6D:
      {
        p.reset( new kp::PartMotion6D() );
        ar & *kp::SmartPtr<kp::PartMotion6D>(p);
        break;
      }
       default:
        std::cout<<"serialization::kp::Part::Ptr not implemented!"<<std::endl;
        break;
    }

  }

}}


/** ArticulatedObject serialization **/
BOOST_SERIALIZATION_SPLIT_FREE(::kp::ArticulatedObject::Ptr)
namespace boost{namespace serialization{

  template<class Archive>
  void save(Archive & ar, const ::kp::ArticulatedObject::Ptr &o, const unsigned int version)
  {
    ar & boost::serialization::base_object<kp::Object> ( *o );
    ar & boost::serialization::base_object<kp::PartMotion6D> ( *o );
    ar & o->part_parameter;

    int num_parts = o->parts.size();
    ar & num_parts;

    for (unsigned i=1; i<o->parts.size(); i++)  // the first part is the object itself
    {
      ar & o->parts[i];
    }
  }

  template <class Archive>
  void load(Archive & ar, ::kp::ArticulatedObject::Ptr& o, const unsigned int version)
  {
    o.reset(new kp::ArticulatedObject());

    ar & boost::serialization::base_object<kp::Object> ( *o );
    ar & boost::serialization::base_object<kp::PartMotion6D> ( *o );
    ar & o->part_parameter;

    int num_parts;
    ar & num_parts;

    if (num_parts > 0)
    {
      kp::Part::Ptr p;
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
