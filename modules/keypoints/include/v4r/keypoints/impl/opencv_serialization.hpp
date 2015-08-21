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
// file: cvmat_serilization.h

#ifndef KP_OPENCV_SERI_HPP
#define KP_OPENCV_SERI_HPP

#include <opencv2/opencv.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/vector.hpp>

namespace boost {
  namespace serialization {

    /** Serialization support for cv::Mat */
    template<class Archive, typename T>
    void serialize(Archive & ar, ::cv::Mat_<T>& m, const unsigned int version)
    {
      if(Archive::is_loading::value == true)
      {
        int cols, rows;
        size_t elem_size, elem_type;

        ar & cols;
        ar & rows;
        ar & elem_size;
        ar & elem_type;

        m.create(rows, cols);

        size_t data_size = m.cols * m.rows * elem_size;
        ar & boost::serialization::make_array(m.ptr(), data_size);
      }
      else
      {
        size_t elem_size = m.elemSize();
        size_t elem_type = m.type();

        ar & m.cols;
        ar & m.rows;
        ar & elem_size;
        ar & elem_type;

        const size_t data_size = m.cols * m.rows * elem_size;
        ar & boost::serialization::make_array(m.ptr(), data_size);
      }
    }
  }
}


BOOST_SERIALIZATION_SPLIT_FREE(::cv::Mat)
namespace boost {
  namespace serialization {

    /** Serialization support for cv::Mat */
    template<class Archive>
    void save(Archive & ar, const ::cv::Mat& m, const unsigned int version)
    {
      size_t elem_size = m.elemSize();
      size_t elem_type = m.type();

      ar & m.cols;
      ar & m.rows;
      ar & elem_size;
      ar & elem_type;

      const size_t data_size = m.cols * m.rows * elem_size;
      ar & boost::serialization::make_array(m.ptr(), data_size);
    }

    /** Serialization support for cv::Mat */
    template <class Archive>
    void load(Archive & ar, ::cv::Mat& m, const unsigned int version)
    {
      int cols, rows;
      size_t elem_size, elem_type;

      ar & cols;
      ar & rows;
      ar & elem_size;
      ar & elem_type;

      m.create(rows, cols, elem_type);

      size_t data_size = m.cols * m.rows * elem_size;
      ar & boost::serialization::make_array(m.ptr(), data_size);
    }

  }
}

BOOST_SERIALIZATION_SPLIT_FREE(cv::KeyPoint)
namespace boost {
    namespace serialization {
        /** Serialization support for cv::KeyPoint */
        template<class Archive>
        void save(Archive &ar, const cv::KeyPoint &p, const unsigned int __attribute__((unused)) version)
        {
            ar & p.pt.x;
            ar & p.pt.y;
            ar & p.size;
            ar & p.angle;
            ar & p.response;
            ar & p.octave;
            ar & p.class_id;
        }

        /** Serialization support for cv::KeyPoint */
        template<class Archive>
        void load(Archive &ar, cv::KeyPoint &p, const unsigned int __attribute__((unused)) version)
        {
            ar & p.pt.x;
            ar & p.pt.y;
            ar & p.size;
            ar & p.angle;
            ar & p.response;
            ar & p.octave;
            ar & p.class_id;
        }
    }
}

namespace boost {
    namespace serialization {

      template<class Archive>
        void serialize(Archive & ar, cv::Point2f &pt, const unsigned int version)
        {
          ar & pt.x;
          ar & pt.y;
        }

    }
}

#endif
