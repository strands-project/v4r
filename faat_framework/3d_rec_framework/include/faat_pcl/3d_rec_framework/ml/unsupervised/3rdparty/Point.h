/*
 * Point.h
 *
 *  Created on: Feb 14, 2013
 *      Author: aitor
 */

#ifndef POINT_H_
#define POINT_H_

//fast-RNN v2.0
//    Copyright (C) 2012  Roberto J. López-Sastre (robertoj.lopez@uah.es)
//                        Daniel Oñoro-Rubio
//                        Víctor Carrasco-Valdelvira
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//   You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include <iostream>

using namespace std;

template <typename T>

class Point
{
 public:
  // The initializer constructor
  //
 Point(T index, double value):_index(index),_value(value) {}

  // The initializer constructor
  //
 Point():_index(0),_value(0) {}

  // The accessor functions
  //
  T get_index() const { return _index; }
  double get_value() const { return _value; }


  // The modifier functions
  //
  void set_index(T index) { _index = index; }
  void set_value(double v) { _value = v; }

  // A point is "less than" another point if the value is less
  //
  bool operator<(const Point& p) const
  {
    return get_value() < p.get_value();
  }

  // Whether the two points are equal
  //
  bool operator==(const Point& p) const
  {
    return (get_value() == p.get_value());
  }

  // Whether a point is greater than another point
  //
  bool operator>(const Point& p) const
  {
    // A point is greater than the other if the coordinate is greater
    //
    return get_value() > p.get_value();
  }

  // Whether a point is less than or equal
  //
  bool operator<=(const Point& p) const
  {
    return get_value() <= p.get_value();
  }

  // Whether a point is greater than or equal
  //
  bool operator>=(const Point& p) const
  {
    // A point is greater-than or equal if it is greater-than the other
    //
    return get_value() >= p.get_value();
  }

 private:

  // The index
  //
  T _index;

  // The value
  //
  double _value;
};

#endif /* POINT_H_ */
