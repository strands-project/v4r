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

#ifndef KP_SCOPE_TIME_HPP
#define KP_SCOPE_TIME_HPP

#include <cmath>
#include <string>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <v4r/core/macros.h>

namespace v4r
{

class V4R_EXPORTS ScopeTime
{
public:
  std::string title;
  bool print;
  boost::posix_time::ptime start;

  inline ScopeTime (const char* _title, bool _print=true) : title(std::string(_title)), print(_print)
  {
    start = boost::posix_time::microsec_clock::local_time ();
  }

  inline ScopeTime () : title(std::string("")), print(true)
  {
    start = boost::posix_time::microsec_clock::local_time ();
  }

  inline ~ScopeTime ()
  {
    double val = this->getTime ();
    if (print) std::cout << title << ": " << val << "ms\n";
  }

  inline double getTime ()
  {
    boost::posix_time::ptime end = boost::posix_time::microsec_clock::local_time ();
    return (static_cast<double> (((end - start).total_milliseconds ())));
  }

  inline void reset ()
  {
    start = boost::posix_time::microsec_clock::local_time ();
  }
};

} //--END--

#endif




