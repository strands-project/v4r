/**
 *  Copyright (C) 2012
 *    Michael Zillich
 *    Automation and Control Institute
 *    Vienna University of Technology
 *    Gusshausstraße 27-29/E376
 *    1040 Vienn, Austria
 *    zillich(at)acin.tuwien.ac.at
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

/**
 * @file math.h
 * @author Michael Zillich
 * @date March 2013
 * @version 1.0
 * @brief Various small math utility functions
 */

#ifndef V4R_UTILS_MATH_H
#define V4R_UTILS_MATH_H


namespace V4R {

/**
 * similar to isnan() or isinf(), check a float for being zero
 * NOTE: We do not use a template here, as fpclassify() is only defined for
 * float and double.
 */
inline bool iszero(float f)
{
  return std::fpclassify(f) == FP_ZERO;
}

/**
 * similar to isnan() or isinf(), check a double for being zero
 * NOTE: We do not use a template here, as fpclassify() is only defined for
 * float and double.
 */
inline bool iszero(double f)
{
  return std::fpclassify(f) == FP_ZERO;
}

}

#endif

