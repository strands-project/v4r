/******************************************************************************
 * Copyright (c) 2017 Daniel Wolf
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 ******************************************************************************/

#pragma once

#include <array>
#include <vector>

#define LEAF_WEIGHTED
//#define USE_RF_ENERGY
#define COIN_FLIP
#define PI 3.141592654

#ifndef RAD2DEG
#define RAD2DEG(x) ((x) * 57.29577951308232087721)
#endif

#define LOG_ERROR(...)    std::cout << "\033[1;31m" << __VA_ARGS__ << "\033[0m" << std::endl;
#define LOG_INFO(...)     std::cout << "\033[1;33m" << __VA_ARGS__ << "\033[0m" << std::endl;
#define LOG_PLAIN(...)    std::cout << __VA_ARGS__ << std::endl;

typedef std::array<unsigned int, 3> PointIdx;
typedef std::vector<std::array<unsigned int, 3> > PointIndices;
typedef std::vector<std::array<unsigned int, 3> >::iterator PointIdxItr;

typedef std::array<int, 2> ClusterIdx;  // first value is image idx, second value cluster idx
typedef std::vector<ClusterIdx> ClusterIndices;
typedef ClusterIndices::iterator ClusterIdxItr;