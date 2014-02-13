/*
 * common.h
 *
 *  Created on: Aug 7, 2012
 *      Author: aitor
 */

#ifndef OBJECTNESS_COMMON_H_
#define OBJECTNESS_COMMON_H_

struct BBox
{
  int x, y, z; //corner position
  int sx, sy, sz; //extension
  float score;
  int id;
  int angle;
};

inline bool
sortBBoxes (const BBox& d1, const BBox& d2)
{
  return d1.score > d2.score;
}

inline bool
BBoxless (const BBox& d1, const BBox& d2)
{
  return d1.score < d2.score;
}

#endif /* OBJECTNESS_COMMON_H_ */
