/**
 * @file Relation.h
 * @author Richtsfeld
 * @date November 2011
 * @version 0.1
 * @brief Struct relation
 */

#ifndef SURFACE_RELATION_H
#define SURFACE_RELATION_H

#include <vector>

namespace surface
{

struct Relation
{
  int type;                               ///< Type of relation (structural level = 1 / assembly level = 2)
  int id_0;                               ///< id of first feature
  int id_1;                               ///< id of second feature
  std::vector<double> rel_value;          ///< relation values (feature vector)
  std::vector<double> rel_probability;    ///< probabilities of correct prediction
  int groundTruth;                        ///< 0=false / 1=true
  double prediction;                      ///< 0=false / 1=true TODO not neccessary, because we have rel_probability?
  bool remove;                            ///< delete flag (currently unused)
};

}

#endif
