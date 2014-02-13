#ifndef SVM_SCALE_H
#define SVM_SCALE_H

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <math.h>

namespace svm
{
  
  
/**
 * @brief Class SVMScale: 
 */
class SVMScale
{
 
public:
  
private:

  double lower, upper;
  std::string save_filename;
  std::string features_filename;
  std::string features_scaled_filename;
  bool have_save_filename;
  bool have_features_filename;
  bool have_features_scaled_filename;
  
  std::vector<double> feature_max;
  std::vector<double> feature_min;
  
  void scale(int index, double &value);
  
public:
  SVMScale(); 
  ~SVMScale();
  
  void setLower(double _lower) {lower = _lower;};
  void setUpper(double _upper) {upper = _upper;};
  void setSaveFileName(std::string _save_filename) {save_filename = _save_filename; have_save_filename = true;};
  void setFeaturesFileName(std::string _features_filename) {features_filename = _features_filename; have_features_filename = true;};
  void setFeaturesScaledFileName(std::string _features_scaled_filename) {features_scaled_filename = _features_scaled_filename; have_features_scaled_filename = true;};
  
  void compute();
};

}

#endif //SVM_SCALE_H