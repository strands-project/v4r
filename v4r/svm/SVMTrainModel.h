#ifndef SVM_TRAIN_MODEL_H
#define SVM_TRAIN_MODEL_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <string>
#include <vector>
#include <cstring>
#include "svm.h"

namespace svm
{
  
  
/**
 * @brief Class SVMTrainModel: 
 */
class SVMTrainModel
{
 
public:
  
private:

  struct svm_parameter param;  // set by parse_command_line
  struct svm_problem prob;     // set by read_problem
  struct svm_model *model;
  struct svm_node *x_space;
  int cross_validation;
  int nr_fold;

  char input_file_name[1024];
  char model_file_name[1024];

  bool have_input_file_name;
  bool have_model_file_name;
  
public:
  SVMTrainModel(); 
  ~SVMTrainModel() {};
  
  void setSVMType(int _svm_type);
  void setKernelType(int _kernel_type);
  void setDegree(int _degree);
  void setGamma(double _gamma);
  void setCoef0(double _coef0);
  void setNu(double _nu);
  void setCacheSize(double _cache_size);
  void setC(double _C);
  void setEps(double _eps);
  void setP(double _p);
  void setShrinking(int _shrinking);
  void setProbability(int _probability);
  void setCrossValidation(int _nr_fold);
  void setWeight(int _i, float _weight);
  void setInputFileName(std::string _input_file_name);
  void setModelFileName(std::string _input_file_name);
  void setNoPrint(bool _no_print);
  int train(double &RecRate, std::vector<int> &ConfusionTable);

private:
  void readProblem(const char *filename);
  void do_cross_validation(double &RecRate, std::vector<int> &ConfusionTable);
  
};

}

#endif //SVM_TRAIN_MODEL_H