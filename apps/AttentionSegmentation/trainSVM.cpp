/**
 *  Copyright (C) 2012  
 *    Ekaterina Potapova, Andreas Richtsfeld, Johann Prankl, Thomas Mörwald, Michael Zillich
 *    Automation and Control Institute
 *    Vienna University of Technology
 *    Gusshausstraße 25-29
 *    1170 Vienna, Austria
 *    ari(at)acin.tuwien.ac.at
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
 * @file trainSVM.cpp
 * @author Andreas Richtsfeld, Ekaterina Potapova
 * @date January 2014
 * @version 0.1
 * @brief Trains svm based on scaled features.
 */


#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <fstream>

#include <pcl/io/pcd_io.h>

#include <boost/filesystem/fstream.hpp>
#include <boost/filesystem.hpp>

#include "v4r/attention_segmentation/SVMTrainModel.h"

void trainFeatures(double gamma, double C, int kernel, int n_folds, double &RecRate, std::vector<int> &ConfusionTable, 
                   std::string &train_ST_file_name_scaled, std::string &model_file_name)
{
  svm::SVMTrainModel svmTrainModel;
  svmTrainModel.setInputFileName(train_ST_file_name_scaled);
  svmTrainModel.setModelFileName(model_file_name);
  svmTrainModel.setKernelType(kernel);
  svmTrainModel.setSVMType(svm::C_SVC);
  svmTrainModel.setGamma(gamma);
  svmTrainModel.setC(C);
  svmTrainModel.setProbability(1);
  svmTrainModel.setCrossValidation(n_folds);
  svmTrainModel.setNoPrint(true);
  svmTrainModel.train(RecRate,ConfusionTable);
}

void trainSVM(std::string &train_ST_file_name_scaled, std::string &model_file_name)
{
  int kernels[4] = {svm::RBF, svm::LINEAR, svm::POLY, svm::SIGMOID};
  std::vector<std::string> kernel_names;
  kernel_names.resize(4);
  kernel_names.at(0) = "rbf";
  kernel_names.at(1) = "linear";
  kernel_names.at(2) = "polynomial";
  kernel_names.at(3) = "sigmoid";

  printf("Classifier: LibSVM\nParameters (LibSVM): KernelType,C,Gamma,RecRate,ConfusionTable\n--\n");

  double RecRateBest = 0;
  int kernelBest;
  std::vector<int> ConfusionTableBest;
  ConfusionTableBest.resize(4);
  int gammaBest = 0;
  int CBest = 0;

  for(int k = 0; k < 1; ++k)
  {
    for(int C = -5; C <= 15; C = C+2)
    {
      for(int gamma = -15; gamma <= 3; gamma = gamma + 2)
      {
        double RecRate;
        std::vector<int> ConfusionTable;
        ConfusionTable.resize(4);
        trainFeatures(pow(2.0,gamma),pow(2.0,C),kernels[k],2,RecRate,ConfusionTable,train_ST_file_name_scaled,model_file_name);

        printf("%s;%8.6f;%8.6f;%8.6f;%3.1f/%3.1f/%3.1f/%3.1f/\n",kernel_names.at(k).c_str(),pow(2.0,C),pow(2.0,gamma),RecRate,
               (double)(ConfusionTable.at(0)),(double)(ConfusionTable.at(1)),(double)(ConfusionTable.at(2)),(double)(ConfusionTable.at(3)));

        if(RecRate>RecRateBest)
        {
          RecRateBest = RecRate;
          kernelBest = k;
          gammaBest = gamma;
          CBest = C;
          ConfusionTableBest = ConfusionTable;
        }
      }
    }
  }

  printf("--\nGlobalBest (LibSVM):\n");
  printf("%s;%8.6f;%8.6f;%8.6f;%3.1f/%3.1f/%3.1f/%3.1f/\n",kernel_names.at(kernelBest).c_str(),pow(2.0,CBest),pow(2.0,gammaBest),RecRateBest,
         (double)(ConfusionTableBest.at(0)),(double)(ConfusionTableBest.at(1)),(double)(ConfusionTableBest.at(2)),(double)(ConfusionTableBest.at(3)));

  printf("Training final classifier...\n");
  trainFeatures(pow(2.0,gammaBest),pow(2.0,CBest),kernels[kernelBest],0,RecRateBest,ConfusionTableBest,train_ST_file_name_scaled,model_file_name);
  printf("%s;%8.6f;%8.6f;%8.6f;%3.1f/%3.1f/%3.1f/%3.1f/\n",kernel_names.at(kernelBest).c_str(),pow(2.0,CBest),pow(2.0,gammaBest),RecRateBest,
         (double)(ConfusionTableBest.at(0)),(double)(ConfusionTableBest.at(1)),(double)(ConfusionTableBest.at(2)),(double)(ConfusionTableBest.at(3)));

}

void printUsage(char *av)
{
  printf("Usage: %s training_data.txt.scaled model.txt\n"
    " Options:\n"
    "   [-h] ... show this help.\n"
    "   training_data.txt.scaled ... filename with scaled training samples\n"
    "   model.txt                ... output model\n", av);
  std::cout << " Example: " << av << " training_data.txt.scaled model.txt" << std::endl;
}

int main(int argc, char *argv[])
{
  if(argc != 3)
  {
    printUsage(argv[0]);
    exit(0);
  }
  
  std::string train_ST_file_name_scaled = argv[1];
  std::string model_file_name = argv[2];
  
  trainSVM(train_ST_file_name_scaled,model_file_name);
  
  return(0);
}


