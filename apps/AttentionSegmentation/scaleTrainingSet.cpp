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
 * @file scaleTrainingSet.cpp
 * @author Andreas Richtsfeld, Ekaterina Potapova
 * @date January 2014
 * @version 0.1
 * @brief Scales training set.
 */


#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <fstream>

#include <pcl/io/pcd_io.h>

#include <boost/filesystem/fstream.hpp>
#include <boost/filesystem.hpp>

#include "v4r/attention_segmentation/SVMScale.h"

void printUsage(char *av)
{
  printf("Usage: %s training_data.txt lower upper\n"
    " Options:\n"
    "   [-h] ... show this help.\n"
    "   training_data.txt ... filename with training samples\n"
    "   lower             ... lower bound for the scaling\n"
    "   upper             ... uppdr bound for the scaling\n", av);
  std::cout << " Example: " << av << " training_data.txt -1 1" << std::endl;
}

int main(int argc, char *argv[])
{
  if(argc != 4)
  {
    printUsage(argv[0]);
    exit(0);
  }
  
  std::string train_ST_file_name = argv[1];
  std::string train_ST_file_name_scaled = train_ST_file_name + ".scaled";
  std::string scaling_file_name = train_ST_file_name + ".scalingparams";
  
  double lower = atof(argv[2]);
  double upper = atof(argv[3]);
  
  svm::SVMScale svmScale;
  
  svmScale.setLower(lower);
  svmScale.setUpper(upper);
  svmScale.setFeaturesFileName(train_ST_file_name);
  svmScale.setSaveFileName(scaling_file_name);
  svmScale.setFeaturesScaledFileName(train_ST_file_name_scaled);
  svmScale.compute();
  
  //   svmScale.setLower(-1);
  //   svmScale.setUpper(1);
  //   svmScale.setFeaturesFileName("./AS-TrainALL.txt");
  //   svmScale.setSaveFileName("./AS-TrainALL.scalingparams.txt");
  //   svmScale.setFeaturesScaledFileName("./AS-TrainALL.scaled.txt");
  //   svmScale.compute();
  
  return(0);
}


