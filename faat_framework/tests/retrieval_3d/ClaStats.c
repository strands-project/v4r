
/** ClaStats parses a .cla file, verifies that it is correctly formated, and then prints
    out statistics about the classification.

    Usage
    ./CreateClaOverview file.cla

    Copyright 2003 Princeton University

*/
#include <stdio.h>

#include "PSBClaParse.h"





int main(int argc, char *argv[]){
  PSBCategoryList* categories;
  PSBCategory* category;
  int totalClasses, nonEmpty, i, totalModels, largestClass;

  if (argc != 2){
    printf("Usage: ClaStats.exe file.cla\n");
    return 1;

  }
  
  categories = parseFile(argv[1]);

  totalClasses = categories->_numCategories;
  largestClass = totalModels = nonEmpty = 0;

  for(i = 0; i < categories->_numCategories; ++i){
    category = categories->_categories[i];
    if (category->_numModels > 0){
    
      printf("category: %s\n", category->_name);
      nonEmpty++;
      totalModels+= category->_numModels; 
      if (category->_numModels > largestClass){
        largestClass = category->_numModels;
      }
      
      int j;
      for(j=0; j < category->_numModels; j++)
      {
        printf("model: %d\n", category->_models[j]);
      }
    }
  }

  printf("Classification file: %s\n", argv[1]);
  printf("Num classes: %d\n", totalClasses);
  printf("Num non-empty classes: %d\n", nonEmpty);
  printf("Num models: %d\n", totalModels); 
  printf("Average class size (non-empty): %g\n", totalModels/(float)nonEmpty);
  printf("Largest class size: %d\n", largestClass);

  return 0;
}

