/*
 * render_from_ply.cpp
 *
 *  Created on: Mar 18, 2013
 *      Author: aitor
 */

#include <pcl/console/parse.h>
#include <faat_pcl/utils/filesystem_utils.h>
#include <pcl/common/common.h>
#include <pcl/io/pcd_io.h>
#include <pcl/apps/render_views_tesselated_sphere.h>
#include <pcl/filters/voxel_grid.h>
#include <faat_pcl/3d_rec_framework/feature_wrapper/global/esf_estimator.h>
#include <vtkPolyData.h>
#include <vtkTriangle.h>
#include <vtkSmartPointer.h>
#include <vtkCellArray.h>
#include <vtkPLYReader.h>
#include <vtkPolyDataMapper.h>
#include <vtkTransform.h>
#include <vtkTransformFilter.h>
#include <vtkTransformPolyDataFilter.h>
#include <faat_pcl/3d_rec_framework/utils/vtk_model_sampling.h>
#include <faat_pcl/utils/filesystem_utils.h>

#include "PSBClaParse.h"

#define CURR_FORMAT 1
#define PSB "PSB"
#define BASE_CATEGORY "0"




static void error(FILE *file, const char *format, ...);
static void createFullName(char *catNameBuffer, const char *className,
                           const char *parentClassName, PSBCategoryList *categoryList,
                           int maxNumCategories);
static PSBCategory *defineCategory(const char *category, const char* parent, int numModels);
static bool isCategoryDefined(PSBCategoryList *categoryList, const char *category,
                              int currentNumCategories);




extern PSBCategoryList *parseFile(const char *catFile){
    FILE *file;
    char firstBuffer[256], secondBuffer[256], catNameBuffer[256];
    int num, numCategories, currentNumCategories, numFound, i;
    int numModels, currNumModels, currNumTotalModels, numTotalModels;
    PSBCategoryList *categoryList;

    file = fopen(catFile, "r");
    if (file == NULL){
        fprintf(stderr, "trouble opening %s\n", catFile);
        exit(1);
    }

    // check header
    if(fscanf(file, "%s %d", firstBuffer, &num) != 2 || strcmp(PSB, firstBuffer) || num != CURR_FORMAT){
        error(file, "%s is not a PSB file of format %d\n", catFile, CURR_FORMAT);
    }

    // get num categories
    if (fscanf(file, "%d %d", &numCategories, &numTotalModels) != 2){
        error(file, "%s does not list the number of categories or models\n", catFile);
    }

    if (numCategories < 0 || numTotalModels < 0){
        error(file, "%d categories and %d models is invalid\n", numCategories, numTotalModels);
    }


    categoryList = (PSBCategoryList*)malloc(sizeof(PSBCategoryList));
    assert(categoryList != NULL);
    categoryList->_numCategories = numCategories;
    categoryList->_categories = (PSBCategory **)malloc(numCategories * sizeof(PSBCategory*));
    assert(categoryList != NULL);

    currentNumCategories = 0;
    currNumTotalModels = 0;

    while(1){
        numFound = fscanf(file, "%s %s %d", firstBuffer, secondBuffer, &numModels);

        if (numFound == 0 || numFound == EOF){
            break; // end of file
        }else if (numFound == 1){
            error(file, "poorly formated category line, %s\n", firstBuffer);
        }else if (numFound == 2){
            error(file, "poorly formated category line, %s %s\n", firstBuffer, secondBuffer);
        }else if (numFound > 3){
            error(file, "poorly formated category line\n");
        }

        if (numModels < 0){
            error(file, "%d is an invalid number of categories\n", numModels);
        }

        if (isCategoryDefined(categoryList, firstBuffer, currentNumCategories)){
            error(file, "%s is already a defined category\n", firstBuffer);
        }

        if (!isCategoryDefined(categoryList, secondBuffer, currentNumCategories)){
            error(file, "%s is not a defined category\n", secondBuffer);
        }


        createFullName(catNameBuffer, firstBuffer, secondBuffer, categoryList, currentNumCategories);

        categoryList->_categories[currentNumCategories] = defineCategory(firstBuffer, catNameBuffer, numModels);

        printf("processing category %s, ", firstBuffer);

        currNumModels = 0;

        for(i = 0; i < numModels; ++i){
            numFound = fscanf(file,"%d", &num);
            if (numFound != 1){
                error(file, "%s has an incorrect number of models, read %d, should be %d\n", firstBuffer, currNumModels, numModels);
            }
            categoryList->_categories[currentNumCategories]->_models[currNumModels] = num;
            currNumModels++;
            currNumTotalModels++;
        }
        printf("finished.\n");
        ++currentNumCategories;

    }

    if (currentNumCategories != numCategories){
        error(file, "expected %d categories, found %d\n", numCategories, currentNumCategories);
    }

    if (currNumTotalModels != numTotalModels){
        error(file, "expected %d models, found %d\n", numTotalModels, currNumTotalModels);
    }


    if (fclose(file) == EOF){
        printf("trouble closing %s\n", catFile);
    }

    printf("Validated %s, %d categories, %d models\n", catFile, numCategories, currNumTotalModels);
    return categoryList;
}


static void error(FILE *file, const char *format, ...){

    char buffer[256];
    va_list ap;

    va_start(ap, format);
    vsprintf(buffer, format, ap);
    va_end(ap);

    fprintf(stderr, buffer);

    fclose(file);

    exit(1);


}


static PSBCategory* defineCategory(const char *categoryName, const char *fullName, int numModels){
    PSBCategory * category;

    category = (PSBCategory*)malloc(sizeof(PSBCategory));
    assert(category!=NULL);

    category->_models = (int*)malloc(numModels * sizeof(int));
    assert(category->_models != NULL);
    memset(category->_models, 0, numModels * sizeof(int));

    category->_name = strdup(categoryName);
    assert(category->_name != NULL);
    category->_fullName = strdup(fullName);
    assert(category->_fullName != NULL);

    category->_numModels = numModels;
    return category;
}

static bool isCategoryDefined(PSBCategoryList *categoryList, const char *category, int currentNumCategories){

    int i;

    if (strcmp(category, BASE_CATEGORY) == 0){
        return true;
    }

    for(i = 0; i < currentNumCategories; ++i){
        if (strcmp(category, categoryList->_categories[i]->_name)==0) return true;
    }

    return false;
}

static void createFullName(char *catNameBuffer, const char *className,
                           const char *parentClassName, PSBCategoryList *categoryList,
                           int maxNumCategories){

    int j;

    if (strcmp(parentClassName, BASE_CATEGORY) == 0){
        strcpy(catNameBuffer, className);
    }else{
        for(j = 0; j < maxNumCategories; ++j){
            if (strcmp(categoryList->_categories[j]->_name, parentClassName)==0){
                sprintf(catNameBuffer, "%s___%s", categoryList->_categories[j]->_fullName, className);
                return;
            }
        }
        assert(0);
    }

}

namespace bf = boost::filesystem;

int
main (int argc, char ** argv)
{
    std::string file, out_dir;
    std::string models_dir;
    std::string ext = ".ply";

    pcl::console::parse_argument (argc, argv, "-out_dir", out_dir);
    pcl::console::parse_argument (argc, argv, "-file", file);
    pcl::console::parse_argument (argc, argv, "-models_dir", models_dir);
    pcl::console::parse_argument (argc, argv, "-ext", ext);


    PSBCategoryList* categories;
    PSBCategory* category;
    int totalClasses, nonEmpty, i, totalModels, largestClass;

    const char * bb = file.c_str();
    categories = parseFile(bb);

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

            //create category file
            std::stringstream category_dir;
            category_dir << out_dir << "/" << category->_name;
            bf::path cat_dir = category_dir.str();
            if(!bf::exists(cat_dir))
            {
                bf::create_directory(cat_dir);
            }

            std::cout << category_dir.str() << std::endl;

            int j;
            for(j=0; j < category->_numModels; j++)
            {
                printf("model: %d\n", category->_models[j]);
                std::stringstream in_file;
                in_file << models_dir << "/m" << category->_models[j] << ext;

                std::stringstream out_file;
                out_file << category_dir.str() << "/m" << category->_models[j] << ext;
                std::cout << out_file.str() << std::endl;

                bf::path p1(in_file.str());
                bf::path p2(out_file.str());
                bf::copy_file(p1,p2);
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

