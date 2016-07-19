#include <iostream>

#include <boost/filesystem.hpp>
#include <pcl/console/parse.h>

#include <v4r/ml/forest.h>
#include <v4r/ml/node.h>
#include <v4r/io/filesystem.h>


/*
 * Directory structure of training data:

mytrainingdirectory
 - 0001.data
 - 0002.data
 - 0003.data
   ...
 - 1242.data
 - categories.txt

-> for every label (the example above has 1242 labels) there is a corresponding data file containing all feature vectors of this label.
   such a file looks like this:

 2.46917e-05  0.000273396  0.000204452     0.823049     0.170685     0.988113     0.993125
 3.20674e-05  0.000280648  0.000229576     0.829844     0.207543     0.987969     0.992765
 3.73145e-05  0.000279801  0.000257583     0.831597     0.235013     0.987825       0.9925
 ...........  ...........  ...........  ...........  ...........  ...........  ...........

Each row: One feature vector (here 7 dimensional)

IMPORTANT: Values separated by spaces, vectors separated by newline.
           COLUMNS MUST ALL HAVE THE SAME CONSTANT WIDTH!!! CODE IS ASSUMING THAT FOR SPEED UP!
           (here width is 12)

The file categories.txt is necessary to define the label names and has a simple format like this:
1	floor
2	wall
3	ceiling
4	table
5	chair
6	furniture
7	object
*/


bool trainRF(const std::string &training_dir, v4r::RandomForest::Forest &rf);
bool  testRF(const std::string &test_dir, v4r::RandomForest::Forest &rf);

bool trainRF(const std::string &training_dir, v4r::RandomForest::Forest &rf)
{
    std::vector < std::string > files_intern = v4r::io::getFilesInDirectory (training_dir, ".*.data", true);
    if(files_intern.empty() )
    {
        std::cerr << "Folder " << training_dir << " does not exist. " << std::endl;
        return false;
    }

    if (files_intern.size()==0)
    {
        std::cerr << "Could not find any data files in folder " << training_dir << std::endl;
        return false;
    }

    // define labels to be trained
    std::vector<int> labels(files_intern.size());
    for(size_t i=0; i < files_intern.size(); i++)
    {
        labels[i] = i+1;
    }

    // load training data from files
    v4r::RandomForest::ClassificationData trainingData;
    trainingData.LoadFromDirectory(training_dir, labels);

    // train forest
    //   parameters:
    //   ClassificationData data
    //   bool refineWithAllDataAfterwards
    //   int verbosityLevel (0 - quiet, 3 - most detail)
    rf.TrainLarge(trainingData, false, 3);

    // save after training
    rf.SaveToFile("myforest");

    return true;
}

bool testRF(const std::string &test_dir, v4r::RandomForest::Forest &rf)
{
    std::vector < std::string > files_intern = v4r::io::getFilesInDirectory (test_dir, ".*.data", true);
    if( files_intern.empty() )
    {
        std::cerr << "Folder " << test_dir << " does not exist. " << std::endl;
        return false;
    }
    if (files_intern.size()==0)
    {
        std::cerr << "Could not find any data files in folder " << test_dir << std::endl;
        return false;
    }

    for(size_t i=0; i < files_intern.size(); i++)
    {
        std::stringstream filename;
        filename << test_dir << "/" << files_intern[i];
        std::ifstream test_file;
        test_file.open(filename.str().c_str(), std::ifstream::in);

        std::string line;
        while( std::getline(test_file, line))
        {
            std::cout << line << std::endl;
            std::vector<float> featureVector;

            std::stringstream lineStream(line);
            float num;
            while (lineStream >> num) {
                featureVector.push_back(num);
            }

            // Hard classification, returning label ID:
            // assuming featureVector contains values...
            int ID = rf.ClassifyPoint(featureVector);

            std::cout << "Hard classification result: " << ID << std::endl << std::endl;

            // Soft classification, returning probabilities for each label
            std::vector<float> labelProbabilities = rf.SoftClassify(featureVector);

            std::cout << "Soft classification results: " << std::endl;
            for(size_t l=0; l<labelProbabilities.size(); l++)
                std::cout << l << ": " << labelProbabilities[l] << std::endl;

            std::cout << std::endl;
        }
    }
    return true;
}

int main(int argc, char** argv)
{
    std::string training_dir, test_dir;
    pcl::console::parse_argument (argc, argv, "-training_dir", training_dir);
    pcl::console::parse_argument (argc, argv, "-test", test_dir);


    // define Random Forest
    //   parameters:
    //   int nTrees
    //   int maxDepth
    //   float baggingRatio
    //   int nFeaturesToTryAtEachNode
    //   float minInformationGain
    //   int nMinNumberOfPointsToSplit
    v4r::RandomForest::Forest rf(2, -1 , 0.5, 200, 0.02, 5);

    trainRF(training_dir, rf);
    testRF(test_dir, rf);
}
