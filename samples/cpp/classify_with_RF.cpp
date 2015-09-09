#include <iostream>

#include <boost/filesystem.hpp>
#include <pcl/console/parse.h>

#include <v4r/ml/forest.h>
#include <v4r/ml/node.h>
#include <v4r/io/filesystem.h>

bool trainRF(const std::string &training_dir, v4r::RandomForest::Forest &rf);
bool  testRF(const std::string &test_dir, v4r::RandomForest::Forest &rf);

bool trainRF(const std::string &training_dir, v4r::RandomForest::Forest &rf)
{
    std::vector < std::string > files_intern;
    if( v4r::io::getFilesInDirectory (training_dir, files_intern, "", ".*.data", true) == -1)
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

bool testRF(std::string &test_dir, v4r::RandomForest::Forest &rf)
{
    std::vector < std::string > files_intern;

    if( v4r::io::getFilesInDirectory (test_dir, files_intern, "", ".*.data", true) == - 1)
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

            // Soft classification, returning probabilities for each label
            std::vector<float> labelProbabilities = rf.SoftClassify(featureVector);
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
