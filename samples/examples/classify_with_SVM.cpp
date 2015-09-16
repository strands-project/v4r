
#include <iostream>
#include <fstream>
#include <boost/filesystem.hpp>
#include <pcl/console/parse.h>

#include <v4r/ml/svmWrapper.h>

bool trainSVM(const std::string &training_data_file, const std::string &training_label_file, v4r::svmWrapper &svm, size_t &max_label, const std::string &svm_save_path = std::string());
bool testSVM(const std::string &test_data_file, v4r::svmWrapper &svm, size_t max_labels);

bool trainSVM(const std::string &training_data_file, const std::string &training_label_file, v4r::svmWrapper &svm, size_t &max_label, const std::string &svm_save_path)
{
    std::vector<std::vector<double> > data_train;
    std::vector<double> target_train;

    // fill label/target samples
    max_label=0;
    std::ifstream file;
    file.open(training_label_file.c_str(), std::ifstream::in);

    std::string line;
    while( std::getline(file, line))
    {
        std::stringstream lineStream(line);
        double label;
        lineStream >> label;
        target_train.push_back(label);
        if (label > max_label)
            max_label = label;
    }
    file.close();


    // fill training x samples
    file.open(training_data_file.c_str(), std::ifstream::in);
    while( std::getline(file, line))
    {
        std::vector<double> featureVector;

        std::stringstream lineStream(line);
        double feat_elem;
        while (lineStream >> feat_elem) {
            featureVector.push_back(feat_elem);
        }
        data_train.push_back(featureVector);
    }
    file.close();

    svm.initSVM();
    svm.svm_para_.gamma = 1.0f/data_train[0].size();
    svm.svm_para_.C = 1;
    svm.setNumClasses(max_label+1);
    svm.shuffleTrainingData(data_train, target_train);
//    svm.dokFoldCrossValidation(data_train, target_train, 5, exp2(-3), exp2(3), 2, exp2(-10), exp2(4), 4);
//    svm.dokFoldCrossValidation(data_train, target_train, 5, svm.svm_para_.C / 2, svm.svm_para_.C * 2, 1.2, svm.svm_para_.gamma / 2, svm.svm_para_.gamma * 2, 1.2);
    svm.sortTrainingData(data_train, target_train);
    svm.computeSvmModel(data_train, target_train, svm_save_path);

    return true;
}

bool testSVM(const std::string &test_data_file, v4r::svmWrapper &svm, size_t max_labels)
{
    std::vector<std::vector<double> > data_test;
    std::vector<double> target_predict;

    std::ifstream file;

    // fill test x samples
    file.open(test_data_file.c_str(), std::ifstream::in);
    std::string line;
    while( std::getline(file, line))
    {
        std::vector<double> featureVector;

        std::stringstream lineStream(line);
        double feat_elem;
        while (lineStream >> feat_elem) {
            featureVector.push_back(feat_elem);
        }
        data_test.push_back(featureVector);


        ::svm_node *svm_n_test = new ::svm_node[featureVector.size()+1];
        for(size_t feat_attr_id=0; feat_attr_id < featureVector.size(); feat_attr_id++)
        {
            svm_n_test[feat_attr_id].value = featureVector[feat_attr_id];
            svm_n_test[feat_attr_id].index = feat_attr_id+1;
        }
        svm_n_test[featureVector.size()].index = -1;
        double prob[max_labels];
        target_predict.push_back (::svm_predict_probability(svm.svm_mod_, svm_n_test, prob));
    }
    file.close();

    std::string directory;
#ifdef _WIN32
    const size_t last_slash_idx = test_data_file.rfind('\\');
#else
    const size_t last_slash_idx = test_data_file.rfind('/');
#endif

    if (std::string::npos != last_slash_idx)
    {
        directory = test_data_file.substr(0, last_slash_idx);
    }

    std::stringstream test_prediction_out_file;

#ifdef _WIN32
    test_prediction_out_file << directory << "\test_predicted_label.txt";
#else
    test_prediction_out_file << directory << "/test_predicted_label.txt";
#endif

    std::ofstream out_file;
    out_file.open(test_prediction_out_file.str().c_str());

    for(size_t sample_id=0; sample_id < target_predict.size(); sample_id++)
    {
        out_file << target_predict[ sample_id ] << std::endl;
    }
    out_file.close();

    return true;
}

int main(int argc, char** argv)
{

    v4r::svmWrapper svm;
    std::string training_data_file,
            training_label_file,
            test_data_file,
            svm_path = "/tmp/trained_libsvm.model";

    pcl::console::parse_argument (argc, argv, "-training_data_file", training_data_file);
    pcl::console::parse_argument (argc, argv, "-training_label_file", training_label_file);
    pcl::console::parse_argument (argc, argv, "-test_data_file", test_data_file);
    pcl::console::parse_argument (argc, argv, "-save_trainded_svm_to", svm_path);

    size_t max_label; // num classes
    trainSVM(training_data_file, training_label_file, svm, max_label, svm_path);
    testSVM(test_data_file, svm, max_label);
}
