
#include <iostream>
#include <fstream>
#include <boost/filesystem.hpp>
#include <pcl/console/parse.h>

const std::string out_fn = "/tmp/feat+label_libsvm.txt";

bool trainSVM(const std::string &training_data_file, const std::string &training_label_file);

bool trainSVM(const std::string &training_data_file, const std::string &training_label_file)
{
    std::vector<double> labels;

    // fill label/target samples
    std::ifstream file;
    file.open(training_label_file.c_str(), std::ifstream::in);

    std::string line;
    while( std::getline(file, line))
    {
        std::stringstream lineStream(line);
        double label;
        lineStream >> label;
        labels.push_back(label);
    }
    file.close();

    std::ofstream of (out_fn.c_str());
    size_t row = 0;
    // fill training x samples
    file.open(training_data_file.c_str(), std::ifstream::in);
    while( std::getline(file, line))
    {
        std::stringstream lineStream(line);
        of << labels[row++] << " ";
        double feat_elem;
        size_t feat_id = 0;
        while (lineStream >> feat_elem) {
            of << feat_id++ << ":" << feat_elem << " ";
        }
        of << std::endl;
    }
    file.close();

    return true;
}

int main(int argc, char** argv)
{
    std::string training_data_file,
            training_label_file;

    pcl::console::parse_argument (argc, argv, "-training_data_file", training_data_file);
    pcl::console::parse_argument (argc, argv, "-training_label_file", training_label_file);
    trainSVM(training_data_file, training_label_file);
}
