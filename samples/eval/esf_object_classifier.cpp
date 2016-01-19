#include <boost/algorithm/string.hpp>
#include <v4r/io/filesystem.h>
#include <map>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <sstream>
#include <opencv2/opencv.hpp>

void
printUsage(int argc, char ** argv);

void
printUsage(int argc, char ** argv)
{
    (void)argc;
    std::cout << "Not enough input arguments specified. Usage: " << std::endl
              << argv[0] << " labelled_test_directory ground_truth_directory descriptor(esf/cnn/max)" << std::endl;
}

class Category
{
public:
    std::string id_;
    size_t num_examples_;
    size_t true_positives_;
    size_t false_positives_;
    size_t false_negatives_;
    std::map<std::string, size_t> confusing_classes_; //counts how many times this category is confused by another category

    Category( const std::string &id=std::string() ) {
        id_ = id;
        num_examples_ = 0;
        true_positives_ = 0;
        false_positives_ = 0;
        false_negatives_ = 0;
    }

    Category &operator=(Category &rhs) {
        this->id_ = rhs.id_;
        this->num_examples_ = rhs.num_examples_;
        this->true_positives_ = rhs.true_positives_;

        return *this;
    }

    int operator==(const Category &rhs) const{
        if ( this->id_.compare(rhs.id_) == 0)
            return 1;
        return 0;
    }
    int operator<(const Category &rhs) const {
        if ( this->id_.compare(rhs.id_) < 0 )
            return 1;
        return 0;
    }
};

int
main (int argc, char ** argv)
{
    if (argc < 3)
    {
        printUsage(argc, argv);
    }
    std::string test_dir = argv[1];
    std::string gt_dir = argv[2];
    std::string desc = argv[3];
    const std::string error_out_dir = "/tmp/classification_failures";

    std::vector<std::string> rel_files = v4r::io::getFilesInDirectory(test_dir, ".*.anno_test", false);

    std::map<std::string, Category> classes;

    size_t num_correct = 0;

    for (size_t i=0; i < rel_files.size(); i++) {
        const std::string prediction_file = test_dir + "/" + rel_files[i];
        std::string gt_rel = rel_files[i];
        boost::replace_last(gt_rel, ".anno_test", ".anno");
        const std::string ground_truth_file = gt_dir + "/" + gt_rel;

        std::string prediction, prediction_esf, prediction_cnn;
        double confidence_esf, confidence_cnn;
        std::vector<std::string> words;
        std::stringstream ground_truth;
        std::ifstream f(prediction_file.c_str());
        f >> prediction_esf >> confidence_esf >> prediction_cnn >> confidence_cnn;
        f.close();

        prediction = prediction_esf;

        if(desc.compare("cnn") == 0 || (desc.compare("max") == 0 && confidence_cnn > confidence_esf))
            prediction = prediction_cnn;

        for( std::map<std::string, Category>::iterator ii=classes.begin(); ii!=classes.end(); ++ii) {
            std::cout << (*ii).first << ": " << (*ii).second.id_ << std::endl;
        }

        std::map<std::string, Category>::iterator it;
        it = classes.find(prediction);
        if(it == classes.end()) {
            Category c(prediction);
            classes[prediction] = c;
        }

        f.open(ground_truth_file.c_str());
        std::string line;
        std::getline(f, line);
        boost::trim_right(line);
        split( words, line, boost::is_any_of("\t "));

        if(words.size() < 5) {
            std::cerr << "Ground truth annotation file does not meet required format (%d %d %d %d class_label)" << std::endl;
            continue;
        }

        for(size_t word_id=4; word_id<words.size(); word_id++) {
            if (word_id !=4)
                ground_truth << "_";

            ground_truth << words[word_id];
        }
        f.close();

        it = classes.find(ground_truth.str());
        if(it == classes.end()) {
            Category c(ground_truth.str());
            classes[ground_truth.str()] = c;
        }

        std::cout << prediction << " / " << ground_truth.str() << std::endl;

        if (prediction.compare(ground_truth.str())==0) {
            num_correct++,
            classes[prediction].true_positives_ ++;
        }
        else {
            classes[prediction].false_positives_++;
            classes[ground_truth.str()].false_negatives_++;

            std::map<std::string, size_t> &confusing_classes = classes[prediction].confusing_classes_;
            std::map<std::string, size_t>::iterator it_conf;
            it_conf = confusing_classes.find(ground_truth.str());
            if(it_conf == confusing_classes.end()) {
                confusing_classes[ground_truth.str()] = 1;
            }
            else {
                it_conf->second++;
            }

            const std::string error_out_fn = error_out_dir + "/" + rel_files[i];
            v4r::io::createDirForFileIfNotExist(error_out_fn);
            std::ofstream of (error_out_fn.c_str());
            of << ground_truth.str() << " " << prediction_esf << " " <<
                  confidence_esf << " " << prediction_cnn << " " << confidence_cnn;
            of.close();
        }
    }

    cv::Mat_<size_t> confusion_mat (classes.size(), classes.size(), (size_t) 0);

    std::map<std::string, Category>::iterator it;
    for( it=classes.begin(); it!=classes.end(); ++it) {
        int row_id = std::distance(classes.begin(), it);
        std::cout << row_id << ": " << it->first << std::endl;
        confusion_mat(row_id, row_id) = it->second.true_positives_;


        std::map<std::string, size_t>::const_iterator it_c;
        it_c = it->second.confusing_classes_.begin();
        for(it_c = it->second.confusing_classes_.begin(); it_c != it->second.confusing_classes_.end(); ++it_c) {
            size_t num_misclass = it_c->second;
            std::map<std::string, Category>::iterator it_tmp; // to find right matrix element to edit
            it_tmp = classes.find(it_c->first);
            int col_id = std::distance(classes.begin(), it_tmp);
            confusion_mat(row_id, col_id) += num_misclass;
        }
    }

    std::cout << "confusion_mat = "<< std::endl;
    for(int row_id=0; row_id < confusion_mat.rows; row_id++) {
        for (int col_id=0; col_id < confusion_mat.cols; col_id++) {
            std::cout << std::setw(5) << confusion_mat(row_id,col_id) << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Classification rate: " << (double)num_correct / rel_files.size();
    return 0;
}
