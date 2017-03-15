#include <v4r/apps/compute_recognition_rate.h>

// -m /media/Data/datasets/TUW/models/ -t /media/Data/datasets/TUW/validation_set/ -g /media/Data/datasets/TUW/annotations/ -r /home/thomas/recognition_results_eval/

int
main (int argc, char ** argv)
{
    std::vector<std::string> arguments(argv + 1, argv + argc);
    v4r::apps::RecognitionEvaluator e;
    arguments = e.init(arguments);

    if( !arguments.empty() )
    {
        std::cout << "Unused parameters: " << std::endl;
        for(const std::string &p : arguments)
            std::cout << p << " ";
        std::cout << std::endl;
    }

    size_t tp, fp, fn;
    e.compute_recognition_rate(tp,fp,fn);

    float precision = (float)tp / (tp + fp);
    float recall = (float)tp / (tp + fn);
    float fscore = 2 * precision * recall / (precision + recall);

//            double score = 4*recall + precision - median_time_ms*0.01*0.01;    // we want to get as much hypotheses as possible - precision will be improved with verification

    std::cout << "tp: " << tp << ", fp: " << fp << ", fn: " << fn
              << ", precision: " << precision << ", recall: " << recall << ", fscore: " << fscore << std::endl;


    return 0;
}
