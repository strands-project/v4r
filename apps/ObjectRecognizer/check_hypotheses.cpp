// -g /media/Data/datasets/TUW/annotations/ -r /media/Data/datasets/TUW/object_recognition_results_xyz -t /media/Data/datasets/TUW/test_set -m /media/Data/datasets/TUW/models --use_generated_hypotheses -v

#include <v4r/apps/compute_recognition_rate.h>

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
    e.checkIndividualHypotheses();
    return 0;
}
