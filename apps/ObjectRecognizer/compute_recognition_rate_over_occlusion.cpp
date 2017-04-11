// -g /media/Data/datasets/TUW/annotations/ -r /home/thomas/recognition_results_eval/ -m /media/Data/datasets/TUW/models/

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
    float recognition_rate = e.compute_recognition_rate_over_occlusion();
    return 0;
}
