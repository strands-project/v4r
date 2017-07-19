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

    e.setOut_dir( e.getOr_dir() );
    std::cout << e.compute_confusion_matrix() << std::endl;;

    return 0;
}
