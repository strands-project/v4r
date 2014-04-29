
#include "pipeline.h"

using namespace std;

namespace object_modeller
{

void Pipeline::process()
{
    for (size_t i = 0; i < modules.size (); i++)
    {
        std::cout << "------------------------ Processing module " << modules[i]->getModule()->getName() << std::endl;

        modules[i]->getModule()->applyConfig(config);
        modules[i]->process();

        std::cout << "------------------------ Finished module " << modules[i]->getModule()->getName() << std::endl;
    }
}

}
