#include "output/posesWriter.h"

#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>

namespace object_modeller
{
namespace output
{

void PosesWriter::applyConfig(Config &config)
{
    this->outputPath = config.getString(getConfigName(), "outputPath", "./out");
    this->pattern = config.getString(getConfigName(), "pattern", "pose_*.txt");
}

void PosesWriter::process(std::vector<Eigen::Matrix4f> poses)
{
    boost::filesystem::path dir(this->outputPath);
    boost::filesystem::create_directory(dir);

    for(size_t k=0; k < poses.size(); k++)
    {
        std::stringstream filename;
        filename << outputPath << "/";

        int wildcardIndex = pattern.find_first_of("*");

        if (wildcardIndex != -1)
        {
            filename << pattern.substr(0, wildcardIndex);
            filename << setw( 8 ) << setfill( '0' ) << static_cast<int>(k);
            filename << pattern.substr(wildcardIndex + 1);
        }
        else
        {
            int wildcardIndex = pattern.find_last_of(".");

            filename << pattern.substr(0, wildcardIndex);
            filename << setw( 8 ) << setfill( '0' ) << static_cast<int>(k);
            filename << pattern.substr(wildcardIndex);
        }

        std::string scene_name;
        filename >> scene_name;
        std::cout << scene_name << std::endl;
        writeMatrixToFile(scene_name, poses[k]);
    }
}

bool PosesWriter::writeMatrixToFile (std::string file, Eigen::Matrix4f & matrix)
{
    std::ofstream out (file.c_str ());
    if (!out)
    {
        std::cout << "Cannot open file.\n";
        return false;
    }

    for (size_t i = 0; i < 4; i++)
    {
        for (size_t j = 0; j < 4; j++)
        {
            out << matrix (i, j);
            if (!(i == 3 && j == 3))
                out << " ";
        }
    }
    out.close ();

    return true;
}

}
}
