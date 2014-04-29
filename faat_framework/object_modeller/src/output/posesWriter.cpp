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
    this->outputPath = config.getString("posesWriter.outputPath", "./out/");
}

void PosesWriter::process(std::vector<Eigen::Matrix4f> poses)
{
    boost::filesystem::path dir(this->outputPath);
    boost::filesystem::create_directory(dir);

    for(size_t k=0; k < poses.size(); k++)
    {
        std::stringstream temp;
        temp << outputPath << "/pose_";
        temp << setw( 8 ) << setfill( '0' ) << static_cast<int>(k) << ".txt";
        std::string scene_name;
        temp >> scene_name;
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
