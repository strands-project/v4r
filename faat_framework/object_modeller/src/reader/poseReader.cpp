
#include "reader/poseReader.h"

#include <pcl/io/pcd_io.h>

#include <faat_pcl/utils/filesystem_utils.h>

#include <boost/filesystem.hpp>
#include <boost/algorithm/string/predicate.hpp>

namespace object_modeller
{
namespace reader
{

PoseReader::PoseReader(std::string config_name) : InModule(config_name)
{
    registerParameter("sequencePrefix", "Sequence Prefix", &sequencePrefix, std::string(""));
    registerParameter("pattern", "Pattern", &pattern, std::string(".*cloud_.*.pcd"));
    registerParameter(ParameterBase::FOLDER, "inputPath", "Input Path", &inputPath, std::string("./"));
    registerParameter("step", "Step", &step, 1);

    nrSequences = 1;
}

std::string PoseReader::getSequenceFolder()
{
    boost::filesystem::directory_iterator end_iter;
    nrSequences = 0;

    std::string result = inputPath;

    if (sequencePrefix.length() == 0)
    {
        return inputPath;
    }

    for (boost::filesystem::directory_iterator dir_iter(inputPath); dir_iter != end_iter;++dir_iter)
    {
        boost::filesystem::directory_entry &entry = (*dir_iter);

        if (boost::filesystem::is_directory(entry.status()))
        {
            std::string foldername = entry.path().filename().string();

            std::cout << "folder name " << foldername << std::endl;
            if (boost::starts_with(foldername, sequencePrefix))
            {
                nrSequences++;

                std::string targetFolder = sequencePrefix;
                targetFolder.append(boost::lexical_cast<std::string>(activeSequence));

                std::cout << "target folder " << targetFolder << std::endl;

                if (foldername.compare(targetFolder) == 0)
                {
                    result = entry.path().string();
                    result.append("/");
                }
            }
        }
    }

    return result;
}

std::vector<Eigen::Matrix4f> PoseReader::process()
{
    std::string sequenceFolder = getSequenceFolder();

    std::cout << "target folder " << sequenceFolder << std::endl;

    std::vector<std::string> files;

    bf::path input_path = sequenceFolder;

    faat_pcl::utils::getFilesInDirectory(input_path, files, pattern);

    std::cout << "Load pose files from source dir: " << sequenceFolder << std::endl;

    for (size_t i = 0; i < files.size (); i++)
    {
        std::cout << "Load pose file " << files[i] << std::endl;

        files[i].insert(0, sequenceFolder);
    }

    // sort files
    std::sort (files.begin (), files.end ());

    // load point clouds
    std::vector<Eigen::Matrix4f> poses;

    for (size_t i = 0; i < files.size (); i+=step)
    {
        printf("Load pose file: %s\n", files[i].c_str());

        Eigen::Matrix4f pose = readPose(files[i]);

        poses.push_back(pose);
    }

    return poses;
}

Eigen::Matrix4f PoseReader::readPose(std::string file)
{
    Eigen::Matrix4f pose;

    std::ifstream in;
    in.open (file.c_str (), std::ifstream::in);

    char linebuf[1024];
    in.getline (linebuf, 1024);
    std::string line (linebuf);
    std::vector<std::string> strs_2;
    boost::split (strs_2, line, boost::is_any_of (" "));

    int c = 0;
    for (int i = 0; i < 16; i++, c++)
    {
        pose (c / 4, c % 4) = static_cast<float> (atof (strs_2[i].c_str ()));
    }

    return pose;
}

}
}
