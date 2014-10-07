
#include "inputModule.h"

#include <vector>
#include <string>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace object_modeller
{
namespace reader
{

template<class TPointType>
class FileReader : public InModule<std::vector<typename pcl::PointCloud<TPointType>::Ptr> >
{
private:
    std::string sequencePrefix;
    std::string pattern;
    std::string inputPath;
    int step;
    int nrSequences;

public:
    FileReader(std::string config_name="reader") : InModule<std::vector<typename pcl::PointCloud<TPointType>::Ptr> >(config_name)
    {
        ConfigItem::registerParameter("sequencePrefix", "Sequence Prefix", &sequencePrefix, std::string(""));
        ConfigItem::registerParameter("pattern", "Pattern", &pattern, std::string(".*cloud_.*.pcd"));
        ConfigItem::registerParameter(ParameterBase::FOLDER, "inputPath", "Input Path", &inputPath, std::string("./"));
        ConfigItem::registerParameter("step", "Step", &step, 1);
    }

    virtual std::vector<typename pcl::PointCloud<TPointType>::Ptr> process();

    virtual void applyConfig(Config::Ptr config);

    std::string getSequenceFolder()
    {
        boost::filesystem::directory_iterator end_iter;
        nrSequences = 0;

        std::string result = inputPath;

        if (sequencePrefix.length() == 0)
        {
            nrSequences = 1;
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
                    targetFolder.append(boost::lexical_cast<std::string>(Module::activeSequence));

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

    virtual int getNrOutputSequences()
    {
        return nrSequences;
    }

    std::string getName()
    {
        return "PCD File Reader";
    }
};

}
}
