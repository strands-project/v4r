#ifndef MODEL2D_H
#define MODEL2D_H

#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/regex.hpp>
#include <cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <vector>
#include <v4r/utils/filesystem_utils.h>


namespace bf = boost::filesystem;

namespace v4r
{
namespace rec_3d_framework
{
class Model2D
{
public:
    std::string id_;
    std::string class_;
    boost::shared_ptr<cv::Mat>view_;
    std::string view_filename_;

    Model2D()
    {
    }

    bool
    operator== (const Model2D &other) const
    {
        return (id_ == other.id_) && (class_ == other.class_);
    }
};


class Source2D
{
    typedef boost::shared_ptr<Model2D> Model2DTPtr;
protected:
    std::string path_;
    boost::shared_ptr<std::vector<Model2DTPtr> > models_;
    bool load_into_memory_;

public:
    Source2D()
    {
        load_into_memory_ = true;
    }

    boost::shared_ptr<std::vector<Model2DTPtr> >
    getModels () const
    {
        return models_;
    }

    bool
    getModelById (std::string & model_id, Model2DTPtr & m)
    {
        typename std::vector<Model2DTPtr>::iterator it = models_->begin ();
        while (it != models_->end ())
        {
            if (model_id.compare ((*it)->id_) == 0)
            {
                m = *it;
                return true;
            } else
            {
                it++;
            }
        }

        return false;
    }

    void
    setLoadIntoMemory(bool b)
    {
      load_into_memory_ = b;
    }

    boost::shared_ptr<std::vector<Model2DTPtr> >
    getModels (std::string & model_id)
    {
        typename std::vector<Model2DTPtr>::iterator it = models_->begin ();
        while (it != models_->end ())
        {
            if (model_id.compare ((*it)->id_) != 0)
            {
                it = models_->erase (it);
            }
            else
            {
                it++;
            }
        }

        return models_;
    }

    void
    setPath (std::string & path)
    {
        path_ = path;
    }

    void
    loadInMemorySpecificModel(const std::string & dir, Model2D & model)
    {
        std::stringstream pathmodel;
        pathmodel << dir << "/" << model.class_ << "/" << model.id_;
        cv::Mat image = cv::imread(model.view_filename_, CV_LOAD_IMAGE_COLOR);

        std::string directory, filename;
        char sep = '/';
#ifdef _WIN32
        sep = '\\';
#endif

        size_t position = model.view_filename_.rfind(sep);
        if (position != std::string::npos)
        {
            directory = model.view_filename_.substr(0, position);
            filename = model.view_filename_.substr(position+1, model.view_filename_.length()-1);
        }

        *(model.view_) =image;
    }

    void
    loadOrGenerate (const std::string & model_path, Model2D & model)
    {
        model.view_.reset (new cv::Mat() );
        model.view_filename_ = model_path;

        if(load_into_memory_)
        {
            loadInMemorySpecificModel(model_path, model);
        }
    }

    void
    generate ()
    {
        models_.reset (new std::vector<Model2DTPtr>);

        //get models in directory
        std::vector < std::string > folders;
        //std::string ext_v[] = {"jpg", "JPG", "png", "PNG", "bmp", "BMP", "jpeg", "JPEG"};

        v4r::utils::getFoldersInDirectory (path_, "", folders);
        std::cout << "There are " << folders.size() << " folders. " << std::endl;

        for (size_t i = 0; i < folders.size (); i++)
        {
            std::stringstream class_path;
            class_path << path_ << "/" << folders[i];
//            for(size_t ext_id=0; ext_id < sizeof(ext_v)/sizeof(ext_v[0]); ext_id++)
//            {
                std::vector < std::string > filesInRelFolder;
                v4r::utils::getFilesInDirectory (class_path.str(), filesInRelFolder, "", ".*\\.(jpg|JPG|png|PNG|jpeg|JPEG|bmp|BMP)", false);
                std::cout << "There are " <<  filesInRelFolder.size() << " files in folder " << folders[i] << ". " << std::endl;

                for (size_t kk = 0; kk < filesInRelFolder.size (); kk++)
                {
                    Model2DTPtr m(new Model2D());
                    m->class_ = folders[i];
                    m->id_ = filesInRelFolder[kk];

                    std::stringstream model_path;
                    model_path << class_path.str() << "/" << filesInRelFolder[kk];
                    std::string path_model = model_path.str ();
                    std::cout << "Calling loadOrGenerate path_model: " << path_model << ", m_class: " << m->class_ << ", m_id: " << m->id_ << std::endl;
                    loadOrGenerate (path_model, *m);

                    models_->push_back (m);
                }
//            }
        }
    }
};
}
}

#endif // MODEL2D_H
