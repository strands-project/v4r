/******************************************************************************
 * Copyright (c) 2012 Aitor Aldoma
 * Copyright (c) 2016 Thomas Faeulhammer
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 ******************************************************************************/

/**
*
*      @author Aitor Aldoma
*      @author Thomas Faeulhammer (faeulhammer@acin.tuwien.ac.at)
*      @date March, 2012
*      @brief object model database
*/

#ifndef V4R_VIEWS_SOURCE_H_
#define V4R_VIEWS_SOURCE_H_

#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <boost/regex.hpp>
#include <v4r/core/macros.h>
#include <v4r/io/filesystem.h>
#include <v4r/recognition/model.h>

namespace bf = boost::filesystem;

namespace v4r
{

/**
     * \brief Abstract data source class, manages filesystem, incremental training, etc.
     * \author Aitor Aldoma
     */
template<typename PointT>
class V4R_EXPORTS Source
{

protected:
    typedef Model<PointT> ModelT;
    typedef boost::shared_ptr<ModelT> ModelTPtr;

    std::vector<ModelTPtr> models_;
    std::string path_;
    float model_scale_;
    bool load_views_;
    float radius_normals_;
    int resolution_mm_;
    bool compute_normals_;
    bool load_into_memory_;
    std::string view_prefix_;
    std::string indices_prefix_;
    std::string pose_prefix_;
    std::string entropy_prefix_;

    //List of model ids that will be loaded
    std::vector<std::string> model_list_to_load_;

public:
    virtual ~Source() = 0;

    Source(int resolution_mm = 5) {
        resolution_mm_ = resolution_mm;
        load_views_ = true;
        compute_normals_ = false;
        load_into_memory_ = true;
        view_prefix_ = std::string ("cloud_");
        pose_prefix_ = std::string("pose_");
        indices_prefix_ = std::string("object_indices_");
        entropy_prefix_ = std::string("entropy_");
    }

    bool isModelIdInList(const std::string & id)
    {
        if(model_list_to_load_.empty())
            return true;

        for(size_t i=0; i < model_list_to_load_.size(); i++)
        {
            if(id.compare(model_list_to_load_[i]) == 0)
                return true;
        }

        return false;
    }

    void setModelList(const std::vector<std::string> & list)
    {
        model_list_to_load_ = list;
    }

    void
    setLoadIntoMemory(bool b)
    {
        load_into_memory_ = b;
    }

    bool
    getLoadIntoMemory()
    {
        return load_into_memory_;
    }

    virtual void
    loadInMemorySpecificModelAndView(ModelT & model, int view_id)
    {
        (void)model;
        (void)view_id;
        PCL_ERROR("This function is not implemented in this Source class\n");
    }

    virtual void
    loadInMemorySpecificModel(ModelT & model)
    {
        (void)model;
        PCL_ERROR("This function is not implemented in this Source class\n");
    }

    float
    getScale ()
    {
        return model_scale_;
    }

    void
    setRadiusNormals(float r)
    {
        radius_normals_ = r;
        compute_normals_ = true;
    }

    void
    setModelScale (float s)
    {
        model_scale_ = s;
    }

    void
    voxelizeAllModels (int resolution_mm)
    {
        for (size_t i = 0; i < models_.size (); i++)
        {
            models_[i]->getAssembled (resolution_mm);
            if(compute_normals_)
                models_[i]->getNormalsAssembled (resolution_mm);
        }
    }

    /**
       * \brief Generate model representation of the training set, generating views if needed
       */
    virtual void
    generate ()=0;

    /**
       * \brief Get the generated model
       */
    std::vector<ModelTPtr>
    getModels ()
    {
        return models_;
    }

    bool
    getModelById (const std::string & model_id, ModelTPtr & m) const
    {
        for(size_t i=0; i<models_.size(); i++)
        {
            if(models_[i]->id_.compare(model_id)==0)
            {
                m = models_[i];
                return true;
            }
        }
        return false;
    }

    void
    removeDescDirectory (const ModelT &m, const std::string & base_dir, const std::string & descr_name)
    {
        const std::string dir = base_dir + "/" + m.class_ + "/" + m.id_ + "/" + descr_name;

        bf::path desc_dir = dir;
        if (bf::exists (desc_dir))
            bf::remove_all (desc_dir);
    }

    void
    setPath (const std::string & path)
    {
        path_ = path;
    }

    void setLoadViews(bool load)
    {
        load_views_ = load;
    }

    void
    setViewPrefix (const std::string &pre)
    {
        view_prefix_ = pre;
    }

    std::string
    getViewPrefix() const
    {
        return view_prefix_;
    }

    typedef boost::shared_ptr< Source<PointT> > Ptr;
    typedef boost::shared_ptr< Source<PointT> const> ConstPtr;
};
}

#endif /* REC_FRAMEWORK_VIEWS_SOURCE_H_ */
