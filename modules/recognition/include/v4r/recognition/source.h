/******************************************************************************
 * Copyright (c) 2012 Aitor Aldoma
 * Copyright (c) 2017 Thomas Faeulhammer
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

#pragma once

#include <v4r/core/macros.h>
#include <v4r/recognition/model.h>

namespace bf = boost::filesystem;

namespace v4r
{

/**
* \brief Abstract data source class, manages filesystem, incremental training, etc.
* \author Aitor Aldoma, Thomas Faeulhammer
*/
template<typename PointT>
class V4R_EXPORTS Source
{

protected:
    typedef Model<PointT> ModelT;

    std::vector< typename Model<PointT>::ConstPtr > models_;    ///< all models
    std::string path_;
    float model_scale_;
    bool load_views_;
    float radius_normals_;
    bool compute_normals_;
    bool load_into_memory_;
    std::string view_prefix_;
    std::string pose_prefix_;
    std::string indices_prefix_;
    std::string entropy_prefix_;

public:
    Source() :
        model_scale_ ( 1.f ),
        load_views_(true),
        compute_normals_(false),
        load_into_memory_(true),
        view_prefix_("cloud_"),
        pose_prefix_ ("pose_"),
        indices_prefix_ ("object_indices_"),
        entropy_prefix_ ("entropy_")
    { }

    /**
     * @brief Source
     * @param model_database_path path to object model database. This class assumes that each object is stored in a seperate folder. Each of these folders has a folder "/views" with training views in it.
     * Each training view has a pointcloud which filename begins with the string in variable view_prefix,
     * object indices that indicate the object which filename begins with the string in variable indices_prefix,
     * a 4x4 homogenous camera pose that aligns the training views into a common coordinate system when multiplied with
     * each other which filename begins with the string in variable pose_prefix
     * @param has_categories if true, reads a model database used for classification, i.e. there is another top-level folders for each category and
     * inside each category folder there is the same structure as for instance recognition
     */
    Source(const std::string &model_database_path, bool has_categories = false);

    void
    setLoadIntoMemory(bool b)
    {
        load_into_memory_ = b;
    }

    virtual void
    loadInMemorySpecificModel(ModelT & model)
    {
        (void)model;
        PCL_ERROR("This function is not implemented in this Source class\n");
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

    /**
    * \brief Get the generated model
    * \return returns all generated models
    */
    std::vector<typename Model<PointT>::ConstPtr>
    getModels () const
    {
        return models_;
    }

    /**
     * @brief getModelById
     * @param class_id unique identifier of the model category
     * @param model_id unique identifier of the model instance
     * @param m pointer to the object model
     * @return true if model was fount
     */
    typename Model<PointT>::ConstPtr
    getModelById (const std::string & class_id, const std::string & instance_id, bool &found) const
    {
        found = false;
        for(size_t i=0; i<models_.size(); i++)
        {
            if( models_[i]->id_.compare(instance_id) == 0 && models_[i]->class_.compare(class_id) == 0 )
            {
                found = true;
                return models_[i];
            }
        }
        std::cerr << "Model with class: " << class_id << " and instance: " << instance_id << " not found" << std::endl;
        typename Model<PointT>::ConstPtr foo;
        return foo;
    }

    /**
     * @brief addModel add a model to the database
     * @param m model
     */
    void
    addModel(const typename Model<PointT>::ConstPtr &m)
    {
        models_.push_back(m);
    }

    void
    setLoadViews(bool load)
    {
        load_views_ = load;
    }

    typedef boost::shared_ptr< Source<PointT> > Ptr;
    typedef boost::shared_ptr< Source<PointT> const> ConstPtr;
};
}
