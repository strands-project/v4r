/******************************************************************************
 * Copyright (c) 2012 Aitor Aldoma
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
    template<typename PointInT>
    class V4R_EXPORTS Source
    {

    protected:
      typedef Model<PointInT> ModelT;
      typedef boost::shared_ptr<ModelT> ModelTPtr;

      std::vector<ModelTPtr> models_;
      std::string path_;
      float model_scale_;
      bool load_views_;
      float radius_normals_;
      float resolution_;
      bool compute_normals_;
      bool load_into_memory_;

      //List of model ids that will be loaded
      std::vector<std::string> model_list_to_load_;

      void
      getIdAndClassFromFilename (const std::string & filename, std::string & id, std::string & classname)
      {
        std::vector < std::string > strs;
        boost::split (strs, filename, boost::is_any_of ("/\\"));
        std::string name = strs[strs.size () - 1];

        classname = strs[0];
        id = name.substr (0, name.length () - 4);
      }

      void
      createClassAndModelDirectories (const std::string & training_dir, const std::string & class_str, const std::string & id_str)
      {
        std::vector < std::string > strs;
        boost::split (strs, class_str, boost::is_any_of ("/\\"));

        std::stringstream ss;
        ss << training_dir << "/";
        for (size_t i = 0; i < strs.size (); i++)
        {
          ss << strs[i] << "/";
          io::createDirIfNotExist(ss.str ());
        }

        ss << id_str;
        io::createDirIfNotExist(ss.str ());
      }

    public:

      Source(float resolution = 0.001f) {
        resolution_ = resolution;
        load_views_ = true;
        compute_normals_ = false;
        load_into_memory_ = true;
      }

      bool isModelIdInList(const std::string & id)
      {
          if(model_list_to_load_.empty())
              return true;

          for(size_t i=0; i < model_list_to_load_.size(); i++)
          {
              if(id.compare(model_list_to_load_[i]) == 0)
              {
                  return true;
              }
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
      loadInMemorySpecificModelAndView(const std::string & dir, ModelT & model, int view_id)
      {
        (void)dir;
        (void)model;
        (void)view_id;
        PCL_ERROR("This function is not implemented in this Source class\n");
      }

      virtual void
      loadInMemorySpecificModel(const std::string & dir, ModelT & model)
      {
        (void)dir;
        (void)model;
        PCL_ERROR("This function is not implemented in this Source class\n");
      }

      float
      getScale ()
      {
        return model_scale_;
      }

      void
      setRadiusNormals(float r) {
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
       * \brief Generate model representation
       */
      virtual void
      generate (const std::string & training_dir = std::string())=0;

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

      bool
      isModelAlreadyTrained (const ModelT m, const std::string & base_dir, const std::string & descr_name) const
      {
        std::stringstream dir;
        dir << base_dir << "/" << m.class_ << "/" << m.id_ << "/" << descr_name;
        bf::path desc_dir = dir.str ();
        if (bf::exists (desc_dir))
        {
          std::cout << dir.str () << " exists..." << std::endl;
          return true;
        }

        std::cout << dir.str () << " does not exist..." << std::endl;
        return false;
      }

      std::string
      getModelDescriptorDir (const ModelT m, const std::string & base_dir, const std::string & descr_name)
      {
        std::stringstream dir;
        dir << base_dir << "/" << m.class_ << "/" << m.id_ << "/" << descr_name;
        return dir.str ();
      }

      std::string
      getModelDirectory (const ModelT m, const std::string & base_dir)
      {
        std::stringstream dir;
        dir << base_dir << "/" << m.class_ << "/" << m.id_;
        return dir.str ();
      }

      std::string
      getModelClassDirectory (const ModelT m, const std::string & base_dir)
      {
        std::stringstream dir;
        dir << base_dir << "/" << m.class_;
        return dir.str ();
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
      createVoxelGridAndDistanceTransform(float resolution)
      {
        for (size_t i = 0; i < models_.size (); i++)
          models_[i]->createVoxelGridAndDistanceTransform (resolution);
      }
    };
}

#endif /* REC_FRAMEWORK_VIEWS_SOURCE_H_ */
