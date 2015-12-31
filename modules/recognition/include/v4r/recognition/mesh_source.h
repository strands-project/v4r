/******************************************************************************
 * Copyright (c) 2012 Aitor Aldoma, Thomas Faeulhammer
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


#ifndef V4R_MESH_SOURCE_H_
#define V4R_MESH_SOURCE_H_

#include <pcl/io/pcd_io.h>
#include <boost/function.hpp>

#include <v4r/common/faat_3d_rec_framework_defines.h>
#include <v4r/common/miscellaneous.h>
#include <v4r/io/eigen.h>
#include <v4r/io/filesystem.h>
#include <v4r/rendering/depthmapRenderer.h>
#include <v4r/recognition/source.h>

namespace v4r
{
    /**
     * \brief Data source class based on mesh models
     * \author Aitor Aldoma, Thomas Faeulhammer
     * \date March, 2012
     */
    template<typename PointT>
      class V4R_EXPORTS MeshSource : public Source<PointT>
      {
        typedef Source<PointT> SourceT;
        typedef Model<PointT> ModelT;
        typedef boost::shared_ptr<ModelT> ModelTPtr;

        using SourceT::path_;
        using SourceT::models_;
        using SourceT::model_scale_;
        using SourceT::radius_normals_;
        using SourceT::compute_normals_;
        using SourceT::load_into_memory_;

        int tes_level_;
        int resolution_;
        float radius_sphere_;
        bool gen_organized_;
        boost::function<bool (const Eigen::Vector3f &)> campos_constraints_func_;

        pcl::PointCloud<PointT> V4R_EXPORTS renderCloud (const DepthmapRenderer &renderer, float & visible);

      public:

        MeshSource () :
        SourceT ()
        {
          gen_organized_ = false;
          load_into_memory_ = true;
        }

        ~MeshSource(){}

        void
        setTesselationLevel (int lev)
        {
          tes_level_ = lev;
        }

        void
        setCamPosConstraints (boost::function<bool
        (const Eigen::Vector3f &)> & bb)
        {
          campos_constraints_func_ = bb;
        }

        void
        setResolution (int res)
        {
          resolution_ = res;
        }

        void
        setRadiusSphere (float r)
        {
          radius_sphere_ = r;
        }

        void
        loadOrGenerate (const std::string & model_path, ModelT & model);

        void
        generate ()
        {
            std::vector < std::string > files = v4r::io::getFilesInDirectory(path_, ".*.ply", true);
            models_.clear();

            for (size_t i = 0; i < files.size (); i++)
            {
                ModelTPtr m(new ModelT);
                this->getIdAndClassFromFilename (files[i], m->id_, m->class_);

                //check which of them have been trained using training_dir and the model_id_
                //load views, poses and self-occlusions for those that exist
                //generate otherwise
                std::cout << files[i] << std::endl;
                std::string path_model = path_ + "/" + files[i];
                loadOrGenerate (path_model, *m);

                models_.push_back (m);
            }
            std::cout << "End of generate function" << std::endl;
        }

        void
        loadInMemorySpecificModel(ModelT & model);
      };
}

#endif /* REC_FRAMEWORK_MESH_SOURCE_H_ */
