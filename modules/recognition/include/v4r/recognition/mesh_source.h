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
#include <boost/filesystem/convenience.hpp>

#include <v4r/common/faat_3d_rec_framework_defines.h>
#include <v4r/common/miscellaneous.h>
#include <v4r/io/eigen.h>
#include <v4r/io/filesystem.h>
#include <v4r/recognition/source.h>
#include <v4r/rendering/depthmapRenderer.h>

namespace bf = boost::filesystem;

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
      private:
        typedef Source<PointT> SourceT;
        typedef Model<PointT> ModelT;
        typedef boost::shared_ptr<ModelT> ModelTPtr;

        using SourceT::path_;
        using SourceT::models_;
        using SourceT::model_scale_;
        using SourceT::radius_normals_;
        using SourceT::compute_normals_;
        using SourceT::load_into_memory_;
        using SourceT::view_prefix_;
        using SourceT::indices_prefix_;
        using SourceT::pose_prefix_;
        using SourceT::entropy_prefix_;

        int tes_level_;
        int resolution_;
        float radius_sphere_;
        bool gen_organized_;

        std::string mesh_dir_;

        boost::shared_ptr<DepthmapRenderer> renderer_;

        pcl::PointCloud<PointT> V4R_EXPORTS renderCloud (const DepthmapRenderer &renderer_, float & visible);

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
        setMeshDir(const std::string &dir)
        {
            mesh_dir_ = dir;
        }

        void
        loadOrGenerate (const std::string & model_path, ModelT & model);

        void
        getIdAndClassFromFilename (const std::string & filename, std::string & id, std::string & classname) const
        {
          std::vector < std::string > strs;
          boost::split (strs, filename, boost::is_any_of ("/\\"));
          std::string name = strs[strs.size () - 1];

          classname = strs[0];
          id = name.substr (0, name.length () - 4);
        }

        void
        loadInMemorySpecificModel(ModelT & model);

        void
        generate();


        typedef boost::shared_ptr< MeshSource<PointT> > Ptr;
        typedef boost::shared_ptr< MeshSource<PointT> const> ConstPtr;
      };
}

#endif /* REC_FRAMEWORK_MESH_SOURCE_H_ */
