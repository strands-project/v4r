#ifndef V4R_MUMFORD_MV
#define V4R_MUMFORD_MV

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/octree/octree.h>
#include "sv_ms_presegmenter.h"

namespace v4rOCTopDownSegmenter
{

    template<typename PointT>
    class MVMumfordShahPreSegmenter : public SVMumfordShahPreSegmenter<PointT>
    {
        private:
            using SVMumfordShahPreSegmenter<PointT>::cloud_;
            using SVMumfordShahPreSegmenter<PointT>::surface_normals_;
            using SVMumfordShahPreSegmenter<PointT>::vis_segmentation_;
            using SVMumfordShahPreSegmenter<PointT>::label_colors_;
            using SVMumfordShahPreSegmenter<PointT>::supervoxels_labels_cloud_;
            using SVMumfordShahPreSegmenter<PointT>::supervoxels_rgb_;
            using SVMumfordShahPreSegmenter<PointT>::save_impath_;
            using SVMumfordShahPreSegmenter<PointT>::merge_candidates_;
            using SVMumfordShahPreSegmenter<PointT>::adjacent_;
            using SVMumfordShahPreSegmenter<PointT>::supervoxel_resolution_;
            using SVMumfordShahPreSegmenter<PointT>::sigma_;
            using SVMumfordShahPreSegmenter<PointT>::alpha_;
            using SVMumfordShahPreSegmenter<PointT>::nyu_;

            float boundary_radius_;

            int extractSVBoundaries();

            void fillPixelMoves(std::vector<std::vector<PixelMove> > & pixel_moves,
                                std::vector<int> & indices_to_compute,
                                bool vis = false);

            void refinement(std::vector<boost::shared_ptr<Region<PointT> > > & regions);
            /*{
                std::cout << "Refinement not implemented in MV setting..." << std::endl;
            }*/

            void computePlaneError(MergeCandidate<PointT> & m)
            {
                m.planar_error_ = m.computePointToPlaneError(cloud_);
            }

            void computePlaneError(boost::shared_ptr<Region<PointT> > & r)
            {
                r->planar_error_ = r->computePointToPlaneError(cloud_);
            }

            virtual void computeBSplineError(MergeCandidate<PointT> & m, int order, int Cpx, int Cpy)
            {
                m.bspline_error_ = m.computeBsplineErrorUnorganized(cloud_, order, Cpx, Cpy);
            }

            /*void computeBsplineError(MergeCandidate<PointT> & m, )
            {
                m.planar_error_ = m.computePointToPlaneError(cloud_);
            }

            void computePlaneError(boost::shared_ptr<Region<PointT> > & r)
            {
                r->planar_error_ = r->computePointToPlaneError(cloud_);
            }*/

            void visualizeRegions(std::vector<boost::shared_ptr<Region<PointT> > > & regions, int viewport);

            void visualizeSegmentation(std::vector<boost::shared_ptr<Region<PointT> > > & regions, int viewport);

            virtual void projectRegionsOnImage(std::vector<boost::shared_ptr<Region<PointT> > > & regions,
                                               std::string append)
            {
                std::cout << "projectRegionsOnImage not implemented in MV setting..." << std::endl;
            }

            virtual void prepare(std::vector<int> & label_to_idx,
                                 std::vector<std::vector<int> > & indices);

            boost::shared_ptr<typename pcl::octree::OctreePointCloudSearch<PointT> > octree_;

        public:

            MVMumfordShahPreSegmenter();
            ~MVMumfordShahPreSegmenter();

            void setBoundaryRadius(float f)
            {
                boundary_radius_ = f;
            }
    };
}

#endif
