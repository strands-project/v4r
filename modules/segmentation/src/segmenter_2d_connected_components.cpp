#include <v4r/segmentation/segmenter_2d_connected_components.h>
#include <pcl/impl/instantiate.hpp>
#include <glog/logging.h>

namespace v4r
{

template <typename PointT>
void
ConnectedComponentsSegmenter<PointT>::segment()
{
    CHECK(scene_->isOrganized()) << "Scene must be organized to use Connected Components 2D segmentation!";

    clusters_.clear();

    //create new binary pointcloud with intensity values (0 and 1), 0 for non-object pixels and 1 otherwise
    pcl::PointCloud<pcl::PointXYZI>::Ptr binary_cloud (new pcl::PointCloud<pcl::PointXYZI>);

    {
        binary_cloud->width = scene_->width;
        binary_cloud->height = scene_->height;
        binary_cloud->points.resize ( scene_->points.size ());
        binary_cloud->is_dense = scene_->is_dense;

        size_t idx;
        for (size_t i = 0; i <  scene_->points.size (); ++i)
        {
            idx = i;
            binary_cloud->points[idx].getVector4fMap () = scene_->points[idx].getVector4fMap ();
            binary_cloud->points[idx].intensity = 1.f;
        }
    }

    //connected components on the binary image
    std::map<float, float> connected_labels;
    float c_intensity = 0.1f;
    float intensity_incr = 0.1f;

    for (int u = 0; u < int (binary_cloud->width); u++)
    {
        for (int v = 0; v < int (binary_cloud->height); v++)
        {
            if ( binary_cloud->at (u, v).intensity != 0.f )
            {
                //check neighboring pixels, first left and then top
                //be aware of margins
                if ((u - 1) < 0 && (v - 1) < 0) //top-left pixel
                    (*binary_cloud) (u, v).intensity = c_intensity;
                else
                {
                    if ((v - 1) < 0) //top-row, check on the left of pixel to assign a new label or not
                    {
                        int left = check ((*binary_cloud) (u - 1, v), (*binary_cloud) (u, v) );
                        if (left)
                        {
                            //Nothing found on the left, check bigger window
                            bool found = false;
                            for (int kk = 2; kk < param_.wsize_ && !found; kk++)
                            {
                                if ((u - kk) < 0)
                                    continue;

                                bool left = check ( (*binary_cloud) (u - kk, v), (*binary_cloud) (u, v) );
                                if ( !left )
                                    found = true;
                            }

                            if (!found)
                            {
                                c_intensity += intensity_incr;
                                (*binary_cloud) (u, v).intensity = c_intensity;
                            }
                        }
                    }
                    else
                    {
                        if ( (u - 1) == 0 )
                        {
                            //check only top
                            bool top = check ((*binary_cloud) (u, v - 1), (*binary_cloud) (u, v) );
                            if (top)
                            {
                                bool found = false;
                                for (int kk = 2; kk < param_.wsize_ && !found; kk++)
                                {
                                    if ((v - kk) < 0)
                                        continue;

                                    bool top = check ((*binary_cloud) (u, v - kk), (*binary_cloud) (u, v) );
                                    if ( !top )
                                        found = true;
                                }

                                if (!found)
                                {
                                    c_intensity += intensity_incr;
                                    (*binary_cloud) (u, v).intensity = c_intensity;
                                }
                            }

                        }
                        else
                        {
                            //check left and top
                            bool left = check ( (*binary_cloud) (u - 1, v), (*binary_cloud) (u, v) );
                            bool top = check ( (*binary_cloud) (u, v - 1), (*binary_cloud) (u, v) );

                            if ( !left && !top )
                            {
                                //both top and left had labels, check if they are different
                                //if they are, take the smallest one and mark labels to be connected..

                                if ((*binary_cloud) (u - 1, v).intensity != (*binary_cloud) (u, v - 1).intensity)
                                {
                                    float smaller_intensity = (*binary_cloud) (u - 1, v).intensity;
                                    float bigger_intensity = (*binary_cloud) (u, v - 1).intensity;

                                    if ((*binary_cloud) (u - 1, v).intensity > (*binary_cloud) (u, v - 1).intensity)
                                    {
                                        smaller_intensity = (*binary_cloud) (u, v - 1).intensity;
                                        bigger_intensity = (*binary_cloud) (u - 1, v).intensity;
                                    }

                                    connected_labels[bigger_intensity] = smaller_intensity;
                                    (*binary_cloud) (u, v).intensity = smaller_intensity;
                                }
                            }

                            if ( left && top )
                            {
                                //if none had labels, increment c_intensity
                                //search first on bigger window
                                bool found = false;
                                for (int dist = 2; dist < param_.wsize_ && !found; dist++)
                                {
                                    if (((u - dist) < 0) || ((v - dist) < 0))
                                        continue;

                                    bool left = check ((*binary_cloud) (u - dist, v), (*binary_cloud) (u, v) );
                                    bool top = check ((*binary_cloud) (u, v - dist), (*binary_cloud) (u, v) );

                                    if ( !left && !top )
                                    {
                                        if ((*binary_cloud) (u - dist, v).intensity != (*binary_cloud) (u, v - dist).intensity)
                                        {
                                            float smaller_intensity = (*binary_cloud) (u - dist, v).intensity;
                                            float bigger_intensity = (*binary_cloud) (u, v - dist).intensity;

                                            if ((*binary_cloud) (u - dist, v).intensity > (*binary_cloud) (u, v - dist).intensity)
                                            {
                                                smaller_intensity = (*binary_cloud) (u, v - dist).intensity;
                                                bigger_intensity = (*binary_cloud) (u - dist, v).intensity;
                                            }

                                            connected_labels[bigger_intensity] = smaller_intensity;
                                            (*binary_cloud) (u, v).intensity = smaller_intensity;
                                            found = true;
                                        }
                                    }
                                    else if ( !left || !top )
                                    {
                                        //one had label
                                        found = true;
                                    }
                                }

                                if (!found)
                                {
                                    //none had label in the bigger window
                                    c_intensity += intensity_incr;
                                    (*binary_cloud) (u, v).intensity = c_intensity;
                                }
                            }
                        }
                    }
                }

            }
        }
    }

    std::map<float, std::vector<int> > clusters_map;
    for (int i = 0; i < int (binary_cloud->width); i++)
    {
        for (int j = 0; j < int (binary_cloud->height); j++)
        {
            if (binary_cloud->at (i, j).intensity != 0)
            {
                //check if this is a root label...
                auto it = connected_labels.find (binary_cloud->at (i, j).intensity);
                while (it != connected_labels.end ())
                {
                    //the label is on the list, change pixel intensity until it has a root label
                    (*binary_cloud) (i, j).intensity = (*it).second;
                    it = connected_labels.find (binary_cloud->at (i, j).intensity);
                }

                auto it_indices = clusters_map.find ( binary_cloud->at (i, j).intensity );

                if (it_indices == clusters_map.end ())
                {
                    std::vector<int> indices;
                    clusters_map[binary_cloud->at (i, j).intensity] = indices;
                }

                clusters_map[binary_cloud->at (i, j).intensity].push_back (static_cast<int> (j * binary_cloud->width + i));
            }
        }
    }

    clusters_.resize (clusters_map.size ());

    size_t kept = 0;
    for (auto it_indices = clusters_map.begin (); it_indices != clusters_map.end (); ++it_indices)
    {
        if ( (*it_indices).second.size () >= param_.min_cluster_size_)
            clusters_[kept++] = (*it_indices).second;
    }

    clusters_.resize (kept);
}

#define PCL_INSTANTIATE_ConnectedComponentsSegmenter(T) template class V4R_EXPORTS ConnectedComponentsSegmenter<T>;
PCL_INSTANTIATE(ConnectedComponentsSegmenter, PCL_XYZ_POINT_TYPES )

}
