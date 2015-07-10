#include <pcl/console/parse.h>
#include <v4r/ORUtils/filesystem_utils.h>
#include <pcl/common/common.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/transforms.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/apps/dominant_plane_segmentation.h>

struct IndexPoint
{
    int idx;
};

POINT_CLOUD_REGISTER_POINT_STRUCT (IndexPoint,
                                   (int, idx, idx)
                                   )

void readJPPose(std::string filename, Eigen::Matrix4f & matrix)
{
    std::ifstream in;
    in.open (filename.c_str (), std::ifstream::in);

    char linebuf[1024];
    in.getline (linebuf, 1024);
    std::string line (linebuf);
    std::vector < std::string > strs_2;
    boost::split (strs_2, line, boost::is_any_of (" "));

    for (int i = 1; i < 17; i++)
    {
        int idx = i - 1;
        matrix (idx / 4, idx % 4) = static_cast<float> (atof (strs_2[i].c_str ()));
    }

}

void extractObjectIndices(pcl::PointCloud<pcl::PointXYZRGB>::Ptr & cloud,
                          std::vector<int> & indicesss)
{

    typedef pcl::PointXYZRGB PointInT;

    pcl::PointCloud<PointInT>::Ptr cloud_pass (new pcl::PointCloud<PointInT> ());

    pcl::PassThrough<PointInT> filter_pass;
    filter_pass.setKeepOrganized(true);
    filter_pass.setInputCloud(cloud);
    filter_pass.setFilterLimits(-0.2,0.2);
    filter_pass.setFilterFieldName("x");
    filter_pass.filter(*cloud_pass);


    pcl::apps::DominantPlaneSegmentation<PointInT> dps;
    dps.setInputCloud (cloud_pass);
    dps.setMaxZBounds (1.2f);
    dps.setObjectMinHeight (0.005);
    dps.setMinClusterSize (1000);
    dps.setDistanceBetweenClusters (0.02f);

    std::vector<pcl::PointCloud<PointInT>::Ptr> clusters;
    std::vector<pcl::PointIndices> indices;
    dps.setDownsamplingSize (0.02f);
    dps.compute_fast (clusters);
    dps.getIndicesClusters (indices);

    if(indices.size() > 0)
    {
        indicesss = indices[0].indices;
        std::cout << indicesss.size() << std::endl;
    }
}

int
main (int argc, char ** argv)
{
    typedef pcl::PointXYZRGB PointInT;

    std::string directory = "";
    std::string output_directory = "";
    int step = 1;

    pcl::console::parse_argument (argc, argv, "-output_directory", output_directory);
    pcl::console::parse_argument (argc, argv, "-directory", directory);
    pcl::console::parse_argument (argc, argv, "-step", step);

    std::vector<std::string> to_process;
    std::string so_far = "";
    std::string pattern = "cloud.*.pcd";

    v4r::utils::getFilesInDirectory(directory, to_process, so_far, pattern, false);

    std::vector<std::string> poses;
    pattern = ".*pose.*.txt";
    v4r::utils::getFilesInDirectory(directory, poses, so_far, pattern, false);

    std::sort(poses.begin(), poses.end());
    std::sort(to_process.begin(), to_process.end());

    std::vector<pcl::PointCloud<PointInT>::Ptr> original_clouds;
    std::vector<Eigen::Matrix4f> matrix_poses;
    std::vector<pcl::PointCloud<IndexPoint> > object_indices_clouds;
    std::vector<std::vector<int> > indices_vector;

    for(size_t i=0; i < to_process.size(); i+=step)
    {
        std::cout << to_process[i] << " " << poses[i] << std::endl;

        std::stringstream view_file;
        view_file << directory << "/" << to_process[i];
        pcl::PointCloud<PointInT>::Ptr cloud (new pcl::PointCloud<PointInT> ());
        pcl::io::loadPCDFile (view_file.str (), *cloud);

        original_clouds.push_back(cloud);

        std::stringstream pose_file;
        pose_file << directory << "/" << poses[i];

        Eigen::Matrix4f pose;
        readJPPose(pose_file.str(), pose);
        pose = pose.inverse().eval();

        std::cout << pose << std::endl;

        matrix_poses.push_back(pose);

        std::vector<int> indices;
        extractObjectIndices(cloud, indices);

        indices_vector.push_back(indices);

        pcl::PointCloud<IndexPoint> obj_indices_cloud;
        obj_indices_cloud.resize(indices.size());

        for(size_t k=0; k < indices.size(); k++)
        {
            obj_indices_cloud.points[k].idx = indices[k];
        }

        object_indices_clouds.push_back(obj_indices_cloud);
    }

    pcl::PointCloud<PointInT>::Ptr big_cloud(new pcl::PointCloud<PointInT>);

    for(size_t i=0; i < original_clouds.size(); i++)
    {
        pcl::PointCloud<PointInT>::Ptr segmented (new pcl::PointCloud<PointInT> ());
        pcl::copyPointCloud(*original_clouds[i], indices_vector[i], *segmented);
        //*big_cloud += *segmented;

        pcl::PointCloud<PointInT>::Ptr trans (new pcl::PointCloud<PointInT> ());
        pcl::transformPointCloud(*segmented, *trans, matrix_poses[i]);
        *big_cloud += *trans;
    }

    pcl::visualization::PCLVisualizer vis("test");
    pcl::visualization::PointCloudColorHandlerRGBField<PointInT> handler(big_cloud);
    vis.addPointCloud<PointInT>(big_cloud, handler);
    vis.addCoordinateSystem(0.2f);
    vis.spin();


    if(output_directory.compare("") != 0)
    {

        bf::path dir = output_directory;
      if(!bf::exists(dir))
      {
          bf::create_directory(dir);
      }

      //save the data with new poses
      for(size_t i=0; i < original_clouds.size(); i++)
      {
          std::stringstream view_file;
          view_file << output_directory << "/cloud_" << std::setfill ('0') << std::setw (8) << i << ".pcd";
          std::cout << view_file.str() << std::endl;

          pcl::io::savePCDFileBinary (view_file.str (), *(original_clouds[i]));

          std::string file_replaced1 (view_file.str());
          boost::replace_last (file_replaced1, "cloud", "pose");
          boost::replace_last (file_replaced1, ".pcd", ".txt");

          std::cout << file_replaced1 << std::endl;

          //read pose as well
          faat_pcl::utils::writeMatrixToFile(file_replaced1, matrix_poses[i]);

          std::string file_replaced2 (view_file.str());
          boost::replace_last (file_replaced2, "cloud", "object_indices");

          std::cout << file_replaced2 << std::endl;

          pcl::io::savePCDFileBinary (file_replaced2, object_indices_clouds[i]);

      }
    }
}
