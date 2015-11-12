//
//  viewer.cpp
//  LyonViewer
//
//  Created by Simon Schreiberhuber on 01.04.15.
//  Copyright (c) 2015 Simon Schreiberhuber. All rights reserved.
//

#include <iostream>
#include <string>
#include <sstream>
#include <limits>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>


#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <v4r/rendering/depthmapRenderer.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <boost/program_options.hpp>
#include <glog/logging.h>

namespace po = boost::program_options;


int main(int argc, const char * argv[]) {
    std::string file;

    po::options_description desc("Depth-Map Rendering\n======================================\n**Allowed options");
    desc.add_options()
            ("help,h", "produce help message")
            ("input,i", po::value<std::string>(&file)->required(), "directory containing the model .ply file")
    ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help"))
    {
        std::cout << desc << std::endl;
        return false;
    }

    try
    {
        po::notify(vm);
    }
    catch(std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl << std::endl << desc << std::endl;
        return false;
    }

    v4r::DepthmapRenderer renderer(640,480);
    //set the intrinsics of a kinect like sensor
    float fx=535.4;
    float fy=539.2;
    float cx=320.1;
    float cy=247.6;
    renderer.setIntrinsics(fx,fy,cx,cy);

    v4r::DepthmapRendererModel model(file);

    //this lets you test if the model has colored elements(note! no textures supported yet.... only colored polygons)
    if(model.hasColor()){
        std::cout << "IT HAS COLOR!!!"<< std::endl;
    }

    renderer.setModel(&model);

    std::vector<Eigen::Vector3f> sphere= renderer.createSphere(3,0);

    std::cout <<"[";
    for(size_t i=0; i<sphere.size(); i++){

        //get point from list
        Eigen::Vector3f point = sphere[i];
        //get a camera pose looking at the center:
        Eigen::Matrix4f orientation = renderer.getPoseLookingToCenterFrom(point);

        renderer.setCamPose(orientation);
        float visible;
        cv::Mat color;
        cv::Mat depthmap = renderer.renderDepthmap(visible,color);
        std::cout << visible << " % visible "  << std::endl;
        cv::imshow("depthmap",depthmap*0.25);
        cv::imshow("color",color);


        //create and save the according pcd files
        std::stringstream ss;
        ss << "test" << i << ".pcd";
        std::string file = ss.str();
        pcl::io::savePCDFileASCII (file, renderer.renderPointcloudColor(visible));
        std::cerr << "Saved data points to " << file << "." << std::endl;
        cv::waitKey();
    }

    return 0;
}
