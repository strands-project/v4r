/*
    Labeling tool with Slic calculation
    <maintainer email="prodromou@acin.tuwien.ac.at">prodromou</maintainer>
    <maintainer email="faeulhammer@acin.tuwien.ac.at">Faeulhammer</maintainer>

    USAGE:
    key 0-9 > choose Label (0... background, 1-9 object labels)
    left click... label superpixel
    SPACE > start new Slic calculation when moving the Trackbar
    's'-key > save labeled image
 */

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <float.h>
#include <string>
#include <sstream>

#include <boost/filesystem.hpp>

#include <pcl/console/parse.h>
#include <pcl/common/common.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <v4r/common/pcl_opencv.h>
#include <v4r/common/io/filesystem_utils.h>
#include <v4r/segmentation/Slic.h>

using namespace std;
using namespace cv;


// ---- Define various colors for easier access -----------
Vec3b label_color[] = {
    Vec3b(255, 0, 0),
    Vec3b(0, 255, 0),
    Vec3b(0, 0, 255),
    Vec3b(255, 0, 255),
    Vec3b(0, 255, 255),
    Vec3b(255, 255, 0),
    Vec3b(64, 255, 128),
    Vec3b(128, 128, 128),
    Vec3b(255, 255, 255),
    Vec3b(0, 0, 0)
};

class Slic_Labelling
{
protected:
    int
    current_label,
    nSclicKMax,
    nSclicMMax,
    nSclicK,  // superpixels number
    nSclicM,    // compactness factor
    nSclicNumOfLabels;

    bool
    bLeftMousePressed,
    is_pcd;

    Mat_<cv::Vec3b>
    slicSrcMat,
    slicImgDRAWMat;

    Mat_<int>
    pxLabel,
    slicLabelsMat;

    std::vector< std::vector< std::pair<int, int> > > superpx_pxs;

    v4r::Slic slic;

    char cKeyPressed;

    std::string in_path;

    pcl::visualization::PCLVisualizer::Ptr vis;

public:
    Slic_Labelling()
    {

    }

    // ---- Function to calculate Slic on Image -----------
    void CalculateSLIC()
    {
        if (nSclicK > 0 && nSclicM > 0)
        {
            slicSrcMat.copyTo(slicImgDRAWMat);
            slic.segmentSuperpixelNumber(slicSrcMat, slicLabelsMat, nSclicNumOfLabels, nSclicK, nSclicM);
            superpx_pxs.resize(nSclicNumOfLabels);
            for(int superpx_id = 0; superpx_id < nSclicNumOfLabels; superpx_id++)
            {
                superpx_pxs[ superpx_id ].clear();
            }

            for (int row_id = 0; row_id < slicLabelsMat.rows; row_id++)
            {
                for (int col_id = 0; col_id < slicLabelsMat.cols; col_id++)
                {
                    int px_label = slicLabelsMat.at<int> ( row_id, col_id);
                    superpx_pxs[px_label].push_back ( std::pair<int,int> (row_id, col_id));

                    if ( pxLabel( row_id , col_id) ) //pixel already labelled by previous labelling steps
                    {
                        slicImgDRAWMat(row_id , col_id) = label_color[ pxLabel( row_id , col_id) - 1 ];
                    }
                }
            }

            slic.drawContours(slicImgDRAWMat, slicLabelsMat, 255,0,0);
            cout << "number of superpixel patches: "<< nSclicNumOfLabels <<endl;
            cv::imshow("Slic Results",slicImgDRAWMat);
        }
    }


    // ---- Function for the trackbar -----------
    static void Ontrackbar(int, void* = NULL)
    {
    }

    void onMouse_st(int nOnMouseEvent, int nOnMouseX, int nOnMouseY)
    {
        // ---- Left mousebutton to draw the selected label -----------
        if (nOnMouseEvent == EVENT_LBUTTONDOWN || (bLeftMousePressed && nOnMouseEvent == EVENT_MOUSEMOVE))
        {
            bLeftMousePressed = true;
            int clkdLabel = slicLabelsMat(nOnMouseY,nOnMouseX);

    #pragma omp parallel for
            for(size_t px_id = 0; px_id < superpx_pxs[ clkdLabel ].size(); px_id++)
            {
                int row_id = superpx_pxs[ slicLabelsMat(nOnMouseY,nOnMouseX) ][ px_id ].first;
                int col_id = superpx_pxs[ slicLabelsMat(nOnMouseY,nOnMouseX) ][ px_id ].second;

                pxLabel( row_id , col_id) = current_label;
                slicImgDRAWMat(row_id , col_id) = label_color[ current_label - 1];

                if (current_label == 0 )
                    slicImgDRAWMat(row_id , col_id) = slicSrcMat(row_id, col_id);
            }

            // ---- Update the image -----------
            slic.drawContours(slicImgDRAWMat, slicLabelsMat, 255,0,0);
            imshow("Slic Results", slicImgDRAWMat);
            //imshow("Labeled Image", labeledImgMat);
        }
        else if (nOnMouseEvent == EVENT_LBUTTONUP)
        {
            bLeftMousePressed = false;
        }

    }

    static void onMouse_st(int nOnMouseEvent, int nOnMouseX, int nOnMouseY, int nOnMouseFlags, void* this_)
    {
        static_cast<Slic_Labelling*>(this_)->onMouse_st(nOnMouseEvent, nOnMouseX, nOnMouseY);
    }

    static void CheckBox(int state, void* val){
         // do nothing for now
         return;
    }

    int init(const std::string &in_path)// const cv::Mat_< cv::Vec3b> &src_img)
    {
        nSclicNumOfLabels = 0;
        nSclicK = 5000;
        nSclicM = 10;
        current_label = 1;
        nSclicKMax=  10000;
        nSclicMMax = 60;
        cKeyPressed = -1;

        bLeftMousePressed = false;
        is_pcd = false;
        this->in_path = in_path;

        if(in_path.substr(in_path.find_last_of(".") + 1) == "pcd")    //check if given img is a point cloud
        {
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr pPCl(new pcl::PointCloud<pcl::PointXYZRGB>);
            pcl::io::loadPCDFile(in_path, *pPCl);
            if (pPCl->isOrganized())
            {
                PCLOpenCV::ConvertPCLCloud2Image<pcl::PointXYZRGB> (pPCl, slicSrcMat);
                is_pcd = true;
            }
            else
            {
                std::cerr << "Point cloud is not organized! Aborting..." << std::endl;
                return -1;
            }
        }
        else
        {
            slicSrcMat = imread(in_path);
        }

        // ---- Create Windows -----------
        namedWindow("Slic Results", cv::WINDOW_NORMAL);
        cv::namedWindow("Labelled Image", 1);
        pxLabel = cv::Mat::zeros(slicSrcMat.rows, slicSrcMat.cols, CV_64F);

        // ---- Create Trackbars -----------
        char cTrackbarName[50];
        sprintf(cTrackbarName, "superpixel max %d", nSclicKMax);
        createTrackbar(cTrackbarName, "Slic Results", &nSclicK, nSclicKMax, this->Ontrackbar);
        sprintf(cTrackbarName, "compactness max %d", nSclicMMax);
        createTrackbar(cTrackbarName, "Slic Results", &nSclicM, nSclicMMax, this->Ontrackbar);


        // ---- Load, Show and Copy src image to global Mat -----------
        imshow("Slic Results", slicSrcMat);

        CalculateSLIC();
        setMouseCallback("Slic Results", onMouse_st, this);

        return 0;
    }

    void run()
    {
        while(1)
        {
            if (isdigit(cKeyPressed))
            {
                current_label = cKeyPressed - '0';
                cout << "Label " << cKeyPressed << endl;
            }

            switch (cKeyPressed)
            {
            // ---- Save labeled img to disk -----------
            case 's':
            {
                cout << "labelling done..." << endl;
                return;
            }

                // ---- Reload slic in main window with SPACE -----------
            case 32:
            {
                CalculateSLIC();
            }
            }
            cKeyPressed = cvWaitKey(1);
        }
    }

    Mat_<int> getPxLabel() const
    {
        return pxLabel;
    }

    void show_and_save_file()
    {
        boost::filesystem::path p(in_path);
        std::string filename_only, filename_with_ext, dir, ext;


        filename_with_ext = p.filename().string(); // file.ext
        filename_only = p.stem().string();
#ifdef WIN32
        dir = in_path.substr(0, in_path.find_last_of('\\'));
#else
        dir = in_path.substr(0, in_path.find_last_of('/'));
#endif
        ext = in_path.substr(in_path.find_last_of('.'));


        for(int label_id = 1; label_id < static_cast<int> (sizeof(label_color)/sizeof(label_color[0]) ); label_id++)
        {
            size_t num_points = 0;

            std::stringstream label_path;
            label_path << dir << "/" << filename_only << "_object_" << label_id << ext;

            if(is_pcd)    //check if given img is a point cloud
            {
                pcl::PointCloud<pcl::PointXYZRGB>::Ptr pLabelledPCl(new pcl::PointCloud<pcl::PointXYZRGB>);
                pcl::io::loadPCDFile(in_path, *pLabelledPCl);

                assert(pxLabel.rows == pLabelledPCl->height && pxLabel.cols == pLabelledPCl->width);

                for (int row_id = 0; row_id < pxLabel.rows; row_id++)
                {
                    for (int col_id = 0; col_id < pxLabel.cols; col_id++)
                    {
                        if ( pxLabel.at<int>(row_id, col_id) != label_id)
                        {
                            pLabelledPCl->at(col_id, row_id).x = std::numeric_limits<float>::quiet_NaN();
                            pLabelledPCl->at(col_id, row_id).y = std::numeric_limits<float>::quiet_NaN();
                            pLabelledPCl->at(col_id, row_id).z = std::numeric_limits<float>::quiet_NaN();
                        }
                        else
                            num_points++;
                    }
                }

                if ( num_points )
                {
                    if(!vis)
                    {
                        vis.reset(new pcl::visualization::PCLVisualizer("segmented object"));
                    }
                    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb_handler (pLabelledPCl);
                    vis->addPointCloud<pcl::PointXYZRGB> (pLabelledPCl, rgb_handler, label_path.str());
                    vis->spin();
                    vis->close();

                    std::cout << "Saving segmented point cloud to " << label_path.str() << "." << std::endl;
                    pcl::io::savePCDFileBinary<pcl::PointXYZRGB>(label_path.str(), *pLabelledPCl);
                }
            }
            else
            {
                cv::Mat_ <cv::Vec3b> slicLabelledMat = cv::Mat_<cv::Vec3b>::zeros(slicSrcMat.rows, slicSrcMat.cols);
                for (int row_id = 0; row_id < pxLabel.rows; row_id++)
                {
                    for (int col_id = 0; col_id < pxLabel.cols; col_id++)
                    {
                        if ( pxLabel.at<int>(row_id, col_id) == label_id)
                        {
                            num_points++;
                            slicLabelledMat.at<Vec3b>(row_id, col_id) = slicSrcMat.at<Vec3b>(row_id, col_id);
                        }
                    }
                }
                if(num_points)
                {
                    cv::imshow("Labelled Image", slicLabelledMat );
                    cv::waitKey(0);
                    imwrite(label_path.str(), slicSrcMat);
                }
            }
        }
    }

};


// ---- Main function -----------
int main(int argc, char *argv[])
{
    Slic_Labelling slic_lbl;

    std::string img_path;
    pcl::console::parse_argument (argc, argv, "-img_path", img_path);

    bf::path img_path_bf = img_path;
    std::vector < std::string > files_intern;
    if (v4r::io::getFilesInDirectory(img_path, files_intern, "", ".*\\.(jpg|JPG|png|PNG|jpeg|JPEG|bmp|BMP|pcd|PCD)", true) != -1)
    {
        for(size_t file_id=0; file_id < files_intern.size(); file_id++)
        {
            std::stringstream image_path_ss;
            image_path_ss << img_path << "/" << files_intern[file_id];

            slic_lbl.init( image_path_ss.str() );
            slic_lbl.run();
            slic_lbl.show_and_save_file();
        }
    }
    else if (bf::is_regular_file(img_path_bf))
    {
        std::cout << "Loading file " << img_path << std::endl;

        slic_lbl.init( img_path);
        slic_lbl.run();
        slic_lbl.show_and_save_file();
    }
    else
        std::cerr << "Image path is not valid." << std::endl;

    return 0;
}
