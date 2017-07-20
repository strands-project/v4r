# Prepare NYU Depth Datasets for 3D Entangled Forest

The scripts in this directory provide a convenient way to download and convert the NYU Depth Datasets [1,2] by Silberman et al. into a form that can be processed by a 3D Entangled Forest, i.e. point clouds readable by PCL. The scripts further create auxilary data, such as a file with the camera pitch and roll angles obtained from the accelerometer values, required by the training procedure. Overall, the following data is created:

* XYZRGB pointclouds
* XYZL pointclouds containing groundtruth (already with the >800 labels mapped down to 13 labels bed, blind, bookshelf, cabinet, ceiling, floor, picture, sofa, table, tv, wall, window, background for version 1, and the 13 labels bed, object, chair, furniture, ceiling, floor, wall deco, sofa, table, wall, window, bookshelf, tv for version 2 of the dataset)
* cam_angles.txt file: Contains roll and pitch angles of the kinect per frame (in rad). Calculated from the accelerometer values provided in the dataset using a calibration file from Silberman.
* index files: index files containing all frame number of training resp. test sets of the dataset, as required by the 3DEF training stage.
* label names file: file containing names of 13 labels as required by the 3DEF training stage
* color code file: file containing the colorization scheme for the labels as used in [3]  

## Prerequisites

MATLAB needs to be installed since the dataset is provided in .mat files and the scrips are MATLAB scripts.
MATLAB's mex compiler needs to be set up (enter `mex -setup` in MATLAB) for the small uwrite tool to compile when the scripts are executed.

## Usage

The scripts are called `prepare_v1_dataset.m` and `prepare_v2_dataset.m`, respectively.
Optionally, the names of the output directories and output files can be adapted.
The generated output can individually be switched on and off using the `save...` parameters.

```matlab
% OUTPUT
pointclouds_dir = 'pointclouds';
lbl_pointclouds_dir = 'labeled_pointclouds';
cameraangles_file = 'cam_angles.txt';
trainingindex_file = 'indextraining.txt';
testindex_file = 'indextest.txt';
colorcode_file = 'color_code.txt';

savelabels = 1;         % save labeled pointclouds?
savepointclouds = 1;    % save rgb pointclouds?
savecameraangles = 1;   % save camera angles file?
savelabelnames = 1;     % save label names file?
saveindexfiles = 1;     % save index files?
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

```

During execution, if necessary, the scripts automatically download the .mat files
containing the dataset as well as the one containing the train/test splits.
To create the pointcloud files, up to 8 parallel threads are used.

## References

[1] N. Silberman and R. Fergus. Indoor scene segmentation using a structured light sensor. In
IEEE International Conference on Computer Vision Workshops (ICCVW), 2011.

[2] N. Silberman, D. Hoiem, P. Kohli, and R. Fergus. Indoor segmentation and support inference
from RGBD images. In European Conference on Computer Vision (ECCV), 2012.

[3] Daniel Wolf, Johann Prankl and Markus Vincze. Enhancing Semantic Seg-
mentation for Robotics: The Power of 3-D Entangled Forests. Robotics
and Automation Letters, IEEE, vol.1, no.1, 2016.
