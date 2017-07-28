# Semantic Segmentation using 3D Entangled Forests (3DEF)
This tutorial first describes how to learn a new 3D Entangled Forest classifier from training data and how to use it for semantic segmentation of 3D pointclouds. We also provide already learned classifiers for the NYU Depth datasets.

# Learning of a new 3DEF classifier
## 1. Preparation of NYU Depth Datasets
To get and convert the NYU Depth Datasets from their original MATLAB format to PCL point clouds and create the necessary auxilary files for training, you can use the scripts provided in [v4r/scripts/NYU_Depth_Dataset](../../scripts/NYU_Depth_Dataset).

## 2. Create training data
To learn a new classifier, first the training data needs to be extracted from a dataset. This is done with the tool `create_3DEF_trainingdata`:
```
./create_3DEF_trainingdata -i input-pointcloud-dir -l groundtruth-dir -x indexfile -g camera-angles-file -o output-dir [more options]
```
where the following directories and files need to be specified:
* `input-pointcloud-dir`: directory containing the XYZRGB input pointclouds
* `groundtruth-dir`: directory containing the XYZL labeled groundtruth pointclouds with the same names as the XYZRGB pointclouds
* `indexfile`: text file containing the list of training files to process, without file extension. E.g. if the training files are named "0001.pcd", "0002.pcd" etc. the index file would look like this:
  ```
  0001
  0002
  ...
  ```
  For NYUv1 and v2 the file `indextraining` is automatically created by the scripts used in step 1.
* `camera-angles-file`: text file containing the roll and pitch angles of the camera for each frame (in rad). Format is one line with [roll] [pitch] per frame, e.g.:
  ```
  0.323245   0.124923
  -0.232424  0.013434
  ...
  ```
  These angles are only used as a fallback if the floor fit fails. First, the tool tries to find the floor plane by fitting a plane to all points labeled floor (the labelID of the floor can be defined with parameter `floor-label`, the default ID of 6 is the setting for the NYU Depth Datasets). If there are not enough floor points to fit a plane in a frame, the camera angles are taken from this angle file, and the height is defined by a constant defined in the code (TODO: make this a parameter as well).
  For NYUv1 and v2 these files are automatically created by the scripts used in step 1.
* `output-dir`: output directory to put the resulting training data

To get a list of all parameters simply start the tool without any command line arguments or use `--help`.

### 2.1 Training data output
The generated training data consists of 4 directories containing files for each frame:
* `clustering`: output from the segmenter. XYZL pointclouds where each point of the scene is assigned a segmentID. For each frame, there is also an `_organized` version created, which can be used in the evaluation tool to generate organized results for nice 2D visualization of the labeling results.
* `labels`: groundtruth labels mapped from points to the segments calculated by the segmentation method. Each segment is assigned the 1 label that has the most overlap with the pointwise groundtruth labeling.
* `pairwise`: pairwise distances (Euclidean, point-to-plane, inverse point-to-plane, horizontal angles and vertical angles) between all segments. Required for the 3D entangled features.
* `unary`: unary features for each segment of each frame.

`labels`, `pairwise` and `unary` files are stored in binary format to speed up loading time for training.

## 3. Train the classifier
After training data has been generated, the tool `train3DEF` runs the training procedure to get a new 3DEF classifier:
```
./train3DEF -i trainingdata-dir -x indexfile  -n labelnames-file -o output-file [more options]
```
* `trainingdata-dir`: the directory containing the training data computed in step 2.
* `indexfile`: text file containing the list of training files to process, without file extension. Usually same as for step 2.
* `labelnames-file`: text file with a list of strings with the class label names. For NYUv1 and v2 these are automatically created by the scripts used in step 1.
* `output-file`: where to store the resulting 3DEF classifier

To get a list of all parameters simply start the tool without any command line arguments or use `--help`.
The default parameters are the ones used in the paper [1].

Please note that the training procedure, depending on the chosen parameters, can last for hours. To speed up the process for the first experiments, e.g. the number of trees could be reduced (parameter `trees`, default: 60).

# Run classification
The usage of the classifier is demonstrated with the `classification_demo` tool, which takes an input pointcloud and outputs the corresponding semantic segmentation result:
```
./classification_demo -f classifier-file -i input-pointcloud -c color-file -o output-pointcloud [more options]
```
* `classifier-file`: file containing the 3DEF classifier.
* `input-pointcloud`: XYZRGB input pointcloud
* `color-file`: text file containing the color code for the labels. Each line contains the 3 RGB values, where line 1 contains the color for unlabeled points (usually 0 0 0), line 2 the color for label 1, line 3 for label 2 etc. The color code files used in [1] are automatically created by the scripts from step 1.
* `output-pointcloud`: where the resulting XYZRGBL pointcloud should be stored. RGB channels contain the label color, the L channel the labelID.

To get reasonable results, the camera pose w.r.t. ground plane also needs to be provided using the following parameters:
* `cam-height`: camera height above ground in meters (default: 1.0)
* `cam-pitch`: camera pitch angle in degrees (default: 0.0)
* `cam-roll`: camera roll angle in degrees (default: 0.0)

To get a list of all parameters simply start the tool without any command line arguments or use `--help`.

# Learned classifiers ready to use
For the NYU Depth Datasets, we provide already learned classifier files you can directly use with the demo tool, such that you do not need to run the entire training pipeline before. The files can be found [here](https://repo.acin.tuwien.ac.at/tmp/permanent/daniel_wolf_3DEF) and are grouped by the dataset and label set they have been trained on:
* `v1_13_labels`: NYU v1 Dataset, using 13 labels bed, blind, bookshelf, cabinet, ceiling, floor, picture, sofa, table, tv, wall, window, background (label set from [2])
* `v2_13_labels`: NYU v2 Dataset, using 13 labels bed, object, chair, furniture, ceiling, floor, wall deco, sofa, table, wall, window, bookshelf, tv (label set from [3])
* `v2_4_labels`: NYU v2 Dataset, using 4 structural labels ground, struct, furniture, props (label set from [4])

# Other apps
* `merged_supervoxels_demo`: Demo of the segmentation method based on merging supervoxels
* `evaluate3DEF`: Runs an evaluation of a learned 3DEF on a given dataset, calculating the confusion matrix for each frame
* `merge3DEF`: Tool to merge multiple forest to one larger forest (useful if training happens in stages, e.g. 20 trees at a time)
* `updateleafs3DEF`: Tool to refine a learned forest by passing the whole training set through each tree again, updating the label distributions in the trees. This generally improves the class avg. accuracy of the classifier
* `analyze3DEF`: Tool to analyze a learned 3DEF by creating histograms of picked features etc.

# References
[1] Daniel Wolf, Johann Prankl and Markus Vincze. Enhancing Semantic Segmentation for Robotics: The Power of 3-D Entangled Forests. Robotics and Automation Letters, IEEE, vol.1, no.1, 2016.

[2] N. Silberman and R. Fergus. Indoor scene segmentation using a structured light sensor. In
IEEE International Conference on Computer Vision Workshops (ICCVW), 2011.

[3] C. Couprie, C. Farabet, L. Najman, and Y. LeCun. Indoor semantic segmentation using depth
information. In International Conference on Learning Representations (ICLR), 2013.

[4] N. Silberman, D. Hoiem, P. Kohli, and R. Fergus. Indoor segmentation and support inference
from RGBD images. In European Conference on Computer Vision (ECCV), 2012.