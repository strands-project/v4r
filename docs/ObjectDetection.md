# Multi-modal RGB-D Object Instance Recognizer

In this tutorial you will learn how to detect objects in 2.5 RGB-D point clouds. The proposed multi-pipeline recognizer will detect 3D object models and estimate their 6DoF pose in the camera's field of view.


## Object Model Database  
The model folder structure is structured as: 

```
object-model-database  
│
└───object-name-1
│   │    [3D_model.pcd]
│   │
│   └─── views
│        │   cloud_000000.pcd
│        │   cloud_00000x.pcd
│        │   ...
│        │   object_indices_000000.txt
│        │   object_indices_00000x.txt
│        │   ...
│        │   pose_000000.txt
│        │   pose_00000x.txt
│        │   ...
│   │
│   └─── [trained-pipeline-1]*
│        │   ...
│   │
│   └─── [trained-pipeline-x]*
│        │   ...
│   
└───object-name-2
│   │    [3D_model.pcd]
│   │
│   └─── views
│        │   cloud_000000.pcd
│        │   cloud_00000x.pcd
│        │   ...
│        │   object_indices_000000.txt
│        │   object_indices_00000x.txt
│        │   ...
│        │   pose_000000.txt
│        │   pose_00000x.txt
│        │   ...
│   │
│   └─── [trained-pipeline-1]*
│        │   ...
│   │
│   └─── [trained-pipeline-x]*
│        │   ...
│        │   ...
│ ...
```
`* this data / folder will be generated`

Objects are trained from several training views. Each training view is represented as 
 * an organized point cloud (`cloud_xyz.pcd`), 
  * the segmentation mask of the object (`object indices_xyz.txt`) ,
  * and a camera pose ( `pose_xyz.pcd`) that aligns the point cloud into a common coordinate system when multiplied by the given 4x4 homogenous transform. 
  
The training views for each object are stored inside a `view`folder that is part of the object model folder. The object model folder is named after the object and contains all information about the object, i.e. initially contains the `views` and a complete 3D model ( `3D_model.pcd` ) of the object that is in the same coordinate system as the one in views. The complete model is used for hypotheses verification and visualization purposes. 

At the first run of the recognizer (or whenever retraining is desired), the object model will be trained with the features of the selected pipeline. The training might take some time and extracted features and keypoints are stored in a subfolder called after its feature descriptor (e.g. sift keypoints for object-name-1 will be stored in `model-database/object-name-1/sift/keypoints.pcd`). If these files already exist, they will be loaded directly from disk and skip the training stage. Please note that the trained directory needs to be deleted whenever you update your object model (e.g. by adding/removing training views), or the argument `--retrain` set when calling the program.

If you have not obtained appropriate data yet, you can download example model files together with test clouds  by running
```
./scripts/get_TUW.sh
```
from your v4r root directory. The files (2.43GB) will be extracted into `data/TUW`.

## Usage
Assuming you built the ObjectRecognizer app, you can now run the recognizer. If you run it for the first time, it will automatically extract the descriptors of choice from your model data (`-m`). 
The test directory can be specified by the argument `-t`. The program accepts either a specific file or a folder with test clouds. To use 2D features, these test clouds need to be organized. 

Example command:

```
./build/bin/ObjectRecognizer -m data/TUW/TUW_models -t data/TUW/test_set -z 2.5 --or_do_sift true --or_do_shot false --or_remove_planes 1 -v
```

## Parameters:
 Parameters for the recognition pipelines and the hypotheses verification are loaded from the `cfg/`folder by default but most of them can also be set from the command line. To list available command line arguments, call the recognizer with the additional flag `-h`.
 
Note that all parameters will be initialized by the values in the `cfg/*.xml` files first, and updated with the given command line arguments. Therefore, command line arguments will always overwrite the parameters from the XML file!

 
 Let us now describe some of the parameters:
 
### Input filter
The **cut-off distance** in meter for the object detection is defined by the command line argument `-z ` or line 
 ```
 <chop_z_>2.0</chop_z_>
 ```
 in `cfg/multipipeline_config.xml`. Objects further away than this value, will be neglected. Note that for typical consumer RGB-D cameras sensor noise increases significantly with the distance to the camera.

If the objects in your test configuration are always standing on a clearly visible support plane (e.g. ground floor or table top), we recommend to also remove points on a plane by adding the command line argument `--or_remove_planes 1` (together with the minimum number of plane inlier points) or editing following entries in `cfg/multipipeline_config.xml` to
```
<remove_planes_>1</remove_planes_>
<min_plane_inliers_>20000</min_plane_inliers_>
```

### Recognition pipelines
The recognizer uses several recognition pipelines in parallel to allow different types of objects to be detected (textured/non-textured, distinctive/uniform shape, under occlusion and clutter).
 
#### Local pipeline 
To enable sift (enabled by default), add command line argument ` --or_do_sift true`
or
```
<do_sift_>1</do_sift_>
<sift_config_xml_>cfg/sift_config.xml</sift_config_xml_>
```
Additional parameters for the SIFT pipeline are then defined in `cfg/sift_config.xml`. Inside `cfg/sift_config.xml`, a crucial parameter is line
```
	<knn_>5</knn_>
```
that defines **number of nearest neighbors for feature matching**. Each keypoint will then be matched to its `knn_` nearest features in the model database. For small model databases or higher precision, consider to decrease this number. For large model databases or higher recall, consider to increase this number.

Similarly, to enable SHOT, add command line argument ` --or_do_shot true`
or
```
<do_shot_>1</do_shot_>
<shot_config_xml_>cfg/shot_config.xml</shot_config_xml_>
```
Besidese the `knn_` parameter, SHOT also requires an explicit **keypoint extraction method** to be set. You can set the keypoint extraction method in `cfg/mulitpipeline_config.xml` by
```
	<shot_keypoint_extractor_method_>1</shot_keypoint_extractor_method_>
```
Currently, following types are available
 * uniform sampling (type: 1)
 * ISS (type: 2)
 * NARF (type: 4)
 * Harris3D (type: 8)
 
 If you select uniform sampling, please consider to filter keypoints on planar surface patches by editing these lines in `cfg/shot_config.xml`:
 
```
	<filter_planar_>1</filter_planar_>
	<planar_support_radius_>0.04</planar_support_radius_>
	<threshold_planar_>0.02</threshold_planar_>
```
The first line enables planar filtering, the second line defines the support radius for deciding about the planarity, and the third parameter is the curvature threshold for this point to be considered as planar.


##### Geometric Consistency Grouping
To allow multiple instances of an object to be detected in the same scene, feature matches will be grouped into geometric consistent clusters. The **minimum cluster size** is hereby defined by command line argument `--or_cg_thresh`or by entry
```
<cg_thresh_>4</cg_thresh_>
```
in `cfg/multipipeline_config.xml`. This number needs to be at least 3. A higher number gives more reliable pose estimates, but potentially reduces recall and frame rate. Since in general there will be more keypoints extracted from high resolution images, consider increasing this threshold to avoid false positives and long runtimes. For each geometric consistent cluster a pose is then estimated using Singular Value Decomposition.


#### Global pipeline 
If your scene configuration allows objects to be segmented, you can also consider using the global recognition pipeline which classifies the object as a whole. In the default configuration, first planes are removed (must be enabled, see above), and then remaining points are clustered based on Euclidean distance. Each cluster is then segmented by the specified classifer(s). Our generic framework, allows multiple classifiers to be used for classifying each cluster (generating multiple hypotheses). Let us give an example for `multipipeline_config.xml` for a single classifier (which should be sufficient for many cases):
```
	<global_recognition_pipeline_config>
		<count>1</count>
		<item_version>0</item_version>
		<item>cfg/global_config.xml</item>
:
:
	<global_feature_types_>
		<count>1</count>
		<item_version>0</item_version>
		<item>3232</item>
:
:
	<classification_methods_>
		<count>1</count>
		<item_version>0</item_version>
		<item>2</item>
```
The first group (global_recognition_pipeline_config) specifies the config XML file used for additional parameter settings for the global pipeline.

The second group  (global_feature_types_) specifies the **type of feature descriptrion**. Currently, following features are available (for a full and updated list of available feature types, please double check the header file `modules/features/include/v4r/features/types.h`):
 * ESF (type: 32)
 * Alexnet (type: 128) -- requires [Caffe library](https://github.com/BVLC/caffe)
 * Simple Shape (type: 1024)
 * Color (type: 2048)

Using an early fusion technique, feature vectors can be concatenated (just by adding the type numbers). In the example above, each cluster is described by concatenating the feature vectors from ESF, Alexnet, Simple Shape and Color into a single feature vector.

The third group (classification_methods_) specififes the **classification method** as defined in `modules/ml/include/v4r/ml/types.h`. Currently, following classification methods are available:
 * Nearest Neighbor (type: 1)
 * SVM (type: 2)
 
 If you selected SVM as your classification method, please also set **SVM specific command line arguments**:
  * `--svm_do_scaling 1` (enables scaling of feature attributes to normalize range between 0 and 1)
  * `--svm_do_cross_validation 3` (does 3-fold cross validation on the SVM parameters) 
  * `--svm_kernel_type 0` (0... linear kernel, 2... RBF kernel) - to speed up training, use linear for high-dimensional features
  *  `--svm_filename model.svm` (in case you have already trained your SVM, you can also just load the SVM from file)
  * `--svm_probability 1` (enables probability estimates)

Each item specifies the type of classifer used. 

### Multiview Recognizer
If you want to use multiview recognition, enable the entry

```
<use_multiview_>1</use_multiview_>
``` 
in the `cfg/multipipeline_config.xml` file. 

Additionally, you can also enable 
```
<use_multiview_hv_>1</use_multiview_hv_>
<use_multiview_with_kp_correspondence_transfer_>1</use_multiview_with_kp_correspondence_transfer_>
```
In most settings, we however recommend to leave `use_multiview_with_kp_correspondence_transfer_` disabled to save computation time. 

The object recognizer will then treat all `*.pcd` files within a folder as observations belonging to the same multi-view sequence. An additional requirement of the multi-view recognizer is that all observations need to be aligned into a common reference frame. Therefore, each `*.pcd`file needs to be associated with a relative camera pose (coarse point cloud registration is not done inside the recognizer!!). We use the header fields `sensor_origin` and `sensor_orientation` within the point cloud structure to read this pose. The registration can so be checked by viewing all clouds with the PCL tool `pcd_viewer /path/to/my/mv_observations/*.pcd`. 
Please check if the registration of your clouds is correct befor using the multi-view recognizer!

### Hypotheses Verification (HV)
Generated object hypotheses are verified by the Hypotheses Verification framework which tries to find the subset of generated hypotheses that best explain the input scene in terms of detected objects avoiding potential redundancy. 

Paramters for the Hypotheses Verification stage are stated in `cfg/hv_config.xml`.

For each object hypothesis, the HV stage first computes the visible object points under the current viewpoint. Objects that are less visible than the defined **minimum visible ratio**
```
<min_visible_ratio_>0.05</min_visible_ratio_>
```
will be rejected (in the example above at least 5% of the object need to be visible).

Based on the visible object, it then runs ICP to refine the pose of the obect. The ICP parameters are
```
	<icp_iterations_>30</icp_iterations_>
	<icp_max_correspondence_>0.02</icp_max_correspondence_>
```

Next, a fitness (or confidence) score is computed for each object hypothesis. Hypotheses with a score lower than the **minimum fitness threshold** 
```
<min_fitness_>0.3</min_fitness_>
```
will be rejected. You can adjust this value to allow weak hypotheses to be removed more easily (0... disables individual rejection, 1... rejects all hypotheses).

The score is computed based on the fit of the object hypothesis to the scene with respect to 
 * Euclidean distance in XYZ
 * surface normal orientation
 * color in CIELAB color space

The importance of these terms, can be set by these lines 
```
	<inlier_threshold_xyz_>0.01</inlier_threshold_xyz_>
	<inlier_threshold_normals_dotp_>0.99</inlier_threshold_normals_dotp_>
	<inlier_threshold_color_>20</inlier_threshold_color_>
	<sigma_xyz_>0.003</sigma_xyz_>
	<sigma_normals_>0.05</sigma_normals_>
	<sigma_color_>10</sigma_color_>
	<w_xyz_>0.25</w_xyz_>
	<w_normals_>0.25</w_normals_>
	<w_color_>0.5</w_color_>
```


## Visualization

To **visualize** the results, use command line argument `-v`. 

To **visualize intermediate hypotheses verification** results, you can use `--hv_vis_model_cues` and `--hv_vis_cues`.


## Output
Recognized results will be stored in a single text file in the folder defined by command line argument `-o`. Each detected object is one line starting with name (same as folder name) of the found object model followed by the confidence (between 0 for poor object hypotheses and 1 for very high confidence -- value in brackets), and the object pose as a 4x4 homogenous transformation matrix in row-major order aligning the object represented in the model coordinate system with the current camera view. Example output:
```
object_08 (0.251965): 0.508105 -0.243221 0.826241 0.176167 -0.363111 -0.930372 -0.0505756 0.0303915 0.781012 -0.274319 -0.561043 1.0472 0 0 0 1 
object_10 (0.109282): 0.509662 0.196173 0.837712 0.197087 0.388411 -0.921257 -0.0205712 -0.171647 0.767712 0.335861 -0.545726 1.07244 0 0 0 1 
object_29 (0.616981): -0.544767 -0.148158 0.825396 0.0723312 -0.332103 0.941911 -0.0501179 0.0478761 -0.770024 -0.301419 -0.562326 0.906379 0 0 0 1 
object_34 (0.565967): 0.22115 0.501125 0.83664 0.0674237 0.947448 -0.313743 -0.0625169 -0.245826 0.231161 0.806498 -0.544174 0.900966 0 0 0 1 
object_35 (0.60515): 0.494968 0.0565292 0.86707 0.105458 0.160923 -0.986582 -0.0275425 -0.104025 0.85388 0.153165 -0.497424 0.954036 0 0 0 1 
object_35 (0.589806): -0.196294 -0.374459 0.906228 0.1488 -0.787666 0.610659 0.0817152 -0.331075 -0.583996 -0.697765 -0.414817 1.01101 0 0 0 1 
```
To visualize results, add argument `-v`. This will visualize the input scene, the generated hypotheses and the verified ones (from bottom to top).   
For further parameter information, call the program with `-h` or have a look at the doxygen documents.  

## Evaluation
To evalute the results, you can use the `compute_recognition_rate` program. 

Example input:
```
./build/bin/ObjectRecognizer/compute_recognition_rate  -r /my/results/ -g /my_dataset/annotations/ -t /my_dataset/test_set -m /my_dataset/models --occlusion_thresh 0.95 --rot_thresh 45
```


## References
* https://repo.acin.tuwien.ac.at/tmp/permanent/dataset_index.php

### Single-View Recognition
* Thomas Fäulhammer, Michael Zillich, Johann Prankl, Markus Vincze, 
*A Multi-Modal RGB-D Object Recognizer*, 
IAPR International Conf. on Pattern Recognition (ICPR), Cancun, Mexico, 2016

### Multi-View Recognition
* Thomas Fäulhammer, Michael Zillich and Markus Vincze
*Multi-View Hypotheses Transfer for Enhanced Object Recognition in Clutter,*
IAPR International Conference on Machine Vision Applications (MVA), Tokyo, Japan, 2015

or, if used with keypoint correspondence transfer:

* Thomas Fäulhammer, Aitor Aldoma, Michael Zillich and Markus Vincze,
*Temporal Integration of Feature Correspondences For Enhanced Recognition in Cluttered And Dynamic Environments,*
IEEE International Conference on Robotics and Automation (ICRA), Seattle, WA, USA, 2015
