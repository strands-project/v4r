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

Objects are trained from several training views. Each training view is represented as an organized point cloud (`cloud_xyz.pcd`), the segmentation mask of the object (`object indices_xyz.txt`) and a camera pose ( `pose_xyz.pcd`) that aligns the point cloud into a common coordinate system when multiplied by the given 4x4 homogenous transform. The training views for each object are stored inside a `view`folder that is part of the object model folder. The object model folder is named after the object and contains all information about the object, i.e. initially contains the `views` and a complete 3D model ( `3D_model.pcd` ) of the object that is in the same coordinate system as the one in views. The complete model is used for hypotheses verification and visualization purposes. 

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
 Parameters for the recognition pipelines and the hypotheses verification are loaded from the `cfg/`folder by default but most of them can also be set from the command line. For a list of available command line parameters, you can use `-h`. Note that all parameters will be initialized by the `cfg/*.xml` files first, and then updated with the given command line arguments. Therefore, command line arguments will always overwrite the parameters from the XML file!

 
 <br>
 Let us now describe some of the parameters:
 
The `-z ` argument defines the **cut-off distance** in meter for the object detection (note that the sensor noise increases with the distance to the camera). Objects further away than this value, will be neglected.

Use ` --or_do_sift true --or_do_shot false` to enable or disable the local recognition pipelines of SIFT and SHOT, respectively.

Computed feature matches will be grouped into geometric consistent clusters from which a pose can be estimated using Singular Value Decomposition. The **cluster size** `--or_cg_thresh` needs to be equal or greater than 3 with a higher number giving more reliable pose estimates (since in general there will be more keypoints extracted from high resolution images, also consider increasing this threshold to avoid false positives and long runtimes). 

If the objects in your test configuration are always standing on a clearly visible surface plane (e.g. ground floor or table top), we recommend to remove points on a plane by setting the parameter `--or_remove_planes 1`.

To **visualize** the results, use command line argument `-v`. 

To **visualize intermediate hypotheses verification** results, you can use `--hv_vis_model_cues`.

```
./build/bin/ObjectRecognizer -m data/TUW/TUW_models -t data/TUW/test_set -z 2.5 --or_do_sift true --or_do_shot false --or_remove_planes 1 -v
```


## Multiview Recognizer
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

## Output
Recognized results will be stored in a single text file, where each detected object is one line starting with name (same as folder name) of the found object model followed by the confidence (between 0 for poor object hypotheses and 1 for very high confidence -- value in brackets), and the object pose as a 4x4 homogenous transformation matrix in row-major order aligning the object represented in the model coordinate system with the current camera view. Example output:
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




## References
* https://repo.acin.tuwien.ac.at/tmp/permanent/dataset_index.php
* Thomas Fäulhammer, Michael Zillich, Johann Prankl, Markus Vincze, "A Multi-Modal RGB-D Object Recognizer", IAPR International Conf. on Pattern Recognition (ICPR), Cancun, Mexico, 2016
