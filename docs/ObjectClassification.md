# Object Classification by ESF

In this tutorial you will learn how to classify objects using the ESF descriptor. In our example, we will render views from mesh files and then classify point clouds by classifying extracted segments. We use the Cat10 models from 3dNet for training and the test clouds of 3dNet for testing.


## Data  
If you have not obtained the data yet, you can download them using the script file. Go to the v4r root directory and run
```
./scripts/get_3dNet_Cat10.sh
```
This will download and extract the Cat10 models (42MB).

If you also want some annotated example point clouds, you can obtain the test set from 3dNet (3.6GB) by running

```
./scripts/get_3dNet_test_data.sh
```

The files will be extracted in the `data/3dNet` directory.

## Usage
Assuming you built the examples samples, you can now run the classifier. If you run it for the first time, it will automatically render views by placing a virtual camera on an artificial sphere around the mesh models in `-i` and store them in the directory given by the `-m` argument. These views are then used for training the classifier, in our case by extracting ESF descriptors. For testing, it will segment the point cloud given by the argument `-t` by your method of choice (default: searching for a dominant plane and running Euclidean clustering for the points above). Each segment is then described by ESF and matched by nearest neighbor search to one of your learned object classes. The results will be stored in a text file which has the same name as the input cloud, just replacing the suffix from `.pcd` to `.anno_test` in the output directory specified by `-o`.  
```
./build/bin/example-esf_object_classifier -i data/3dNet/Cat10_ModelDatabase -m data/3dNet/Cat10_Views -t data/3dNet/Cat10_TestDatabase/pcd_binary/ -o /tmp/3dNet_ESF_results
```

## References
* https://repo.acin.tuwien.ac.at/tmp/permanent/3d-net.org/

*  Walter Wohlkinger, Aitor Aldoma Buchaca, Radu Rusu, Markus Vincze. "3DNet: Large-Scale Object Class Recognition from CAD Models". In IEEE International Conference on Robotics and Automation (ICRA), 2012.
