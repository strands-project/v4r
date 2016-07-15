# Multi-modal RGB-D Object Instance Detector

In this tutorial you will learn how to detect objects in 2.5 RGB-D point clouds. The proposed multi-pipeline recognizer will detect 3D object models and estimate their 6DoF pose in the camera's field of view.


## Data  
If you have not obtained the data yet, you can download (2.43GB) model files and some test clouds from your v4r root directory by running
```
./scripts/get_TUW.sh
```

The files will be extracted in the `data/TUW` directory.

## Usage
Assuming you built the examples samples, you can now run the classifier. If you run it for the first time, it will automatically extract the descriptors of choice from your model data (`-m`). The test directory can be specified by the argument `-t` and the cut-off distance in meter by `-z`. For a full list of available parameters, you can use `-h`.

```
./build/bin/example-object_recognizer -m data/TUW/models -t data/TUW/test_set -z 2.5 --do_sift true --do_shot false --do_esf false --do_alexnet false -v
```

## References
* https://repo.acin.tuwien.ac.at/tmp/permanent/dataset_index.php
* Thomas Fäulhammer, Aitor Aldoma, Michael Zillich, Markus Vincze, "Temporal Integration of Feature Correspondences For Enhanced Recognition in Cluttered And Dynamic Environments", IEEE Int. Conf. on Robotics and Automation (ICRA), 2015
