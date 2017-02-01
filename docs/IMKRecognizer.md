# Keypoint based Object Recognition in Monocular Images

IMKRecognizer is able to recognize objects in monocular images and to estimate the pose using a pnp-method. Object models, i.e. a sequence of RGBD-keyframes, the corresponding poses and the an object mask, can be created with RTMT, which supports OpenNI-Cameras (ASUS, Kinect).      


## Usage
For object recognition the model directory with the object-data stored in sub-directories need to be provided. For a RGBD-keyframe this can look like: “data/models/objectname1/views/frame_xxxx.pcd”. 
If the recognizer is started the first time the keyframes are loaded, interest points are detected, clustered and stored in a concatenated 'imk_objectname1_objectname2.bin' file. In case any recognition parameter is changed this file must be deleted and it will be created newly next time.

```
bin/example-imkRecognizeObject-file --help
bin/example-imkRecognizeObject-file -d data/models/ -n objectname1 objectname2 ... -f ~/testimages/image_%04d.jpg -s 0 -e 5 -t 0.5
```

## References
* J. Prankl, T. Mörwald, M. Zillich, M. Vincze: "Probabilistic Cue Integration for Real-time Object Pose Tracking"; in: Proceedings of the 9th International Conference on Computer Vision Systems, (2013), 10 S.
