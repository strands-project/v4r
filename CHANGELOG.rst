^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package v4r
^^^^^^^^^^^^^^^^^^^^^^^^^

Forthcoming
-----------
* Merge branch 'master' of github.com:strands-project/v4r
* fix Bloom issue with umlauts
* Merge remote-tracking branch 'strands/master'
* Contributors: Thomas Fäulhammer

1.0.8 (2015-09-17)
------------------
* Merge pull request `#28 <https://github.com/strands-project/v4r/issues/28>`_ from strands-project/remove_c++11_flags_and_common_namespace
  remove C++11 flags
* remove C++11 flags
  remove common namespace
  remove duplicated files
  divide samples in examples, evaluation and utility tools (enable examples by default in cmake)
  add Qt Cache files in .gitignore list
* Contributors: Thomas Fäulhammer

1.0.7 (2015-09-16)
------------------
* Merge pull request `#27 <https://github.com/strands-project/v4r/issues/27>`_ from strands-project/new_samples_structure
  New samples structure
* Merge pull request `#26 <https://github.com/strands-project/v4r/issues/26>`_ from strands-project/add-tomgine
  Add tomgine
* new samples structure
* divide samples into examples, tools and evals
* adds ESF classifier using new point cloud rendering based on TomGine (camera pose is not extracted right now)
* Merge pull request `#24 <https://github.com/strands-project/v4r/issues/24>`_ from strands-project/sift_gpu_solution
  Sift gpu solution
* added initial segmentation example
* updated usage output
* added tomgine
* added Random Forest and SVM
* Merge remote-tracking branch 'sergey/add-libsvm' into add-libsvm
* added RandomForest
  fixed some warnings
* Add libsvm 3rd-party library
* Merge branch 'master' into 'master'
  Master
  See merge request !37
* reverted sv recognizer header file because otherwise cg pointer cast caused seg fault
  fixed some warnings
* make SIFT_GPU optional by setting BUILD_SIFTGPU in cmake
* added segmentation dependency to samples
* added binary vector increment
  changed parameter name to avoid confusion in range image computation
* merged
* Merge branch 'master' into 'master'
  Master
  this hopefully includes all the changes from LaMoR Summer School + fixes for the Recognizer
  See merge request !36
* Contributors: Sergey Alexandrov, Thomas Fäulhammer

1.0.6 (2015-09-07)
------------------
* Merge pull request `#23 <https://github.com/strands-project/v4r/issues/23>`_ from strands-project/mergeLAMOR
  Merge lamor
* merged lamor STRANDS
* Merge branch 'master' of github.com:strands-project/v4r into mergeLAMOR
* Merge branch 'master' of rgit.acin.tuwien.ac.at:root/v4r into mergeLAMOR
* added default param for printParams in MV recognizer
  other minor changes
* Update Readme.md
* hopefully fixes bug in ourcvfh with different pcl versions
  view_all_point_clouds_in_folder can now also save images to disk
* Merge branch 'master' into 'master'
  Master
  See merge request !35
* catch SIFT FLANN exception when updating model database
* flann idx now configurable
* Merge branch 'master' into 'master'
  Master
  See merge request !34
* Merge branch 'master' into 'master'
  Master
  See merge request !33
* Contributors: Marc Hanheide, Thomas Fäulhammer

1.0.5 (2015-08-30)
------------------

1.0.4 (2015-08-29)
------------------
* Merge pull request `#22 <https://github.com/strands-project/v4r/issues/22>`_ from strands-project/marc-hanheide-patch-1
  disable C++11
* disable C++11
  see https://github.com/strands-project/v4r_ros_wrappers/commit/0f008ac162ef2319d5685054023ce2c6f0c8db55
* disable C++11
  see https://github.com/strands-project/v4r_ros_wrappers/commit/0f008ac162ef2319d5685054023ce2c6f0c8db55
* Contributors: Marc Hanheide

1.0.3 (2015-08-29)
------------------
* Merge pull request `#21 <https://github.com/strands-project/v4r/issues/21>`_ from strands-project/added_install_commands
  added install targets for apps
* added install targets for apps
* Contributors: Marc Hanheide

1.0.2 (2015-08-29)
------------------
* Merge pull request `#20 <https://github.com/strands-project/v4r/issues/20>`_ from strands-project/marc-hanheide-patch-1
  don't include FREAK headers
* don't include FRAK headers
  as this seems to fail in non-free opencv... see https://github.com/strands-project/v4r_ros_wrappers/pull/3
* Contributors: Marc Hanheide, Michael Zillich

1.0.1 (2015-08-28)
------------------
* fixed some compiler warnings
  fixed out of range bug in GHV RGB2CIELAB when RGB color is white (255,255,255)
  fixed typo in parameter for eval sv
* removed comments in sv recognizer,
  save parameter file in sv recognizer eval
* removed linemod and debug build for recognition
* fixed bug in sv_recognizer
* added EDT include path
* added ground truth annotator as app
  removed unused files in recognition
* added sv recognition sample
  fixed missing chop_z behaviour in single view recognizer
* added sample eval for single view object recognizer
* updated ReadMe
* added libglm-dev as build dependency
* Merge branch 'add-glm-dependency' into 'master'
  Add GLM dependency
  See merge request !32
* Add GLM dependency
* Merge branch 'master' into 'master'
  Master
  See merge request !31
* added cmake files for openni2
* Merge branch 'master' into 'master'
  Master
  See merge request !30
* Merge branch 'fix-u-r' into 'master'
  Fix undefined reference errors (with Octree and 1.7.1)
  See merge request !29
* added qt-opengl-dev as dependency
* added openni in package.xml
* Merge branch 'master' of rgit.acin.tuwien.ac.at:root/v4r
* linked openni libraries to RTMT
  added octree_impl to hopefully solve pcl conflicts with versions <= 1.7.1
* Hopefully fix undefined reference errors (with Octree)
* Merge branch 'add-template-keyword' into 'master'
  Add missing 'template' keyword (causes clang compilation error)
  See merge request !28
* Merge branch 'master' of rgit.acin.tuwien.ac.at:root/v4r
* added RTMT GL libraries again
* Add missing 'template' keyword (causes clang compilation error)
* added binary operations to common
  changed dist calculation for planes available from outside
* fixed QT4 CMake file
* Merge branch 'master' into 'master'
  fixed QT4 CMake file
  See merge request !27
* Merge branch 'master' into 'master'
  added RTMT
  See merge request !26
* added RTMT
* Merge branch 'master' into 'master'
  Master
  See merge request !25
* added possibility to crop image when converting PCD to image
  createDirIfNotExists should now create all directories recursively
  added initial version for pcl segmentation (used in STRANDS in Year1) - not finished
* make parameters double (instead of float) to make it directly accessible via ros getparam function
* Merge branch 'master' into 'master'
  Master
  See merge request !24
* fixed error with Willow Poses
  removed object modelling dependency which is not yet present
* added const specifier for get function
* Merge branch 'master' into 'master'
  Master
  See merge request !23
* Merge branch 'master' of rgit.acin.tuwien.ac.at:t-faeulhammer/v4r
  Conflicts:
  samples/cpp/save_pose_into_pcd.cpp
* added some V4R_EXPORTS in registration module
  removed redundant fast_icp in common module
  added app for 3D reconstruction based on SIFT and MST
  fixed CERES version conflict
  fixed some dependency issues
* fix of last push
* fix of last push
* added definitions for willow_dataset in save_pose_into_pcd sample
* added mask<->indices converter function
  ground truth annotator now also outputs mask for object in first frame
* added initial version for ground truth labelling tool
* del
* added samples folder
* Merge branch 'dynamic_object_learning' of rgit.acin.tuwien.ac.at:t-faeulhammer/v4r into dynamic_object_learning
* fixed some ns
* fixes some namespace issues
* added object learning again
* fixed pcl version conflict with vector of eigen
* Merge branch 'master' of rgit.acin.tuwien.ac.at:t-faeulhammer/v4r
* fixed vector conflict with different PCL versions
* fixed some ns
* Merge branch 'master' into dynamic_object_learning
* changed ns
* fixed wrong macro names for detect CUDA cmake
* Merge branch 'dynamic_object_learning' of http://rgit.acin.tuwien.ac.at/t-faeulhammer/v4r into dynamic_object_learning
* Merge branch 'dynamic_object_learning' of rgit.acin.tuwien.ac.at:t-faeulhammer/v4r into dynamic_object_learning
* added object learning again
* fixes some namespace issues
* Merge branch 'dynamic_object_learning' of rgit.acin.tuwien.ac.at:t-faeulhammer/v4r into dynamic_object_learning
* added object learning again
* fixed wrong cmake macro name
* added object learning again
* del
* Merge branch 'master' into 'master'
  del
  See merge request !22
* Merge branch 'master' into 'master'
  Master
  See merge request !21
* remnoved second layer namespace "rec_3d_framework"
  added some V4R_EXPORTS
  changed some include paths
  removed redundant faat_3d_rec_framework.h file
* Merge branch 'dependency-propagation' of rgit.acin.tuwien.ac.at:alexandrov88/v4r into dependency_propagation
* Print OpenCV and Ceres statuses
* Update find Ceres to export variables
* Implement dependency propagation
* Split filesystem_utils into parts
* Remove duplicate find eigen call
* Properly set variables in FindEDT
* Properly set variables in FindOpenCV
* Properly set variables in FindEigen
* SiftGPU fixup
* Boost fixup
* Change SIFTGPU_INCLUDE_DIR -> SIFTGPU_INCLUDE_DIRS
* Update io module
* Find Boost globally
* Merge branch 'master' into 'master'
  Master
  See merge request !20
* Merge branch 'master' into dynamic_object_learning
* added camera tracker - uff, so many changes!
* updated recognition cmakefile to have correct link to opencv
  fixed some shadow warnings
* fixed some warning and added V4R_EXPORTS
  added openmp in cmake
* Merge branch 'master' into 'master'
  fixed some warning and added V4R_EXPORTS
  added openmp in cmake
  See merge request !19
* Merge branch 'fix-edt' into 'master'
  Build EDT library with -fPIC option
  See merge request !18
* Build EDT library with -fPIC option
* fixed some warnings
  changed default parameter value of sor
* Merge branch 'master' into 'master'
  Master
  See merge request !17
* Merge branch 'master' into dynamic_object_learning
* added object_modelling cmakelists.txt
* added OpenCV as cmake dependency
  added some V4R_EXPORTS
  re-inserted computeOccludedPoints (why was this not merged?? Check other files!)
  added OpenMP cmake c/cxx flags
* fixed warnings of shadowed variables
  using new v4r namespaces
* Merge branch 'master' into dynamic_object_learning
  Conflicts:
  modules/object_modelling/include/v4r/object_modelling/do_learning.h
  modules/object_modelling/include/v4r/object_modelling/model_view.h
  modules/object_modelling/src/do_learning.cpp
  modules/object_modelling/src/visualization.cpp
* updated EDT include path
* Merge remote-tracking branch 'sergey/cmake_updates'
* Create core module, moved macros.h and version.h here
* All modules now explicitly depend on PCL
* Fix EDT
* added missing segmentation dependency
* Merge branch 'master' into 'master'
  added missing segmentation dependency
  See merge request !16
* adapted to new cmake system
* Merge branch 'master' into 'master'
  Master
  See merge request !15
* Merge pull request `#19 <https://github.com/strands-project/v4r/issues/19>`_ from strands-project/new_cmake
  New cmake
* Merge branch 'master' into dynamic_object_learning
  Conflicts:
  modules/CMakeLists.txt
* Fix 3rd party header handling for the case of no-install usage of V4R
* Merge branch 'new_cmake' into 'master'
  New cmake
  See merge request !14
* changed required PCL version to less restrictive 1.7.
  Otherwise, there is a conflict on Jenkins because it only provides package for 1.7.1
* hide recognition module for the time being
* added package.xml again - Jenkins needs it to build the library
  added sergey to maintainer list
* Merge remote-tracking branch 'sergey/master' into new_cmake
  Conflicts:
  modules/recognition/CMakeLists.txt
  modules/registration/CMakeLists.txt
* Fix V4RConfig.cmake for use without installation
* Merge branch 'master' into dynamic_object_learning
* fixed some warnings with redundant typenames and wrong derived signature (& missing) in Recognition
  fixed missing EDT dependency in Registration
* Merge branch 'master' into 'master'
  Master
  See merge request !13
* updated supervoxel clustering method
  added some function docs
  optional parameter for pairwise transform refinement
* filtering smooth clusters works -- without visualization
* smooth clusters work now --- with visualization for debug
* Miscellaneous should not depend on keypoints
* Revert "(Temporarily) move miscellaneous to keypoints because it depends on them"
  This reverts commit 8b4bf90048750c95bae136b9b65dbb890c8c900e.
* Add V4R_EXPORTS here and there
* Merge branch 'master' into dynamic_object_learning
* pcl::copyPointCloud now also accepts binary obj mask
* beautify code - moved from indices to mask
  added parameter filter_planes_only (not working for value false yet)
* (Temporarily) move miscellaneous to keypoints because it depends on them
* Solve undefined reference problem
* Export 3rdparty include directories
* Remove compatibility stuff
* Finalize SiftGPU support
* table filtering working now as expected...
  removed some unnecessary includes
* temporary commit for visualizing table planes supported by object mask
* Another fix for SiftGPU
* Merge branch 'revert_merge_request' into 'master'
  Revert "Merge branch 'dynamic_object_learning' into 'master'"
  This reverts commit 87d034a1a8c8763657ca59ff08f9ec95a5d1c4be, reversing
  changes made to d183d5143b68e70de0e678a3d0659fae2a85a731.
  See merge request !12
* Revert "Merge branch 'dynamic_object_learning' into 'master'"
  This reverts commit 87d034a1a8c8763657ca59ff08f9ec95a5d1c4be, reversing
  changes made to d183d5143b68e70de0e678a3d0659fae2a85a731.
* Trying to add SiftGPU
* Fix EDT
* Remove SiftGPU sources
* Fix EDT third-party dependency
* Merge branch 'master' into dynamic_object_learning
* fixed some warnings
  added occlusion reasoning for two clouds (optional tf) which return occlusion mask
* Merge branch 'dynamic_object_learning' into 'master'
  Dynamic object learning
  See merge request !11
* added parameter for statistical outlier removal (mean=50 and stddevmul=1 didn't work well on asus_box)
  fixed bug in CreateMaskFromVecIndices
  there seems to be still a problem in occlusion reasoning
* Add new build system, migrate common and segmentation modules
* Get rid of legacy build system stuff
* Merge branch 'master' into dynamic_object_learning
* Merge branch 'master' into 'master'
  fixed warning of unused variable in SLICO
  fixed visualization issue when called multiple times
  See merge request !10
* fixed warning of unused variable in SLICO
  fixed visualization issue when called multiple times
* updated region growing such that it does not use points already neglected by plane extractor
  fixed visualizition issue when calling the visualization service more than once
* Merge branch 'master' into 'master'
  Master
  See merge request !9
* Merge branch 'master' into dynamic_object_learning
* added ceres version check
  updated McLMIcp.cpp to use new fixes from aitor
* include devil dependency
* changed to right rosdep key for glew
* Merge branch 'master' into dynamic_object_learning
* added some dependencies
* Merge branch 'master' into 'master'
  Master
  See merge request !8
* Merge branch 'master' into dynamic_object_learning
* removed aitor from maintainer list
* Merge remote-tracking branch 'strands/package_xml'
* Merge branch 'master' into dynamic_object_learning
* added parameter class for noise model based integration
  changed Eigen::Vector4f vector for correct allocation
* indices are now stored in a more generic way
  visualization now also includes noise model
  added Statistical Outlier Removal for initial indices
  added logical operator for binary masks
  TODO: visualization does only work for first service call
* added opencv dependency
* fixed dependencies to the correct rosdep keys
* added a first package.xml
* MST is now a parameter
  plane indices are stored as a vector of a vector now - (otherwise high cost occured in callgrind profiler)
  updated clear function
* Merge branch 'master' into dynamic_object_learning
* createDirIfNotExist function is now in common
* fixed problem with nan values (recursive absolute pose computation based on spanning tree implementation was not correct)
* minimum spanning tree is working now... there are nan values transferred to nearest neighbor search -> still needs to be fixed!
* bug fix - should be back to STRANDS review demo state
* Merge branch 'master' into 'master'
  Master
  See merge request !7
* Merge branch 'master' into dynamic_object_learning
* fixed some linking problems... fixed bug in setCloudPose (last element was not set to 1)
  made code clang compatible...
* tmp commit
* Merge branch 'master' into dynamic_object_learning
* fixed linking error, updated some namespaces
* tmp commit
* Merge branch 'master' into dynamic_object_learning
* changed some recognition files to use new filesystem namespace
* tmp commit
* Merge branch 'master' into 'master'
  Master
  See merge request !6
* temporary commit of dynamic object learning. not compiling yet!
* deleted remaining temp(~) files
* added keypoint files needed for object learning
* added clustertonormals from keypointTools
* add initial version of keypoints
* Merge branch 'master' into 'master'
  Master
  See merge request !5
* some fixes to merge to master
* Merge remote-tracking branch 'v4r_root/master'
  Conflicts:
  3rdparty/metslib/CMakeLists.txt
  CMakeLists.txt
  cmake/v4rutils.cmake
  cmake/v4rutils.cmake~
  modules/common/CMakeLists.txt
  modules/common/include/v4r/common/noise_model_based_cloud_integration.h
  modules/common/include/v4r/common/noise_models.h
  modules/common/src/noise_model_based_cloud_integration.cpp
  modules/common/src/noise_models.cpp
  modules/recognition/include/v4r/recognition/boost_graph_extension.h
  modules/recognition/include/v4r/recognition/ghv.h
  modules/recognition/include/v4r/recognition/multiview_object_recognizer_service.h
  modules/recognition/src/boost_graph_extension.cpp
  modules/recognition/src/boost_graph_visualization_extension.cpp
  modules/recognition/src/multiview_object_recognizer_service.cpp
  modules/segmentation/CMakeLists.txt
* remove ~
* .
* .
* tmp commit
* Merge branch 'master' into 'master'
  Added multiview recognizer. renamed some namespaces.
  See merge request !4
* Added multiview recognizer. renamed some namespaces.
* Merge branch 'master' into 'master'
  Master
  See merge request !3
* Fixed merge conflict
* Initial commit. For some reason if segmentation app is compiled - there is a linking problem with pcl. Namespaces are a mess!
* initial commit
* upd
* Merge branch 'master' into 'master'
  update readme
  See merge request !1
* update readme
* Add new file
* Init commit
* Contributors: Marc Hanheide, Markus Bajones, Sergey Alexandrov, Thomas Faeulhammer, Thomas Fäulhammer
