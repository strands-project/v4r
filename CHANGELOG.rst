^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package v4r
^^^^^^^^^^^^^^^^^^^^^^^^^

1.4.1 (2016-02-01)
------------------
* Merge pull request `#58 <https://github.com/strands-project/v4r/issues/58>`_ from strands-project/fix1
  initialize counter variable
* initialize counter variable
* Merge pull request `#57 <https://github.com/strands-project/v4r/issues/57>`_ from strands-project/remove_c+11_from_header
  remove c++11 construct in header file
* remove c++11 construct in header file
* Merge pull request `#56 <https://github.com/strands-project/v4r/issues/56>`_ from strands-project/fix1
  Fix1
* add siftgpu as optional dependency in RTMT
* copy uniform_sampling files from PCL 1.7.2 to make V4R also compile on PCL 1.8
* updated RTMT noise model parameters
* Merge remote-tracking branch 'v4r_root/master'
* Merge branch 'dynamic_object_learning' into 'master'
  Dynamic object learning
  See merge request !50
* Merge branch 'master' into 'master'
  Master
  See merge request !49
* Contributors: Thomas Fäulhammer

1.4.0 (2016-01-27)
------------------
* Merge pull request `#55 <https://github.com/strands-project/v4r/issues/55>`_ from strands-project/new_recognition_resolved_merge_conflict
  New recognition resolved merge conflict
* Merge branch 'new_recognition'
* Merge remote-tracking branch 'strands/master'
* change default values
* fix noise model based cloud integration
* make opencv sift instantiation conditional on siftgpu presence
* integrate parse console arguments into library
* Merge pull request `#54 <https://github.com/strands-project/v4r/issues/54>`_ from taketwo/speed-up
  Speed-up info collection in NMBasedCloudIntegration
* uses more parallelization
* Merge remote-tracking branch 'sergey_strands/speed-up' into new_recognition
* Speed-up info collection in NMBasedCloudIntegration
  Pre-compute the number of points and resize big_cloud_info\_ only once.
  This achieves > 2x speed-up in Debug mode.
* tmp commit to test siftgpu
* some beauty
* add present of model in view variable for go3d
  change default noise model param
* parallelize add models function in go3d
* some beauty
* normalize all components of LAB colors in range -1 to 1
* put color transform into seperate class
* remove a few pointers and add parallel block
  refactor code for merging feature correspondences in multiview recognizer
* fix conditional compilation with -DWITH_SIFTGPU=OFF
* remove hough_3d as it is not used and within PCL (maybe other version though)
* remove accidentally added build folder
* remove template parameters FeatureT and DistT for local recognizer/estimator
  save descriptors as binary text file on disk
* getting rid of some pointers
  move duplicated functions in a common file
* make multipipeline recognizer code parallel
* parallelize correspondence grouping
* make converttoflann create its flann data internally (to make interfacing simpler)
* hopefully solves range_check_error during correspondence grouping
  refactored some code
* add missing ifdef HAVE_SIFTGPU
* fix interface problem in IOL and avoid deprecated interface
* Merge pull request `#52 <https://github.com/strands-project/v4r/issues/52>`_ from strands-project/add_citation_license_file
  add citation, license and authors file
* add citation, license and authors file
* Merge pull request `#51 <https://github.com/strands-project/v4r/issues/51>`_ from strands-project/build-fixes
  Build fixes
* Merge remote-tracking branch 'severin/build-fixes'
  Conflicts:
  samples/examples/object_recognizer_new.cpp
* Merge pull request `#49 <https://github.com/strands-project/v4r/issues/49>`_ from strands-project/fix_siftgpu_problem_in_IOL
  Fix siftgpu problem in iol
* use HAVE_SIFTGPU to check if siftgpu is available on system in object modelling module
* rename dynamic object learning to incremental object learning
* Added missing header 'boost/format.hpp' in a few examples
* [cmake] ObjectGroundTruthAnnotator requires VTK
* [cmake] Ensure v4r compiles without ceres at CMake level
  Note that V4R *does not yet* compile without ceres due to
  modules/reconstruction/include/v4r/reconstruction/impl/ReprojectionError.hpp
  requiring ceres.h
* Properly guards omp.h includes so that the project compile without OpenMP support
* [cmake] Cosmetic in CMakeLists
* [cmake] Use pkg-config to find OpenNI2
  The Debian package for libopenni2 provides a .pc but no
  FindOpenNI2.cmake
* [cmake] FindOpenGL does not return the version
* [cmake] Added support for compiling with Qt5
  Note that CMake option WITH_QT needs to be enabled,
  and *WIT_QT4* needs to be disabled.
* [cmake] Enable WITH_QT by default
* Merge pull request `#44 <https://github.com/strands-project/v4r/issues/44>`_ from strands-project/dynamic_object_learning
  Dynamic object learning
* make compatible to new v4r interfaces
* Merge branch 'dol_rebased'
* Merge branch 'master' of github.com:strands-project/v4r
* fix deprecated warning
* remnants from RAL paper
* fixed some bugs
* fix of fix
* fixed bug in evaluation - don't test on same set as object was trained
* write eval recognition for uncontrolled scenes to test on controlled ones
  added visualization of all learnt models
* fix wrong parameter type
* added file to test model coverage
* skip patrol run for which no object has been modelled
* recognition evaluation with respect to coverage for controlled runs
* Merge branch 'recognition_dev' into dol_rebased
  Conflicts:
  modules/recognition/include/v4r/recognition/impl/local_recognizer.hpp
* taken view file is now correct
* forgot to undo temporary change
* added evaluation tool for recognition performance measure of partial model coverage
* add FindX11 and FindXRandR
* fixed error when training views do not begin with 0
* recognition evaluation for partial model and for offline data more or less ready
* added first evaluation tool to compute recognition performance with respect to percentage of visible model
* added visualize_masked_pcd again
* Merge branch 'recognition_dev' into dol_rebased
* Merge branch 'dol_rebased' of rgit.acin.tuwien.ac.at:t-faeulhammer/v4r into dol_rebased
  Conflicts:
  samples/examples/dynamic_object_learning.cpp
  samples/icra16/eval_dol_gt.cpp
  samples/icra16/eval_dynamic_object_learning_with_mask_pertubation.cpp
* add eval
  use boost program options
* adapt code to make rebase compile
* rebase commit
* added noise level evaluation for initial mask ICRA16
* added eval for inital mask evaluation
  added for icra16 singleview
* fixed sv eval when test_dir is not present
* fixed bug in icra sv eval, when csv file has only 2 columns
* eval almost ready
* added icra vis
* seperate post-processing and save to disk in object learning
* fixed wrong output file if name of mask is mask.txt only
* removed overhead computation when sift based camera pose estimation is disabled
* fixed ground truth labelling
* fixed color in add text
* just addded a const
* removing nan points in initial mask - otherwise seg fault when after erosion not enough points
* included plane merge
  moved logical stuff to common module
  added plane visualization
  added plane properties
* added function to write images to disk for intermediate steps
* make ratio parameter accessible from outside for occluded and object supported points
* sort files before evaluation and output debug info
* added some V4R_EXPORTS in registration module
  removed redundant fast_icp in common module
  added app for 3D reconstruction based on SIFT and MST
  fixed CERES version conflict
  fixed some dependency issues
* up
* add Willow Dataset definition for save_pose_into_pcd sample
* set sensor pose to identity in eval to show right visiualization
* parameters can now also be set in constructor
  initial eval code now in samples (should be moved somewhere else later on)
* moved mask<->indices conversion function into v4r common module
* added object_modelling again
* adapt code to make rebase compile
* rebase commit
* added noise level evaluation for initial mask ICRA16
* added eval for inital mask evaluation
  added for icra16 singleview
* fixed sv eval when test_dir is not present
* fixed bug in icra sv eval, when csv file has only 2 columns
* eval almost ready
* added icra vis
* seperate post-processing and save to disk in object learning
* fixed wrong output file if name of mask is mask.txt only
* removed overhead computation when sift based camera pose estimation is disabled
* fixed ground truth labelling
* fixed color in add text
* just addded a const
* removing nan points in initial mask - otherwise seg fault when after erosion not enough points
* included plane merge
  moved logical stuff to common module
  added plane visualization
  added plane properties
* added function to write images to disk for intermediate steps
* make ratio parameter accessible from outside for occluded and object supported points
* sort files before evaluation and output debug info
* added some V4R_EXPORTS in registration module
  removed redundant fast_icp in common module
  added app for 3D reconstruction based on SIFT and MST
  fixed CERES version conflict
  fixed some dependency issues
* up
* add Willow Dataset definition for save_pose_into_pcd sample
* set sensor pose to identity in eval to show right visiualization
* parameters can now also be set in constructor
  initial eval code now in samples (should be moved somewhere else later on)
* moved mask<->indices conversion function into v4r common module
* added object_modelling again
* Contributors: Sergey Alexandrov, Séverin Lemaignan, Thomas Fäulhammer

1.3.1 (2016-01-13)
------------------
* Merge pull request `#43 <https://github.com/strands-project/v4r/issues/43>`_ from strands-project/fix_classifier
  Fix classifier
* fix global classifier error when reading from new model database file structure
* build utility tools by default
* Merge remote-tracking branch 'simon/master'
* Merge pull request `#42 <https://github.com/strands-project/v4r/issues/42>`_ from strands-project/remove_glfw3_dependency
  Remove glfw3 dependency
* remove some output messages
* remove glfw3 dependency and use X* libraries only
  fixed some deprecated interfaces warnings
  added some build /run dependency for openni
* Added code for a proper destructor
* cleaned up some code
* Merge remote-tracking branch 'simon/master'
* Merge remote-tracking branch 'simon/change_glfw_to_old'
* removed the need for glfw and changed everything to work with only x11 dependencies
* Merge remote-tracking branch 'origin/master'
* Merge remote-tracking branch 'v4r_root/master'
* Merge remote-tracking branch 'strands/master'
* merged
* use openni instead of openni2
* Merge remote-tracking branch 'v4r_root/recognition_dev'
* Merge branch 'recognition_update' into 'master'
  Recognition update
  See merge request !2
* Merge branch 'recognition_update' into 'master'
  Recognition update
  See merge request !45
* Contributors: Simon Schreiberhuber, Thomas Fäulhammer

1.3.0 (2016-01-08)
------------------

1.2.0 (2016-01-08)
------------------
* Merge pull request `#40 <https://github.com/strands-project/v4r/issues/40>`_ from strands-project/recognition_dev
  Recognition dev
* use openni instead of openni2
* rename object tracker
* updated object tracker and RTMT saves tracking model correctly
* seperated normal computation methods into new file
  using using namespace v4r in samples for (i) brevity, (ii) conformity with ROS wrappers
  changed some deprecated interfaces
  split header files into impl (.hpp) files mainly to avoid c++11 neccessity when including with ROS
* temporary backup commit
* noise model based cloud integration update (also moved to registration module)
  uses properties (1) lateral noise, (2) axial noise, (3) distance in px to depth discontinuity
* backup commit
* first try for new noise modelling
* Merge branch 'fix_KeypointSlamRGBD' into recognition_dev
* fixed problem when training views do not start with cloud_000000.pcd
  TODO: re-initialize problem still exists (if training database is altered, flann matrix will be wrong - have to remove *_flann.idx manually right now)
  fixed trigraph warnings
* use absolute value when checking reprojected poitns in ground truth annotation
  added parameters for noise model based integration demo program
* since image2 only takes integer values, we do not need to interpolate (checked by Hannes)
* add zero padding in interpolationfunction to avoid assertion error in Debug mode
  add fix from Hannes
* fix seg fault when dist_coeffs is 2x4 matrix instead of 1x8
* small fix (avoid ourcvfh)
* rewrite noise model based integration so that it uses really equation from Nguyen et al paper.
* tmp commit
* tmp commit
* replaced a few integer and long unsigned integer by size_t to hopefully make it working on 32bit machines
  added visualization functions for hypotheses verification
* some more changes in pcl2opencv
* change pcl2opencv interfaces
* added tools again
* Merge remote-tracking branch 'strands/master' into add_v4r_exports
* added a few more V4R_EXPORTS (visibility attribute) for classes
  added cmake_minimum_required version (cmake 2.8.8 apparently can not handle url hash tags)
* add v4r_export for tomita
* Contributors: Thomas Fäulhammer

1.1.1 (2015-11-23)
------------------
* Merge pull request `#37 <https://github.com/strands-project/v4r/issues/37>`_ from strands-project/add_glm_run_dependency
  add glm also as run dependency
* add glm also as run dependency
* Contributors: Thomas Fäulhammer

1.1.0 (2015-11-20)
------------------
* Merge pull request `#35 <https://github.com/strands-project/v4r/issues/35>`_ from strands-project/recognition_update
  Recognition update
* Merge remote-tracking branch 'v4r_root/recognition_update' into recognition_update1
* Merge branch 'fix_glfw3' into 'recognition_update'
  Fix glfw3 and undefined references to X*
  See merge request !47
* add multiple X11 linker libs
* Fix variable names in examples CMakeLists
* Merge remote-tracking branch 'sergey/fix-glfw3' into recognition_update1
* Export 3rdparty include directories into config file
* undo insert HAVE_V4R_RENDERING
* add some x*libraries in package.xml to hopefully solve undefined references
* Merge remote-tracking branch 'sergey/fix-glfw3' into recognition_update1
* added description for go3d parameter
* Properly add GLFW3 third-party library
* Merge branch 'fixes-for-recognition-update' into 'recognition_update'
  Fixes for recognition update
  This fixes a few compilation problems in the current recognition update branch.
  See merge request !46
* Fix "invalid suffix 'd' on floating constant" error
* Add missing dependency (rendering depends on GLM)
* added glog as dependency (otherwise linking potentially fails)
* updated parameters for sv recognition
* added conversion function from point cloud to fixed sized image
  removed unused parameters in global estimator
  changed namespace of pclopencv to v4r
* computing histogram size by sizeof to make esf estimator compatible with PCL 1.7.1
* remove template parameter Feature from global classifier and make it a vector instead
  added esf object classifier again
* tmp commit
* Merge remote-tracking branch 'simon/recognition_update' into recognition_update1
  Conflicts:
  modules/rendering/src/depthmapRenderer.cpp
* tmp commit (conditional saving of pcd as xyz or xyzrgb) before merging simons update
* Cleaned up the code and sorted out some culprits.
* fixed datatype for colormap
  fixed some warnings
  added program options for radius, subdivision, camera intrinsics,...
* added glGetError queries.
* added rendering + example
  added glew, glfw find package
* updated some more parameter descriptions
* renamed occlusion reasoning into zbuffering and removed second layer namespace
  seperated classes into seperate files
  renamed boost graph extenstion into multi-view representation
  fixed shot recognizer (removed indices), parameters are now written to file
* added GLOG dependency
  using boost program option for object recognizer examples and Ground-truth annotator
* use integer key for model assembly (instead of float) - resolution for it is now a parameter
  temporary included visualization for pose refinement
* parameters are now double (instead of float) to be easily accessible from outside via ros getparam
  default parameters change
  updated ground truth annotator and evaluations for recognizer to new framework
* added clear multiview data
* Properly export template instantiations in EDT
* Fix METSlib third-party library
* removed visualization reminiscent in single-view recognizer
* fixed wrong index computation in 3D occupancy grid
  removed siftgpu library from necessary dependency in reconstruction app
* fixed wrong angle difference calculation when clustering object hypotheses [TODO: make parameter accesible from outside]
* (hopefully) fixes crash when no valid recognition model is found
  merging close hypotheses is now possible (generate less hypotheses, refines pose by these larger correspondence set)
* using mask instead of indices in go3d addModels function
  increased default occlusion threshold
  can be compiled with clang again
* fixed multiplane segmentation in unorganized point clouds (TODO: downsample cloud)
  replaced USE_SIFT_GPU definitions with HAVE_SIFTGPU
  v4r_config.h now presents status of HAVE_SIFTGPU
* added pcl version of ClusterNormalsToPlane (works for unorganized point clouds now)
  TODO: fix multiplane segmentation method
* install metslib header files
  fixed go3d
  createVoxelGridAndDistanceTransforms is now called inside generate (for registered views source) TODO: Check influence of resolution paramter!
  added some description and licenses
* temporary commit with GO3D visualization
* fixed wrong transformation of keypoints when using -transfer_feature_matches 1
* added mising tracking dependency
* recognizer:
  - added license
  - removed unused variables
  - moved internally used public methods to protected
* go3d implemented but results not satisfying (parameter not good?)
* pruningGrap in Multiview Object Recognizer is working
  [TODO: Finish Go3D Hypothesis Verification Integration]
* failed try of point cloud rendering with vtk
* when using compute_mst, it crashes after using pruneGraph
* absolute pose computation seems to work
* absolute pose computation seems to work
* added merging of feat correspondences
* tmp commit
* temporary commit (single-view recognizer correspondence grouping seems broken)
* adding parameter classes
  remove redundant variables
  getting rid of singleview_object_recognizer class
  local estimator uses normal estimator from v4r common now
  Reimplementation of multiview recognizer just started (NOT WORKING / COMPILING)
* single view object recognizer almost ready
* tmp commit
  getting rid of redundnant single_view object recognizer class
* correspondences in recognizer are now stored as indexes to original cloud
  this should reduce memory requirement
  New parameter class for Hypotheses Verification methods (different results to before - TODO: double check default parameters!)
* only training dir parameter neccessary any more
  improved code readability
* temporary commit (signatures not initialized) otherwise it seems to work
* overall update of training procedure
* recognizer structure sift parameter was named inconsistently
  fixed some warnings
* this includes the changes from gitlab v4r version made by @alexandrox88
  - fixes assimp in tomgine
  - remove ipp
  adds object tracking
  fixes a few warnings
* SOMETHING SEEMS TO BE WRONG WITH THE TRANSFORMS
  namespace update
  polishing multiview recognizer
  add libsvm as system dependency
* merged remove_tomgine
* Merge branch 'master' of rgit.acin.tuwien.ac.at:root/v4r
* Merge branch 'remove-ipp' into 'master'
  Remove all mentions of IPP (Intel Performance Primitives)
  Remove all mentions of IPP (Intel Performance Primitives). This remained from OpenCV scripts.
  See merge request !43
* Remove all mentions of IPP (Intel Performance Primitives)
* Merge branch 'fix-tomgine-assimp' into 'master'
  Fix Assimp dependency
  This merge request fixes missing Assimp include in Tomgine and updates the CMake script for finding the package.
  See merge request !42
* Update Assimp finder script
* Add missing AssImp include in tomgine
* Fix a few warnings in tomgine
* Merge branch 'master' into 'master'
  Master
  created a tracking module and added the monocular object tracker from RTMT
  See merge request !41
* Merge branch 'find-system-libsvm' into 'master'
  Add CMake commands to detect system installation of LibSVM
  The possibility to build LibSVM from source is preserved, but has to be enabled by setting BUILD_LIBSVM option (which is now off by default).
  See merge request !40
* added monocular camera pose tracker (lk/ keypoint based) from RTMT
* Merge branch 'master' of rgit.acin.tuwien.ac.at:root/v4r
* test
* test
* mv test
* just a test file
* Contributors: Johann Prankl, Markus Bajones, Sergey Alexandrov, Thomas Fäulhammer, simon.schreiberhuber@gmx.net

1.0.11 (2015-10-14)
-------------------
* Merge pull request `#34 <https://github.com/strands-project/v4r/issues/34>`_ from strands-project/remove_tomgine
  temporary remove Tomgine and everything related to it (i.e. object cl…
* also comment computeCentroid in single-view object recognizer
* comment computeCentroid to silence error
* temporary remove Tomgine and everything related to it (i.e. object classification)
* Contributors: Thomas Fäulhammer

1.0.10 (2015-09-21)
-------------------
* Merge pull request `#31 <https://github.com/strands-project/v4r/issues/31>`_ from strands-project/namespace_update
  Namespace update
* namespace update
  polishing multiview recognizer
  add libsvm as system dependency
* Merge remote-tracking branch 'sergey/find-system-libsvm' into namespace_update
* Add CMake commands to detect system installation of LibSVM
  The possibility to build LibSVM from source is preserved, but has to be
  enabled by setting BUILD_LIBSVM option (which is now off by default).
* rename multiview_object_recognizer
  silence unused variable warning
  removed unneccessary point cloud copy
  normal method now a parameter
* Merge branch 'master' into 'master'
  Master
  See merge request !39
* Merge branch 'master' into 'master'
  Master
  See merge request !38
* Contributors: Sergey Alexandrov, Thomas Fäulhammer

1.0.9 (2015-09-17)
------------------
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
