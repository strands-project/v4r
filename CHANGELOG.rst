^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package v4r
^^^^^^^^^^^^^^^^^^^^^^^^^

1.4.9 (2017-07-28)
------------------
* merge master into release
* fix changelog, package.xml
* Merge branch 'fix-rpath-libsvm' into 'master'
  Fix installation of 3rd party shared libs
  Closes #29
  See merge request !202
* Merge branch 'ceres_3rd_party' into 'master'
  Include and install Ceres as 3rd party
  See merge request !201
* Add CMake command to install libsvm.so
* Add 3rd party library directory to the CMAKE_INSTALL_RPATH
* fixed old naming issue
* install ceres as 3rd party library
  Because of Ubuntu not fixing a bug in their ceres package [1]  we need to add it as a 3rd party lib.
  [1] https://bugs.launchpad.net/ubuntu/+source/ceres-solver/+bug/1596296
* Merge branch 'remove-include' into 'master'
  Remove useless include
  See merge request !200
* Remove useless include
* Update .gitlab-ci.yml
* Update .gitlab-ci.yml
* update changelog
* Test: Enable deb generation on Ubuntu 16.04
* merge
* Merge branch 'master' into 'master'
  RGB-D cloud filtering using a sequence of clouds, (batch version, average of depth and colour)
  See merge request !197
* Merge branch 'add_semantic_segmentation' into 'master'
  add semantic segmentation module + apps and scripts for NYU Datasets
  See merge request !198
* add semantic segmentation module + apps and scripts for NYU Datasets
  bugfix: renamed one function
  adapted command line arguments
  adapted scripts, moved some files from git to repo server
  Update Readme.md
  Update Readme.md
  removed unused include files (esp. openni2_grabber.h)
  add readme file for semantic segmentation apps
  update scripts and readme
  add matlab scripts for nyu depth dataset conversion
  add apps for semantic_segmentation
  add module semantic_segmentation
* Merge branch 'master' of rgit.acin.tuwien.ac.at:root/v4r
* changed name of tsf and added example
* Merge branch 'master' of rgit.acin.tuwien.ac.at:hannes-prankl/v4r
* tsf batch filtering
* Merge branch 'fix_global_pipeline_param_bug' into 'master'
  Fix global pipeline param bug
  See merge request !196
* Merge branch 'master' of rgit.acin.tuwien.ac.at:root/v4r
* Merge branch 'forward_declare_caffe_net' into 'master'
  Forward declare caffe net
  See merge request !195
* fix bug when global pipeline parameters are not completely defined
* Merge branch 'fix_caffe' into 'master'
  Fix caffe
  Closes #25
  See merge request !194
* comment pcl conflict
* Merge branch 'forward_declare_caffe_net' of rgit.acin.tuwien.ac.at:t-faeulhammer/v4r
* comment definitions
* forward declare caffe net
  avoid caffe include
* Merge branch 'master' of rgit.acin.tuwien.ac.at:root/v4r
* fix missing include paths
* Merge branch 'opencv_fix' into 'master'
  Fix OpenCV Bug for unordered sets
  See merge request !192
* Add Caffe Path hints
* Fix OpenCV Bug for unordered sets
* Merge branch 'master' of rgit.acin.tuwien.ac.at:root/v4r
* bug fix: turn off mapping -> tsf-filtering still collected frames
* Merge branch 'master' into 'master'
  Master
  See merge request !190
* bug fix:
  - TSFGlobalCloudFilteringSimple: set width of the point cloud (if not ec-filtered)
  - example: parameter setting: do not ec-filter
* added modul: camera tracking and mapping (using s temporal smoothing flter - TSF)
* Merge branch 'master' of rgit.acin.tuwien.ac.at:root/v4r
* add multiview keypoint correspondence visualization
* Merge branch 'pcl_mesh_renderer' into 'master'
  Pcl mesh renderer
  See merge request !189
* removed loading of pcl files with pcl::io::loadPLYFile. This method 1. tends to fail and 2. does not exist on the continous integration system.
* Cleanup of the changes. Finally got rid of the flipped coordinate system bug.
* Added a lot of debug code but principially found the bugs causing the projection to be inverted. TODO: remove debug code!!
* Merge branch 'master' of rgit.acin.tuwien.ac.at:root/v4r
* no idea (must be minor)
* Added new constructor accepting pcl meshes.
* update doc
* update doc
* Merge branch 'add_change_detection' into 'master'
  Add change detection
  See merge request !188
* update doc
* update CMakeLists
* remove Eval app
* fix error when reading floating point occlusion values
* avoid running into pcl bug because corr_rejector ransac is not reset
* remove unused method
* make aligned
* :lipstick:
* read training data despite loading model from file
* add more verbosity
* allow to load SVM model from file
* update default param
* update default svm parameter range
* fix seg fault when saving model
* save trained svm to current working directory
* add svm scaling
* allow setting kp_transfer
* allow to sub-sample views to speed up evaluations
* allow to set knn from command line
* allow to do icp outside HV
* write refined pose
* remove condition
* remove opencv sift as it is integrated in sift local estimator
* weight feature vector so they approximately scale to 1
* add some boost program options
* remove unused icp parameter
* update min dist for cluster factor default parameter value
* remove debug
* update parameter
* temp debug info
* remove debugging information again
* remove exit
* bf
* add more debug info
* temporary for debugging
* update
* more info
* :lipstick:
* fix compile error
* up
* temporary verbose logging
* :lipstick:
* up
* update visualization
* update params
* use openmp for zbuffering
* fix min_dist_for_cluster parameter for gc
* add multiview visualization for kp transfer
* avoid redundant recognition rate computations
* use full model for z-buffering in HV
* add timing and do some common pre-checks
* update author order
* :lipstick:
* pre-compute search radius
* set rendering default background-color to white
* add missing pieces for remove non-upright objects
* serialize hypotheses
* add option to remove hypotheses not standing upright on a suppport plane
* do not output confusion matrix twice
* allow subsampling during rendering
* remove unused normal computation for model
  should be done outside
* Merge branch 'fix_normals' into 'master'
  fix nasty bug in normals computation (flip always towards viewpoint)
  See merge request !187
* use auto
* use auto
* revert normal referencing
* add missing declarations
* revert normal referencing
* fix visualization bug when skip verification is on
* use reference
* use reference
* fix nasty bug in normals computation (flip always towards viewpoint)
* fix doxygen comments
* visualize curvature information
* fix background color bug
* allow to save images to disk
* remove leftover for visualiization
* add option in xml
* just to make sure
* add ICRA version of multi-view classifier
* add missing piece for param gc grouping
* get recognition pipelines
* allow to disable correspondence clustering
* make deep copy
* fix bug when views are empty
* add try catch
* :lipstick:
* update params
* add keypoint redundancy check
* use tanh function for xyz score and re-name parameter
* temporarily add some hack to avoid running into bug for outlier cloud visualization
* add some more information for timing
* fix some timing measurement bug
* make shuffle views a parameter
* add reading of computation time
* :lipstick:
* fix bug in recognition rate compuation when dot product becomes 1+eps
* add multiview evaluation
* fix nasty bug in normals computation (flip always towards viewpoint)
* add normals visualization
* fix multiview and icp stuff, add max views parameter
* backup of some old code
* remove some deprecated stuff
* partially fix multi-view registration
* remove unused template parameter
* add xtion depth mask
* add some logs
* add global config
* update params
* :lipstick:
* update rec error computation and visualize errors
* fix visualization
* fix recognition rate bug
* add missing files
* update dependency list
* update color and normals comparison
* make HV param private
* add dependencies
* fix bug when hypothesis does not explain any scene point
* :lipstick:
* remove empty appearance module
* adapt L channel
* make fitness threshold dependend on visibility
* :lipstick:
* put params in seperate file
* :lipstick:
* Merge remote-tracking branch 'root/master' into add_change_detection
* :lipstick:
* update url for siftgpu
* update timings log
* re-arrangechecks to hopefully speed-up a bit
* fix rec file output and add timings
* fix visualization
* reset mv
* improve performance and brevity of noise model based cloud integration
* add copyrights
* integrate change detection again
* add param for mutliview hv
* use seperate parameter class for noise model based registration + fix remaining const madness
* add noise model based cloud integration
  not quite working yet
* :lipstick:
* const madness
* update multiview
  save verification status intrinsically
  correctly transfer hypotheses across views
* add verified member
* make objectrecognizer mv compatible
* update param
* allow online mv rec
* opti model - structure (no impl)
* 1.4.5
* Merge branch 'release' of rgit.acin.tuwien.ac.at:markus-bajones/v4r into release
* update changelog
* set ros distro
* update changelog
* Contributors: Daniel Wolf, Johann Prankl, Markus Bajones, Sergey Alexandrov, Simon Schreiberhuber, Thomas Faeulhammer, Thomas Fäulhammer, bajo

1.4.8 (2017-04-11)
------------------
* Merge remote-tracking branch 'upstream/master' into release
* Merge branch 'update_siftgpu_url' into 'master'
  update url for siftgpu
  Closes #24
  See merge request !186
* update url for siftgpu
* Merge branch 'add_eval' into 'master'
  Add eval
  See merge request !185
* up
* update params
* just to make sure
* :lipstick:
* use inner product for normal comparison
* up
* use pcl_version macro
* update param
* update z-buffering
* use dotproduct for normals comparison
* use dist_l2 enum from opencv 2.4
* remove highgui include
* update params
* use vis params
* use logarithm
* :lipstick:
* use param
* update pose refinement and visible cloud computation
* remove color normalization
* use pcl stopwatch to allow verbosity setting
* use tanh for color comparison
* use right color comparison function
* fix missing header
* reset viewpoint by default to fix visual
* small hack to have consistent number of parameters
* hopefully fixes compile error
* allow inconsistent xml file and just warn
* exit when param not okay
* use planar filter
* use integer for verbosity
* initialize depth registration mask
* add accidentally deleted header declaration
* improve readability
* fix wrong sort direction
* fix some warnings
* use surface normals for z-buffering
* rename parameter
* update hv parameter
* add verbosity
* fix
* fix compile error
* :lipstick:
* add confusion matrix computation to evaluate global pipeline
* reset sensor orientation before segmentation
* use distance instead of z component for cut-off
* updae eval to allow setting multitoken elements
* make remove planes parameter accessible from outside
* use distance instead of z component for cut-off
* save parameters to file
* add initialization
* use RGB transform histogram for global color estimator and add parameter class
* make histogram row major
* use built-in svm cross validation
* fix compile error
* update global recognizer (mainly add visualization)
* update alexnet integration
* update global recognizer to allow for multiple feature descriptors per cluster
  + add visualization option
* add to previous commit
* use eigen vector instead of PointT type as min max point
* add ourcvfh as an option for global concat estimator
* use normal estimator from outside
* make camera a class variable
* remove virtual where not neccessary
* add ourcvfh estimator
* update global pose estimation and add visualization
  TODO: check if everything works correctly
* update desc
* compute discrepancy between centroids of view and centroid of full object model
* compute model centroid
* fix taking wrong norms (should be sqrt)
* use visualization parameter
* add parameter for coordinate axis scale
* add utility function for aligning two vectors
* Merge remote-tracking branch 'root/master' into add_eval
* Merge branch 'add-gtest' into 'master'
  Add GTest
  See merge request !177
* add pose refinement to app
* Merge branch 'add_eval' into 'master'
  Add eval
  See merge request !182
* Merge branch 'fix_iol_runtime_error' into 'master'
  fix throwing runtime_error
  See merge request !181
* Merge branch 'fix_bug' into 'master'
  change include order to fix opencv bug
  Closes #23
  See merge request !180
* fix throwing runtime_error
* fix throwing runtime_error
* update default config (elongation can now be part of the feature vector)
* update header information
* :lipstick:
* awesome global concate descriptor
* remove parameter remnant
* :lipstick:
* fix hist range check
* add color to concat descriptor
* fix type
* add global color descriptor
* add concatenation of global descriptor
* :lipstick:
* add a simple global shape estimator
* fix visualization bug
* remove log copy as glog does not support this apparently
* write function to query boost xml value
* use glog more frequently
* remove volatile remnant
* reset normals
* :lipstick:
* make svm param gamma init to 1/num_featurse by default if not otherwise set
* add unfiltered global results to vis
* make global rec visualization a parameter
* fix coordinate system vis issue
* fix global recognizers plane alignment
* allow to reurn normals from segmentation
* update opencv sift
* use l1 norm
* output recognition rate
* revert accidental commit
* fix wrong endif for rops
* fix sift
* fix rops comile error on pcl < 1.7.2
* allow to visualize keypoints
* update params
* allow having multiple keypoint extractor + change default params
* update xml
* comment try catch
* fix bug
* tmp commit
* :performance:
* use organized normal computation by default
* remove redundant object mask check
* fix sift opengl issue
  TODO: use master thread for sift to use openmp
* allow having multiple support radii
* :lipstick:
* remove keypoint cloud and processed cloud from local estimator
* Merge remote-tracking branch 'root/master' into add_eval
* Merge branch 'add_rotation_check' into 'master'
  Add rotation check
  See merge request !179
* change include order to fix opencv bug
* change include order to fix opencv bug
* remove points not belonging to object during initialization
* visualize normals in processed cloud
* add rops as features
  not tested yet
* make shot support radius a boost paramater
* remove nan points before  shot computation
* remove keypoint indices if the normals are not finite and estimator needs normals
* use omp
* fix segfault
* change default params
* allow setting boundary type
* avoid keypoint cloud copy if not neccessary
* add narf parameters
* fix bug
* remove config files as they are copied all the time anyway
* add parameter for shot and normal estimation
* increase normal densitiy in visualization
* remove check
* change default param
* :lipstick:
* add check if file exists when reading xml files
* use stl vector and :lipstick:
* :lipstick:
* :lipstick:
* :lipstick:
* make normal computation method a parameter
* another try with rootsift
* add missing piece
* add rootsift as parameter
* add l2 normalization
* fix
* try root sift
* revert test
* test
* fix eval stopping criteria
* fix compile error
* fix hash
* fix compile error on pcl 1.8 (take 2)
* fix compile error on pcl 1.8 (take 2)
* fix compile error on pcl 1.8
* fix compile error on pcl 1.8
* add eval
* remove eval
* add missing executable in cmake
* add point types include to gcg to hopefully fix compile error on trusty
* add inidividiual hypotheses check with ground-truth data + :lipstick:
* Merge branch 'add_param' into 'master'
  add angular threshold parameter
  See merge request !176
* temp commit
* compute normals when init features
* allow graph-based correspondence grouping
* use cv::Vec3b
* also use stl vectors for thresholds
* fix not removing already used parameter
* make gcg input const
* make retrain a parameter for outside
* also use 2d stl vector instead of mat for center
  + fix some warnings
* use 2d stl vector instead of cv mat to fix memory leak
* fix compile error
* Add GTest as a third-party library
* add angular threshold parameter
* increase model resolution for visualization
* make compute recognition rate an apps library
* remove plane extractor from global and use segmentation app
* :wrench:
* :sparkles:
* add angular threshold parameter
* Add protection from calling target_include_directories with empty list
* Remove INSTALL_TESTS option from CMake
* Install only modules marked as PUBLIC
* Merge remote-tracking branch 'refs/remotes/root/master'
  Conflicts:
  modules/segmentation/src/plane_utils.cpp
  modules/segmentation/src/segmentation_utils.cpp
* Merge branch 'fix_pcl_visualization_issue' into 'master'
  fix opening multiple pcl visualization windows
  See merge request !173
* Merge branch 'add_simons_plane_extractor' into 'master'
  add simons plane extractor
  See merge request !172
* Merge branch 'add_hellinger_kernel' into 'master'
  add hellinger kernel
  See merge request !171
* init params
* add boost parameter
* remove redundant param init
* init params
* init params
* add boost parameter
* add boost parameter
* :lipstick:
* fix compile error
* cleanup
* fix wrong index
* flip normals always towards viewpoint
* only vector4f now
* temp commit
* temp commit
* temp commit
* tmp commit
* temp commit
* :lipstick:
* remove redundancy
* reduce redundancy
* reduce redundancy
* use plane utility functions
* use global functions
* use const
* remove cam
* add some plane utils
* :lipstick:
* fix compile error
* cleanup
* fix wrong index
* flip normals always towards viewpoint
* only vector4f now
* temp commit
* temp commit
* temp commit
* tmp commit
* temp commit
* :lipstick:
* remove redundancy
* reduce redundancy
* use plane utility functions
* use global functions
* rename config variable name
* use const
* remove cam
* add some plane utils
* add seg params
* add processed cloud to visualization and use original cloud for hv
* update cloud segmenter
* allow title change
* use boost paths
* use boost filesystems paths instead of string
  will be casted anyway
* remove visualization output
* extract plane indices and increase performance
* fix opening multiple pcl visualization windows
  define visualizer static
* enable plane removal by default
* use seperate cloud for processing input
* fix opening multiple pcl visualization windows
  define visualizer static
* extract plane indices and increase performance
* add hellinger kernel
* add simons plane extractor
* update recognizer to use new segmentation with explicit plane extraction
* add hellinger kernel
* add simons plane extractor
* Merged branch master into add_evaluation_for_diss
* Merge branch 'master' into 'master'
  ec filter of small clusters - some beautification
  See merge request !169
* ec filter of small clusters - some beautification
* Merge branch 'update_io' into 'master'
  Update io
  See merge request !167
* Merge branch 'master' into 'master'
  Master
  See merge request !168
* RTMT: remove small clusters before storing the global model
* Merge branch 'master' of rgit.acin.tuwien.ac.at:root/v4r
* use random search for hyperparameter
* prepare for local search
* fix performance bug
* put changes in seperate file and automatically add pairwise changes
* add functionality to set output dir
* add eval in cmake
* do not visualize by default
* fix missing occlusion threshold init
* fix bug in visualization
* init commit
* add xml config for multipipeline
* :lipstick:
* add parameter class for multipipeline recognizer
* add remove folder
* update folder copy
* Merge branch 'up_recognition_rate_computation' into 'master'
  Up recognition rate computation
  See merge request !164
* add chi-square
* add parameter for num trees of kdtree in feature matching
* use object centroid for translation error computation
* make eval compatible to new annotation format
* object models are read directly from folder instead from init source
* Merge branch 'up_io' into 'master'
  remove deprecated functions and make paths windows compatible
  See merge request !165
* Merge branch 'revert_disabling_apps' into 'master'
  revert disabling of some apps and examples
  See merge request !166
* revert disabling of some apps and examples
* Merge branch 'master' of rgit.acin.tuwien.ac.at:root/v4r
* minor: config/ test ImGD-Descriptor
* remove deprecated functions and make paths windows compatible
* Merge branch 'fix_rf_training' into 'master'
  bug fix in random forest training
  See merge request !163
* Merge branch 'rec_eval' into 'master'
  Rec eval
  See merge request !162
* fix pcl version madness
* avoid not available getkeypointsindices in PCL < 1.7.2
* remove conditional pcl instantiation
* lm optimization of the poses (proj. + depth) -- tested
* allow to skip verification
* Merged branch master into normal_estimation
* update keypoint example demo
* fix bug when indices are empty
* update default param
* :lipstick:
* update normal computations
* add z adaptive
* init normal class
* fix noise model based cloud integration when object mask is not available
* fix missing init
* update segmentation
  - use vector<int> instead of pcl::pointindices
  - seperate plane extraction and segmentation
  - rename files
  - move some definitions into .cpp
* tmp commit
* rename variables
* remove siftgpu dependency
* remove pcl 1_8 dependency
* adds example code for keypoint extraction and filtering
* add dependencies
* Merge branch 'add_3rd_partry_libraries_as_example_dependencies' into 'master'
  add dependencies
  See merge request !159
* make segment pcd an app
* fix seg fault when input contains nan points
* fix wrong if conditions
* remove siftgpu dependency
* pnp and/or depth ransac version (not tested)
* Merge branch 'fix_noise_model_base_integration' into 'master'
  fix noise model based cloud integration when object mask is not available
  See merge request !158
* add dependencies
* fix noise model based cloud integration when object mask is not available
* fix redundant typename in keypoint init
* Merged branch master into rename_params
* Merge branch 'master' of rgit.acin.tuwien.ac.at:root/v4r
* live version (capture with opencv)
* bug fix in random forest training
  boost random generator is not thread safe, do not access it from within openmp parallel section
* update segmentation
  - use vector<int> instead of pcl::pointindices
  - seperate plane extraction and segmentation
  - rename files
  - move some definitions into .cpp
* tmp commit
* Merged branch master into rename_params
* rename variables
* :lipstick:
* Contributors: Daniel Wolf, Johann Prankl, Markus Bajones, Sergey Alexandrov, Thomas Faeulhammer, Thomas Fäulhammer

1.4.7 (2017-02-23)
------------------
* Merge branch 'upstream_master' into release
* Merge branch 'fix_wrong_sift_init' into 'master'
  fix missing init
  See merge request !157
* fix missing init
* Merge branch 'master' into 'master'
  added additional parameter for imkRecognizer to specify bin filename
  See merge request !155
* changed imkRecognizer example to handle user filename
* Merge branch 'master' of rgit.acin.tuwien.ac.at:root/v4r
* added additional parameter to set file name for imk-bin-file
* Merge branch 'master' into 'master'
  param change: more accurate flann configuration
  See merge request !154
* param change: more accurate flann
* Contributors: Edith Langer, Johann Prankl, Markus Bajones, Thomas Faeulhammer

1.4.6 (2017-02-14)
------------------
* Merged branch master into release
* remove auto keyword as this causes undesired behavior on some machines
* Contributors: Thomas Fäulhammer, Unknown

1.4.5 (2017-02-13)
------------------
* Merge branch 'release' into 'release'
  Release 1.4.4-internal
  See merge request !152
* Contributors: Markus Bajones

1.4.4 (2017-02-13)
------------------
* Merge branch 'fixed_missing_parameter_init' into 'master'
  fix initialize parameters
  See merge request !151
* Merge branch 'ceres-warning' into 'master'
  Make sure that found Ceres library is a shared object
  See merge request !150
* Make sure that found Ceres library is a shared object
  Otherwise will get relocation error at linker stage.
* fix initialize parameters
* Merge branch 'all_updates' into 'master'
  Segmentation, ML and Keypoint extractor updates
  See merge request !147
* Merge branch 'update_segmentation' into 'master'
  update segmentation
  See merge request !146
* Merge branch 'ci-ceres-shared' into 'master'
  Build Ceres in shared mode on CI server
  See merge request !149
* Build Ceres in shared mode on CI server
* :lipstick:
* update classifiers
  - write a global iniitialization method that allows to initialize any classifier by its type
  - move parameters outside class to not make them templates
* update keypoint extractors
  - write a global iniitialization method that allows to initialize any keypoint extractor by its type
  - move parameters outside class to not make them templates
* move parameter class
* remove segmenter.cpp
* update segmentation
  - write a global iniitialization method that allows to initialize any segmenter by its type
  - use visualization from utils
  - move parameters outside class to not make them templates
* Merge branch 'new_recognizer' into 'master'
  New recognizer
  See merge request !141
* put files into v4r module apps to allow to use it as a library
  fixes alignment issues with potentially wrong library versions of PCL, OpenCV, Eigen, Boost etc
* add pointer check
* add missing header file to cmake
* make classes aligned for fixed-sized vectorizable eigen objects
* Merged branch master into new_recognizer
* Merge branch 'fix_opencv3_issue' into 'master'
  fixes #19 (duplicate base type for cv::Feature2D)
  Closes #19
  See merge request !145
* fix warnings
* Merged branch fix_opencv3_issue into new_recognizer
* fixes #19 (duplicate base type for cv::Feature2D)
  supposed to fix https://rgit.acin.tuwien.ac.at/root/v4r/issues/19
* update readme
* fix some warnings
* do smooth cluster check in global optimization, move visualization option away from param, cleanup
* fix visualization
* move hv visualization in seperate file/class, update hv evaluation
  + cleanup
* add is_outlier visualization
  + :lipstick:
* fix visualization bug
* :lipstick:
* use auto and fix indentation
* make model resolution consistent with scene resolution
* fix model color in vis
* fix filename
* fix compile error
* initial commit for mv recognition
* fix visualization issue for ghv
  + fix warning
* fix missing camera depth registration mask init for xml constructor
* fix hv crash and wrong filename input
* :lipstick:
* read camera parameters from xml file
* remove unused recognition files
* make path windows compatible
* update compute recognition rate
* fix error on PCL 1.7.2
* update annotation format
* fix pcl 1.7 error
* use conditional clustering from 3rdParty folder
* new recognizer
  this commit is too huge to comment... sorry a few months work that I can't split up anymore :-P
* create class for zBuffering, PCL-2-OpenCV converter and Occlusion Reasoning
* use pcl correspondence grouping instead of a copy
* Merge branch 'find-devil' into 'master'
  Look for DevIL library and fail if not found when building SiftGPU
  Closes #20
  See merge request !144
* Look for DevIL library and fail if not found when building SiftGPU
* Merge branch 'master' into 'master'
  IMKRecognizer: create model from indices
  See merge request !142
* Merge branch 'cmake-cleanup' into 'master'
  Cmake cleanup
  See merge request !143
* Remove some occurrences of IOS, WINRT, and ANDROID in CMakeLists
* Remove options and config variables inherited from OpenCV
* Remove unused CHECK_MODULE CMake macro
* Fix defaulting to Release mode
* load indices files (additional to to masks) for modelling
* added temporal filtering to RTMT-modelling
* Merge branch 'update_zBuff_and_OccReasoning' into 'master'
  Update z buff and occ reasoning
  See merge request !138
* Merge branch 'feat' into 'master'
  Feat
  See merge request !137
* Merge branch 'master' into 'master'
  IMKRecognizer: keypoint based, monocular object recognizer
  See merge request !132
* Merge branch 'fix_warning' into 'master'
  fixes shadowed auto variable warning
  See merge request !140
* Merge branch 'add_db_creation_from_plyfiles' into 'master'
  creates a recognition database from ply files in folder
  See merge request !139
* removed debug cout
* merged
* Merge branch 'master' of rgit.acin.tuwien.ac.at:root/v4r
* Merge branch 'fix_some_shadow_warnings' into 'master'
  fix some warnings (shadowed and unused variables)
  See merge request !136
* Merge branch 'update_ml' into 'master'
  make svm class better encapsulated
  See merge request !133
* create class for zBuffering, PCL-2-OpenCV converter and Occlusion Reasoning
* fixes shadowed auto variable warning
* creates a recognition database from ply files in folder
  (used for new recongition database format)
* add pcl visualization parameter class
* Merged branch update_ml into master
* Merged branch feat into master
* Merged branch fix_some_shadow_warnings into master
* add shift histogramm function
  + some :lipstick:
* update pcl serialization
* update camera class to include depth registration mask
* update point cloud properties computation
* use new indices
* add image crop utility function
* some sift updates
* fix some warnings (shadowed variables)
* Merged branch cleanup_segmentation into master
* remove files
* Merged branch cleanup_segmentation into master
* clean up segmentation and add plane utils
  removes specific segmenation example from modules
* make svm class better encapsulated
  put files into utility file, add const
* add cmake file to find vtk (copied from opencv)
* clean up segmentation and add plane utils
  removes specific segmenation example from modules
* Update ObjectDetection.md
* make svm class better encapsulated
  put files into utility file, add const
* Merge branch 'update_pcl-1_8' into 'master'
  put pcl files only availble in PCL 1.8 into 3rdparty PCL folder
  See merge request !128
* Merge branch 'fix_gitignore' into 'master'
  remove docs from blacklist
  Closes #16
  See merge request !131
* Merge branch 'master' of rgit.acin.tuwien.ac.at:root/v4r
* Merge branch 'fix_warnings' into 'master'
  fix some warnings
  See merge request !130
* remove docs from blacklist
* MIT license, colour confidence value, documentation
* MIT license header
* remove unused pcl trunk files
* Merge branch 'fix_opencv3_error' into 'master'
  fix compile error on OpenCV3
  See merge request !129
* Fix compile error
* remove commment
* fix some warnings
* fix some more missing pcl_1_8 bits
* up
* Merge branch 'fix_opencv3_error' into update_pcl-1_8
* fix compile error on OpenCV3
* use uniform sampling from PCL 1.7 since 1.8 doesn't seem to provide indices
* getting mad
* up
* namespace driving me crazy
* add 2d
* some more dashes
* fix some namespace problems
* c++ does not seem to like dashes too much
* put pcl files only availble in PCL 1.8 into 3rdparty PCL folder
* keypoint based monocular object recognizer
* Merge branch 'master' of rgit.acin.tuwien.ac.at:root/v4r
* Merge branch 'update_Cam' into 'master'
  add fov to camera
  See merge request !126
* remove redundant vertical fiel of view member variable
* Merge branch 'master' of rgit.acin.tuwien.ac.at:root/v4r
* Merge branch 'debian_packages' into 'master'
  create debian packages if commit was tagged.
  See merge request !110
* add fov to camera
* Merge branch 'doxy_and_some_minor_beauty' into 'master'
  Doxy and some minor beauty
  See merge request !125
* Merge branch 'use_constptr' into 'master'
  use constptr where appropriate
  See merge request !124
* Merge branch 'fix_uniform_sampling' into 'master'
  make uniform sampling work on PCL versions 1.8+
  See merge request !123
* some doxygen fixes and :lipstick:
* use constptr when neccessary
* make uniform sampling work on PCL versions 1.8+
* Merge branch 'master' of rgit.acin.tuwien.ac.at:root/v4r
* Merge branch 'some_minor_fixes' into 'master'
  Some minor fixes
  See merge request !122
* Merge branch 'ceres_fix' into 'master'
  fix ceres version issue with renamed parameter
  See merge request !121
* add function for principal component analysis (elongation / centroid)
* neglect .autosave files
* fix  quaternion computation in case not normalized
* add focal length to boost program options
* remove openni include
* fix ceres version issue with renamed parameter
* Merge branch 'master' of rgit.acin.tuwien.ac.at:root/v4r
* Merge branch 'feat1' into 'master'
  Feat1
  This adds some functions used for recognition / hypotheses verification
  See merge request !118
* Merge branch 'add_serialization' into 'master'
  Add boost serialization for common pcl types
  See merge request !117
* Merge branch 'add_doxygen_config' into 'master'
  add doxygen config file to be able to generate Doxygen based documentation
  See merge request !116
* Merge branch 'use_pcl_eigen' into 'master'
  remove redundant copy of PCL file
  cleans up v4r repository (redundant PCL copy - PCL is a required dependency anyway)
  See merge request !115
* Merged branch master into debian_packages
* add computeMaskFromImageMap function
* minor code reduction
* add initial histogram equalizer
  not tested yet
* add colorcomparison method enum
* remove unused std_msgs serialization
* add boost serialization for common pcl point cloud types
* some more redundant pcl copies replaced
* add doxygen config file to be able to generate Doxygen based documentation
* fix namespace issue
* replace v4r/common/eigen.h with pcl/eigen.h
* Merge branch 'master' of rgit.acin.tuwien.ac.at:root/v4r
* Merge branch 'warningfix_pcl' into 'master'
  Fixed warnings in PCL
  So, this should fix every warning from external libraries. I did this by adding the necessary SYSTEM parameter to the according "(target_)include_directories" calls. As far as i see it it does not suppress any warnings in our own headers (as intended) but due to my lack of understanding of CMAKE and our V4R structure i suggest somebody might want to test it.
  See merge request !113
* Fixed another mistake... credits goes to Sergey.
* Fixed missing }
* removed comments.
* Being more selective at suppressing warnings.
  Being more selective at suppressing warnings as now only warnings from external header files will be suppressed.
* Merge branch 'warningfix_keypoints' into 'master'
  Fixed some Warnings in Keypoints
  As the title says. I hope i broke no real functionality.
  See merge request !100
* Update V4RUtils.cmake
* Suppress warnings in external headers.
  Added the SYSTEM property to V4RUtils.cmake where it is needed to suppress pcl warnings of external include files.
* fix to lowercase path name
* rename header files to be included in the packaging process
* update changelog
* Merged branch master into debian_packages
* Merge remote-tracking branch 'v4r-master/master'
* Update .gitlab-ci.yml
  Install sed, use it to compile on 8 cores
* Update .gitlab-ci.yml
  Make sure v4r is installed in /usr
* Merged branch master into master
* Merge branch 'attention' into 'master'
  Integrate attention based segmentation Now works on Ubuntu 14.04 and 16.04
  @michael-zillich-1 @msuchi Can I get some feedback before I merge this?
  See merge request !112
* disable deb packages on xenial for now. issue with libceres to blame
* Adaptions for v4r source code for compiling under Ubuntu 16.04.
  + changed include from "cvmath" to <cvmath>
  + changed namespace of isnan to std::isnan
  this is tested for opencv 2.4 and pcl 1.7.2 which have to be set when launchi9ng cmake:
  cmake -DPCL_DIR=<path to pcl> -DOpenCV_DIR=<oath to opencv 2.4>
* Merged origin/attention_segmentation into master
* Merge branch 'Update_Contributing.md' into 'master'
  Update contributing.md
  Added usage of  setup.sh to CONTRIBUTING.md
  See merge request !111
* need to install devscripts
* need to install python-bloom before we can use it
* need to install python-bloom before we can use it
* Update CONTRIBUTING.md
* update gitlab-ci file
* Added dependency installation "how to" to Contributing.md
* create debian packages if commit was tagged.
* Merged branch master into master
* Merge branch 'ubuntu16.04+opencv3' into 'master'
  Ubuntu16.04+opencv3
  @ghalmets
  See merge request !103
* pass 2 parameters to setup. ubuntu and ros codename
* Removed some warnings in keypoints.
* escape variables
* update setup.sh and gitlab-ci.yml to be more generic
* gitlab's lint checker says it is OK now. Let's see.
* fix gitlab syntax after strange merge issue
* Merge remote-tracking branch 'refs/remotes/upstream/master'
  Conflicts:
  .gitlab-ci.yml
* Merged branch master into master
* Merge branch 'bug_transposed_rendering' into 'master'
  Changed camera matrix input and output of the pointcloud generation class. Now i…
  This is the bugfix to: https://rgit.acin.tuwien.ac.at/root/v4r/issues/13
  See merge request !98
* Merge branch 'Install_dependencies' into 'master'
  Adding Setup.sh
  Added Setup.sh to v4r for a more convenient dependency installation.
  .gitlab-ci.yml was edited to use setup.sh to keep the script in the CI loop.
  Workflow:
  `git clone git@my-awesome-v4r-repo`
  `cd v4r`
  `./setup.sh`
  `mkdir build && cd build`
  `cmake ..`
  `make -j8`
  See merge request !96
* Update Readme.md
* Update Readme.md to rgit and added usage of ./setup.sh
* Changed camera matrix input and output of the pointcloud generation class. Now it is not transposed, or does not has to be transposed anymore.
* Update .gitlab-ci.yml
* Added Setup.sh
* Add script for first build
  Installing rosdep and dependencies, building v4r.
* update apps to include all programs for attention based segmentation
* update attention_segmentation module
* Add first sample app for attention based segmentation! Yes it works.
* fix header files
* fix cmake mistake
* add and activate opennurbs and on_nurbs
* shifted around attention_* files
* Merged branch master into master
* fix cmake file
* add opennurbs as build option for V4R
* reflect change of opennurbs directory
* add cmake find file for openNurbs
* moved opennurbs to 3rdparty
* delete autosave file
* change dependencies
* bring in all the files for attention segmentation -HACK
* Update examples after eputils merge
* Small changes because of the eputils merge into attention_segmentation
* We no longer need/have a v4r_eputils module
* move eputils into attention_segmentation
* Add examples for attention based segmentation
* Adapt to new v4r structure for attention based segmentation
  mainly namespace changes, V4R_EXPORTS, etc.
* Add missing files for eputils
* Adapt to new v4r structure
  namespaces, V4R_EXPORTS, etc.
* small changes to bring attention_segmentation into the new v4r structure
* small changes to bring eputils into the new v4r structure
* Inital copy of attention_segmentation from v4r svn
* Inital copy of eputils from v4r svn
* Merge branch 'master' into 'master'
  v4r now compiles with OpenCV 2.x and 3.1 on Ubuntu 14.04 and 16.04
  See merge request !94
* Merge branch 'master' of rgit.acin.tuwien.ac.at:root/v4r
* v4r now compiles with OpenCV 2.x and 3.1 on Ubuntu 14.04 and 16.04
* Merge branch 'master' into 'master'
  temporal filter (tracks pose and integrates several rgb-d frames)
  incl. bug fix: default param in common/occlusion_reasoning.cpp
  See merge request !93
* Merge branch 'master' into 'master'
  Update of Contribution.md with results from Structure Workshop.
  I have merged the minutes of V4R structure workshop into the Contribution.md
  See merge request !92
* Update CONTRIBUTING.md
* Fixed some Typos
* Update CONTRIBUTING.md
* temporal filter (tracks pose and integrates several rgb-d frames)
* Update CONTRIBUTING.md
* Update CONTRIBUTING.md
* Update CONTRIBUTING.md
* Update CONTRIBUTING.md Update description v4r exports
* Update CONTRIBUTING.md minor changes
* Update CONTRIBUTING.md: Formating
* Update CONTRIBUTING.md: added sections: "Structure", "Documentation", and "How to Build V4R?".
* Manually set PCL_APPS_LIBRARY
* Merged branch ubuntu16.04+opencv3 into ubuntu16.04+opencv3
* I think this is it.
* say yes to apt-get. all the time
* No debug symbols installed
* build ceres from source
* We can force the dpkg installation
* handle install with apt-get force
* Next Ubuntu hack
* fix stupid Ubuntu typo
* Introduce hack because Ubuntu
* rosdep really needs sudo. install it.
* No sudo in xenial image
* No sudo in xenial image
* fix ubuntu version. and again.
* fix ubuntu version
* Also build on Ubuntu 16.04
* fix: no default values
* I think this is it.
* say yes to apt-get. all the time
* No debug symbols installed
* build ceres from source
* We can force the dpkg installation
* handle install with apt-get force
* Next Ubuntu hack
* fix stupid Ubuntu typo
* Introduce hack because Ubuntu
* rosdep really needs sudo. install it.
* No sudo in xenial image
* No sudo in xenial image
* fix ubuntu version. and again.
* fix ubuntu version
* Also build on Ubuntu 16.04
* Merge branch 'Test' into 'master'
  Update AUTHORS
  See merge request !89
* Merge branch 'libsvm' into 'master'
  fix libsvm dependency in package.xml
  See merge request !90
* fix libsvm dependency in package.xml
* Update AUTHORS
* Merge branch 'add_boost_dependency' into 'master'
  Update hypotheses_verification.cpp (wtf? I just commited this change)
  See merge request !88
* Update hypotheses_verification.cpp
* Update hypotheses_verification.cpp (wtf? I just commited this change)
* Merge branch 'master' into 'master'
  Master
  See merge request !82
* Merge branch 'add_boost_dependency' into 'master'
  add missing boost dependency
  See merge request !86
* add missing boost dependency
* Go back to use standard ubuntu trusty docker image
  This is easier to support in the future.
* Merged branch master into master
* Merged branch master into master
* Merge branch 'fix_vector_type' into 'master'
  Fix vector type
  See merge request !79
* Update hypotheses_verification.cpp
* Update hypotheses_verification.cpp
* fix vector type
* Merge branch 'master' of rgit.acin.tuwien.ac.at:root/v4r
* fix vector type for new histogram interface
* Update Readme.md
* Merge branch 'master' into 'master'
  Use docker image that has those dependencies already installed
  See merge request !78
* Use docker image that has those dependencies already installed
* Merge branch 'master' into 'master'
  Update .gitlab-ci.yml
  See merge request !77
* Update .gitlab-ci.yml
* Merged branch master into master
* Update .gitlab-ci.yml
* Update .gitlab-ci.yml
* Merge branch 'some_fixes' into 'master'
  Some fixes
  See merge request !56
* Update .gitlab-ci.yml
* Merged branch master into master
* Update .gitlab-ci.yml
  yet again. ;-)
* Update .gitlab-ci.yml
* Merge branch 'master' of rgit.acin.tuwien.ac.at:root/v4r
* put some header defintions into cpp files and remove .hpp files
* Update .gitlab-ci.yml
* put miscellaneous functions into more specific files
* Update v4r_style_guide.md
* fix merge conflict
* Merged branch master into master
* added: only small inline functions
* Update CONTRIBUTING.md
* Update CONTRIBUTING.md
* Update CONTRIBUTING.md
* Merged branch master into master
* added: keep pull requests short
* fixed typo
* fixed typo
* clean up .gitlab-ci.yml
* Merge branch 'master' of rgit.acin.tuwien.ac.at:root/v4r
* add depdendencies description
* Update .gitlab-ci.yml
* Merge branch 'master' of rgit.acin.tuwien.ac.at:root/v4r
* add contributing and style_format doc files
* Update package.xml
  test if this compiles now
* Update .gitlab-ci.yml
  Continue on rosdep errors. Arrrrrr
* Update .gitlab-ci.yml
  fix syntax
* Update .gitlab-ci.yml
  specify the ROS version (needed to resolve packages from package.xml)
* Update .gitlab-ci.yml
  We need wget as well.
* Update .gitlab-ci.yml
* Update .gitlab-ci.yml
  Seems like we need cmake after all
* Add .gitlab-ci.yml
  First try
* update color transformation and comparison
* use boost::dynamic_bitset instead of vector<bool>, add camera class, put some definitions into header files + some code refactoring
* fix roi when reaching boundary
* Merge branch 'master' of rgit.acin.tuwien.ac.at:root/v4r
* Merge branch 'master' into 'master'
  Ubuntu 16.04 compatibility
  See merge request !54
* Merge pull request #67 from strands-project/ubuntu1604_compatibility
  Ubuntu1604 compatibility
* Merge remote-tracking branch 'hannes/master'
* add pcl time header
* Merge remote-tracking branch 'v4r-master/master'
* add bounding box function
* seperate definitions from some header files to reduce compile time
* fix some warnings
* Merge pull request #66 from strands-project/sync_from_gitlab
  Sync from gitlab
* Merge branch 'master' of rgit.acin.tuwien.ac.at:root/v4r
* Merge branch 'new_try' into 'master'
  New try
  See merge request !53
* Merge pull request #64 from strands-project/new_try
  [WIP] New try
* add timing for pose refinement
* update citation file
* put pcl_opencv functions from header into implementation file
* some code optimization
* some changes for compiling with Ubuntu 16.04
* some beauty
* add script for obtaining alexNet CNN
* use const
* make destructors virtual for virtual classes
* remove empty file
* remove broken files
* put test data into directory
* fix existing directory warning
* add docs for recognition
  update get_TUW script
* add script for downloading 3dnet test data
* add doc for RTMT
* include scripts to obtain training data from TUW and 3dNet
* include missing mean substraction in alexnet feature estimation
* update for shape cnn classifier to work
* fix wrong model assembly resolution
* fix compilation errors for eval and app pieces
* fix missing clear of indices when no keypoints are detected
* remove voxelgriddistancetransform method call
* remove default typename in createIndicesFromMask method to allow usage without c++11
* add global hypotheses non-maxima surpression to rejection method
* group hypotheses by segmentation cluster
* add online averaging function
* add hyp stuff (should have been staged earlier)
* remove EDT stuff
* check if all views are trained during initialization (not just if directory exist)
* put boost program options directly into parameter classes, merge ghv with hypotheses verification class
* make seperate table class
* minor fixes for save pose into pcd
* update some visualization functions in recognition
* remove sift based alignment in incremental object learning class
* use new segmentation class and provide combined feature, keypoint and segmentation layer
* hopefully fixing Caffe optional dependency
* up pcl version
* fix compilation error caused by addcoordinatesystem if used with PCL < 1.7.2
* add esf classifier again
* fix typo in openmp call
* fix some warnings
* fix bug in optional dependening on caffe and rendering
* change default params and do not instantiate harris and iss keypoint extractor on PCL versions < 1.7.2 (as keypoint indices is not available for these versions)
* make recognition library dependency to rendering and Caffe optional
* move some hpps into cpps
* skip recongition rate computation if output file already exists
* add nicer visualization for paper
* add todo comment
* add eval for rec rate over occlusion
* fix crop bug in pcl opencv conversion
* fix min fitness threshold scaling
* flip table plane towards viewpoint and make parameter for min points accessible
* make resolution as an integer in mm
* add coordinate system for visualizing recognition results
* fix bug in color conversion
* change default parameter for svm cross validation
* make smooth segmentation parameter scale with depth
* avoid table plane filtering for initialization in local recognizer
* add parameter options for smooth clustering
* add dense SIFT option (not tested yet and only available for SIFTGPU)
* add smooth clustering and linear thresholding of model fitness threshold (with visibliity)
* use multi-plane segmentation for local recognizer to find *heighest* table plane
* fix visualization for recognition rate computation when cloud sensor header is changed
* temporary remove parallel call of recognizer
  QH6205 qhull error (qh_initqhull_start): qh_qh already defined.  Call qh_save_qhull() first
* fix bug in compute recognition rate
* ignore multiview and esf object classifier for now
* make model fitness threshold adaptive to visible ratio (TODO: write a proper function)
* use bigger rendering points in model cues visualization
* fix wrong sigma for AB color components
* remove table plane debug visualization
* rename some recognition init parameters
* reset view to do not mess up visualization in evaluation recognition example
* add option to just compute dominant plane (without clustering)
* fix bug with multiple call to recognize if recognizer is local
* add all the options for initialization
* make local recognizer more modular
* fix bug in knn_sift initialization
* add missing iostream include in classifier.h
* add opencv sift option again (NOTE: Not tested yet)
* remove keypoint rounding stuff in sift
* rewrite local estimator interfaces
* remove redundant files, take into acccount sign ambiguity of eigen vectors for global recognizer
* fix bug with missing normal computation
* migrated feature estimator changes (except eigen matrix). kinda working but only for first test view it seems
* add global recognizer
* add ourcvfh pcl trunk version, fix view all point clouds in folder
* merging svmwrapper, classifier classes, keypoint extractors... still working
* merged many things from rec_optimization_isolated branch (hyp generation still working - verificaiton not)
* add point cloud to eigen matrix conversion with indices
* add ptr for gcg
* remove old comments
* add ptr for gcg
* fix merge conflict
* add vector sort with indices return
* add some histogram functions
* add cielab2rgb transformation
* some code polish in graph based geometric consistency grouping
* avoid some warnings
* add visualization of model keypoints
* fix visualization of correspondences
* remove global requirement for samples to have all modules enabled
  it now only looks for the individual dependency of each sample and compiles just the ones which meet their dependencies
* using parameter class for gcg when gcg is used... small code polish
* addition to previous commit
* fix color conversion compilation error in case of PointXYZ instantiation
* make ghv compile for PointXYZ type instantation as well
* fix error with color retrieval in verification code
* optimize speed
* fix bug in model assembly
* remove parameter output
* speed up verification evaluation by compressing scene fitting matrix
* add recognition rate evaluation
* make it compile for PointXYZ as well
* compute visible model cloud on multiple views
* merge hv_go3d into ghv (not ready yet)
  optimize visible model cloud computation in verification
* add depth image computation in zBuffering class
  (remove XYZRGBA template instantition)
* split code into more functions, add omp sections again, and some minor beauty
* add replace moves again by checking pairwise intersection matrix
* enhance pairwise intersection computation by fixing smoothing, speeding up computation and adding erosion
* remove some more obsolete code
* use new verification cost function and remove obsolete code pieces
* add smoothing function to zbuffering (does not work properly though)
* add function to remove column or row from eigen matrix
* fix compiler error in change detection module
* implement pairwise intersection computation in verification algorithm
* add rendering function in zbuffering (explicit)
* use local point color to compare color
* delete obsolet member variables
* delete count active hypotheses function in verificitation (as it is not used anyway)
* make update function use member variables instead of having to pass them as an argument
* do not use weights for outliers - just ratio of number of outliers compared to visible points
* clip noise model based radius for inliers search
* rename variable and do label check earlier to avoid redundant processing
* fix seg fault when not using icp for pose refinement
* reset camera view point in object recognizer to avoid messing up visualization
* clip max viewing angle in noise model to 80 degrees to avoid huge noise terms (was 85)
* use noise model for model explained points
* fix ignore color even if exists check
* fix wrong use of row and column counter in self zbuffering module
* do incremental smooth clustering via noise model (not finished yet)
* make visualize go cues a switch parameter
* add a static function to query noise level for a single point
* temp commit
* Integration of change detection into recognition module
* Annotation of changes in GT data
* Change detection module added
* Compilation fix: duplicated pragma
* add merge for multiview
* normalize optimization variables
* fix multipipeline merging of hypotheses when disabled. Also skip merging of ident hypothesis
* fixed self occlusion reasoning
* add pose refinement
  fix noise model based cloud integration for just one input cloud as well as for no indices given
* fixed points on plane side variable in ghv
* working again
* explained and unexplained points seem okay
* fix merge conflict
* fix merge conflict
* Merge pull request #63 from taketwo/remove-x86
  Remove all mentions of x86 and x86_64 in CMake scripts
* Remove all mentions of x86 and x86_64 in CMake scripts
* Merge branch 'master' of rgit.acin.tuwien.ac.at:root/v4r
* use object indices also for unfiltered registered model cloud and only save filtered input clouds if debug option is set
* Merge branch 'master' into 'master'
  added quick fixed to handle some range check exceptions
  needs proper handling soon
  See merge request !51
* 1.3.3
* 1.3.2
* Merge remote-tracking branch 'upstream/master'
* add missing Xxf86vm lib
* Merge remote-tracking branch 'remotes/upstream/recognition_update'
* Contributors: Georg, Georg Halmetschlager-Funek, Johann Prankl, Markus Bajones, Markus Suchi, Martin Velas, Michael Zillich, Sergey Alexandrov, Simon Schreiberhuber, Thomas Faeulhammer, Thomas Fäulhammer

* Merge branch 'attention' into 'master'
  Integrate attention based segmentation Now works on Ubuntu 14.04 and 16.04
  @michael-zillich-1 @msuchi Can I get some feedback before I merge this?
  See merge request !112
* Adaptions for v4r source code for compiling under Ubuntu 16.04.
  + changed include from "cvmath" to <cvmath>
  + changed namespace of isnan to std::isnan
  this is tested for opencv 2.4 and pcl 1.7.2 which have to be set when launchi9ng cmake:
  cmake -DPCL_DIR=<path to pcl> -DOpenCV_DIR=<oath to opencv 2.4>
* Merged origin/attention_segmentation into master
* Merge branch 'Update_Contributing.md' into 'master'
  Update contributing.md
  Added usage of  setup.sh to CONTRIBUTING.md
  See merge request !111
* Update CONTRIBUTING.md
* Added dependency installation "how to" to Contributing.md
* Merged branch master into master
* Merge branch 'ubuntu16.04+opencv3' into 'master'
  Ubuntu16.04+opencv3
  @ghalmets
  See merge request !103
* pass 2 parameters to setup. ubuntu and ros codename
* escape variables
* update setup.sh and gitlab-ci.yml to be more generic
* gitlab's lint checker says it is OK now. Let's see.
* fix gitlab syntax after strange merge issue
* Merge remote-tracking branch 'refs/remotes/upstream/master'
  Conflicts:
  .gitlab-ci.yml
* Merge branch 'bug_transposed_rendering' into 'master'
  Changed camera matrix input and output of the pointcloud generation class. Now i…
  This is the bugfix to: https://rgit.acin.tuwien.ac.at/root/v4r/issues/13
  See merge request !98
* Merge branch 'Install_dependencies' into 'master'
  Adding Setup.sh
  Added Setup.sh to v4r for a more convenient dependency installation.
  .gitlab-ci.yml was edited to use setup.sh to keep the script in the CI loop.
  Workflow:
  `git clone git@my-awesome-v4r-repo`
  `cd v4r`
  `./setup.sh`
  `mkdir build && cd build`
  `cmake ..`
  `make -j8`
  See merge request !96
* Update Readme.md
* Update Readme.md to rgit and added usage of ./setup.sh
* Changed camera matrix input and output of the pointcloud generation class. Now it is not transposed, or does not has to be transposed anymore.
* Update .gitlab-ci.yml
* Added Setup.sh
* Add script for first build
  Installing rosdep and dependencies, building v4r.
* update apps to include all programs for attention based segmentation
* update attention_segmentation module
* Add first sample app for attention based segmentation! Yes it works.
* fix header files
* fix cmake mistake
* add and activate opennurbs and on_nurbs
* shifted around attention\_* files
* Merged branch master into master
* fix cmake file
* add opennurbs as build option for V4R
* reflect change of opennurbs directory
* add cmake find file for openNurbs
* moved opennurbs to 3rdparty
* delete autosave file
* change dependencies
* bring in all the files for attention segmentation -HACK
* Update examples after eputils merge
* Small changes because of the eputils merge into attention_segmentation
* We no longer need/have a v4r_eputils module
* move eputils into attention_segmentation
* Add examples for attention based segmentation
* Adapt to new v4r structure for attention based segmentation
  mainly namespace changes, V4R_EXPORTS, etc.
* Add missing files for eputils
* Adapt to new v4r structure
  namespaces, V4R_EXPORTS, etc.
* small changes to bring attention_segmentation into the new v4r structure
* small changes to bring eputils into the new v4r structure
* Inital copy of attention_segmentation from v4r svn
* Inital copy of eputils from v4r svn
* Merge branch 'master' into 'master'
  v4r now compiles with OpenCV 2.x and 3.1 on Ubuntu 14.04 and 16.04
  See merge request !94
* v4r now compiles with OpenCV 2.x and 3.1 on Ubuntu 14.04 and 16.04
* Merge branch 'master' into 'master'
  temporal filter (tracks pose and integrates several rgb-d frames)
  incl. bug fix: default param in common/occlusion_reasoning.cpp
  See merge request !93
* Merge branch 'master' into 'master'
  Update of Contribution.md with results from Structure Workshop.
  I have merged the minutes of V4R structure workshop into the Contribution.md
  See merge request !92
* Update CONTRIBUTING.md
* Fixed some Typos
* Update CONTRIBUTING.md
* temporal filter (tracks pose and integrates several rgb-d frames)
* Update CONTRIBUTING.md
* Update CONTRIBUTING.md
* Update CONTRIBUTING.md
* Update CONTRIBUTING.md Update description v4r exports
* Update CONTRIBUTING.md minor changes
* Update CONTRIBUTING.md: Formating
* Update CONTRIBUTING.md: added sections: "Structure", "Documentation", and "How to Build V4R?".
* Manually set PCL_APPS_LIBRARY
* Merged branch ubuntu16.04+opencv3 into ubuntu16.04+opencv3
* I think this is it.
* say yes to apt-get. all the time
* No debug symbols installed
* build ceres from source
* We can force the dpkg installation
* handle install with apt-get force
* Next Ubuntu hack
* fix stupid Ubuntu typo
* Introduce hack because Ubuntu
* rosdep really needs sudo. install it.
* No sudo in xenial image
* No sudo in xenial image
* fix ubuntu version. and again.
* fix ubuntu version
* Also build on Ubuntu 16.04
* fix: no default values
* I think this is it.
* say yes to apt-get. all the time
* No debug symbols installed
* build ceres from source
* We can force the dpkg installation
* handle install with apt-get force
* Next Ubuntu hack
* fix stupid Ubuntu typo
* Introduce hack because Ubuntu
* rosdep really needs sudo. install it.
* No sudo in xenial image
* No sudo in xenial image
* fix ubuntu version. and again.
* fix ubuntu version
* Also build on Ubuntu 16.04
* Merge branch 'Test' into 'master'
  Update AUTHORS
  See merge request !89
* Merge branch 'libsvm' into 'master'
  fix libsvm dependency in package.xml
  See merge request !90
* fix libsvm dependency in package.xml
* Update AUTHORS
* Merge branch 'add_boost_dependency' into 'master'
  Update hypotheses_verification.cpp (wtf? I just commited this change)
  See merge request !88
* Update hypotheses_verification.cpp
* Update hypotheses_verification.cpp (wtf? I just commited this change)
* Merge branch 'master' into 'master'
  Master
  See merge request !82
* Merge branch 'add_boost_dependency' into 'master'
  add missing boost dependency
  See merge request !86
* add missing boost dependency
* Go back to use standard ubuntu trusty docker image
  This is easier to support in the future.
* Merged branch master into master
* Merged branch master into master
* Merge branch 'fix_vector_type' into 'master'
  Fix vector type
  See merge request !79
* Update hypotheses_verification.cpp
* Update hypotheses_verification.cpp
* fix vector type
* Merge branch 'master' of rgit.acin.tuwien.ac.at:root/v4r
* fix vector type for new histogram interface
* Update Readme.md
* Merge branch 'master' into 'master'
  Use docker image that has those dependencies already installed
  See merge request !78
* Use docker image that has those dependencies already installed
* Merge branch 'master' into 'master'
  Update .gitlab-ci.yml
  See merge request !77
* Update .gitlab-ci.yml
* Merged branch master into master
* Update .gitlab-ci.yml
* Update .gitlab-ci.yml
* Merge branch 'some_fixes' into 'master'
  Some fixes
  See merge request !56
* Update .gitlab-ci.yml
* Merged branch master into master
* Update .gitlab-ci.yml
  yet again. ;-)
* Update .gitlab-ci.yml
* Merge branch 'master' of rgit.acin.tuwien.ac.at:root/v4r
* put some header defintions into cpp files and remove .hpp files
* Update .gitlab-ci.yml
* put miscellaneous functions into more specific files
* Update v4r_style_guide.md
* fix merge conflict
* Merged branch master into master
* added: only small inline functions
* Update CONTRIBUTING.md
* Update CONTRIBUTING.md
* Update CONTRIBUTING.md
* Merged branch master into master
* added: keep pull requests short
* fixed typo
* fixed typo
* clean up .gitlab-ci.yml
* Merge branch 'master' of rgit.acin.tuwien.ac.at:root/v4r
* add depdendencies description
* Update .gitlab-ci.yml
* Merge branch 'master' of rgit.acin.tuwien.ac.at:root/v4r
* add contributing and style_format doc files
* Update package.xml
  test if this compiles now
* Update .gitlab-ci.yml
  Continue on rosdep errors. Arrrrrr
* Update .gitlab-ci.yml
  fix syntax
* Update .gitlab-ci.yml
  specify the ROS version (needed to resolve packages from package.xml)
* Update .gitlab-ci.yml
  We need wget as well.
* Update .gitlab-ci.yml
* Update .gitlab-ci.yml
  Seems like we need cmake after all
* Add .gitlab-ci.yml
  First try
* update color transformation and comparison
* use boost::dynamic_bitset instead of vector<bool>, add camera class, put some definitions into header files + some code refactoring
* fix roi when reaching boundary
* Merge branch 'master' of rgit.acin.tuwien.ac.at:root/v4r
* Merge branch 'master' into 'master'
  Ubuntu 16.04 compatibility
  See merge request !54
* Merge pull request #67 from strands-project/ubuntu1604_compatibility
  Ubuntu1604 compatibility
* Merge remote-tracking branch 'hannes/master'
* add pcl time header
* Merge remote-tracking branch 'v4r-master/master'
* add bounding box function
* seperate definitions from some header files to reduce compile time
* fix some warnings
* Merge pull request #66 from strands-project/sync_from_gitlab
  Sync from gitlab
* Merge branch 'master' of rgit.acin.tuwien.ac.at:root/v4r
* Merge branch 'new_try' into 'master'
  New try
  See merge request !53
* Merge pull request #64 from strands-project/new_try
  [WIP] New try
* add timing for pose refinement
* update citation file
* put pcl_opencv functions from header into implementation file
* some code optimization
* some changes for compiling with Ubuntu 16.04
* some beauty
* add script for obtaining alexNet CNN
* use const
* make destructors virtual for virtual classes
* remove empty file
* remove broken files
* put test data into directory
* fix existing directory warning
* add docs for recognition
  update get_TUW script
* add script for downloading 3dnet test data
* add doc for RTMT
* include scripts to obtain training data from TUW and 3dNet
* include missing mean substraction in alexnet feature estimation
* update for shape cnn classifier to work
* fix wrong model assembly resolution
* fix compilation errors for eval and app pieces
* fix missing clear of indices when no keypoints are detected
* remove voxelgriddistancetransform method call
* remove default typename in createIndicesFromMask method to allow usage without c++11
* add global hypotheses non-maxima surpression to rejection method
* group hypotheses by segmentation cluster
* add online averaging function
* add hyp stuff (should have been staged earlier)
* remove EDT stuff
* check if all views are trained during initialization (not just if directory exist)
* put boost program options directly into parameter classes, merge ghv with hypotheses verification class
* make seperate table class
* minor fixes for save pose into pcd
* update some visualization functions in recognition
* remove sift based alignment in incremental object learning class
* use new segmentation class and provide combined feature, keypoint and segmentation layer
* hopefully fixing Caffe optional dependency
* up pcl version
* fix compilation error caused by addcoordinatesystem if used with PCL < 1.7.2
* add esf classifier again
* fix typo in openmp call
* fix some warnings
* fix bug in optional dependening on caffe and rendering
* change default params and do not instantiate harris and iss keypoint extractor on PCL versions < 1.7.2 (as keypoint indices is not available for these versions)
* make recognition library dependency to rendering and Caffe optional
* move some hpps into cpps
* skip recongition rate computation if output file already exists
* add nicer visualization for paper
* add todo comment
* add eval for rec rate over occlusion
* fix crop bug in pcl opencv conversion
* fix min fitness threshold scaling
* flip table plane towards viewpoint and make parameter for min points accessible
* make resolution as an integer in mm
* add coordinate system for visualizing recognition results
* fix bug in color conversion
* change default parameter for svm cross validation
* make smooth segmentation parameter scale with depth
* avoid table plane filtering for initialization in local recognizer
* add parameter options for smooth clustering
* add dense SIFT option (not tested yet and only available for SIFTGPU)
* add smooth clustering and linear thresholding of model fitness threshold (with visibliity)
* use multi-plane segmentation for local recognizer to find *heighest* table plane
* fix visualization for recognition rate computation when cloud sensor header is changed
* temporary remove parallel call of recognizer
  QH6205 qhull error (qh_initqhull_start): qh_qh already defined.  Call qh_save_qhull() first
* fix bug in compute recognition rate
* ignore multiview and esf object classifier for now
* make model fitness threshold adaptive to visible ratio (TODO: write a proper function)
* use bigger rendering points in model cues visualization
* fix wrong sigma for AB color components
* remove table plane debug visualization
* rename some recognition init parameters
* reset view to do not mess up visualization in evaluation recognition example
* add option to just compute dominant plane (without clustering)
* fix bug with multiple call to recognize if recognizer is local
* add all the options for initialization
* make local recognizer more modular
* fix bug in knn_sift initialization
* add missing iostream include in classifier.h
* add opencv sift option again (NOTE: Not tested yet)
* remove keypoint rounding stuff in sift
* rewrite local estimator interfaces
* remove redundant files, take into acccount sign ambiguity of eigen vectors for global recognizer
* fix bug with missing normal computation
* migrated feature estimator changes (except eigen matrix). kinda working but only for first test view it seems
* add global recognizer
* add ourcvfh pcl trunk version, fix view all point clouds in folder
* merging svmwrapper, classifier classes, keypoint extractors... still working
* merged many things from rec_optimization_isolated branch (hyp generation still working - verificaiton not)
* add point cloud to eigen matrix conversion with indices
* add ptr for gcg
* remove old comments
* add ptr for gcg
* fix merge conflict
* add vector sort with indices return
* add some histogram functions
* add cielab2rgb transformation
* some code polish in graph based geometric consistency grouping
* avoid some warnings
* add visualization of model keypoints
* fix visualization of correspondences
* remove global requirement for samples to have all modules enabled
  it now only looks for the individual dependency of each sample and compiles just the ones which meet their dependencies
* using parameter class for gcg when gcg is used... small code polish
* addition to previous commit
* fix color conversion compilation error in case of PointXYZ instantiation
* make ghv compile for PointXYZ type instantation as well
* fix error with color retrieval in verification code
* optimize speed
* fix bug in model assembly
* remove parameter output
* speed up verification evaluation by compressing scene fitting matrix
* add recognition rate evaluation
* make it compile for PointXYZ as well
* compute visible model cloud on multiple views
* merge hv_go3d into ghv (not ready yet)
  optimize visible model cloud computation in verification
* add depth image computation in zBuffering class
  (remove XYZRGBA template instantition)
* split code into more functions, add omp sections again, and some minor beauty
* add replace moves again by checking pairwise intersection matrix
* enhance pairwise intersection computation by fixing smoothing, speeding up computation and adding erosion
* remove some more obsolete code
* use new verification cost function and remove obsolete code pieces
* add smoothing function to zbuffering (does not work properly though)
* add function to remove column or row from eigen matrix
* fix compiler error in change detection module
* implement pairwise intersection computation in verification algorithm
* add rendering function in zbuffering (explicit)
* use local point color to compare color
* delete obsolet member variables
* delete count active hypotheses function in verificitation (as it is not used anyway)
* make update function use member variables instead of having to pass them as an argument
* do not use weights for outliers - just ratio of number of outliers compared to visible points
* clip noise model based radius for inliers search
* rename variable and do label check earlier to avoid redundant processing
* fix seg fault when not using icp for pose refinement
* reset camera view point in object recognizer to avoid messing up visualization
* clip max viewing angle in noise model to 80 degrees to avoid huge noise terms (was 85)
* use noise model for model explained points
* fix ignore color even if exists check
* fix wrong use of row and column counter in self zbuffering module
* do incremental smooth clustering via noise model (not finished yet)
* make visualize go cues a switch parameter
* add a static function to query noise level for a single point
* temp commit
* Integration of change detection into recognition module
* Annotation of changes in GT data
* Change detection module added
* Compilation fix: duplicated pragma
* add merge for multiview
* normalize optimization variables
* fix multipipeline merging of hypotheses when disabled. Also skip merging of ident hypothesis
* fixed self occlusion reasoning
* add pose refinement
  fix noise model based cloud integration for just one input cloud as well as for no indices given
* fixed points on plane side variable in ghv
* working again
* explained and unexplained points seem okay
* fix merge conflict
* fix merge conflict
* Merge pull request #63 from taketwo/remove-x86
  Remove all mentions of x86 and x86_64 in CMake scripts
* Remove all mentions of x86 and x86_64 in CMake scripts
* Merge branch 'master' of rgit.acin.tuwien.ac.at:root/v4r
* use object indices also for unfiltered registered model cloud and only save filtered input clouds if debug option is set
* Merge branch 'master' into 'master'
  added quick fixed to handle some range check exceptions
  needs proper handling soon
  See merge request !51
* 1.3.3
* 1.3.2
* Merge remote-tracking branch 'upstream/master'
* add missing Xxf86vm lib
* Merge remote-tracking branch 'remotes/upstream/recognition_update'
* Contributors: Georg, Georg Halmetschlager-Funek, Johann Prankl, Markus Bajones, Markus Suchi, Martin Velas, Michael Zillich, Sergey Alexandrov, Simon Schreiberhuber, Thomas Fäulhammer

1.4.3 (2016-02-26)
------------------

1.4.2 (2016-02-26)
------------------
* Merge pull request `#60 <https://github.com/strands-project/v4r/issues/60>`_ from strands-project/strands
  some quick fixes regarding range check exceptions, need proper fix eventually
* Merge branch 'master' of github.com:strands-project/v4r into strands
* added quick fixed to handle some range check exceptions
  needs proper handling soon
* Merge pull request `#59 <https://github.com/strands-project/v4r/issues/59>`_ from strands-project/fix_range_error_when_using_hv_use_histogram_specification
  Update ghv.h
* Update ghv.h
* Contributors: Michael Zillich, Thomas Fäulhammer, mzillich

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
