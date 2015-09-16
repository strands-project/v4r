set(the_description "TomGine")
add_definitions(-DLINUX)
add_definitions(-DGL_GLEXT_PROTOTYPES)
v4r_define_module(tomgine REQUIRED v4r_core pcl opencv assimp boost opengl)
