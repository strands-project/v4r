set(the_description "I/O")

v4r_add_module(io v4r_core)

v4r_module_include_directories(${Boost_INCLUDE_DIRS})
v4r_glob_module_sources()

v4r_create_module(${Boost_LIBRARIES})

