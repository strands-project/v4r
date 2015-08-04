set(the_description "I/O")

v4r_add_module(io REQUIRED v4r_core OPTIONAL boost eigen)
v4r_module_include_directories()
v4r_glob_module_sources()
v4r_create_module()

