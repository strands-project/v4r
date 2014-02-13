include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include 
                    ${CMAKE_CURRENT_SOURCE_DIR}/../3d_rec_framework/include 
                    ${CMAKE_CURRENT_SOURCE_DIR}/../recognition/include
                    ${CMAKE_CURRENT_SOURCE_DIR}/../registration/include
                    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/
                    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/include)

add_executable(create_class_structure create_class_structure.cpp)
target_link_libraries(create_class_structure ${PCL_LIBRARIES} faat_pcl_recognition faat_pcl_utils faat_pcl_3d_rec_framework)


