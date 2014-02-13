set(SUBSYS_NAME registration)
set(LIB_NAME ${PROJECT_NAME}_${SUBSYS_NAME})
set(SUBSYS_DESC "FAAT registration")
SET(FOO_LIBRARY_TYPE SHARED)

if(FAAT_PCL_USE_CUDA)

    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include
                        ../utils/include ../recognition/include
                        ../utils/
                        ${CUDA_TOOLKIT_INCLUDE}
                        ${CMAKE_CURRENT_SOURCE_DIR}/../3d_rec_framework/include)

    set(registration_incs
      include/faat_pcl/${SUBSYS_NAME}/icp_with_gc.h
      include/faat_pcl/${SUBSYS_NAME}/fast_icp_with_gc.h
      include/faat_pcl/${SUBSYS_NAME}/visibility_reasoning.h
      include/faat_pcl/${SUBSYS_NAME}/registration_utils.h
      include/faat_pcl/${SUBSYS_NAME}/mv_lm_icp.h
      include/faat_pcl/${SUBSYS_NAME}/lm_icp.h
    )

    set(registration_impl_incs
      include/faat_pcl/${SUBSYS_NAME}/impl/icp_with_gc.hpp
      include/faat_pcl/${SUBSYS_NAME}/impl/fast_icp_with_gc.hpp
      include/faat_pcl/${SUBSYS_NAME}/impl/visibility_reasoning.hpp
    )

    set(registration_srcs
      src/icp_with_gc.cpp
      src/visibility_reasoning.cpp
      src/lm_icp.cpp
      src/mv_lm_icp.cpp
      src/fast_icp_with_gc.cpp
    )

    cuda_add_library(${PROJECT_NAME}_cuda_registration SHARED src/cuda/icp_with_gc.cu include/faat_pcl/${SUBSYS_NAME}/cuda/icp_with_gc.h)

    add_library(${LIB_NAME} ${FOO_LIBRARY_TYPE} ${registration_srcs} ${registration_impl_incs} ${registration_incs})
    target_link_libraries(${LIB_NAME} ${PCL_LIBRARIES} ${PROJECT_NAME}_utils ${PROJECT_NAME}_recognition ${PROJECT_NAME}_cuda_registration levmar splm boost_regex)

    #add_executable(icp_with_gc test/icp_with_gc.cpp)
    #target_link_libraries(icp_with_gc ${PCL_LIBRARIES} faat_pcl_utils faat_pcl_registration boost_regex)

    add_executable(fast_icp_with_gc test/fast_icp_with_gc.cpp)
    target_link_libraries(fast_icp_with_gc ${PROJECT_NAME}_registration boost_regex)

    add_executable(icp_nl test/icp_lm.cpp)
    target_link_libraries(icp_nl ${PCL_LIBRARIES} ${PROJECT_NAME}_utils ${PROJECT_NAME}_registration boost_regex)

    add_executable(icp_nl_multi_view test/multi_view_icp_lm.cpp)
    target_link_libraries(icp_nl_multi_view ${PCL_LIBRARIES} ${PROJECT_NAME}_utils ${PROJECT_NAME}_registration boost_regex)

    add_executable(icp_nl_multi_view_rgb test/multi_view_icp_lm_rgb.cpp)
    target_link_libraries(icp_nl_multi_view_rgb ${PCL_LIBRARIES} ${PROJECT_NAME}_utils ${PROJECT_NAME}_registration boost_regex)

else(FAAT_PCL_USE_CUDA)
	
	include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include 
                    ../utils/include ../recognition/include
                    ../utils/
                    ${CMAKE_CURRENT_SOURCE_DIR}/../3d_rec_framework/include) 

	set(registration_incs
	  include/faat_pcl/${SUBSYS_NAME}/visibility_reasoning.h
	)
	
	set(registration_impl_incs
          include/faat_pcl/${SUBSYS_NAME}/impl/icp_with_gc.hpp
	  include/faat_pcl/${SUBSYS_NAME}/impl/visibility_reasoning.hpp
	)
	
	set(registration_srcs
	  src/visibility_reasoning.cpp
	  src/lm_icp.cpp
	  src/mv_lm_icp.cpp
	  src/fast_icp_with_gc.cpp
	)
		       
	add_library(${LIB_NAME} ${FOO_LIBRARY_TYPE} ${registration_srcs} ${registration_impl_incs} ${registration_incs})
        target_link_libraries(${LIB_NAME} ${PCL_LIBRARIES} ${PROJECT_NAME}_utils ${PROJECT_NAME}_recognition levmar splm boost_regex)
	
	#add_executable(fast_icp_with_gc test/fast_icp_with_gc.cpp)  
	#target_link_libraries(fast_icp_with_gc ${PCL_LIBRARIES} faat_pcl_utils faat_pcl_registration boost_regex)
	
	add_executable(icp_nl test/icp_lm.cpp)  
        target_link_libraries(icp_nl ${PCL_LIBRARIES} ${PROJECT_NAME}_utils ${PROJECT_NAME}_registration boost_regex)
	
	add_executable(icp_nl_multi_view test/multi_view_icp_lm.cpp)  
        target_link_libraries(icp_nl_multi_view ${PCL_LIBRARIES} ${PROJECT_NAME}_utils ${PROJECT_NAME}_registration boost_regex)
	
	add_executable(icp_nl_multi_view_rgb test/multi_view_icp_lm_rgb.cpp)  
        target_link_libraries(icp_nl_multi_view_rgb ${PCL_LIBRARIES} ${PROJECT_NAME}_utils ${PROJECT_NAME}_registration boost_regex)
	
endif(FAAT_PCL_USE_CUDA)
