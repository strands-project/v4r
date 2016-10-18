set(ON_NURBS_INCLUDES
	closing_boundary.h
	fitting_curve_2d_apdm.h
	fitting_curve_2d_asdm.h
	fitting_curve_2d_atdm.h
	fitting_curve_2d_pdm.h
	fitting_curve_2d_sdm.h
	fitting_curve_2d_tdm.h
	fitting_curve_2d.h
	fitting_curve_pdm.h
	fitting_cylinder_pdm.h
	fitting_sphere_pdm.h
	fitting_surface_im.h
	fitting_surface_pdm.h
	fitting_surface_tdm.h
	global_optimization_pdm.h
	global_optimization_tdm.h
	nurbs_data.h
	nurbs_solve.h
	nurbs_tools.h
	sequential_fitter.h
	sparse_mat.h
	triangulation.h)

set(ON_NURBS_SOURCES
	closing_boundary.cpp
	fitting_curve_2d_apdm.cpp
	fitting_curve_2d_asdm.cpp
	fitting_curve_2d_atdm.cpp
	fitting_curve_2d_pdm.cpp
	fitting_curve_2d_sdm.cpp
	fitting_curve_2d_tdm.cpp
	fitting_curve_2d.cpp
	fitting_curve_pdm.cpp
	fitting_cylinder_pdm.cpp
	fitting_sphere_pdm.cpp
	fitting_surface_im.cpp
	fitting_surface_pdm.cpp
	fitting_surface_tdm.cpp
	global_optimization_pdm.cpp
	global_optimization_tdm.cpp
	nurbs_tools.cpp
	sequential_fitter.cpp
	sparse_mat.cpp
	triangulation.cpp)
	
SET(USE_UMFPACK 0 CACHE BOOL "Use UmfPack for solving sparse systems of equations (e.g. in surface/on_nurbs)" )
IF(USE_UMFPACK)
	set(ON_NURBS_SOURCES ${ON_NURBS_SOURCES} nurbs_solve_umfpack.cpp)
	set(ON_NURBS_LIBRARIES ${ON_NURBS_LIBRARIES} cholmod umfpack)
ELSE(USE_UMFPACK)
	set(ON_NURBS_SOURCES ${ON_NURBS_SOURCES} nurbs_solve_eigen.cpp)
ENDIF(USE_UMFPACK)

