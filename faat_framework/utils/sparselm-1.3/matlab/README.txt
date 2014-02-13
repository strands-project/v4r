This directory contains a matlab MEX interface to sparseLM. This interface
has been tested with Matlab v. 6.5 R13 under linux and v. 7.4 R2007 under Windows.

FILES
The following files are included:
sparselm.c:       C MEX-file for sparseLM
sparselm_spsolvr: C MEX-file for mapping sparse solver names to integers
Makefile: UNIX makefile for compiling the above sources using mex.
          Linking only against CHOLMOD is provided, modify
          appropriately to include more sparse solvers.
sparselm.m: Documentation for the sparselm MEX-file.
splmdemo.m: Demonstration of using the MEX interface; run as matlab < splmdemo.m

*.m: Matlab functions implementing various objective functions and their Jacobians.
     For instance, chRosen.m implements the objective function for Rosenbrock's chained
     problem and jacchRosen.m implements its Jacobian.



COMPILING
- On Linux/Unix, use the provided Makefile. For matlab versions 7.3 or later, 
  variable HAVE_LARGE_ARRAYS should be set to "yes" in the Makefile.

- Under Windows, use the provided Makefile.win as a basis. For matlab versions
  7.3 or later, variable HAVE_LARGE_ARRAYS should be set to "yes" in the Makefile.

TESTING
After compiling, execute splmdemo.m with matlab < splmdemo.m 
