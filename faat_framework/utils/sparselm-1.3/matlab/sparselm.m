% SPARSELM  matlab MEX interface to the sparseLM sparse non-linear least squares minimization
% library available from http://www.ics.forth.gr/~lourakis/sparseLM/
% 
% Usage: sparselm can be used in any of the following ways:
% [ret, popt, info]=sparselm(fname, jacname, 'an', p0, ncnst, x, jnnz, jtjnnz, itmax, opts, ...)
% [ret, popt, info]=sparselm(fname, jacname, 'zp', p0, ncnst, x, jnnz, jtjnnz, itmax, opts, ...)
% [ret, popt, info]=sparselm(fname, jacname, 'anzp', S, p0, ncnst, x, jnnz, jtjnnz, itmax, opts, ...)
% [ret, popt, info]=sparselm(fname, p0, ncnst, x, jnnz, jtjnnz, itmax, opts, ...)
%
%  
% The dots at the end denote any additional, problem specific data that are passed uninterpreted to
% all invocations of fname and jacname, see below for details.
%
% In the following, the word "vector" is meant to imply either a row or a column vector.
%
% required input arguments:
% - fname: String defining the name of a matlab function implementing the function to be minimized.
%      fname will be called as fname(p, ...), where p denotes the parameter vector and the dots any
%      additional data passed as extra arguments during the invocation of sparselm (refer to the
%      chained Rosenbrock function problem (chRosen) in splmdemo.m for an example).
%
% - p0: vector of doubles holding the initial parameters estimates.
%
% - ncnst: number of p0's elements whose values should be kept fixed to their initial values.
%
% - x: vector of doubles holding the measurements vector.
%
% - jnnz: number of nonzero elements in the Jacobian J.
%
% - jtjnnz: number of nonzero elements in the approximate Hessian J^T*J, -1 if unknown.
%
% - itmax: maximum number of iterations.
%
% - opts: vector of doubles specifying the minimization parameters, as follows:
%      opts(1) scale factor for the initial damping factor
%      opts(2) stopping threshold for ||J^T e||_inf
%      opts(3) stopping threshold for ||Dp||_2
%      opts(4) stopping threshold for ||e||_2
%      opts(5) step used in finite difference approximation to the Jacobian.
%      opts(6) sparse solver to be used for solving the augmented normal equations.
%      If an empty vector (i.e. []) is specified, defaults are used.
%  
% optional input arguments:
% - jacname: String defining the name of matlab function implementing the Jacobian of function fname.
%      jacname will be called as jacname(p, ...) where p is again the parameter vector and the dots
%      denote any additional data passed as extra arguments to the invocation of sparselm. If omitted,
%      the Jacobian's zero pattern is detected automatically and its numerical values are approximated
%      with finite differences through repeated invocations of fname.
%
% - jactype: String defining the type of Jacobian returned by jacname. It should be one of the following:
%      'an' specifies that jacname analytically computes the Jacobian of fname. This type assumes that
%           no (nonzero) elements of the Jacobian become zero during the minimization; in the opposite
%           case, use 'anzp' explained next.
%      'anzp' specifies that jacname analytically computes the Jacobian of fname (just like 'an' above)
%             and additionally, S is a read-only sparse matrix that explicitly defines the Jacobian's
%             nonzero pattern. This type is necessary to cater for matlab's lack of guarantees that the
%             number of nonzero elements in a matrix does not change over time: if an entry of a sparse
%             matrix becomes zero, then matlab automatically adjusts the former so that no storage is
%             wasted for the zero entry. sparseLM, on the other hand, assumes a static sparse structure
%             for the Jacobian throughout a minimization. Therefore, matrix S is used to facilitate the
%             copying of elements between two sparse matrices with different structure: All nonzero
%             elements from the matlab Jacobian are passed to their appropriate locations in sparseLM
%             and explicit zeros are passed in place of elements marked as non-zeros in S but being
%             numerically zero (and, therefore, missing) in the matlab Jacobian. Matrix S is unchanged
%             during the minimization.
%      'zp' specifies that jacname computes just the nonzero pattern for the Jacobian of fname.
%
%      If omitted, a default of 'an' is assumed.
%
% - S:  read-only sparse matrix explicitly defining the Jacobian's nonzero pattern. Used in combination
%       with 'anzp' explained above.
%
% output arguments
% - ret: return value of sparselm, corresponding to the number of iterations if successful, -1 otherwise.
%
% - popt: estimated minimizer, i.e. minimized parameters vector.
%
% - info: optional array of doubles, which upon return provides information regarding the minimization.
%      See splm.c for more details.
%
 
error('sparselm.m is used only for providing documentation to sparselm; make sure that sparselm.c has been compiled using mex');
