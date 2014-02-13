/* ////////////////////////////////////////////////////////////////////////////////
// 
//  Matlab MEX file for the sparse Levenberg - Marquardt minimization algorithm
//  Copyright (C) 2008-2011  Manolis Lourakis (lourakis at ics forth gr)
//  Institute of Computer Science, Foundation for Research & Technology - Hellas
//  Heraklion, Crete, Greece.
//
//  This program is free software; you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation; either version 2 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//////////////////////////////////////////////////////////////////////////////// */

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <math.h>
#include <string.h>
#include <ctype.h>

#include <splm.h>

#include <mex.h>

/**
#define DEBUG
**/


#define __ANJAC        0  /* analytic Jacobian */
#define __ZPJAC        1  /* approximate Jacobian, only zero pattern supplied */     
#define __NOJAC        2  /* approximate Jacobian, zero pattern to be guessed */  
#define __ANZPJAC      3  /* analytic Jacobian, zero pattern supplied */

#define __MAX__(A, B)     ((A)>=(B)? (A) : (B))

struct mexdata {
  /* matlab names of the fitting function & its Jacobian */
  char *fname, *jacname;

  /* sparse matrix defining the structure of an analytic Jacobian */
  mxArray *jacstruct;

  /* binary flags specifying if input p0 is a row or column vector */
  int isrow_p0;

  /* rhs args to be passed to matlab. rhs[0] is reserved for
   * passing the parameter vector. If present, problem-specific
   * data are passed in rhs[1], rhs[2], etc
   */
  mxArray **rhs;
  int nrhs; /* >= 1 */
};

/* display printf-style error messages in matlab */
static void matlabFmtdErrMsgTxt(char *fmt, ...)
{
char  buf[256];
va_list args;

	va_start(args, fmt);
	vsprintf(buf, fmt, args);
	va_end(args);

  mexErrMsgTxt(buf);
}

/* display printf-style warning messages in matlab */
static void matlabFmtdWarnMsgTxt(char *fmt, ...)
{
char  buf[256];
va_list args;

	va_start(args, fmt);
	vsprintf(buf, fmt, args);
	va_end(args);

  mexWarnMsgTxt(buf);
}

static void func(double *p, double *hx, int m, int n, void *adata)
{
mxArray *lhs[1];
double *mp, *mx;
register int i;
struct mexdata *dat=(struct mexdata *)adata;

  /* prepare to call matlab */
  mp=mxGetPr(dat->rhs[0]);
  for(i=m; i-->0;  )
    mp[i]=p[i];
    
  /* invoke matlab */
  mexCallMATLAB(1, lhs, dat->nrhs, dat->rhs, dat->fname);

  /* copy back results & cleanup */
  mx=mxGetPr(lhs[0]);
  for(i=n; i-->0;  )
    hx[i]=mx[i];

  /* delete the matrix created by matlab */
  mxDestroyArray(lhs[0]);
}

static void jacfunc(double *p, struct splm_ccsm *jac, int m, int n, void *adata)
{
mxArray *lhs[1];
double *mp;
double *mval, *val;
int *rowidx, *colptr;
#ifdef USE_LARGEARRAYDIMS
mwIndex *mrowidx, *mcolptr;
#else
int *mrowidx, *mcolptr;
#endif /* USE_LARGEARRAYDIMS */
register int i;
struct mexdata *dat=(struct mexdata *)adata;
int nnz;

  /* prepare to call matlab */
  mp=mxGetPr(dat->rhs[0]);
  for(i=m; i-->0;  )
    mp[i]=p[i];

  /* invoke matlab */
  mexCallMATLAB(1, lhs, dat->nrhs, dat->rhs, dat->jacname);
    
  /* copy back results & cleanup */
  mval=mxGetPr(lhs[0]);
  mrowidx=mxGetIr(lhs[0]);
  mcolptr=mxGetJc(lhs[0]);
  nnz=mcolptr[m]; /* the value returned by mxGetNzmax(lhs[0]) is >= nnz! */

  val=jac->val;
  rowidx=jac->rowidx;
  colptr=jac->colptr;

  if(nnz!=jac->nnz)
    matlabFmtdErrMsgTxt("sparseLM: MATLAB Jacobian does not have the expected number of nonzeros! [%d != %d].", nnz, jac->nnz);

  for(i=nnz; i-->0;  ){
    rowidx[i]=mrowidx[i];
    val[i]=mval[i];
  }

  for(i=m+1; i-->0;  )
    colptr[i]=mcolptr[i];
      
  /* delete the matrix created by matlab */
  mxDestroyArray(lhs[0]);
}

/* as above but without copying values */
static void jacfunc_novalues(double *p, struct splm_ccsm *jac, int m, int n, void *adata)
{
mxArray *lhs[1];
double *mp;
int *rowidx, *colptr;
#ifdef USE_LARGEARRAYDIMS
mwIndex *mrowidx, *mcolptr;
#else
int *mrowidx, *mcolptr;
#endif /* USE_LARGEARRAYDIMS */
register int i;
struct mexdata *dat=(struct mexdata *)adata;
int nnz;

  /* prepare to call matlab */
  mp=mxGetPr(dat->rhs[0]);
  for(i=m; i-->0;  )
    mp[i]=p[i];

  /* invoke matlab */
  mexCallMATLAB(1, lhs, dat->nrhs, dat->rhs, dat->jacname);
    
  /* copy back results & cleanup */
  mrowidx=mxGetIr(lhs[0]);
  mcolptr=mxGetJc(lhs[0]);
  nnz=mcolptr[m]; /* the value returned by mxGetNzmax(lhs[0]) is >= nnz! */

  rowidx=jac->rowidx;
  colptr=jac->colptr;

  if(nnz!=jac->nnz)
    matlabFmtdErrMsgTxt("sparseLM: MATLAB Jacobian does not have the expected number of nonzeros! [%d != %d].", nnz, jac->nnz);

  for(i=nnz; i-->0;  )
    rowidx[i]=mrowidx[i];

  for(i=m+1; i-->0;  )
    colptr[i]=mcolptr[i];
      
  /* delete the matrix created by matlab */
  mxDestroyArray(lhs[0]);
}

/* as jacfunc but with sparse structure provided explicitly */
static void jacfunc_structure(double *p, struct splm_ccsm *jac, int m, int n, void *adata)
{
mxArray *lhs[1];
double *mp;
double *mval, *val;
int *rowidx, *colptr;
#ifdef USE_LARGEARRAYDIMS
mwIndex *mrowidx, *mcolptr, *Srowidx, *Scolptr;
mwSize mnnz, Sm, Sn, Snnz;
#else
int *mrowidx, *mcolptr, *Srowidx, *Scolptr;
int mnnz, Sm, Sn, Snnz;
#endif /* USE_LARGEARRAYDIMS */
register int i, j, col; /* resp. source (i.e. matlab), destination (i.e. SPLM) & column index */
struct mexdata *dat=(struct mexdata *)adata;
int nnz;

  /* prepare to call matlab */
  mp=mxGetPr(dat->rhs[0]);
  for(i=m; i-->0;  )
    mp[i]=p[i];

  /* invoke matlab */
  mexCallMATLAB(1, lhs, dat->nrhs, dat->rhs, dat->jacname);
    
  /* copy back results & cleanup */
  mval=mxGetPr(lhs[0]);
  mrowidx=mxGetIr(lhs[0]);
  mcolptr=mxGetJc(lhs[0]);
  mnnz=mcolptr[m]; /* the value returned by mxGetNzmax(lhs[0]) is >= nnz! */

  val=jac->val;
  rowidx=jac->rowidx;
  colptr=jac->colptr;
  nnz=jac->nnz;

  if((int)mnnz>nnz)
    matlabFmtdErrMsgTxt("sparseLM: MATLAB Jacobian has more nonzeros than expected! [%d > %d].", (int)mnnz, nnz);

  /* retrieve structure matrix */
  Sm=mxGetM(dat->jacstruct);
  Sn=mxGetN(dat->jacstruct);
  Srowidx=mxGetIr(dat->jacstruct);
  Scolptr=mxGetJc(dat->jacstruct);
  Snnz=Scolptr[Sn];
  if(m!=(int)Sn || n!=(int)Sm)
    matlabFmtdErrMsgTxt("sparseLM: MATLAB structure Jacobian has invalid dimensions! [expected %dx%d, got %dx%d].", n, m, (int)Sm, (int)Sn);
  if(nnz!=(int)Snnz)
    matlabFmtdErrMsgTxt("sparseLM: MATLAB structure Jacobian does not have the expected number of nonzeros! [%d != %d].", (int)Snnz, nnz);

  /* copy structure */
  for(i=nnz; i-->0;  )
    rowidx[i]=(int)Srowidx[i];
  for(i=m+1; i-->0;  )
    colptr[i]=(int)Scolptr[i];

  /* copy the nonzero elements of the matlab Jacobian to jac assuming that its
   * structure is a subset of jac, i.e. some of the assumed nonzero entries in
   * jac might not have corresponding nonzero entries from the matlab Jacobian.
   *
   * Inspired from copyto() in IPOPT's sparsematrix.cpp
   * see also https://projects.coin-or.org/Ipopt/wiki/MatlabInterface
   */
  for(i=j=col=0; col<m; ++col){
    for(  ; i<(int)mcolptr[col+1]; ++i, ++j){
      /* copy all nonzeros in column: check whether the source row & column
       * match those of the destination. The first term checks row indices,
       * the last two the column ones. While there is a mismatch, the
       * current element in destination is cleared and the destination index
       * is advanced.
       *
       * Note that this code will fail if the matlab array contains more
       * nonzeros than the SPLM one! - this is supposed to be ensured by
       * the user.
       */
      while(!( ((int)mrowidx[i]==rowidx[j]) && (j>=colptr[col]) && (j<colptr[col+1]) )){
        val[j++]=0.0; /* element (rowidx[j], col) not present in matlab Jacobian */

#if 0   /* bounds checking removed for better performance */
        if(j>=nnz)
          matlabFmtdErrMsgTxt("sparseLM: MATLAB Jacobian has an unexpected nonzero element at (%d, %d)!\n",
                                (int)mrowidx[i]+1, col+1);
#endif
      }

      /* at this point row & column indices match, hence the source entry is
       * copied to the destination
       */
      val[j]=mval[i];
    }      
  }

  /* delete the matrix created by matlab */
  mxDestroyArray(lhs[0]);
}

/* check the supplied matlab function and its Jacobian. Returns 1 on error, 0 otherwise */
static int checkFuncAndJacobian(double *p, int  m, int n, int testjac, struct mexdata *dat)
{
mxArray *lhs[1];
register int i;
int ret=0;
double *mp;

  mexSetTrapFlag(1); /* handle errors in the MEX-file */

  mp=mxGetPr(dat->rhs[0]);
  for(i=0; i<m; ++i)
    mp[i]=p[i];

  /* attempt to call the supplied func */
  i=mexCallMATLAB(1, lhs, dat->nrhs, dat->rhs, dat->fname);
  if(i){
    fprintf(stderr, "sparseLM: error calling '%s'.\n", dat->fname);
    ret=1;
  }
  else if(!mxIsDouble(lhs[0]) || mxIsComplex(lhs[0]) || !(mxGetM(lhs[0])==1 || mxGetN(lhs[0])==1) ||
      __MAX__(mxGetM(lhs[0]), mxGetN(lhs[0]))!=n){
    fprintf(stderr, "sparseLM: '%s' should produce a real vector with %d elements (got %d).\n",
                    dat->fname, m, __MAX__(mxGetM(lhs[0]), mxGetN(lhs[0])));
    ret=1;
  }
  /* delete the matrix created by matlab */
  mxDestroyArray(lhs[0]);

  if(testjac){
    /* attempt to call the supplied jac  */
    i=mexCallMATLAB(1, lhs, dat->nrhs, dat->rhs, dat->jacname);
    if(i){
      fprintf(stderr, "sparseLM: error calling '%s'.\n", dat->jacname);
      ret=1;
    }
    else if(!mxIsSparse(lhs[0]) || mxIsComplex(lhs[0]) || mxGetM(lhs[0])!=n || mxGetN(lhs[0])!=m){
      fprintf(stderr, "sparseLM: '%s' should produce a real %dx%d sparse matrix (got %dx%d).\n",
                      dat->jacname, n, m, mxGetM(lhs[0]), mxGetN(lhs[0]));
      ret=1;
    }
    /* delete the matrix created by matlab */
    mxDestroyArray(lhs[0]);
  }

  mexSetTrapFlag(0); /* on error terminate the MEX-file and return control to the MATLAB prompt */

  return ret;
}

/*
[ret, p, info]=sparselm(f, j, 'an', p0, ncnst, x, jnnz, jtjnnz, itmax, opts, ...), when j analytically computes the Jacobian of f
[ret, p, info]=sparselm(f, j, 'zp', p0, ncnst, x, jnnz, jtjnnz, itmax, opts, ...), when j computes just the nonzero pattern for the Jacobian of f
[ret, p, info]=sparselm(f, p0, ncnst, x, jnnz, jtjnnz, itmax, opts, ...), when the Jacobian's nonzero pattern is detected automatically. USE CAUTIOUSLY!
[ret, p, info]=sparselm(f, j, 'anzp', S, p0, ncnst, x, jnnz, jtjnnz, itmax, opts, ...), when j analytically computes the Jacobian of f, S explicitly provides j's sparsity pattern
*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *Prhs[])
{
register int i;
register double *pdbl;
mxArray **prhs=(mxArray **)&Prhs[0];
struct mexdata mdata;
int len, status;
double *p, *p0, *ret, *x;
int m, n, jtyp, ncnst, jnnz, jtjnnz, itmax, nopts, nextra;
char *jdescr;
double opts[SPLM_OPTS_SZ]={SPLM_INIT_MU, SPLM_STOP_THRESH, SPLM_STOP_THRESH, SPLM_STOP_THRESH, SPLM_DIFF_DELTA, SPLM_CHOLMOD};
double info[SPLM_INFO_SZ];

  /* parse input args; start by checking their number */
  if((nrhs<9))
    matlabFmtdErrMsgTxt("sparseLM: at least 9 input arguments required (got %d).", nrhs);
  if(nlhs>3)
    matlabFmtdErrMsgTxt("sparseLM: too many output arguments (max. 3, got %d).", nlhs);
  else if(nlhs<2)
    matlabFmtdErrMsgTxt("sparseLM: too few output arguments (min. 2, got %d).", nlhs);
    
  /* note that in order to accommodate optional args, prhs & nrhs are adjusted accordingly below */

  /** func **/
  /* first argument must be a string, i.e. a char row vector */
  if(mxIsFunctionHandle(prhs[0]))
    mexErrMsgTxt("sparseLM: function handles for first argument not supported, convert to string with func2str.");
  if(mxIsChar(prhs[0])!=1)
    mexErrMsgTxt("sparseLM: first argument must be a string.");
  if(mxGetM(prhs[0])!=1)
    mexErrMsgTxt("sparseLM: first argument must be a string (i.e. char row vector).");
  /* store supplied name */
  len=mxGetN(prhs[0])+1;
  mdata.fname=mxCalloc(len, sizeof(char));
  status=mxGetString(prhs[0], mdata.fname, len);
  if(status!=0)
    mexErrMsgTxt("sparseLM: not enough space. String is truncated.");

  /* check whether second (optional) argument is a string */
  if(mxIsChar(prhs[1])==1){
    /** fjac **/
    if(mxGetM(prhs[1])!=1)
      mexErrMsgTxt("sparseLM: second argument must be a string (i.e. char row vector).");
    /* store supplied name */
    len=mxGetN(prhs[1])+1;
    mdata.jacname=mxCalloc(len, sizeof(char));
    status=mxGetString(prhs[1], mdata.jacname, len);
    if(status!=0)
      mexErrMsgTxt("sparseLM: not enough space. String is truncated.");
    mdata.jacstruct=NULL;

    /** jtyp **/
    /* the third (optional) argument must be a string */
    if(mxIsChar(prhs[2]) && mxGetM(prhs[2])==1){
      /* examine supplied type */
      len=mxGetN(prhs[2])+1;
      jdescr=mxCalloc(len, sizeof(char));
      status=mxGetString(prhs[2], jdescr, len);
      if(status!=0)
        mexErrMsgTxt("sparseLM: not enough space. String is truncated.");

      for(i=0; jdescr[i]; ++i)
        jdescr[i]=tolower(jdescr[i]);

      if(!strncmp(jdescr, "anzp", 4)) jtyp=__ANZPJAC;
      else if(!strncmp(jdescr, "an", 2)) jtyp=__ANJAC;
      else if(!strncmp(jdescr, "zp", 2)) jtyp=__ZPJAC;
      else matlabFmtdErrMsgTxt("sparseLM: unknown Jacobian type '%s'.", jdescr);
      mxFree(jdescr);

      prhs+=2;
      nrhs-=2;

      /* if an analytical Jacobian with provided structure has been specified,
       * the structure must be in a sparse matrix pointed to by the next argument
       */
      if(jtyp==__ANZPJAC){
        if(!mxIsSparse(prhs[1]) || mxIsComplex(prhs[1]))
          mexErrMsgTxt("sparseLM: Jacobian structure argument must be a real sparse matrix.");

        mdata.jacstruct=prhs[1];

        ++prhs;
        --nrhs;
      }
    }
    else{
      /* mexErrMsgTxt("sparseLM: type of Jacobian must be a string."); */

      /* assume analytic Jacobian */
      jtyp=__ANJAC;
      ++prhs;
      --nrhs;
    }
  }
  else{ /* no Jacobian function supplied */
    if(mxIsFunctionHandle(prhs[1]))
      mexErrMsgTxt("sparseLM: function handles for Jacobian not supported, convert to string with func2str.");
    
    jtyp=__NOJAC;
    mdata.jacname=NULL;
  }

  /** p0 **/
  /* the second required argument must be a real row or column vector */
  if(!mxIsDouble(prhs[1]) || mxIsComplex(prhs[1]) || !(mxGetM(prhs[1])==1 || mxGetN(prhs[1])==1) || mxIsSparse(prhs[1]))
    mexErrMsgTxt("sparseLM: p0 must be a dense real vector.");
  p0=mxGetPr(prhs[1]);
  /* determine if we have a row or column vector and retrieve its 
   * size, i.e. the number of parameters
   */
  if(mxGetM(prhs[1])==1){
    m=mxGetN(prhs[1]);
    mdata.isrow_p0=1;
  }
  else{
    m=mxGetM(prhs[1]);
    mdata.isrow_p0=0;
  }
  /* copy input parameter vector to avoid destroying it */
  p=mxMalloc(m*sizeof(double));
  for(i=0; i<m; ++i)
    p[i]=p0[i];

  /** ncnst **/
  /* the third required argument must be a scalar */
  if(!mxIsDouble(prhs[2]) || mxIsComplex(prhs[2]) || mxGetM(prhs[2])!=1 || mxGetN(prhs[2])!=1)
    mexErrMsgTxt("sparseLM: ncnst must be a scalar.");
  ncnst=(int)mxGetScalar(prhs[2]);

  /** x **/
  /* the fourth required argument must be a real row or column vector */
  if(!mxIsDouble(prhs[3]) || mxIsComplex(prhs[3]) || !(mxGetM(prhs[3])==1 || mxGetN(prhs[3])==1) || mxIsSparse(prhs[3]))
    mexErrMsgTxt("sparseLM: x must be a dense real vector.");
  x=mxGetPr(prhs[3]);
  n=__MAX__(mxGetM(prhs[3]), mxGetN(prhs[3]));

  /** jnnz **/
  /* the fifth required argument must be a scalar */
  if(!mxIsDouble(prhs[4]) || mxIsComplex(prhs[4]) || mxGetM(prhs[4])!=1 || mxGetN(prhs[4])!=1)
    mexErrMsgTxt("sparseLM: jnnz must be a scalar.");
  jnnz=(int)mxGetScalar(prhs[4]);

  /** jtjnnz **/
  /* the sixth argument must be a scalar */
  if(!mxIsDouble(prhs[5]) || mxIsComplex(prhs[5]) || mxGetM(prhs[5])!=1 || mxGetN(prhs[5])!=1)
    mexErrMsgTxt("sparseLM: jtjnnz must be a scalar.");
  jtjnnz=(int)mxGetScalar(prhs[5]);
    
  /** itmax **/
  /* the seventh required argument must be a scalar */
  if(!mxIsDouble(prhs[6]) || mxIsComplex(prhs[6]) || mxGetM(prhs[6])!=1 || mxGetN(prhs[6])!=1)
    mexErrMsgTxt("sparseLM: itmax must be a scalar.");
  itmax=(int)mxGetScalar(prhs[6]);
    
  /** opts **/
  /* if present, the eighth argument must be a real row or column vector */
  if(nrhs>=8){
    if(!mxIsDouble(prhs[7]) || mxIsComplex(prhs[7]) || (!(mxGetM(prhs[7])==1 || mxGetN(prhs[7])==1) &&
                                                        !(mxGetM(prhs[7])==0 && mxGetN(prhs[7])==0)))
      mexErrMsgTxt("sparseLM: opts must be a real vector.");
    pdbl=mxGetPr(prhs[7]);
    nopts=__MAX__(mxGetM(prhs[7]), mxGetN(prhs[7]));

    ++prhs;
    --nrhs;

    if(nopts!=0){ /* if opts==[], nothing needs to be done and the defaults are used */
      if(nopts>SPLM_OPTS_SZ)
        matlabFmtdErrMsgTxt("sparseLM: opts must have at most %d elements, got %d.", SPLM_OPTS_SZ, nopts);
      else if(nopts<SPLM_OPTS_SZ)
        matlabFmtdWarnMsgTxt("sparseLM: only the %d first elements of opts specified, remaining set to defaults.", nopts);
      for(i=0; i<nopts; ++i)
        opts[i]=pdbl[i];
    }
    /* else using defaults */
  }
  /* else using defaults */


  /* arguments below this point are assumed to be extra arguments passed
   * to every invocation of the fitting function and its Jacobian
   */

extraargs:
  /* handle any extra args and allocate memory for
   * passing the current parameter estimate to matlab
   */
  nextra=nrhs-7;
  mdata.nrhs=nextra+1;
  mdata.rhs=(mxArray **)mxMalloc(mdata.nrhs*sizeof(mxArray *));
  for(i=0; i<nextra; ++i)
    mdata.rhs[i+1]=(mxArray *)prhs[nrhs-nextra+i]; /* discard 'const' modifier */
#ifdef DEBUG
  fflush(stderr);
  fprintf(stderr, "sparseLM: %d extra args\n", nextra);
#endif /* DEBUG */

  if(mdata.isrow_p0){ /* row vector */
    mdata.rhs[0]=mxCreateDoubleMatrix(1, m, mxREAL);
  }
  else{ /* column vector */
    mdata.rhs[0]=mxCreateDoubleMatrix(m, 1, mxREAL);
  }

  /* ensure that the supplied function & Jacobian are as expected */
  if(checkFuncAndJacobian(p, m, n, jtyp!=__NOJAC, &mdata)){
    status=SPLM_ERROR;
    goto cleanup;
  }

  /* invoke sparseLM */
  switch(jtyp){
    case __ANJAC:
      status=sparselm_derccs(func, jacfunc, p, x, m, ncnst, n, jnnz, jtjnnz, itmax, opts, info, (void *)&mdata);
      break;
    case __ZPJAC:
      status=sparselm_difccs(func, jacfunc_novalues, p, x, m, ncnst, n, jnnz, jtjnnz, itmax, opts, info, (void *)&mdata);
      break;
    case __ANZPJAC:
      status=sparselm_derccs(func, jacfunc_structure, p, x, m, ncnst, n, jnnz, jtjnnz, itmax, opts, info, (void *)&mdata);
      break;
    default:
      status=sparselm_difccs(func, NULL, p, x, m, ncnst, n, jnnz, jtjnnz, itmax, opts, info, (void *)&mdata);
      break;
  }

#ifdef DEBUG
  fflush(stderr);
  printf("sparseLM: minimization returned %d in %g iter, reason %g\n\tSolution: ", status, info[5], info[6]);
  for(i=0; i<m; ++i)
    printf("%.7g ", p[i]);
  printf("\n\n\tMinimization info:\n\t");
  for(i=0; i<SPLM_INFO_SZ; ++i)
    printf("%g ", info[i]);
  printf("\n");
#endif /* DEBUG */

  /* copy back return results */
  /** ret **/
  plhs[0]=mxCreateDoubleMatrix(1, 1, mxREAL);
  ret=mxGetPr(plhs[0]);
  ret[0]=(double)status;

  /** popt **/
  plhs[1]=(mdata.isrow_p0==1)? mxCreateDoubleMatrix(1, m, mxREAL) : mxCreateDoubleMatrix(m, 1, mxREAL);
  pdbl=mxGetPr(plhs[1]);
  for(i=0; i<m; ++i)
    pdbl[i]=p[i];

  /** info **/
  if(nlhs>2){
    plhs[2]=mxCreateDoubleMatrix(1, SPLM_INFO_SZ, mxREAL);
    pdbl=mxGetPr(plhs[2]);
    for(i=0; i<SPLM_INFO_SZ; ++i)
      pdbl[i]=info[i];
  }

cleanup:
  /* cleanup */
  mxDestroyArray(mdata.rhs[0]);

  mxFree(mdata.fname);
  if(mdata.jacname) mxFree(mdata.jacname);
  mxFree(p);
  mxFree(mdata.rhs);

  if(status==SPLM_ERROR)
    mexWarnMsgTxt("sparseLM: optimization returned with an error!");
}
