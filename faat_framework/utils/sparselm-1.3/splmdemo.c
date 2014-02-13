/////////////////////////////////////////////////////////////////////////////////
//// 
////  Demonstration driver program for the sparseLM optimization package.
////  Copyright (C) 2008-2010  Manolis Lourakis (lourakis at ics forth gr)
////  Institute of Computer Science, Foundation for Research & Technology - Hellas
////  Heraklion, Crete, Greece.
////
////  This program is free software; you can redistribute it and/or modify
////  it under the terms of the GNU General Public License as published by
////  the Free Software Foundation; either version 2 of the License, or
////  (at your option) any later version.
////
////  This program is distributed in the hope that it will be useful,
////  but WITHOUT ANY WARRANTY; without even the implied warranty of
////  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
////  GNU General Public License for more details.
////
///////////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <splm.h>


#define DIV(n, m)     ((n) / (m))

/* Sample functions to be minimized with sparseLM and their Jacobians. See
 * "Sparse Test Problems for Unconstrained Optimization" available
 * at http://www3.cs.cas.cz/ics/reports/v1064-10.ps
 */


/* Chained Rosenbrock function */
// initial point: p[i] =  -1.2 for i odd, 1.0 for i even, minimum at (1, 1, ..., 1)
void chainedRosenbrock(double *p, double *hx, int m, int n, void *adata)
{
register int k, k1, i;

  for(k=0; k<n; ++k){
    k1=k+1; // k is zero-based, convert to one-based
    i=DIV(k1+1, 2) - 1; // convert i to zero-based
    if(k1%2==1) // k1 odd
      hx[k]=10.0*(p[i]*p[i]-p[i+1]);
    else // k1 even
      hx[k]=p[i]-1.0;
  }
}

/* analytic CRS Jacobian for chainedRosenbrock() */
void chainedRosenbrock_anjacCRS(double *p, struct splm_crsm *jac, int m, int n, void *adata)
{
register int k, k1, i;
int l;

  for(k=l=0; k<n; ++k){
    jac->rowptr[k]=l;
    k1=k+1; // k is zero-based, convert to one-based
    i=DIV(k1+1, 2) - 1; // convert i to zero-based
    if(k1%2==1){ // k1 odd, hx[k]=10*(p[i]*p[i]-p[i+1])
      jac->val[l]=20.0*p[i]; jac->colidx[l++]=i;
      jac->val[l]=-10.0; jac->colidx[l++]=i+1;
    }
    else { // k1 even, hx[k]=p[i]-1.0
      jac->val[l]=1.0; jac->colidx[l++]=i;
    }
  }
  jac->rowptr[n]=l;
}

/* analytic CCS Jacobian for chainedRosenbrock() */
void chainedRosenbrock_anjacCCS(double *p, struct splm_ccsm *jac, int m, int n, void *adata)
{
register int k, k1, i;
int l;
struct splm_crsm jac_crs;

  /* allocate and fill-in a CRS Jacobian... */
  splm_crsm_alloc(&jac_crs, n, m, jac->nnz);

  for(k=l=0; k<n; ++k){
    jac_crs.rowptr[k]=l;
    k1=k+1; // k is zero-based, convert to one-based
    i=DIV(k1+1, 2) - 1; // convert i to zero-based
    if(k1%2==1){ // k1 odd, hx[k]=10*(p[i]*p[i]-p[i+1])
      jac_crs.val[l]=20.0*p[i]; jac_crs.colidx[l++]=i;
      jac_crs.val[l]=-10.0; jac_crs.colidx[l++]=i+1;
    }
    else { // k1 even, hx[k]=p[i]-1.0
      jac_crs.val[l]=1.0; jac_crs.colidx[l++]=i;
    }
  }
  jac_crs.rowptr[n]=l;

  /* ...convert to CCS */
  splm_crsm2ccsm(&jac_crs, jac);
  splm_crsm_free(&jac_crs);
}

/* analytic CCS Jacobian for chainedRosenbrock(). Jacobian is first constructed using sparse triplet format */
void chainedRosenbrock_anjacCCS_ST(double *p, struct splm_ccsm *jac, int m, int n, void *adata)
{
register int k, k1, i;
struct splm_stm jac_st;

  /* allocate and fill-in a ST Jacobian... */
  splm_stm_allocval(&jac_st, n, m, jac->nnz);

  for(k=0; k<n; ++k){
    k1=k+1; // k is zero-based, convert to one-based
    i=DIV(k1+1, 2) - 1; // convert i to zero-based
    if(k1%2==1){ // k1 odd, hx[k]=10*(p[i]*p[i]-p[i+1])
      splm_stm_nonzeroval(&jac_st, k, i, 20.0*p[i]);
      splm_stm_nonzeroval(&jac_st, k, i+1, -10.0);
    }
    else { // k1 even, hx[k]=p[i]-1.0
      splm_stm_nonzeroval(&jac_st, k, i, 1.0);
    }
  }
  /* ...convert to CCS */
  splm_stm2ccsm(&jac_st, jac);
  splm_stm_free(&jac_st);
}

/* zero pattern of CCS Jacobian for chainedRosenbrock(). Jacobian is first constructed using sparse triplet format */
void chainedRosenbrock_zpjacCCS_ST(double *p, struct splm_ccsm *jac, int m, int n, void *adata)
{
register int k, k1, i;
struct splm_stm jac_st;

  /* allocate and fill-in a ST Jacobian... */
  splm_stm_alloc(&jac_st, n, m, jac->nnz);

  for(k=0; k<n; ++k){
    k1=k+1; // k is zero-based, convert to one-based
    i=DIV(k1+1, 2) - 1; // convert i to zero-based
    if(k1%2==1){ // k1 odd, hx[k]=10*(p[i]*p[i]-p[i+1])
      splm_stm_nonzero(&jac_st, k, i);
      splm_stm_nonzero(&jac_st, k, i+1);
    }
    else { // k1 even, hx[k]=p[i]-1.0
      splm_stm_nonzero(&jac_st, k, i);
    }
  }
  /* ...convert to CCS */
  splm_stm2ccsm(&jac_st, jac);
  splm_stm_free(&jac_st);
}



/* Chained Wood function */
static const double sqrt10=3.162277660, sqrt90=9.486832981; /* sqrt(10.0), sqrt(90.0) */

// initial point:  (0, -3, 0, -3, -1, -2, ..., -1, -2),  minimum at (1, 1, ..., 1)
void chainedWood(double *p, double *hx, int m, int n, void *adata)
{
register int k, k1, i;

  for(k=0; k<n; ++k){
    k1=k+1; // k is zero-based, convert to one-based
    i=2*DIV(k1+5, 6)-1 - 1; // convert i to zero-based

    switch(k1%6){
      case 0: hx[k]=(p[i+1] - p[i+3])/sqrt10;
              break;
      case 1: hx[k]=10*(p[i]*p[i] - p[i+1]);
              break;
      case 2: hx[k]=p[i] - 1.0;
              break;
      case 3: hx[k]=sqrt90*(p[i+2]*p[i+2] - p[i+3]);
              break;
      case 4: hx[k]=p[i+2] - 1.0;
              break;
      case 5: hx[k]=sqrt10*(p[i+1] + p[i+3] - 2.0);
              break;
    }
  }
}

void chainedWood_jac(double *p, struct splm_crsm *jac, int m, int n, void *adata)
{
register int k, k1, i;
int l;

  for(k=l=0; k<n; ++k){
    jac->rowptr[k]=l;
    k1=k+1; // k is zero-based, convert to one-based
    i=2*DIV(k1+5, 6)-1 - 1; // convert i to zero-based

    switch(k1%6){
      case 0: //hx[k]=(p[i+1] - p[i+3])/sqrt10;
              jac->val[l]=1.0/sqrt10;  jac->colidx[l++]=i+1;
              jac->val[l]=-1.0/sqrt10; jac->colidx[l++]=i+3;
              break;
      case 1: //hx[k]=10*(p[i]*p[i] - p[i+1]);
              jac->val[l]=20*p[i]; jac->colidx[l++]=i;
              jac->val[l]=-10.0;   jac->colidx[l++]=i+1;
              break;
      case 2: //hx[k]=p[i] - 1.0;
              jac->val[l]=1.0;   jac->colidx[l++]=i;
              break;
      case 3: //hx[k]=sqrt90*(p[i+2]*p[i+2] - p[i+3]);
              jac->val[l]=2.0*sqrt90*p[i+2]; jac->colidx[l++]=i+2;
              jac->val[l]=-sqrt90;           jac->colidx[l++]=i+3;
              break;
      case 4: //hx[k]=p[i+2] - 1.0;
              jac->val[l]=1.0;  jac->colidx[l++]=i+2;
              break;
      case 5: //hx[k]=sqrt10*(p[i+1] + p[i+3] - 2.0);
              jac->val[l]=sqrt10; jac->colidx[l++]=i+1;
              jac->val[l]=sqrt10; jac->colidx[l++]=i+3;
              break;
    }
  }
  jac->rowptr[n]=l;
}


/* Chained Powell singular function */
static const double sqrt5=2.236067978; /* sqrt(5.0) */

// initial point: (3.0, -1.0, 0.0, 1.0, 3.0, -1.0, 0.0, 1.0, ... )  minimum at (0, 0, ..., 0)
void chainedPowell(double *p, double *hx, int m, int n, void *adata)
{
register int k, k1, i;
double tmp;

  for(k=0; k<n; ++k){
    k1=k+1; // k is zero-based, convert to one-based
    i=2*DIV(k1+3, 4)-1 - 1; // convert i to zero-based

    switch(k1%4){
      case 0:
              tmp=p[i] - p[i+3];
              hx[k]=sqrt10*tmp*tmp;
              break;
      case 1:
              hx[k]=p[i] + 10.0*p[i+1];
              break;
      case 2:
              hx[k]=sqrt5*(p[i+2] - p[i+3]);
              break;
      case 3:
              tmp=p[i+1] - 2.0*p[i+2];
              hx[k]=tmp*tmp;
              break;
    }
  }
}

void chainedPowell_jac(double *p, struct splm_crsm *jac, int m, int n, void *adata)
{
register int k, k1, i;
int l;

  for(k=l=0; k<n; ++k){
    jac->rowptr[k]=l;
    k1=k+1; // k is zero-based, convert to one-based
    i=2*DIV(k1+3, 4)-1 - 1; // convert i to zero-based

    switch(k1%4){
      case 0:
              jac->val[l]= 2.0*sqrt10*(p[i] - p[i+3]); jac->colidx[l++]=i;
              jac->val[l]=-2.0*sqrt10*(p[i] - p[i+3]); jac->colidx[l++]=i+3;
              break;
      case 1:
              jac->val[l]=1.0;  jac->colidx[l++]=i;
              jac->val[l]=10.0; jac->colidx[l++]=i+1;
              break;
      case 2:
              jac->val[l]=sqrt5;  jac->colidx[l++]=i+2;
              jac->val[l]=-sqrt5; jac->colidx[l++]=i+3;
              break;
      case 3:
              jac->val[l]= 2.0*(p[i+1] - 2.0*p[i+2]); jac->colidx[l++]=i+1;
              jac->val[l]=-4.0*(p[i+1] - 2.0*p[i+2]); jac->colidx[l++]=i+2;
              break;
    }
  }
  jac->rowptr[n]=l;
}


/* Chained Cragg and Levy function */
// initial point: (1.0, 2.0, 2.0, ... )  minimum 0.920932 at
// (-0.498736, 0.31382, 0.137895, 0.603926, 0.337989, 0.783396, 0.514324, 1.03259, 1.00976, 1), M=10

void chainedCraggLevy(double *p, double *hx, int m, int n, void *adata)
{
register int k, k1, i;
double tmp;

  for(k=0; k<n; ++k){
    k1=k+1; // k is zero-based, convert to one-based
    i=2*DIV(k1+4, 5)-1 - 1; // convert i to zero-based

    switch(k1%5){
      case 0:
              hx[k]=p[i+3] - 1.0;
              break;
      case 1:
              tmp=exp(p[i]) - p[i+1];
              hx[k]=tmp*tmp;
              break;
      case 2:
              tmp=p[i+1] - p[i+2];
              hx[k]=10.0*tmp*tmp*tmp;
              break;
      case 3:
              tmp=cos(p[i+2] - p[i+3]); tmp=tmp*tmp;
              hx[k]=(1.0-tmp)/tmp;
              break;
      case 4:
              tmp=p[i]*p[i];
              hx[k]=tmp*tmp;
              break;
    }
  }
}

/* analytic Jacobian for chainedCraggLevy() */
void chainedCraggLevy_anjac(double *p, struct splm_crsm *jac, int m, int n, void *adata)
{
register int k, k1, i;
int l;
double tmp;

  for(k=l=0; k<n; ++k){
    jac->rowptr[k]=l;
    k1=k+1; // k is zero-based, convert to one-based
    i=2*DIV(k1+4, 5)-1 - 1; // convert i to zero-based

    switch(k1%5){
      case 0:
              jac->val[l]=1.0; jac->colidx[l++]=i+3;
              break;
      case 1:
              tmp=exp(p[i]);
              jac->val[l]= 2.0*(tmp - p[i+1])*tmp;  jac->colidx[l++]=i;
              jac->val[l]=-2.0*(tmp - p[i+1]);      jac->colidx[l++]=i+1;
              break;
      case 2:
              tmp=p[i+1] - p[i+2]; tmp=tmp*tmp;
              jac->val[l]= 30.0*tmp;  jac->colidx[l++]=i+1;
              jac->val[l]=-30.0*tmp;  jac->colidx[l++]=i+2;
              break;
      case 3:
              tmp=sin(p[i+2] - p[i+3])/cos(p[i+2] - p[i+3]);
              jac->val[l]= 2.0*tmp*(1.0 + tmp*tmp); jac->colidx[l++]=i+2;
              jac->val[l]=-2.0*tmp*(1.0 + tmp*tmp); jac->colidx[l++]=i+3;
              break;
      case 4:
              jac->val[l]=4.0*p[i]*p[i]*p[i]; jac->colidx[l++]=i;
              break;
    }
  }
  jac->rowptr[n]=l;
}

/* zero pattern only for the Jacobian of chainedCraggLevy() */
void chainedCraggLevy_zpjac(double *p, struct splm_crsm *jac, int m, int n, void *adata)
{
register int k, k1, i;
int l;

  for(k=l=0; k<n; ++k){
    jac->rowptr[k]=l;
    k1=k+1; // k is zero-based, convert to one-based
    i=2*DIV(k1+4, 5)-1 - 1; // convert i to zero-based

    switch(k1%5){
      case 0:
              jac->colidx[l++]=i+3;
              break;
      case 1:
              jac->colidx[l++]=i;
              jac->colidx[l++]=i+1;
              break;
      case 2:
              jac->colidx[l++]=i+1;
              jac->colidx[l++]=i+2;
              break;
      case 3:
              jac->colidx[l++]=i+2;
              jac->colidx[l++]=i+3;
              break;
      case 4:
              jac->colidx[l++]=i;
              break;
    }
  }
  jac->rowptr[n]=l;
}


#define MAX(n, m)     (((n) >= (m))? (n) : (m))

/* Sizes for the test problems */

/* Chained Rosenbrock function */
#define   MCHROSEN   10 // must be even
#define   NCHROSEN    2*(MCHROSEN-1)
/* Chained Wood function */
#define   MCHWOOD   10 // must be even
#define   NCHWOOD    3*(MCHWOOD-2)
/* Chained Powell singular function */
#define   MCHPOWELL   10 // must be even
#define   NCHPOWELL    2*(MCHPOWELL-2)
/* Chained Cragg and Levy function */
#define   MCHCRALEV   10 // must be even
#define   NCHCRALEV    5*(MCHCRALEV-2)/2

int main()
{
register int i;
double opts[SPLM_OPTS_SZ], info[SPLM_INFO_SZ];
double p[MAX(MCHROSEN, MAX(MCHWOOD, MAX(MCHPOWELL, MCHCRALEV)))];
int m, n, ret;
int problem, nnz;
char *probname[]={
      "chained Rosenbrock function",
      "chained Wood function",
      "chained Powell singular function",
      "chained Cragg and Levy function",
};

  opts[0]=SPLM_INIT_MU; opts[1]=SPLM_STOP_THRESH; opts[2]=SPLM_STOP_THRESH;
  opts[3]=SPLM_STOP_THRESH;
  opts[4]=SPLM_DIFF_DELTA; // relevant only if finite difference approximation to Jacobian is used
  opts[5]=SPLM_CHOLMOD; // use CHOLMOD
  //opts[5]=SPLM_PARDISO; // use PARDISO

  /* uncomment the appropriate line below to select a minimization problem */
  problem=
      0; // chained Rosenbrock function
      //1; //Chained Wood function
      //2; //Chained Powell singular function
      //3; //Chained Cragg and Levy function

  switch(problem){
  default: fprintf(stderr, "unknown problem specified (#%d)!\n", problem);
           exit(1);
  break;

  case 0:
  /* chained Rosenbrock function */
    m=MCHROSEN; n=NCHROSEN; nnz=3*NCHROSEN/2;
    for(i=0; i<MCHROSEN; i+=2){ // when zero-based is odd, one-based is even!
      p[i]=-1.2;
      p[i+1]=1.0;
    }

#if 0
    /* Jacobian error checking. Results to be taken with a grain of salt!
     * (cf. comments preceding sparselm_chkjac_core() in splm_misc.c)
     */
    { double err[NCHROSEN];

    sparselm_chkjaccrs(chainedRosenbrock, chainedRosenbrock_anjacCRS, p, m, n, nnz, NULL, err); // check CRS Jacobian
    //sparselm_chkjacccs(chainedRosenbrock, chainedRosenbrock_anjacCCS, p, m, n, nnz, NULL, err); // check CCS Jacobian
    for(i=0; i<NCHROSEN; ++i)
      printf("Gradient %d is %s (%g)\n", i,
          (err[i]==0.0)? "wrong" : ((err[i]==1.0)? "correct" : ((err[i]<0.5)? "probably wrong" : "probably correct")),
          err[i]);
    }
#endif

    /* different ways of minimizing Rosenbrock's chained function are demonstrated below */
    ret=sparselm_dercrs(chainedRosenbrock, chainedRosenbrock_anjacCRS, p, NULL, m, 0, n, nnz, -1, 1000, opts, info, NULL); // CRS Jacobian
    //ret=sparselm_derccs(chainedRosenbrock, chainedRosenbrock_anjacCCS, p, NULL, m, 0, n, nnz, -1, 1000, opts, info, NULL); // CCS Jacobian
    /* CCS Jacobian constructed in ST fmt */
    //ret=sparselm_derccs(chainedRosenbrock, chainedRosenbrock_anjacCCS_ST, p, NULL, m, 0, n, nnz, -1, 1000, opts, info, NULL);
    /* CCS Jacobian, its zero pattern is constructed in ST fmt */
    //ret=sparselm_difccs(chainedRosenbrock, chainedRosenbrock_zpjacCCS_ST, p, NULL, m, 0, n, nnz, -1, 1000, opts, info, NULL);
    /* no Jacobian, its zero pattern to be detected */
    //ret=sparselm_difccs(chainedRosenbrock, NULL, p, NULL, m, 0, n, nnz, -1, 1000, opts, info, NULL);
  break;

  case 1:
  /* chained Wood function */
    m=MCHWOOD; n=NCHWOOD; nnz=NCHWOOD/6*10;
    for(i=0; i<MCHWOOD; ++i){
      int i1=i+1; // convert to zero based

      switch(i1%2){
        case 1: p[i]=(i1<=4)? -3.0 : -2.0;
              break;
        case 0: p[i]=(i1>=4)? -1.0 : 0.0;
              break;
      }
    }

    ret=sparselm_dercrs(chainedWood, chainedWood_jac, p, NULL, m, 0, n, nnz, -1, 1000, opts, info, NULL);
  break;
    
  case 2:
  /* chained Powell singular function */
    m=MCHPOWELL; n=NCHPOWELL; nnz=NCHPOWELL*2;
    for(i=0; i<MCHPOWELL; ++i){
      int i1=i+1; // convert to zero based

      switch(i1%4){
        case 1: p[i]=3.0;
              break;
        case 2: p[i]=-1.0;
              break;
        case 3: p[i]=0.0;
              break;
        case 0: p[i]=1.0;
              break;
      }
    }

    ret=sparselm_dercrs(chainedPowell, chainedPowell_jac, p, NULL, m, 0, n, nnz, -1, 1000, opts, info, NULL);
  break;

  case 3:
  /* chained Cragg and Levy function */
    m=MCHCRALEV; n=NCHCRALEV; nnz=NCHCRALEV/5*8;
    for(i=1, p[0]=1.0; i<MCHCRALEV; ++i)
      p[i]=2.0;

    /* different ways of minimizing Cragg and Levy's chained function are demonstrated below */
    ret=sparselm_dercrs(chainedCraggLevy, chainedCraggLevy_anjac, p, NULL, m, 0, n, nnz, -1, 1000, opts, info, NULL); // analytic Jacobian
    //ret=sparselm_difcrs(chainedCraggLevy, chainedCraggLevy_zpjac, p, NULL, m, 0, n, nnz, -1, 1000, opts, info, NULL); // Jacobian's zero pattern
    //ret=sparselm_difcrs(chainedCraggLevy, NULL, p, NULL, m, 0, n, nnz, -1, 1000, opts, info, NULL); // Jacobian's zero pattern to be detected
  break;

  }

  printf("Results for %s:\n", probname[problem]);
  printf("sparseLM returned %d in %g iter, reason %g\nSolution: ", ret, info[5], info[6]);
  for(i=0; i<m; ++i)
    printf("%.7g ", p[i]);
  printf("\n\nMinimization info:\n");
  for(i=0; i<SPLM_INFO_SZ; ++i)
    printf("%g ", info[i]);
  printf("\n");

  exit(0);
}
