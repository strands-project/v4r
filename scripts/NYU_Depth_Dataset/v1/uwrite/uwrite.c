/*                           -*- Mode: C -*-
 *
 * uwrite
 *
 * like fwrite, but will write a double value to a uint8 array
 * A=UWRITE(D, PREC)
 * D is an array of read doubles (standard matlab array)
 * which is written out to A in the PREC format
 *
 *
 * Author          : Sridhar Anandakrishnan <sak@essc.psu.edu>
 * Created On      : Mon Jul 29 12:30:39 2002
 * Last Modified By: Sridhar Anandakrishnan
 * Last Modified On: Mon Jul 29 12:57:11 2002
 * Update Count    : 11
 * Status          : Unknown, Use with caution!
 *
    Copyright (c) 2002 Sridhar Anandakrishnan, sak@essc.psu.edu

 */

/* System Headers */
#include <stdio.h>
#include <math.h>
#include "mex.h"

/* Local Headers */

/* Macros */
#define NDIMS 2

/* File-scope variables */

/* External variables */

/* External functions */

/* Structures and Unions */
union U {
  unsigned char UC;		/* 8 */
  unsigned short int USI;	/* 16 */
  unsigned long int ULI;	/* 32 */
  signed char SC;		/* 8 */
  signed short int SSI;		/* 16 */
  signed long int SLI;		/* 32 */
  float F;			/* 32 */
  double D;			/* 64 */
  char in[8];			/* input array */
} u;

enum precs {SCHAR, UCHAR, INT8, INT16, INT32, 
	    UINT8, UINT16, UINT32,
	    FLOAT32, FLOAT64, DOUBLE};
int prsz[]={1, 1, 1, 2, 4,
	    1, 2, 4,
	    4, 8, 8};		/* prsz[prec] */

/* Signal Catching Functions */

/* Functions */

/* Main */

void
mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{
  int        dims[] = {1, sizeof(double)};
  int        i, m, n, size, prec, precstrlen;
  unsigned char *pu8, *pd;	/* pointer to input and output */
  char *precstr="double";		/* precision specifier */
  double *d;			/* return value */

  /* Examine input (right-hand-side) arguments. */
  /*     mexPrintf("\n\nThere are %d right-hand-side argument(s).\n", nrhs); */
  if(nrhs<2)
    prec=DOUBLE;		/* default output prec */
  else {
    precstrlen=(mxGetM(prhs[1]))*mxGetN(prhs[1]) + 1;
    precstr=mxCalloc(precstrlen, sizeof(char));
    if(mxGetString(prhs[1], precstr, precstrlen))
      mexErrMsgTxt("Not enough space\n");
    /* windows c compiler choked on switch-case */
    if(!strcmp(precstr,"schar")) prec=SCHAR;
    else if(!strcmp(precstr,"uchar")) prec=UCHAR;
    else if(!strcmp(precstr,"int8")) prec=INT8;
    else if(!strcmp(precstr,"int16")) prec=INT16;
    else if(!strcmp(precstr,"int32")) prec=INT32;
    else if(!strcmp(precstr,"uint8")) prec=UINT8;
    else if(!strcmp(precstr,"uint16")) prec=UINT16;
    else if(!strcmp(precstr,"uint32")) prec=UINT32;
    else if(!strcmp(precstr,"float32")) prec=FLOAT32;
    else if(!strcmp(precstr,"float64")) prec=FLOAT64;
    else if(!strcmp(precstr,"double")) prec=DOUBLE;
    else mexErrMsgTxt("Unknown precision specifier");
  }
  /*     mexPrintf("precision is %s: %d", precstr, prec); */

  if(nrhs<1)
    mexErrMsgTxt("\n\nOne input required.");
  m=mxGetM(prhs[0]);
  n=mxGetN(prhs[0]);

  size=m*n;
    
  /* Examine output (left-hand-side) arguments. */
  /*     mexPrintf("\n\nThere are %d left-hand-side argument(s).\n", nlhs); */

  dims[0]=size*prsz[prec];
  dims[1]=1;

  /* make output uint array */
  plhs[0]=mxCreateNumericArray(NDIMS, dims, mxUINT8_CLASS, mxREAL);
  pu8 = (unsigned char *)mxGetPr(plhs[0]);
  /* get input data */
  d = mxGetPr(prhs[0]);

  /*     mexPrintf("d=%f\n", d[0]); */

  /* cast the double into the right type */
  for(i=0;i<size;i++) {
    if(prec==UCHAR || prec==INT8) {		/* SC */
      u.SC=d[i];
    } else if (prec==INT16) {	                /* SSI */
      u.SSI=d[i];
    } else if (prec==INT32) {			/* SLI */
      u.SLI=d[i];
    } else if (prec==UINT8 || prec==UCHAR) {		/* UC */
      u.UC=d[i];
    } else if (prec==UINT16) {		/* USI */
      u.USI=d[i];
    } else if (prec==UINT32) {		/* ULI */
      u.ULI=d[i];
    } else if (prec==FLOAT32)	{		/* F */
      u.F=d[i];
    } else if (prec==FLOAT64 || prec==DOUBLE) {	/* D */
      u.D=d[i];
    } else {
      mexErrMsgTxt("Unknown precision");
    }
    /* copy the bytes out of the union */
    memcpy(pu8+i*prsz[prec], u.in, prsz[prec]);
  }
}

