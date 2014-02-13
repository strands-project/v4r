/* ////////////////////////////////////////////////////////////////////////////////
// 
//  Matlab MEX utility for the sparse Levenberg - Marquardt minimization algorithm
//  Copyright (C) 2011  Manolis Lourakis (lourakis at ics forth gr)
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

/* Map sparse solver alphanumeric names to indeger codes for use in the options
 * argument of sparselm. E.g. sparselm_spsolv('cholmod') returns SPLM_CHOLMOD (=1)
 */

#include <string.h>
#include <ctype.h>

#include <mex.h>

#include <splm.h>


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
register int i;
int len, spsolver;
char buf[512], *str;
double *ret;

  if(nrhs!=1)
    mexPrintf("%s: only one argument expected! (got %d)\n", mexFunctionName(), nrhs);

  if(mxIsChar(prhs[0])!=1){
    sprintf(buf, "%s: argument must be a string.", mexFunctionName());
    mexErrMsgTxt(buf);
  }

  if(mxGetM(prhs[0])!=1){
    sprintf(buf, "%s: argument must be a string (i.e. char row vector).", mexFunctionName());
    mexErrMsgTxt(buf);
  }

  /* get supplied name */
  len=mxGetN(prhs[0])+1;
  str=mxCalloc(len, sizeof(char));
  i=mxGetString(prhs[0], str, len);
  if(i!=0){
    sprintf(buf, "%s: not enough space. String is truncated.", mexFunctionName());
    mexErrMsgTxt(buf);
  }

  for(i=0; str[i]; ++i)
    str[i]=tolower(str[i]);

  if(!strcmp(str, "cholmod")) spsolver=SPLM_CHOLMOD;
  else if(!strcmp(str, "csparse")) spsolver=SPLM_CSPARSE;
  else if(!strcmp(str, "ldl")) spsolver=SPLM_LDL;
  else if(!strcmp(str, "umfpack")) spsolver=SPLM_UMFPACK;
  else if(!strcmp(str, "ma77")) spsolver=SPLM_MA77;
  else if(!strcmp(str, "ma57")) spsolver=SPLM_MA57;
  else if(!strcmp(str, "ma47")) spsolver=SPLM_MA47;
  else if(!strcmp(str, "ma27")) spsolver=SPLM_MA27;
  else if(!strcmp(str, "pardiso")) spsolver=SPLM_PARDISO;
  else if(!strcmp(str, "dss")) spsolver=SPLM_DSS;
  else if(!strcmp(str, "superlu")) spsolver=SPLM_SuperLU;
  else if(!strcmp(str, "taucs")) spsolver=SPLM_TAUCS;
  else if(!strcmp(str, "spooles")) spsolver=SPLM_SPOOLES;
  else if(!strcmp(str, "mumps")) spsolver=SPLM_MUMPS;
  else {
    sprintf(buf, "%s: unknown sparse solver '%s' specified.", mexFunctionName(), str);
    mexErrMsgTxt(buf);
  }

  plhs[0]=mxCreateDoubleMatrix(1, 1, mxREAL);
  ret=mxGetPr(plhs[0]);
  ret[0]=(double)spsolver;

  mxFree(str);
}
