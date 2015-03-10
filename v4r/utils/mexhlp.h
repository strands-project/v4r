/***************************************************************************
 *   Copyright (C) 2010 by Markus Bader                                    *
 *   markus.bader@tuwien.ac.at                                             *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/
/**
 * @file mexhlp.h
 * @author Markus Bader
 * @brief Helper functions for Matlab mex
 **/

#ifndef AK_MXHPL_HPP
#define AK_MXHPL_HPP

#include "mex.h"
#include <vector>
#include <string>
#include <iostream>


namespace V4R {

inline mxArray *getPtr2MxArray ( const void *p ) {
    mxArray *pArray = mxCreateNumericMatrix ( 1, 1, mxUINT32_CLASS, ( mxComplexity ) 0 );
    unsigned int *pDes = ( unsigned int * ) mxGetPr ( pArray );
    *pDes = ( unsigned int ) p;
    return pArray;
}

inline void *getMxArray2Ptr ( const mxArray *pMx ) {
    void *ptr = NULL;
    if ( mxIsUint32 ( pMx ) ) {
        unsigned int p = * ( ( unsigned int* ) mxGetPr ( pMx ) );
        ptr = ( void * ) p;
    } else {
        std::cerr << "wrong pointertype!\n";
    }
    if ( ptr == NULL ) mexErrMsgTxt ( "Object is NULL !\n" );
    return ptr;
}

inline double getMxArray2Double ( const mxArray *pMx ) {
    double v = mxGetScalar ( pMx );
    return v;
}

inline int getMxArray2Int ( const mxArray *pMx ) {
    return ( int ) mxGetScalar ( pMx );
}

inline bool getMxArray2Bool ( const mxArray *pMx ) {
    return ( bool ) mxGetScalar ( pMx );
}

inline mxArray *getInt2MxArray ( int taskID ) {
    mxArray *pArray = mxCreateNumericMatrix ( 1, 1, mxINT32_CLASS, ( mxComplexity )  0 );
    int *pDes = ( int * ) mxGetPr ( pArray );
    *pDes = taskID;
    return pArray;
}

inline mxArray *getVector2MxArray ( const std::vector<std::string> &v ) {
    char **ppStr = new char*[v.size()];
    for ( int i = 0; i < v.size(); i++ ) {
       ppStr[i] = (char*) v[i].c_str();
    }
    mxArray *pList = mxCreateCharMatrixFromStrings((mwSize)v.size(), (const char **)ppStr); 
		delete ppStr;
    return pList;
}
inline mxArray *getVector2MxArray ( const std::vector<double> &v ) {
    mxArray *pMx = mxCreateDoubleMatrix ( v.size(), 1, mxREAL );
    double *p = ( double * ) mxGetPr ( pMx );
    for ( int i = 0; i < v.size(); i++ ) {
        p[i] = v[i];
    }
    return pMx;
}


inline mxArray *getVector2MxArray ( const std::vector<float> &v ) {
    mxArray *pMx = mxCreateDoubleMatrix ( v.size(), 1, mxREAL );
    double *p = ( double * ) mxGetPr ( pMx );
    for ( int i = 0; i < v.size(); i++ ) {
        p[i] = ( double ) v[i];
    }
    return pMx;
}

inline mxArray *getRGB2MxArray ( unsigned char *pSrc, unsigned int width, unsigned int height ) {
    mwSize dims[] = {height, width, 3 };
    mxArray *pMx = mxCreateNumericArray ( 3, dims, mxUINT8_CLASS, ( mxComplexity ) 0 );
    unsigned char *pDes = ( unsigned char* )  mxGetPr ( pMx );
    unsigned int o = width*height;
    unsigned int o2 =  o+o;
    for ( unsigned int h = 0; h < height; h++ ) {
        for ( unsigned int w = 0; w < width; w++ ) {
            unsigned int indexDes = ( h+w*height );
            unsigned int indexSrc = ( h*width+w ) * 3;
            pDes[indexDes] = pSrc[indexSrc];
            pDes[indexDes + o] = pSrc[indexSrc + 1];
            pDes[indexDes + o2] = pSrc[indexSrc + 2];
        }
    }
    return pMx;
}
}
#endif
