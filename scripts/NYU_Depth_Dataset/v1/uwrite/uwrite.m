%UWRITE Write binary data to char vector.
%   S = UWRITE(A,PRECISION) writes the elements of matrix A
%   to a char column vector S, translating MATLAB values to the specified
%   precision. The data are written in column order.
%
%   PRECISION controls the form and size of the result.  See the list
%   of allowed precisions under UREAD.  'bitN' and 'ubitN' are not supported.  
%
%   For example,
%
%       s=uwrite(magic(5),'int32')
%
%   creates a 100-byte uint8 column vector S, containing the 25 elements of the
%   5-by-5 magic square, stored as 4-byte integers.
%
%   See also UREAD, FWRITE, FREAD, FPRINTF, SAVE, DIARY.
%
%   Copyright (c) 2002 Sridhar Anandakrishnan, sak@essc.psu.edu
