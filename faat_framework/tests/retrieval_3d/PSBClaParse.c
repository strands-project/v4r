// Philip Shilane
// Copyright 2003 Princeton University 
//
// This utility verifies the format of a .cla file.
// A valid cla file has:
// PSB FormatNum
// numCategories numModels
// category parentCategory numModels
// one model id per line
//
// Where parentCategory and numModels can be 0 (zero).
// All top level categories should specify a parent category of 0 (zero).
// Categories and numbers cannot have white space within the name.
// Parent categories must be defined before children categories, 0 is already defined.
// All category names must be unique.
// Arbitrary whitespace is otherwise allowed. 
// The .cla file should be in text format and can have up to 128 characters per line.
//
// This parser only handles PSB format 1.
//

#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <stdlib.h>
#include <assert.h>




