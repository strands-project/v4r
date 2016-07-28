# 1. Naming
## 1.1. Files

All files should be *under_scored*.

* Header files have the extension .h
* Templated implementation files have the extension .hpp
* Source files have the extension .cpp


## 1.2. Directories

All directories and subdirectories should be under_scored.

* Header files should go under include/v4r/module_name/
* Templated implementation files should go under include/v4r/module_name/impl/
* Source files should go under src/

## 1.3. Includes

All include statement are made with <chevron_brackets>, e.g.:
```cpp
#include <pcl/module_name/file_name.h>
#incluce <pcl/module_name/impl/file_name.hpp>
```
Also keep the list of includes clean and orderly by first including system level, then external libraries, then v4r internal. 
I.e. as a general rule: the more general include files go first. Sort includes within a group alphabetically. 

## 1.4. Defines & Macros & Include guards

Macros should all be *ALL_CAPITALS_AND_UNDERSCORED*. 
To avoid the problem of double inclusion of header files, guard each header file with a `#pragma once` statement placed just past the MIT license.
```cpp
// the license
#pragma once
// the code
```
You might still encounter include guards in existing files which names are mapped from their include name, e.g.: v4r/filters/bilateral.h becomes V4R_FILTERS_BILATERAL_H_.
```cpp
// the license

#ifndef V4R_MODULE_NAME_FILE_NAME_H_
#define V4R_MODULE_NAME_FILE_NAME_H_

// the code

#endif
```

## 1.5. Namespaces
Put all your code into the namespace *v4r* and avoid using sub-namespaces whenever possible.

```cpp
namespace v4r
{
    ...
}
```

## 1.6. Classes / Structs

Class names (and other type names) should be *CamelCased* . Exception: if the class name contains a short acronym, the acronym itself should be all capitals. Class and struct names are preferably nouns: `PFHEstimation` instead of `EstimatePFH`.

Correct examples:
```cpp
class ExampleClass;
class PFHEstimation;
```

## 1.7. Functions / Methods

Functions and class method names should be *camelCased*, and arguments are *under_scored*. Function and method names are preferably verbs, and the name should make clear what it does: `checkForErrors()` instead of `errorCheck()`, `dumpDataToFile()` instead of `dataFile()`.

Correct usage:
```cpp
int
applyExample (int example_arg);
```
## 1.8. Variables

Variable names should be *under_scored*.
```cpp
int my_variable;
```
Give meaningful names to all your variables.

## 1.8.1. Constants

Constants should be *ALL_CAPITALS*, e.g.:

```cpp
const static int MY_CONSTANT = 1000;
```

## 1.8.2. Member variables

Variables that are members of a class are *under_scored_*, with a trailing underscore added, e.g.:
```cpp
int example_int_;
```

# 2. Indentation and Formatting

V4R uses a variant of the Qt style formatting. The standard indentation for each block is 4 spaces. If possible, apply this measure for your tabs and other spacings.
## 2.1. Namespaces

In a header file, the contents of a namespace should be indented, e.g.:

```cpp
namespace v4r
{
    class Foo
    {
        ...
    };
}
```
## 2.2. Classes

The template parameters of a class should be declared on a different line, e.g.:

```cpp
template <typename T>
class Foo
{
    ...
}
```

## 2.3. Functions / Methods

The return type of each function declaration must be placed on a different line, e.g.:

```cpp
void
bar ();
```

Same for the implementation/definition, e.g.:

```cpp
void
bar ()
{
  ...
}
```

or
```cpp
void
Foo::bar ()
{
  ...
}
```

or

```cpp
template <typename T> void
Foo<T>::bar ()
{
  ...
}
```

## 2.4. Braces

Braces, both open and close, go on their own lines, e.g.:
```cpp
if (a < b)
{
    ...
}
else
{
    ...
}
```
Braces can be omitted if the enclosed block is a single-line statement, e.g.:
```cpp
if (a < b)
    x = 2 * a;
```


# 3. Structuring
## 3.1. Classes and API

For most classes in V4R, it is preferred that the interface (all public members) does not contain variables and only two types of methods:

* The first method type is the get/set type that allows to manipulate the parameters and input data used by the class.
* The second type of methods is actually performing the class functionality and produces output, e.g. compute, filter, segment.

## 3.2. Passing arguments

For getter/setter methods the following rules apply:

* If large amounts of data needs to be set it is preferred to pass either by a *const* reference or by a *const* boost shared_pointer instead of the actual data.
* Getters always need to pass exactly the same types as their respective setters and vice versa.
* For getters, if only one argument needs to be passed this will be done via the return keyword. If two or more arguments need to be passed they will all be passed by reference instead.

For the compute, filter, segment, etc. type methods the following rules apply:

*  The output arguments are preferably non-pointer type, regardless of data size.
* The output arguments will always be passed by reference.

## 3.3 Use const
To allow clients of your class to immediately see which variable can be altered and which are used for read only access, define input arguments *const*. The same applies for member functions which do not change member variables.

## 3.3 Do not clutter header files
To reduce compile time amongst others, put your definitions into seperate .cpp or .hpp files. Define functions inline in the header only when they are small, say, 10 lines or fewer.

## 3.4 Fix warnings
To keep the compiler output non-verbose and reduce potential conflicts, avoid warnings produced by the compiler. If you encounter warnings from other parts in the library, either try to fix them directly or report them using the issue tracker.

## 3.5 Check input range and handle exceptions
To avoid confusing runtime errors and undefined behaviour, check the input to your interfaces for potential conflicts and provide meaningful error messages. If possible, try to catch these exceptions and return in a well-defined state.

## 3.6 Document
V4R uses [Doxygen](https://www.stack.nl/~dimitri/doxygen/manual/docblocks.html) for documentation of classes and interfaces. Any code in V4R must be documented using Doxygen format.

## 3.7 Template your classes/functions whenever appropriate  
As many classes and functions in V4R depend on e.g. templated point clouds, allow your classes to accommodate with the various types by using template classes and functions.
