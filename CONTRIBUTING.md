# Contributing to V4R

Please take a moment to review this document in order to make the contribution
process easy and effective for everyone involved.

Following these guidelines helps to communicate that you respect the time of
the developers managing and developing this open source project. In return,
they should reciprocate that respect in addressing your issue or assessing
patches and features.


## Dependencies
V4R is an open-source project with the goal to be easily installed on different platforms by providing released Debian packages for Ubuntu systems. To allow packaging the V4R library, all dependencies need to be defined in `package.xml`. The required names for specific packages can be found [here](https://github.com/strands-project/rosdistro/blob/strands-devel/rosdep/base.yaml) or [here](https://raw.githubusercontent.com/ros/rosdistro/master/rosdep/base.yaml). Packages not included in this list need to be added as [3rdparty libraries](https://rgit.acin.tuwien.ac.at/root/v4r/wikis/how-to-add-third-party-dependency) to V4R. Whenever possible, try to depend on packaged libraries. This especially applies to PCL and OpenCV. Currently this means contribute your code such that it is compatible to PCL 1.7.2 and OpenCV 2.4.9.  
Also, even though V4R stands for Vision for Robotics, our library is independent of ROS. If you need a ROS component, put your core algorithms into this V4R library and create wrapper interfaces in the seperate [v4r_ros_wrappers repository](https://github.com/strands-project/v4r_ros_wrappers).

### Dependency Installation (Ubuntu 14.04 & Ubuntu 16.04)
If you are running a freshly installed Ubuntu 14.04 or Ubuntu 16.04, you can use 'setup.sh' to install all necessary dependencies. Under the hood './setup.sh' installs a phyton script which associates the dependencies listed in the package.xml file with corresponding debian packages and installs them right away.

**Ubuntu 14.04, 'Trusty'**
```
cd /wherever_v4r_is_located/v4r
./setup.sh
```
**Ubuntu 16.04, 'Xenial'**
```
cd /wherever_v4r_is_located/v4r
./setup.sh xenial kinetic
```

After executing ./setup.sh within your v4r root directory you should be able to compile v4r on your machine with

```
cd /wherever_v4r_is_located/v4r
mkdir build && cd build
cmake ..
make -j8
```

If you don't want to use 'setup.sh' for any wired reason, click [here](#HowToBuild) to jump to *"How to Build V4R without using './setup.sh'? (Ubuntu 14.04)"*.

## Using the issue tracker

The issue tracker for [internal](https://rgit.acin.tuwien.ac.at/root/v4r/issues) or [STRANDS](https://github.com/strands-project/v4r/issues) are
the preferred channel for submitting [pull requests](#pull-requests) and
[bug reports](#bugs), but please respect the following
restrictions:

* Please **do not** derail or troll issues. Keep the discussion on topic and
  respect the opinions of others.


<a name="pull-requests"></a>
## Pull requests

Pull requests let you tell others about changes you've pushed to a repository on GitHub. Once a pull request is sent, interested parties can review the set of changes, discuss potential modifications, and even push follow-up commits if necessary. Therefore, this is the preferred way of pushing your changes - **do not** push your changes directly onto the master branch!
Also, keep your pull requests small and focussed on a specific issue/feature. Do not accumulate months of changes into a single pull request! Such a pull request can not be reviewed!

Good pull requests - patches, improvements, new features - are a fantastic
help. They should remain focused in scope and avoid containing unrelated
commits.


<a name="checklist"></a>
### Checklist

Please use the following checklist to make sure that your contribution is well
prepared for merging into V4R:

1. Source code adheres to the coding conventions described in [V4R Style Guide](docs/v4r_style_guide.md).
   But if you modify existing code, do not change/fix style in the lines that
   are not related to your contribution.

2. Commit history is tidy (no merge commits, commits are [squashed](http://davidwalsh.name/squash-commits-git)
   into logical units).

3. Each contributed file has a [license](#license) text on top.


<a name="bugs"></a>
## Bug reports

A bug is a _demonstrable problem_ that is caused by the code in the repository.
Good bug reports are extremely helpful - thank you!

Guidelines for bug reports:

1. **Check if the issue has been reported** &mdash; use GitHub issue search.

2. **Check if the issue has been fixed** &mdash; try to reproduce it using the
   latest `master` branch in the repository.

3. **Isolate the problem** &mdash; ideally create a reduced test
   case.

A good bug report shouldn't leave others needing to chase you up for more
information. Please try to be as detailed as possible in your report. What is
your environment? What steps will reproduce the issue? What would you expect to
be the outcome? All these details will help people to fix any potential bugs.
After the report of a bug, a responsible person will be selected and informed to solve the issue.

<a name="license"></a>
## License

V4R is [MIT licensed](LICENSE.txt), and by submitting a patch, you agree to
allow V4R to license your work under the terms of the MIT
License. The corpus of the license should be inserted as a C++ comment on top
of each `.h` and `.cpp` file:

```cpp
/******************************************************************************
 * Copyright (c) 2016 Firstname Lastname
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 ******************************************************************************/
```

Please note that if the academic institution or company you are affiliated with
does not allow to give up the rights, you may insert an additional copyright
line.

<a name="structure"></a>
## Structure
The repostiory consists of several folders and files containing specific parts of the library. This section gives a short introduction to the most important ones.

### ./3rdparty
See Dependencies.

### ./apps
Bigger code examples and tools (=more than only one file) such as RTMT.
Apps depend on modules.

### ./cmake
Several cmake macros.

### ./docs
Tutorials and further documentations.

### ./modules
Contains all core components of the library and is organized in logical sub folders which are further called 'packages'.
A package holds the source files which are located in './src'. 
The corresponding header files are located in './include/v4r/package_name/'

By following this structure, new modules can be easily added to the corresponding CMakeLists.txt with
```cpp
v4r_define_module(package_name REQUIRED components)
```
i.e. 
```cpp
v4r_define_module(change_detection REQUIRED v4r_common pcl opencv)
```

* ./modules/common &mdash; anything that can be reused by other packages

* ./modules/core &mdash; core is used by every module and does only include macros -> To make your modules visible to other modules you have to add the macros and 'V4R_EXPORTS' to your header files. 
```cpp
#include <v4r/core/macros.h>
...
class V4R_EXPORTS ...
```

### samples 
*./samples/exsamples: short code pieces that demonsrate how to use a module.
*./samples/tools: small tools with only one file

### Other Files
**CITATION.md** &mdash; This files includes bibTex encoded references.
They can be used to cite the appropriate modules if you use V4R in your work.

**CONTRIBUTING.md** &mdash; The file you read at the moment.


<a name="Documentation"></a>
## Documentation
ALWAYS document your code. We use Doxygen. A nice introduction to Doxygen can be found [here](https://www.stack.nl/~dimitri/doxygen/manual/docblocks.html).

The Doxygen documentation has to be compiled localy on your system for the moment.
However, it will be available *online* on gitlab quiet soon.
Bajo will find a nice solution for that using the CI system.

##  <a name="HowToBuild"></a>How to Build V4R without using './setup.sh'? (Ubuntu 14.04)

As mentioned allready, V4R is using a package.xml file and rosdep to install all necessary dependencies. This can be easily done using ./setup.sh.
However, if you don't want to use 'setup.sh' for any reason follow the instructions below.

1. Do all the necessary git magic to download v4r on your machine, then open a console.
2. Install and initialize rosdep, cmake and build-essential
```
sudo apt-get update
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu trusty main" > /etc/apt/sources.list.d/ros-latest.list'
wget http://packages.ros.org/ros.key -O - | sudo apt-key add -
sudo apt-get update
sudo apt-get install -y python-rosdep build-essential cmake
sudo rosdep init
rosdep update
```
3. Install dependencies and compile v4r
```
cd /wherever_v4r_is_located/v4r
rosdep install --from-paths . -i -y -r --rosdistro indigo
mkdir build && cd build
cmake ..
make -j8
```



