#######################################################################
#######################################################################
##                  CMakeLists.txt for PySYCL
#######################################################################
#######################################################################
## This file is part of the PySYCL software for SYCL development in
## Python.  It is licensed under the MIT licence.  A copy of
## this license, in a file named LICENSE.md, should have been
## distributed with this file.  A copy of this license is also
## currently available at "http://opensource.org/licenses/MIT".
##
## Unless explicitly stated, all contributions intentionally submitted
## to this project shall also be under the terms and conditions of this
## license, without any additional terms or conditions.
#######################################################################
#######################################################################

cmake_minimum_required (VERSION 3.13)
project(PySYCL)

#######################################################################
## Version of C++
#######################################################################
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

#######################################################################
## Build types
#######################################################################
include(CMake/BuildType.cmake)

#######################################################################
## Include dependencies for each module
#######################################################################
add_library(pysycl_project_dependencies INTERFACE)

#######################################################################
## PyBind
#######################################################################
include(CMake/PyBind11.cmake)

#######################################################################
## Subdirectories
#######################################################################
add_subdirectory(Device)
add_subdirectory(Vector)

#######################################################################
## Doxygen documentation
#######################################################################
include(CMake/Doxygen.cmake)

#######################################################################
## Sphinx documentation
#######################################################################
FIND_PROGRAM(SPHINX_API_EXE NAMES sphinx-apidoc)
FIND_PROGRAM(SPHINX_BUILD_EXE NAMES sphinx-build)

set(SPHINX_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/../sphinx")
set(SPHINX_BUILD "${CMAKE_CURRENT_SOURCE_DIR}/../sphinx/_build")
set(SPHINX_MODULE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../build/pysycl")
set(SPHINX_PAGES_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/../sphinx/_build/*")
set(SPHINX_PAGES "${CMAKE_CURRENT_SOURCE_DIR}/../docs/")

add_custom_target(sphinx
  COMMAND ${SPHINX_API_EXE} -o ${SPHINX_SOURCE} ${SPHINX_MODULE_DIR}
  COMMAND ${SPHINX_BUILD_EXE} -b html ${SPHINX_SOURCE} ${SPHINX_BUILD}
  COMMENT "Generating Sphinx documentation"
)

add_custom_target(github-pages
  COMMAND cp -r ${SPHINX_PAGES_SOURCE} ${SPHINX_PAGES}
  COMMENT "Generating Sphinx documentation for github pages"
)

