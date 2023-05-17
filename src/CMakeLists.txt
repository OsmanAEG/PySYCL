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
## Doxygen documentation
#######################################################################
find_package(Doxygen)

if(DOXYGEN_FOUND)
  set(DOXYGEN_OUTPUT_DIRECTORY ../doxygen)
  set(DOXYGEN_COLLABORATION_GRAPH YES)
  set(DOXYGEN_EXTRACT_ALL YES)
  set(DOXYGEN_CLASS_DIAGRAMS YES)
  set(DOXYGEN_HIDE_UNDOC_RELATIONS NO)
  set(DOXYGEN_HAVE_DOT YES)
  set(DOXYGEN_CLASS_GRAPH YES)
  set(DOXYGEN_CALL_GRAPH YES)
  set(DOXYGEN_CALLER_GRAPH YES)
  set(DOXYGEN_COLLABORATION_GRAPH YES)
  set(DOXYGEN_BUILTIN_STL_SUPPORT YES)
  set(DOXYGEN_EXTRACT_PRIVATE YES)
  set(DOXYGEN_EXTRACT_PACKAGE YES)
  set(DOXYGEN_EXTRACT_STATIC YES)
  set(DOXYGEN_EXTRACT_LOCALMETHODS YES)
  set(DOXYGEN_UML_LOOK YES)
  set(DOXYGEN_UML_LIMIT_NUM_FIELDS 50)
  set(DOXYGEN_TEMPLATE_RELATIONS YES)
  set(DOXYGEN_DOT_GRAPH_MAX_NODES 100)
  set(DOXYGEN_MAX_DOT_GRAPH_DEPTH 0)
  set(DOXYGEN_DOT_TRANSPARENT YES)
  doxygen_add_docs(doxygen ../src/Device/SYCL_Device.cpp)
else()
  message("Doxygen needs to be installed to generate the doxygen documentation")
endif()

#######################################################################
## Subdirectories
#######################################################################
add_subdirectory(Device)
add_subdirectory(Vector)

#######################################################################
## Sphinx documentation
#######################################################################
FIND_PROGRAM(SPHINX_API_EXE NAMES sphinx-apidoc)
FIND_PROGRAM(SPHINX_BUILD_EXE NAMES sphinx-build)
set(SPHINX_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/../docs")
set(SPHINX_BUILD "${CMAKE_CURRENT_SOURCE_DIR}/../docs/_build")
set(SPHINX_MODULE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../build/PySYCL")

add_custom_target(sphinx
  COMMAND ${SPHINX_API_EXE} -o ${SPHINX_SOURCE} ${SPHINX_MODULE_DIR}
  COMMAND ${SPHINX_BUILD_EXE} -b html ${SPHINX_SOURCE} ${SPHINX_BUILD}
  COMMENT "Generating Sphinx documentation"
)

