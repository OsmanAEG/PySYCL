#######################################################################
#######################################################################
##                  src/CMake/PythonUnitTest.cmake
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

#######################################################################
## Find Python
#######################################################################
find_package(Python COMPONENTS Interpreter REQUIRED)

#######################################################################
## Creating custom target for running Python unit tests
#######################################################################
add_custom_target(run_python_tests
  COMMAND ${CMAKE_COMMAND} -E echo "Running Python tests..."
  COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/Tester_Python.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
