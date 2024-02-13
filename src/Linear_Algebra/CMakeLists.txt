#######################################################################
#######################################################################
##                  CMakeLists.txt for PySYCL
##                    subdirectory: Linear_Algebra
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
# Setting the name of the PyBind module
#######################################################################
option(PYSYCL_Linear_Algebra "Compile the PySYCL Linear_Algebra module" ON)
if(PYSYCL_Linear_Algebra)
  PySYCL_add_pybind11_module(linalg Linear_Algebra_Python_Module.cpp)
endif()