# ######################################################################
# ######################################################################
# # CMakeLists.txt for PySYCL
# # subdirectory : Array2D
# ######################################################################
# ######################################################################
# # This file is part of the PySYCL software for SYCL development in
# # Python.It is licensed under the MIT licence.A copy of
# # this license, in a file named LICENSE.md, should have been
# # distributed with this file.A copy of this license is also
# # currently available at "http://opensource.org/licenses/MIT".
# #
# # Unless explicitly stated, all contributions intentionally submitted
# # to this project shall also be under the terms and conditions of this
# # license, without any additional terms or conditions.
# ######################################################################
# ######################################################################

# ######################################################################
# # Setting the name of the PyBind module
# ######################################################################
option(PYSYCL_Array2D "Compile the PySYCL Array2D module" ON)

if(PYSYCL_Array2D)
  PySYCL_add_pybind11_module(array_2d Array2D_Python_Module.cpp)
endif()