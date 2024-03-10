///////////////////////////////////////////////////////////////////////
// This file is part of the PySYCL software for SYCL development in
// Python.  It is licensed under the MIT licence.  A copy of
// this license, in a file named LICENSE.md, should have been
// distributed with this file.  A copy of this license is also
// currently available at "http://opensource.org/licenses/MIT".
//
// Unless explicitly stated, all contributions intentionally submitted
// to this project shall also be under the terms and conditions of this
// license, without any additional terms or conditions.
///////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////
/// \file
/// \brief Python module for array in PySYCL.
///////////////////////////////////////////////////////////////////////
#include <pybind11/pybind11.h>

#include "Array_Python_Module.h"

namespace py = pybind11;

///////////////////////////////////////////////////////////////////////
// Array module for PySYCL
///////////////////////////////////////////////////////////////////////
PYBIND11_MODULE(array, m){
  m.doc() = R"delim(
    Array module for PySYCL
      This module provides classes and functions for creating PySYCL arrays.
    )delim";

  array_module(m);
}