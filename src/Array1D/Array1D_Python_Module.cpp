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
/// \brief Python module for array_1d in PySYCL.
///////////////////////////////////////////////////////////////////////
#include "Array1D_Python_Module.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

///////////////////////////////////////////////////////////////////////
// Array1D module for PySYCL
///////////////////////////////////////////////////////////////////////
PYBIND11_MODULE(array_1d, m) {
  m.doc() = R"delim(
    Array1D module for PySYCL
      This module provides classes and functions for creating PySYCL 1D arrays.
    )delim";

  array_1d_module(m);
}
