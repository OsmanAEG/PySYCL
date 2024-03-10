#ifndef ARRAY_PYTHON_MODULE_H
#define ARRAY_PYTHON_MODULE_H

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
/// \brief Python module for an array object in PySYCL.
///////////////////////////////////////////////////////////////////////
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Array.h"

namespace py = pybind11;

using Scalar_T = float;

///////////////////////////////////////////////////////////////////////
// Device class and functions
///////////////////////////////////////////////////////////////////////
void array_module(py::module& m){
  m.def("array", &pysycl::array_selector<Scalar_T>, R"delim(
  )delim");
}

#endif //ARRAY_PYTHON_MODULE_H
