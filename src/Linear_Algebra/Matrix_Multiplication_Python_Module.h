#ifndef MATRIX_MULTIPLICATION_PYTHON_MODULE_H
#define MATRIX_MULTIPLICATION_PYTHON_MODULE_H

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
/// \brief Python module for a matrix multiplication in PySYCL.
///////////////////////////////////////////////////////////////////////
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Matrix_Multiplication.h"

namespace py = pybind11;

///////////////////////////////////////////////////////////////////////
// Matrix multiplication function
///////////////////////////////////////////////////////////////////////
void matmul_module(py::module& m){
  m.def("matmul", &pysycl::matmul, R"delim(
  )delim");
}

#endif //MATRIX_MULTIPLICATION_PYTHON_MODULE_H