#ifndef ARRAY2D_MATRIX_MULTIPLICATION_PYTHON_MODULE_H
#define ARRAY2D_MATRIX_MULTIPLICATION_PYTHON_MODULE_H

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
/// \brief Python module for array2D matrix multiplication in PySYCL.
///////////////////////////////////////////////////////////////////////

#include <pybind11/pybind11.h>

#include "Array2D_Matrix_Multiplication.h"

namespace py = pybind11;

using Array2D_Explicit_type = pysycl::Array2D_Explicit;
using Array2D_Shared_type   = pysycl::Array2D_Shared;

///////////////////////////////////////////////////////////////////////
// Array2D Matrix Multiplication functions
///////////////////////////////////////////////////////////////////////
void array2dmatrixmultiplication_module(py::module& m){
  // Multiply two 2D arrays together (basic)
  m.def("matmul", &pysycl::matrix_multiplication_Array2D<Array2D_Explicit_type>, R"delim(
    Description
      Multiply two 2D arrays together.

    Parameters
      arr2D_1 : pysycl.array2D.array2D_explicit or pysycl.array2D.array2D_shared
        First array to multiply.
      arr2D_2 : pysycl.array2D.array2D_explicit or pysycl.array2D.array2D_shared
        Second array to multiply.
      A : float (optional)
        Scalar value to multiply the first array by.
      B : float (optional)
        Scalar value to multiply the second array by.

    Returns
      pysycl.array2D.array2D_explicit or pysycl.array2D.array2D_shared
        Array containing the result of the matrix multiplication.

    Example
      Copy
      ----
      >>> import numpy as np
      >>> from pysycl import device
      >>> from pysycl import array2D
      >>> Q = device.device_object()
      >>> arr2D_1 = array2D.array2D_shared(10, 8, Q)
      >>> arr2D_2 = array2D.array2D_shared(10, 8, Q)
      >>> arr2D_3 = array2D.matrix_multiplication(arr2D_1, arr2D_2)
    )delim",
    py::arg("arr2D_1"),
    py::arg("arr2D_2"),
    py::arg("A")=1.0f,
    py::arg("B")=1.0f,
    py::arg("b") = 4,
    py::arg("kernel_key") = std::string("default"));

  m.def("matmul", &pysycl::matrix_multiplication_Array2D<Array2D_Shared_type>, R"delim(
    matmul: shared version
  )delim",
  py::arg("arr2D_1"),
  py::arg("arr2D_2"),
  py::arg("A")=1.0f,
  py::arg("B")=1.0f,
  py::arg("b") = 4,
  py::arg("kernel_key") = std::string("default"));
}

#endif // ARRAY2D_MATRIX_MULTIPLICATION_PYTHON_MODULE_H
