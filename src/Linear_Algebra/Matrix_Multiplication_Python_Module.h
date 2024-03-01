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
    Description
      This function evaluates a matrix multiplication and returns the result.

    Parameters
      A : pysycl.array2D
        The first matrix for multiplication.

      B : pysycl.array2D
        The second matrix for multiplication.

      wg_size : int
        Optional: Work group size

    Example
      >>> import pysycl
      >>>
      >>> M = 4000
      >>> N = 800
      >>> P = 2500
      >>>
      >>> A = pysycl.array_2d(M, N)
      >>> B = pysycl.array_2d(N, P)
      >>>
      >>> A.fill(8.0)
      >>> B.fill(3.0)
      >>> C = pysycl.linalg.matmul(A, B)
      >>>
      >>> C.mem_to_cpu()
      >>> print(C[30, 50])
      19200.0
  )delim",
  py::arg("A"),
  py::arg("B"),
  py::arg("wg_size") = 4);

  m.def("tiled_matmul", &pysycl::tiled_matmul, R"delim(
    Description
      This function evaluates a matrix multiplication and returns the result.

    Parameters
      A : pysycl.array2D
        The first matrix for multiplication.

      B : pysycl.array2D
        The second matrix for multiplication.

      wg_size : int
        Optional: Work group size

    Example
      >>> C = pysycl.linalg.tiled_matmul(A, B)
  )delim",
  py::arg("A"),
  py::arg("B"),
  py::arg("wg_size") = 4);
}

#endif //MATRIX_MULTIPLICATION_PYTHON_MODULE_H