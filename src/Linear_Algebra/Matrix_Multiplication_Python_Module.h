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

///////////////////////////////////////////////////////////////////////
/// pybind11
///////////////////////////////////////////////////////////////////////
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

///////////////////////////////////////////////////////////////////////
/// local
///////////////////////////////////////////////////////////////////////
#include "../Array2D/Array2D.h"
#include "Matrix_Multiplication.h"

namespace py = pybind11;

///////////////////////////////////////////////////////////////////////
// Matrix multiplication function
///////////////////////////////////////////////////////////////////////
template<typename Scalar_T>
void bind_matmul_module(py::module &m) {
  using Array2D_T = pysycl::Array2D<Scalar_T>;
  m.def("matmul", &pysycl::matmul<Array2D_T>, R"delim(
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
        py::arg("A"), py::arg("B"), py::arg("C"), py::arg("tile_size"));
}

void matmul_module(py::module &m) {
  bind_matmul_module<double>(m);
  bind_matmul_module<float>(m);
  bind_matmul_module<int>(m);
}

#endif // MATRIX_MULTIPLICATION_PYTHON_MODULE_H