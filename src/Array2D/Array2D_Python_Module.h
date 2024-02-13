#ifndef ARRAY2D_PYTHON_MODULE_H
#define ARRAY2D_PYTHON_MODULE_H

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
/// \brief Python module for an array_2d object in PySYCL.
///////////////////////////////////////////////////////////////////////
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Array2D.h"

#include "../Device/Device_Instance.h"

namespace py = pybind11;

using Array2D_T = pysycl::Array2D;

///////////////////////////////////////////////////////////////////////
// Device class and functions
///////////////////////////////////////////////////////////////////////
void array_2d_module(py::module& m){
  py::class_<Array2D_T>(m, "array_2d_init", R"delim(
    Description
      This class creates a PySYCL array_2d object.
    )delim")
    .def(py::init<int, int, pysycl::Device_Instance>(), R"delim(
      Default Constructor
        Constructor that creates a 2D PySYCL array.

      Parameters
        cols: int
          Number of columns.
        rows: int
          Number of rows.
        device:
          Optional: The target PySYCL device.

        Returns
          A PySYCL 2D array.

        Example
          >>> import pysycl
          >>> A = pysycl.array_2d.array_2d_init(10, 12)
      )delim",
      py::arg("rows"),
      py::arg("cols"),
      py::arg("device"))
    .def("num_rows", &Array2D_T::num_rows, R"delim(
      Description
        This function returns the number of rows.

      Returns
        The number of rows.

      Example
        >>> rows = A.num_rows()
        >>> print(rows)
        12
      )delim")
    .def("num_cols", &Array2D_T::num_cols, R"delim(
      Description
        This function returns the number of columns.

      Returns
        The number of columns.

      Example
        >>> rows = A.num_cols()
        >>> print(cols)
        10
      )delim")
    .def("fill", &Array2D_T::fill, R"delim(
      Description
        This function fills the array with a constant value

      Parameters
        C : double
          Some scalar constant.

      Example
        >>> A.fill(45.0)
        >>> print(A[9, 7])
        45.0
      )delim")
    .def("__getitem__", [](Array2D_T &self, std::pair<int, int> idx){
      return self(idx.first, idx.second);})
    .def("__setitem__", [](Array2D_T &self, std::pair<int, int> idx, double val){
      self(idx.first, idx.second) = val;})
    .def("__add__",      [](Array2D_T &a, Array2D_T &b) -> Array2D_T {return a + b;})
    .def("__iadd__",     [](Array2D_T &a, Array2D_T &b){return a + b;})
    .def("__sub__",      [](Array2D_T &a, Array2D_T &b) -> Array2D_T {return a - b;})
    .def("__isub__",     [](Array2D_T &a, Array2D_T &b){return a - b;})
    .def("__mul__",      [](Array2D_T &a, Array2D_T &b) -> Array2D_T {return a * b;})
    .def("__imul__",     [](Array2D_T &a, Array2D_T &b){return a * b;})
    .def("__truediv__",  [](Array2D_T &a, Array2D_T &b) -> Array2D_T {return a / b;})
    .def("__itruediv__", [](Array2D_T &a, Array2D_T &b){return a / b;});
}

#endif //ARRAY2D_PYTHON_MODULE_H