#ifndef ARRAY1D_PYTHON_MODULE_H
#define ARRAY1D_PYTHON_MODULE_H

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
/// \brief Python module for an array_1d object in PySYCL.
///////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////
/// pybind11
///////////////////////////////////////////////////////////////////////
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

///////////////////////////////////////////////////////////////////////
/// local
///////////////////////////////////////////////////////////////////////
#include "../Device/Device_Instance.h"
#include "Array1D.h"

namespace py = pybind11;

using Scalar_T = float;
using Array1D_T = pysycl::Array1D<Scalar_T>;

///////////////////////////////////////////////////////////////////////
// Device class and functions
///////////////////////////////////////////////////////////////////////
void array_1d_module(py::module &m) {
  py::class_<Array1D_T>(m, "array_1d", R"delim(
    Description
      This class creates a PySYCL array_1d object.
    )delim")
      .def(py::init<int, pysycl::Device_Instance &>(), R"delim(
      Default Constructor
        Constructor that creates a 1D PySYCL array.

      Parameters
        size: int
          Number of elements.
        device:
          Optional: The target PySYCL device.

        Returns
          A PySYCL 1D array.

        Example
          >>> import pysycl
          >>> A = pysycl.array_1d.array_1d_init(10)
      )delim",
           py::arg("size"), py::arg("device"))
      .def(py::init<py::array_t<Scalar_T>, pysycl::Device_Instance &>(),
           R"delim(
      NumPy Constructor
        Constructor that creates a 1D PySYCL array from a 1D NumPy array.

      Parameters
        np_arr: numpy.array()
          numpy array.
        device:
          Optional: The target PySYCL device.

        Returns
          A PySYCL 1D array.

        Example
          >>> import pysycl
          >>> import numpy as np
          >>> A = pysycl.array_1d(np.random.rand(N).astype(np.float32))
      )delim",
           py::arg("np_arr"), py::arg("device"))
      .def("get_size", &Array1D_T::get_size, R"delim(
      Description
        This function returns the number of elements.

      Returns
        The number of elements.

      Example
        >>> size = A.get_size()
        >>> print(size)
        10
      )delim")
      .def("fill", &Array1D_T::fill, R"delim(
      Description
        This function fills the array with a constant value.

      Parameters
        C : float
          Some scalar constant.

      Example
        >>> A.fill(45.0)
        >>> A.mem_to_cpu()
        >>> print(A[9])
        45.0
      )delim",
           py::arg("C"))
      .def("mem_to_gpu", &Array1D_T::mem_to_gpu, R"delim(
      Description
        This function copies array memory from CPU to GPU.

      Example
        >>> A.mem_to_gpu()
      )delim")
      .def("mem_to_cpu", &Array1D_T::mem_to_cpu, R"delim(
      Description
        This function copies array memory from GPU to CPU.

      Example
        >>> A.mem_to_cpu()
      )delim")
      .def("max", &Array1D_T::max, R"delim(
      Description
        This function finds the maximum value in the array.

      Returns
        The maximum value.

      Example
        >>> max = A.max()
      )delim")
      .def("min", &Array1D_T::min, R"delim(
      Description
        This function finds the minimum value in the array.

      Returns
        The minimum element value.

      Example
        >>> min = A.min()
      )delim")
      .def("sum", &Array1D_T::sum, R"delim(
      Description
        This function finds the sum of all element values in the array.

      Returns
        The sum of all element values.

      Example
        >>> sum = A.sum()
      )delim")
      .def("__getitem__", [](Array1D_T &self, int i) { return self(i); })
      .def("__setitem__",
           [](Array1D_T &self, int i, Scalar_T val) { self(i) = val; })
      .def("__add__",
           [](const Array1D_T &a, const Array1D_T &b) -> Array1D_T {
             return a + b;
           })
      .def("__iadd__",
           [](const Array1D_T &a, const Array1D_T &b) { return a + b; })
      .def("__sub__",
           [](const Array1D_T &a, const Array1D_T &b) -> Array1D_T {
             return a - b;
           })
      .def("__isub__",
           [](const Array1D_T &a, const Array1D_T &b) { return a - b; })
      .def("__mul__",
           [](const Array1D_T &a, const Array1D_T &b) -> Array1D_T {
             return a * b;
           })
      .def("__imul__",
           [](const Array1D_T &a, const Array1D_T &b) { return a * b; })
      .def("__truediv__",
           [](const Array1D_T &a, const Array1D_T &b) -> Array1D_T {
             return a / b;
           })
      .def("__itruediv__",
           [](const Array1D_T &a, const Array1D_T &b) { return a / b; });
}

#endif // ARRAY1D_PYTHON_MODULE_H
