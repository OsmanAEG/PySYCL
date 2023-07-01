#ifndef ARRAY2D_PYBIND_MODULE_H
#define ARRAY2D_PYBIND_MODULE_H

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
/// \brief Python module for array2D in PySYCL.
///////////////////////////////////////////////////////////////////////

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "Array2D_Object.h"
#include "../Device/Device_Object.h"

namespace py = pybind11;

PYBIND11_MODULE(array2D, m){
  m.doc() = R"delim(
    Array2D module in PySYCL
      This module contains the Array2D class for use in PySYCL.
  )delim";

  /////////////////////////////////////////////////////////////////////
  // Array2D class and functions
  /////////////////////////////////////////////////////////////////////
  py::class_<pysycl::Array2D_Object>(m, "array2D_object", R"delim(
    Description
      This class creates a PySYCL Array2D object.
    )delim")
    .def(py::init<int, int, pysycl::Device_Object>(), R"delim(
      Description
        Constructor that creates an Array2D object in PySYCL.

      Constructor Parameters
        M : int
          Number of rows in the array.
        N : int
          Number of columns in the array.
        device : pysycl.device_object
          SYCL device to use for allocation and calculation.

      Returns
        Array2D
          Array2D object in PySYCL.

      Example
        Copy
        ----
        >>> from pysycl import device
        >>> from pysycl import array2D
        >>> Q = device.device_object()
        >>> arr = array2D.array2D(10, 8, Q)
      )delim",
      py::arg("M"),
      py::arg("N"),
      py::arg("device"))
    .def("copy_device_to_host", &pysycl::Array2D_Object::copy_device_to_host, R"delim(
      Description
        Copies the data from the device to the host.

      Parameters
        None

      Returns
        None

      Example
        Copy
        ----
        >>> arr.copy_device_to_host()
      )delim")
    .def("copy_host_to_device", &pysycl::Array2D_Object::copy_host_to_device, R"delim(
      Description
        Copies the data from the host to the device.

      Parameters
        None

      Returns
        None

      Example
        Copy
        ----
        >>> arr.copy_host_to_device()
      )delim")
    .def("get_host_data", &pysycl::Array2D_Object::get_host_data, R"delim(
      Description
        Returns the host data as a numpy array.

      Parameters
        None

      Returns
        numpy.ndarray
          Numpy array containing the host data.

      Example
        Copy
        ----
        >>> arr.copy_device_to_host()
        >>> py_arr = arr.get_host_data()
      )delim")
    .def("fill", &pysycl::Array2D_Object::fill, R"delim(
      Description
        Fills the array with a given value.

      Parameters
        value : float
          Value to fill the array with.

      Returns
        None

      Example
        Copy
        ----
        >>> arr.fill(5.0)
      )delim",
      py::arg("value"))
    .def("fill_element_host", &pysycl::Array2D_Object::fill_element_host, R"delim(
      Description
        Fills a single element in the array with a given value.

      Parameters
        i : int
          Row index of the element to fill.
        j : int
          Column index of the element to fill.
        value : float
          Value to fill the array with.

      Returns
        None

      Example
        Copy
        ----
        >>> arr.fill_element_host(6, 4, 8.0)
      )delim",
      py::arg("value"),
      py::arg("i"),
      py::arg("j"))
    .def("sum_reduction", &pysycl::Array2D_Object::sum_reduction, R"delim(
      Description
        Sums the elements in the array.

      Parameters
        None

      Returns
        float
          Sum of the elements in the array.

      Example
        Copy
        ----
        >>> arr_sum = arr.sum_reduction()
      )delim")
    .def("min_reduction", &pysycl::Array2D_Object::min_reduction, R"delim(
      Description
        Finds the minimum value in the array.

      Parameters
        None

      Returns
        float
          Minimum value in the array.

      Example
        Copy
        ----
        >>> arr_min = arr.min_reduction()
      )delim")
    .def("max_reduction", &pysycl::Array2D_Object::max_reduction, R"delim(
      Description
        Finds the maximum value in the array.

      Parameters
        None

      Returns
        float
          Maximum value in the array.

      Example
        Copy
        ----
        >>> arr_max = arr.max_reduction()
      )delim");
}

#endif // #ifndef ARRAY2D_PYBIND_MODULE_H