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

#include "Array2D_Explicit.h"
#include "Array2D_Shared.h"
#include "Array2D_Operations.h"
#include "../Device/Device_Object.h"

using Array2D_Explicit_type = pysycl::Array2D_Explicit;
using Array2D_Shared_type = pysycl::Array2D_Shared;

namespace py = pybind11;

PYBIND11_MODULE(array2D, m){
  m.doc() = R"delim(
    Array2D module in PySYCL
      This module contains the classes and functions that pertain to the
      creation of PySYCL 2D array objects and their manipulation.
  )delim";

  /////////////////////////////////////////////////////////////////////////////
  // Array2D_Explicit class and functions
  /////////////////////////////////////////////////////////////////////////////
  py::class_<pysycl::Array2D_Explicit>(m, "array2D_explicit", R"delim(
    Description
      This class is used to create a 2D array object in PySYCL. It is the
      explicit version of the Array2D class, meaning that the user has to
      explicitly move data between the host and device.
  )delim")
  .def(py::init<int, int, pysycl::Device_Object>(), R"delim(
    Description
      Constructor for the Array2D_Explicit class.

    Constructor Parameters
      rows : int
        Number of rows in the array.
      cols : int
        Number of columns in the array.
      device : pysycl.device.device_object
        SYCL device object to use for allocation and manipulation.

    Returns
      array2D_explicit
        array2D_explicity object in pysycl.array2D.

    Example
      Copy
      ----
      >>> from pysycl import device
      >>> from pysycl import array2D
      >>> Q = device.device_object()
      >>> arr2D = array2D.array2D_explicit(10, 8, Q)
    )delim",
    py::arg("rows"),
    py::arg("cols"),
    py::arg("device"))
  .def("number_of_rows", &pysycl::Array2D_Explicit::number_of_rows, R"delim(
    Description
      Get the number of rows in the array.

    Parameters
      None

    Returns
      int
        Number of rows in the array.

    Example
      Copy
      ----
      >>> from pysycl import device
      >>> from pysycl import array2D
      >>> Q = device.device_object()
      >>> arr2D = array2D.array2D_explicit(10, 8, Q)
      >>> arr2D.get_rows()
      10
    )delim")
  .def("number_of_cols", &pysycl::Array2D_Explicit::number_of_cols, R"delim(
    Description
      Get the number of columns in the array.

    Parameters
      None

    Returns
      int
        Number of columns in the array.

    Example
      Copy
      ----
      >>> from pysycl import device
      >>> from pysycl import array2D
      >>> Q = device.device_object()
      >>> arr2D = array2D.array2D_explicit(10, 8, Q)
      >>> arr2D.get_cols()
      8
    )delim")
  .def("get_device", &pysycl::Array2D_Explicit::get_device, R"delim(
    Description
      Get the SYCL device object associated with the array.

    Parameters
      None

    Returns
      pysycl.device.device_object
        SYCL device object associated with the array.

    Example
      Copy
      ----
      >>> from pysycl import device
      >>> from pysycl import array2D
      >>> Q = device.device_object()
      >>> arr2D = array2D.array2D_explicit(10, 8, Q)
      >>> Q_arr2D = arr2D.get_device()
    )delim")
  .def("copy_host_to_device", &pysycl::Array2D_Explicit::copy_host_to_device, R"delim(
    Description
      Copy the host array to the device array.

    Parameters
      None

    Returns
      None

    Example
      Copy
      ----
      >>> from pysycl import device
      >>> from pysycl import array2D
      >>> Q = device.device_object()
      >>> arr2D = array2D.array2D_explicit(10, 8, Q)
      >>> arr2D.copy_host_to_device()
    )delim")
  .def("copy_device_to_host", &pysycl::Array2D_Explicit::copy_device_to_host, R"delim(
    Description
      Copy the device array to the host array.

    Parameters
      None

    Returns
      None

    Example
      Copy
      ----
      >>> from pysycl import device
      >>> from pysycl import array2D
      >>> Q = device.device_object()
      >>> arr2D = array2D.array2D_explicit(10, 8, Q)
      >>> arr2D.copy_device_to_host()
    )delim")
  .def("set_host_value", &pysycl::Array2D_Explicit::set_host_value, R"delim(
    Description
      Set the value of an element in the host array.

    Parameters
      row : int
        Row index of the element to set.
      col : int
        Column index of the element to set.
      value : float
        Value to set the element to.

    Returns
      None

    Example
      Copy
      ----
      >>> from pysycl import device
      >>> from pysycl import array2D
      >>> Q = device.device_object()
      >>> arr2D = array2D.array2D_explicit(10, 8, Q)
      >>> arr2D.set_host_value(2, 3, 5.0)
    )delim",
    py::arg("row"),
    py::arg("col"),
    py::arg("value"))
  .def("set_host_value", &pysycl::Array2D_Explicit::set_host_value, R"delim(
    Description
      Set the value of an element in the host array.

    Parameters
      row : int
        Row index of the element to set.
      col : int
        Column index of the element to set.
      value : float
        Value to set the element to.

    Returns
      None

    Example
      Copy
      ----
      >>> from pysycl import device
      >>> from pysycl import array2D
      >>> Q = device.device_object()
      >>> arr2D = array2D.array2D_explicit(10, 8, Q)
      >>> arr2D.get_host_value(2, 3)
    )delim",
    py::arg("row"),
    py::arg("col"),
    py::arg("value"))
  .def("get_host_data", &pysycl::Array2D_Explicit::get_host_data, R"delim(
    Description
      Get the host array data.

    Parameters
      None

    Returns
      numpy.ndarray
        Numpy array containing the host array data.

    Example
      Copy
      ----
      >>> import numpy as np
      >>> from pysycl import device
      >>> from pysycl import array2D
      >>> Q = device.device_object()
      >>> arr2D = array2D.array2D_explicit(10, 8, Q)
      >>> host_data = arr2D.get_host_data()
      >>> print(host_data)
      [[0. 0. 0. 0. 0. 0. 0. 0.]
       [0. 0. 0. 0. 0. 0. 0. 0.]
       [0. 0. 0. 0. 0. 0. 0. 0.]
       [0. 0. 0. 0. 0. 0. 0. 0.]
       [0. 0. 0. 0. 0. 0. 0. 0.]
       [0. 0. 0. 0. 0. 0. 0. 0.]
       [0. 0. 0. 0. 0. 0. 0. 0.]
       [0. 0. 0. 0. 0. 0. 0. 0.]
       [0. 0. 0. 0. 0. 0. 0. 0.]
       [0. 0. 0. 0. 0. 0. 0. 0.]]
    )delim");

  /////////////////////////////////////////////////////////////////////////////
  // Array2D_Shared class and functions
  /////////////////////////////////////////////////////////////////////////////
  py::class_<pysycl::Array2D_Shared>(m, "array2D_shared", R"delim(
    Description
      This class is used to create a 2D array object in PySYCL. It is the
      shared memory version of the Array2D class. It is used to create a 2D
      array that is shared between the host and device. Data movement between
      the host and device is done implicitly.
  )delim")
  .def(py::init<int, int, pysycl::Device_Object>(), R"delim(
    Description
      Constructor for the Array2D_Shared class.

    Constructor Parameters
      rows : int
        Number of rows in the array.
      cols : int
        Number of columns in the array.
      device : pysycl.device.device_object
        SYCL device object to use for allocation and manipulation.

    Returns
      array2D_shared
        array2D_shared object in pysycl.array2D.

    Example
      Copy
      ----
      >>> from pysycl import device
      >>> from pysycl import array2D
      >>> Q = device.device_object()
      >>> arr2D = array2D.array2D_shared(10, 8, Q)
    )delim",
    py::arg("rows"),
    py::arg("cols"),
    py::arg("device"))
  .def("number_of_rows", &pysycl::Array2D_Shared::number_of_rows, R"delim(
    Description
      Get the number of rows in the array.

    Parameters
      None

    Returns
      int
        Number of rows in the array.

    Example
      Copy
      ----
      >>> from pysycl import device
      >>> from pysycl import array2D
      >>> Q = device.device_object()
      >>> arr2D = array2D.array2D_shared(10, 8, Q)
      >>> arr2D.get_rows()
      10
    )delim")
  .def("number_of_cols", &pysycl::Array2D_Shared::number_of_cols, R"delim(
    Description
      Get the number of columns in the array.

    Parameters
      None

    Returns
      int
        Number of columns in the array.

    Example
      Copy
      ----
      >>> from pysycl import device
      >>> from pysycl import array2D
      >>> Q = device.device_object()
      >>> arr2D = array2D.array2D_shared(10, 8, Q)
      >>> arr2D.get_cols()
      8
    )delim")
  .def("get_device", &pysycl::Array2D_Shared::get_device, R"delim(
    Description
      Get the SYCL device object associated with the array.

    Parameters
      None

    Returns
      pysycl.device.device_object
        SYCL device object associated with the array.

    Example
      Copy
      ----
      >>> from pysycl import device
      >>> from pysycl import array2D
      >>> Q = device.device_object()
      >>> arr2D = array2D.array2D_shared(10, 8, Q)
      >>> Q_arr2D = arr2D.get_device()
    )delim")
  .def("set_value", &pysycl::Array2D_Shared::set_value, R"delim(
    Description
      Set the value of an element in the host array.

    Parameters
      row : int
        Row index of the element to set.
      col : int
        Column index of the element to set.
      value : float
        Value to set the element to.

    Returns
      None

    Example
      Copy
      ----
      >>> from pysycl import device
      >>> from pysycl import array2D
      >>> Q = device.device_object()
      >>> arr2D = array2D.array2D_shared(10, 8, Q)
      >>> arr2D.set_value(2, 3, 5.0)
    )delim",
    py::arg("row"),
    py::arg("col"),
    py::arg("value"))
  .def("get_value", &pysycl::Array2D_Shared::get_value, R"delim(
    Description
      Set the value of an element in the host array.

    Parameters
      row : int
        Row index of the element to set.
      col : int
        Column index of the element to set.
      value : float
        Value to set the element to.

    Returns
      None

    Example
      Copy
      ----
      >>> from pysycl import device
      >>> from pysycl import array2D
      >>> Q = device.device_object()
      >>> arr2D = array2D.array2D_shared(10, 8, Q)
      >>> arr2D.get_value(2, 3)
    )delim",
    py::arg("row"),
    py::arg("col"))
  .def("get_data", &pysycl::Array2D_Shared::get_data, R"delim(
    Description
      Get the host array data.

    Parameters
      None

    Returns
      numpy.ndarray
        Numpy array containing the host array data.

    Example
      Copy
      ----
      >>> import numpy as np
      >>> from pysycl import device
      >>> from pysycl import array2D
      >>> Q = device.device_object()
      >>> arr2D = array2D.array2D_shared(10, 8, Q)
      >>> data = arr2D.get_data()
      >>> print(data)
      [[0. 0. 0. 0. 0. 0. 0. 0.]
       [0. 0. 0. 0. 0. 0. 0. 0.]
       [0. 0. 0. 0. 0. 0. 0. 0.]
       [0. 0. 0. 0. 0. 0. 0. 0.]
       [0. 0. 0. 0. 0. 0. 0. 0.]
       [0. 0. 0. 0. 0. 0. 0. 0.]
       [0. 0. 0. 0. 0. 0. 0. 0.]
       [0. 0. 0. 0. 0. 0. 0. 0.]
       [0. 0. 0. 0. 0. 0. 0. 0.]
       [0. 0. 0. 0. 0. 0. 0. 0.]]
    )delim");

  /////////////////////////////////////////////////////////////////////////////
  // Array2D Operational functions
  /////////////////////////////////////////////////////////////////////////////
  m.def("add_array2D", &pysycl::add_Array2D<Array2D_Explicit_type>, R"delim(
    Description
      Add two 2D arrays together.

    Parameters
      arr2D_1 : pysycl.array2D.array2D_explicit or pysycl.array2D.array2D_shared
        First array to add.
      arr2D_2 : pysycl.array2D.array2D_explicit or pysycl.array2D.array2D_shared
        Second array to add.
      A : float (optional)
        Scalar value to multiply the first array by.
      B : float (optional)
        Scalar value to multiply the second array by.

    Returns
      pysycl.array2D.array2D_explicit or pysycl.array2D.array2D_shared
        Array containing the result of the addition.

    Example
      Copy
      ----
      >>> import numpy as np
      >>> from pysycl import device
      >>> from pysycl import array2D
      >>> Q = device.device_object()
      >>> arr2D_1 = array2D.add_array2D(10, 8, Q)
      >>> arr2D_2 = array2D.add_array2D(10, 8, Q)
      >>> arr2D_3 = array2D.add_array2D(arr2D_1, arr2D_2)
    )delim",
    py::arg("arr2D_1"),
    py::arg("arr2D_2"),
    py::arg("A") = 1.0f,
    py::arg("B") = 1.0f);

  m.def("add_array2D", &pysycl::add_Array2D<Array2D_Shared_type>, R"delim(
    add_array2D: shared version.
  )delim",
    py::arg("arr2D_1"),
    py::arg("arr2D_2"),
    py::arg("A") = 1.0f,
    py::arg("B") = 1.0f);
}

#endif // ARRAY2D_PYBIND_MODULE_H