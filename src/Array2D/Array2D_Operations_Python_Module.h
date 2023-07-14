#ifndef ARRAY2D_OPERATIONS_PYTHON_MODULE_H
#define ARRAY2D_OPERATIONS_PYTHON_MODULE_H

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
/// \brief Python module for array2D operations in PySYCL.
///////////////////////////////////////////////////////////////////////

#include <pybind11/pybind11.h>

#include "Array2D_Operations.h"

namespace py = pybind11;

///////////////////////////////////////////////////////////////////////
// Array2D Operational functions
///////////////////////////////////////////////////////////////////////
void array2doperations_module(py::module& m){
  // Add two 2D arrays together (basic)
  m.def("add", &pysycl::add_Array2D<Array2D_Explicit_type>, R"delim(
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
      >>> arr2D_1 = array2D.array2D_shared(10, 8, Q)
      >>> arr2D_2 = array2D.array2D_shared(10, 8, Q)
      >>> arr2D_3 = array2D.add(arr2D_1, arr2D_2)
    )delim",
    py::arg("arr2D_1"),
    py::arg("arr2D_2"),
    py::arg("A") = 1.0f,
    py::arg("B") = 1.0f);

  m.def("add", &pysycl::add_Array2D<Array2D_Shared_type>, R"delim(
    add shared version.
  )delim",
    py::arg("arr2D_1"),
    py::arg("arr2D_2"),
    py::arg("A") = 1.0f,
    py::arg("B") = 1.0f);

  // Subtract two 2D arrays together (basic)
  m.def("sub", &pysycl::subtract_Array2D<Array2D_Explicit_type>, R"delim(
    Description
      Subtract two 2D arrays together.

    Parameters
      arr2D_1 : pysycl.array2D.array2D_explicit or pysycl.array2D.array2D_shared
        First array to subtract.
      arr2D_2 : pysycl.array2D.array2D_explicit or pysycl.array2D.array2D_shared
        Second array to subtract.
      A : float (optional)
        Scalar value to multiply the first array by.
      B : float (optional)
        Scalar value to multiply the second array by.

    Returns
      pysycl.array2D.array2D_explicit or pysycl.array2D.array2D_shared
        Array containing the result of the subtraction.

    Example
      Copy
      ----
      >>> import numpy as np
      >>> from pysycl import device
      >>> from pysycl import array2D
      >>> Q = device.device_object()
      >>> arr2D_1 = array2D.array2D_shared(10, 8, Q)
      >>> arr2D_2 = array2D.array2D_shared(10, 8, Q)
      >>> arr2D_3 = array2D.subtract(arr2D_1, arr2D_2)
    )delim",
    py::arg("arr2D_1"),
    py::arg("arr2D_2"),
    py::arg("A") = 1.0f,
    py::arg("B") = 1.0f);

  m.def("sub", &pysycl::subtract_Array2D<Array2D_Shared_type>, R"delim(
    sub: shared version.
  )delim",
    py::arg("arr2D_1"),
    py::arg("arr2D_2"),
    py::arg("A") = 1.0f,
    py::arg("B") = 1.0f);

  // Multiply two 2D arrays together (basic)
  m.def("mul", &pysycl::multiply_Array2D<Array2D_Explicit_type>, R"delim(
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
        Array containing the result of the multiply.

    Example
      Copy
      ----
      >>> import numpy as np
      >>> from pysycl import device
      >>> from pysycl import array2D
      >>> Q = device.device_object()
      >>> arr2D_1 = array2D.array2D_shared(10, 8, Q)
      >>> arr2D_2 = array2D.array2D_shared(10, 8, Q)
      >>> arr2D_3 = array2D.multiply(arr2D_1, arr2D_2)
    )delim",
    py::arg("arr2D_1"),
    py::arg("arr2D_2"),
    py::arg("A") = 1.0f,
    py::arg("B") = 1.0f);

  m.def("mul", &pysycl::multiply_Array2D<Array2D_Shared_type>, R"delim(
    mul: shared version.
  )delim",
    py::arg("arr2D_1"),
    py::arg("arr2D_2"),
    py::arg("A") = 1.0f,
    py::arg("B") = 1.0f);

  // Divide two 2D arrays together (basic)
  m.def("div", &pysycl::divide_Array2D<Array2D_Explicit_type>, R"delim(
    Description
      Divide two 2D arrays together.

    Parameters
      arr2D_1 : pysycl.array2D.array2D_explicit or pysycl.array2D.array2D_shared
        First array to divide.
      arr2D_2 : pysycl.array2D.array2D_explicit or pysycl.array2D.array2D_shared
        Second array to divide.
      A : float (optional)
        Scalar value to divide the first array by.
      B : float (optional)
        Scalar value to divide the second array by.

    Returns
      pysycl.array2D.array2D_explicit or pysycl.array2D.array2D_shared
        Array containing the result of the divide.

    Example
      Copy
      ----
      >>> import numpy as np
      >>> from pysycl import device
      >>> from pysycl import array2D
      >>> Q = device.device_object()
      >>> arr2D_1 = array2D.array2D_shared(10, 8, Q)
      >>> arr2D_2 = array2D.array2D_shared(10, 8, Q)
      >>> arr2D_3 = array2D.divide(arr2D_1, arr2D_2)
    )delim",
    py::arg("arr2D_1"),
    py::arg("arr2D_2"),
    py::arg("A") = 1.0f,
    py::arg("B") = 1.0f);

  m.def("div", &pysycl::divide_Array2D<Array2D_Shared_type>, R"delim(
    div: shared version.
  )delim",
    py::arg("arr2D_1"),
    py::arg("arr2D_2"),
    py::arg("A") = 1.0f,
    py::arg("B") = 1.0f);
}

#endif // ARRAY2D_OPERATIONS_H

