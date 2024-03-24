#ifndef DEVICE_MANAGER_PYTHON_MODULE_H
#define DEVICE_MANAGER_PYTHON_MODULE_H

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
/// \brief Python module for device instance in PySYCL.
///////////////////////////////////////////////////////////////////////

/// Device Management in PySYCL

///////////////////////////////////////////////////////////////////////
// pybind11
///////////////////////////////////////////////////////////////////////
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

///////////////////////////////////////////////////////////////////////
// local
///////////////////////////////////////////////////////////////////////
#include "Device_Manager.h"

namespace py = pybind11;

///////////////////////////////////////////////////////////////////////
// Device class and functions
///////////////////////////////////////////////////////////////////////
void device_manager_module(py::module &m) {
  m.def("get_device_manager", &pysycl::Device_Manager::get_device_manager,
        py::return_value_policy::reference, R"delim(
      Description
        This function returns the device manager.

      Returns
        The PySYCL device manager.

      Example

      )delim");
  m.def("get_device", &pysycl::get_device, py::return_value_policy::reference,
        R"delim(
      Description
        This function returns the device.

      Returns
        The PySYCL device.

      Example

      )delim");
  py::class_<pysycl::Device_Manager>(m, "device_manager", R"delim(
    Description
      This class creates a PySYCL device manager.
    )delim")
      .def("get_device", &pysycl::Device_Manager::get_device,
           py::return_value_policy::reference, R"delim(
      Description
        This function returns a device from the manager.

      Returns
        The PySYCL device.

      Example
        >>> print(my_device.vendor())
        NVIDIA Corporation
      )delim",
           py::arg("platform_index") = 0, py::arg("device_index") = 0);
}

#endif // DEVICE_MANAGER_PYTHON_MODULE_H
