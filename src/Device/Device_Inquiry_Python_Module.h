#ifndef DEVICE_INQUIRY_PYTHON_MODULE_H
#define DEVICE_INQUIRY_PYTHON_MODULE_H

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
/// \brief Python module for device inquiry in PySYCL.
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
#include "Device_Inquiry.h"

namespace py = pybind11;

///////////////////////////////////////////////////////////////////////
// Device inquiry function
///////////////////////////////////////////////////////////////////////
void device_inquiry_module(py::module &m) {
  m.def("get_device_list", &pysycl::get_device_list, R"delim(
    Description
      This function returns a list of all available devices.

    Returns
      A list of available PySYCL devices.

    Example
      >>> import pysycl
      >>> my_devices = device.device_inquiry()
      >>> print(my_devices)
      ['NVIDIA GeForce RTX 3060 Laptop GPU [0, 0]']
  )delim");
}

#endif // DEVICE_INQUIRY_PYTHON_MODULE_H