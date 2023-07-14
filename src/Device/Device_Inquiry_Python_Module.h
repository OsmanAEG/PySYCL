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

#include <pybind11/pybind11.h>

#include "Device_Inquiry.h"

namespace py = pybind11;

///////////////////////////////////////////////////////////////////////
// Device inquiry functions
///////////////////////////////////////////////////////////////////////

void deviceinquiry_module(py::module& m){
  m.def("platform_list", &pysycl::platform_list, R"delim(
    .. figure:: _static/images/platforms.png
      :scale: 50 %
      :alt: Common Platforms

      Common Platforms

    Description
      This function returns a list of available SYCL platforms.

    Returns
      list
        A list of available SYCL platforms.

    Example
      Copy
      ----
      >>> from pysycl import device
      >>> print(device.platform_list())
      ['NVIDIA CUDA BACKEND ( platform index = 0)']
  )delim")
  .def("device_list", &pysycl::device_list, R"delim(
    .. figure:: _static/images/gpu.png
      :scale: 50 %
      :alt: Device Selection

      Device Selection(GPU, CPU, FPGA)

    Description
      This function returns a list of available SYCL devices.

    Returns
      list
        A list of available SYCL devices.

    Example
      Copy
      ----
      >>> from pysycl import device
      >>> print(device.device_list(0))
      ['NVIDIA GeForce RTX 3060 Laptop GPU ( device index = 0)']
  )delim"),
  py::arg("platform_index") = 0;
}

#endif // DEVICE_INQUIRY_PYTHON_MODULE_H