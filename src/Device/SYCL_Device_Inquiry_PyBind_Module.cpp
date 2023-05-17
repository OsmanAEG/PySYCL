#ifndef SYCL_DEVICE_INQUIRY_PYBIND_MODULE_CPP
#define SYCL_DEVICE_INQUIRY_PYBIND_MODULE_CPP

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
/// \brief Python module for SYCL device inquiry in PySYCL.
///////////////////////////////////////////////////////////////////////

#include "SYCL_Device_Inquiry.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

PYBIND11_MODULE(device_inquiry, m) {
  namespace py = pybind11;

  m.doc() = R"delim(
    SYCL device inquiry sub-module for PySYCL
      This sub-module provides available platforms and devices for use.
  )delim";

  m.def("platform_list", &pysycl::platform_list, R"delim(
    Description
      This function returns a list of available SYCL platforms.

    Returns
      list
        A list of available SYCL platforms.

    Example
      >>> from pysycl import device_inquiry
      >>> device_inquiry.platform_list()
      [NVIDIA CUDA BACKEND, Intel(R) OpenCL, Intel(R) Level-Zero]
  )delim");

  m.def("device_list", &pysycl::device_list, R"delim(
    Description
      This function returns a list of available SYCL devices for a given platform.

    Parameters
      platform_index: int, default = 0
        Optional: Index for the sycl platform to select.

    Returns
      list
        A list of available SYCL devices for the given platform.

    Example
      >>> from pysycl import device_inquiry
      >>> device_inquiry.device_list(0)
      [NVIDIA Geforce GTX 1080, NVIDIA Geforce GTX 1080 Ti, NVIDIA Geforce GTX 1080 Ti]
  )delim"),
  py::arg("platform_index") = 0;
}

#endif // #ifndef SYCL_DEVICE_INQUIRY_PYBIND_MODULE_CPP