#ifndef DEVICE_PYTHON_MODULE_H
#define DEVICE_PYTHON_MODULE_H

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
/// \brief Python module for device in PySYCL.
///////////////////////////////////////////////////////////////////////

#include <pybind11/pybind11.h>

#include "Device_Object_Python_Module.h"
#include "Device_Inquiry_Python_Module.h"

namespace py = pybind11;

///////////////////////////////////////////////////////////////////////
// Device module for PySYCL
///////////////////////////////////////////////////////////////////////

PYBIND11_MODULE(device, m){
  m.doc() = R"delim(
    Device module for PySYCL
      This module provides classes and functions for selecting SYCL devices.
  )delim";

  deviceobject_module(m);
  deviceinquiry_module(m);
}

#endif // DEVICE_PYTHON_MODULE_H