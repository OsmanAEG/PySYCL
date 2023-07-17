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
/// \brief Python module for array2D in PySYCL.
///////////////////////////////////////////////////////////////////////

#include <pybind11/pybind11.h>

#include "Array2D_Explicit/Array2D_Explicit_Python_Module.h"
#include "Array2D_Shared/Array2D_Shared_Python_Module.h"
#include "Array2D_Operations/Array2D_Operations_Python_Module.h"
#include "Array2D_Matrix_Multiplication/Array2D_Matrix_Multiplication_Python_Module.h"

namespace py = pybind11;

///////////////////////////////////////////////////////////////////////
// Array2D module for PySYCL
///////////////////////////////////////////////////////////////////////

PYBIND11_MODULE(array2D, m){
  m.doc() = R"delim(
    Array2D module for PySYCL
      This module provides classes and functions for creating and
      manipulating 2D arrays in PySYCL.
  )delim";

  array2dexplicit_module(m);
  array2dshared_module(m);
  array2doperations_module(m);
  array2dmatrixmultiplication_module(m);
}

#endif // ARRAY2D_PYTHON_MODULE_H