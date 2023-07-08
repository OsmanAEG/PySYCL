#ifndef ARRAY2D_OPERATIONS_H
#define ARRAY2D_OPERATIONS_H

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
/// \brief Array2D Operations in PySYCL. This header file contains
///        the many operations that can be performed on the Array2D
///        class and classes.
///////////////////////////////////////////////////////////////////////

#include <CL/sycl.hpp>
#include <vector>

#include "../Device/Device_Object.h"
#include "Array2D_Explicit.h"
#include "Array2D_Shared.h"

///////////////////////////////////////////////////////////////////////
/// \addtogroup Array2D
/// @{
namespace pysycl {
///////////////////////////////////////////////////////////////////////
/// \brief Add two Array2D objects together.
/// \param[in] arr2D_1 First Array2D to add.
/// \param[in] arr2D_2 Second Array2D to add.
/// \param[in] A Constant multiplier for the first Array2D.
/// \param[in] B Constant multiplier for the second Array2D.
/// \return The result of the addition.
template <typename Array2D_type>
Array2D_type add_Array2D(const Array2D_type &arr2D_1,
                         const Array2D_type &arr2D_2,
                         const float &A = 1.0f,
                         const float &B = 1.0f) {
  // Check that the two arrays are the same size
  if (arr2D_1.number_of_rows() != arr2D_2.number_of_rows() ||
      arr2D_1.number_of_cols() != arr2D_2.number_of_cols()) {
    throw std::runtime_error("Array2D objects must be the same size to add.");
  }

  // Check that the two arrays are on the same device
  if (arr2D_1.get_device().queue() != arr2D_2.get_device().queue()) {
    throw std::runtime_error("Array2D objects must be on the same device to add.");
  }

  const auto rows   = arr2D_1.number_of_rows();
  const auto cols   = arr2D_1.number_of_cols();
  const auto device = arr2D_1.get_device();

  auto Q = device.queue();

  // Create a new Array2D object to store the result
  Array2D_type result(rows, cols, device);

  // Create a command group to perform the addition
  Q.submit([&](sycl::handler &h){
    const auto data_1 = arr2D_1.get_data_ptr();
    const auto data_2 = arr2D_2.get_data_ptr();
    const auto data_r = result.get_data_ptr();

    h.parallel_for(sycl::range<2>(rows, cols), [=](sycl::id<2> idx){
      int i = idx[0];
      int j = idx[1];

      data_r[i*cols + j] = A*data_1[i*cols + j] + B*data_2[i*cols + j];
    });
  });

  return result;
}

/// @} end "Array2D" doxygen group
} // namespace pysycl

#endif // ARRAY2D_OPERATIONS_H
