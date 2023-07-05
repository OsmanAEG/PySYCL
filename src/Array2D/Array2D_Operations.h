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
/// \brief Array2D operations in PySYCL. These operations receive
///        two individual arrays and return a single array.
///////////////////////////////////////////////////////////////////////

#include <CL/sycl.hpp>
#include <vector>

#include "../Device/Device_Object.h"
#include "../Array2D/Array2D_Object.h"

///////////////////////////////////////////////////////////////////////
/// \addtogroup Array2D
/// @{

namespace pysycl {
///////////////////////////////////////////////////////////////////////
/// \brief Add two 2D arrays together.
/// \param[in] array2D_1 First array to add.
/// \param[in] array2D_2 Second array to add.
/// \return Array2D object containing the sum of the two arrays.
Array2D_Object add_Arrays2D(const Array2D_Object &array2D_1,
                            const Array2D_Object &array2D_2,
                            const float &A = 1.0,
                            const float &B = 1.0){
  // Check that the arrays are the same size.
  if(array2D_1.get_rows()    != array2D_2.get_rows() ||
     array2D_1.get_columns() != array2D_2.get_columns()){
    throw std::runtime_error("Arrays are not the same size.");
  }

  if(array2D_1.get_device().queue() != array2D_2.get_device().queue()){
    throw std::runtime_error("Arrays are not on the same device.");
  }

  const auto M      = array2D_1.get_rows();
  const auto N      = array2D_1.get_columns();
  const auto device = array2D_1.get_device();

  auto Q = device.queue();

  // Create a new array to store the result.
  Array2D_Object result(M, N, device);

  // Create a command group to perform the addition.
  Q.submit([&](sycl::handler &h){
    const auto data_1 = array2D_1.get_data_device();
    const auto data_2 = array2D_2.get_data_device();
    const auto data_3 = result.get_data_device();

    h.parallel_for(M*N, [=](sycl::id<1> idx){
      data_3[idx] = A*data_1[idx] + B*data_2[idx];
    });
  });

  return result;
}

///////////////////////////////////////////////////////////////////////
/// \brief Subtract two 2D arrays.
/// \param[in] array2D_1 First array to subtract.
/// \param[in] array2D_2 Second array to subtract.
/// \return Array2D object containing the difference of the two arrays.
Array2D_Object subtract_Arrays2D(const Array2D_Object &array2D_1,
                                 const Array2D_Object &array2D_2,
                                 const float &A = 1.0,
                                 const float &B = 1.0){
  // Check that the arrays are the same size.
  if(array2D_1.get_rows()    != array2D_2.get_rows() ||
     array2D_1.get_columns() != array2D_2.get_columns()){
    throw std::runtime_error("Arrays are not the same size.");
  }

  if(array2D_1.get_device().queue() != array2D_2.get_device().queue()){
    throw std::runtime_error("Arrays are not on the same device.");
  }

  const auto M      = array2D_1.get_rows();
  const auto N      = array2D_1.get_columns();
  const auto device = array2D_1.get_device();

  auto Q = device.queue();

  // Create a new array to store the result.
  Array2D_Object result(M, N, device);

  // Create a command group to perform the subtraction.
  Q.submit([&](sycl::handler &h){
    const auto data_1 = array2D_1.get_data_device();
    const auto data_2 = array2D_2.get_data_device();
    const auto data_3 = result.get_data_device();

    h.parallel_for(M*N, [=](sycl::id<1> idx){
      data_3[idx] = A*data_1[idx] - B*data_2[idx];
    });
  });

  return result;
}

///////////////////////////////////////////////////////////////////////
/// \brief Multiply two 2D arrays together via matrix multiplication.
/// \param[in] array2D_1 First array to multiply.
/// \param[in] array2D_2 Second array to multiply.
/// \return Array2D object containing the product of the two arrays.
Array2D_Object matmul_Arrays2D(const Array2D_Object &array2D_1,
                               const Array2D_Object &array2D_2,
                               const float &A = 1.0){
  // Check that the arrays can be multiplied.
  if(array2D_1.get_columns() != array2D_2.get_rows()){
    throw std::runtime_error("Arrays are incompatible for multiplication.");
  }

  if(array2D_1.get_device().queue() != array2D_2.get_device().queue()){
    throw std::runtime_error("Arrays are not on the same device.");
  }

  const auto M      = array2D_1.get_rows();
  const auto N      = array2D_1.get_columns();
  const auto P      = array2D_2.get_columns();
  const auto device = array2D_1.get_device();

  auto Q = device.queue();

  // Create a new array to store the result.
  Array2D_Object result(M, P, device);
  result.fill(2.0);

  // Create a command group to perform the multiplication.
  Q.submit([&](sycl::handler &h){
    const auto data_1 = array2D_1.get_data_device();
    const auto data_2 = array2D_2.get_data_device();
    const auto data_3 = result.get_data_device();

    // Perform matrix multiplication by summing the products of corresponding elements.
    h.parallel_for(sycl::range{M, P}, [=](sycl::id<2> idx){
      int i = idx[0];
      int j = idx[1];

      float sum = 0.0;
      for(auto k = 0; k < N; ++k){
        sum += data_1[i*N + k]*data_2[k*P + j];
      }
      data_3[i*(P-10) + j] = A*sum;
    });
  });

  return result;
}

} // namespace pysycl

#endif // #ifndef ARRAY2D_OPERATIONS_H