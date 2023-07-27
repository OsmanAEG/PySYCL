#ifndef ARRAY2D_MATRIX_MULTIPLICATION_H
#define ARRAY2D_MATRIX_MULTIPLICATION_H

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
/// \brief Array2D Matrix Multiplication in PySYCL. This header file
///        contains the matrix multiplication kernels for Array2D
///        objects.
///////////////////////////////////////////////////////////////////////

#include <CL/sycl.hpp>
#include <vector>

#include "../../Device/Device_Object.h"
#include "../../Math/Math_Functions.h"
#include "../Array2D_Explicit/Array2D_Explicit.h"
#include "../Array2D_Shared/Array2D_Shared.h"

///////////////////////////////////////////////////////////////////////
/// \addtogroup Array2D
/// @{
namespace pysycl {
///////////////////////////////////////////////////////////////////////
/// \brief Array2D Mat-Mul.
/// \param[in] arr2D_1 First Array2D for operation.
/// \param[in] arr2D_2 Second Array2D for operation.
/// \param[in] A Constant multiplier for the first Array2D.
/// \param[in] B Constant multiplier for the second Array2D.
/// \return The result of the operation.
template <typename Array2D_type>
void basic_matrix_multiplication_Array2D(sycl::queue Q,
                                         const Array2D_type &arr2D_1,
                                         const Array2D_type &arr2D_2,
                                         Array2D_type &result,
                                         const int &M,
                                         const int &N,
                                         const int &P,
                                         const float &A = 1.0f,
                                         const float &B = 1.0f){
  Q.submit([&](sycl::handler &h){
    const auto data_1 = arr2D_1.get_data_ptr();
    const auto data_2 = arr2D_2.get_data_ptr();
    const auto data_r = result.get_data_ptr();

    h.parallel_for(sycl::range<2>(M, P), [=](sycl::id<2> idx){
      const auto i = idx[0];
      const auto j = idx[1];

      float c_ij = 0.0f;

      for(int k = 0; k < N; ++k) {
        c_ij += data_1[i*N + k] * data_2[k*P + j];
      }

      data_r[i*P + j] = c_ij;
    });
  });
}

///////////////////////////////////////////////////////////////////////
/// \brief Array2D Mat-Mul (ND-Range).
/// \param[in] arr2D_1 First Array2D for operation.
/// \param[in] arr2D_2 Second Array2D for operation.
/// \param[in] A Constant multiplier for the first Array2D.
/// \param[in] B Constant multiplier for the second Array2D.
/// \param[in] b Local work group size.
/// \return The result of the operation.
template <typename Array2D_type>
void Array2D_matrix_multiplication_Array2D_ND(sycl::queue Q,
                                         const Array2D_type &arr2D_1,
                                         const Array2D_type &arr2D_2,
                                         Array2D_type &result,
                                         const int &M,
                                         const int &N,
                                         const int &P,
                                         const float &A = 1.0f,
                                         const float &B = 1.0f,
                                         const int &b = 4){
  Q.submit([&](sycl::handler &h){
    const auto data_1 = arr2D_1.get_data_ptr();
    const auto data_2 = arr2D_2.get_data_ptr();
    const auto data_r = result.get_data_ptr();

    auto global = sycl::range<2>(M, P);
    auto local = sycl::range<2>(b, b);

    h.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> it){
      const int i = it.get_global_id(0);
      const int j = it.get_global_id(1);

      double sum = 0.0;

      for(int k = 0; k < N; ++k){
        sum += data_1[i*N + k] * data_2[k*P + j];
      }

      data_r[i*P + j] = sum;
    });
  });
}



///////////////////////////////////////////////////////////////////////
/// \brief Array2D Mat-Mul (Handler).
/// \param[in] arr2D_1 First Array2D for operation.
/// \param[in] arr2D_2 Second Array2D for operation.
/// \param[in] A Constant multiplier for the first Array2D.
/// \param[in] B Constant multiplier for the second Array2D.
/// \param[in] b Local work group size.
/// \param[in] kernel_key The kernel key to use for the operation.
/// \return The result of the operation.
template <typename Array2D_type>
Array2D_type matrix_multiplication_Array2D(const Array2D_type &arr2D_1,
                                           const Array2D_type &arr2D_2,
                                           const float &A = 1.0f,
                                           const float &B = 1.0f,
                                           const int &b = 4,
                                           std::string kernel_key = "default"){
  // Check that the two arrays are the same size
  if (arr2D_1.number_of_cols() != arr2D_2.number_of_rows()) {
    throw std::runtime_error("Array2D objects have incompatible dimensions!");
  }

  // Check that the two arrays are on the same device
  if (arr2D_1.get_device().queue() != arr2D_2.get_device().queue()) {
    throw std::runtime_error("Array2D objects must be on the same device.");
  }

  const auto M = arr2D_1.number_of_rows();
  const auto N = arr2D_1.number_of_cols();
  const auto P = arr2D_2.number_of_cols();

  const auto device = arr2D_1.get_device();

  auto Q = device.queue();

  // Create a new Array2D object to store the result
  Array2D_type result(M, P, device);

  if(kernel_key == "default"){
    basic_matrix_multiplication_Array2D(Q, arr2D_1, arr2D_2, result, M, N, P, A, B);
    return result;
  }
  else if(kernel_key == "nd"){
    Array2D_matrix_multiplication_Array2D_ND(Q, arr2D_1, arr2D_2,
                                             result, M, N, P, A, B, b);
    return result;
  }
  else{
    throw std::runtime_error("Invalid kernel key.");
  }
}

} // namespace pysycl

#endif // ARRAY2D_MATRIX_MULTIPLICATION_H