#ifndef MATRIX_MULTIPLICATION_H
#define MATRIX_MULTIPLICATION_H

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
/// \brief Matrix Multiplication in PySYCL.
///////////////////////////////////////////////////////////////////////
#include <CL/sycl.hpp>

#include "../Array2D/Array2D.h"
#include "../Device/Device_Instance.h"

///////////////////////////////////////////////////////////////////////
/// \addtogroup Linear_Algebra
/// @{
namespace pysycl {

///////////////////////////////////////////////////////////////////////
/// \brief Matrix multiplication function
/// \param[in] A The first Array2D that is being multiplied.
/// \param[in] B The second Array2D that is being multiplied.
/// \return The Array2D product of the matrix multiplication.
Array2D matmul(Array2D& A, Array2D& B){
  if(A.num_cols() != B.num_rows()) throw std::runtime_error("ERROR: Incompatible Array2D dimensions.");

  const bool same_platform_idx = A.get_device().get_platform_index() == B.get_device().get_platform_index();
  const bool same_device_idx = A.get_device().get_device_index() == B.get_device().get_device_index();

  if(!same_platform_idx || !same_device_idx) throw std::runtime_error("ERROR: Incompatible PySYCL device.");

  const auto M = A.num_rows();
  const auto N = A.num_cols();
  const auto P = B.num_cols();

  const size_t wg_size = sqrt(A.get_device().get_max_workgroup_size());

  Array2D C(M, P, A.get_device());
  auto Q = A.get_device().get_queue();

  Q.submit([&](sycl::handler& h){
    const size_t global_size_M = ((M + wg_size - 1)/wg_size)*wg_size;
    const size_t global_size_P = ((P + wg_size - 1)/wg_size)*wg_size;

    sycl::range<2> global{global_size_M, global_size_P};
    sycl::range<2> local{wg_size, wg_size};

    auto data_1   = A.get_device_data_ptr();
    auto data_2   = B.get_device_data_ptr();
    auto data_new = C.get_device_data_ptr();

    h.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> it){
      const auto i = it.get_global_id(0);
      const auto j = it.get_global_id(1);

      if(i >= M || j >= P) return;

      double c_ij = 0.0;

      for(int n = 0; n < N; ++n){
        c_ij += data_1[i*N + n]*data_2[n*P + j];
      }

      data_new[i*P + j] = c_ij;
    });
  }).wait();

  return C;
}

/// @} // end "Linear_Algebra" doxygen group

} // namespace pysycl

#endif // MATRIX_MULTIPLICATION_H