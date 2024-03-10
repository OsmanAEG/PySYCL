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

///////////////////////////////////////////////////////////////////////
/// sycl
///////////////////////////////////////////////////////////////////////
#include <CL/sycl.hpp>

///////////////////////////////////////////////////////////////////////
/// local
///////////////////////////////////////////////////////////////////////
#include "../Device/Device_Instance.h"

///////////////////////////////////////////////////////////////////////
/// \addtogroup Linear_Algebra
/// @{
namespace pysycl {

///////////////////////////////////////////////////////////////////////
/// \brief Tiled Matrix multiplication function.
/// \param[in] A The first Array2D that is being multiplied.
/// \param[in] B The second Array2D that is being multiplied.
/// \param[in] C The resulting Array2D.
/// \param[in] wg_size The tile size.
template <typename Array2D_type>
void matmul(Array2D_type &A, Array2D_type &B, Array2D_type &C,
            const size_t &wg_size) {
  if (A.num_cols() != B.num_rows()) {
    throw std::runtime_error("ERROR: Incompatible Array2D dimensions.");
  }

  if (C.num_rows() != A.num_rows() || C.num_cols() != B.num_cols()) {
    throw std::runtime_error("ERROR: Incompatible Array2D dimensions.");
  }

  const bool same_platform_idx =
      A.get_platform_index() == B.get_platform_index();
  const bool same_device_idx = A.get_device_index() == B.get_device_index();

  if (!same_platform_idx || !same_device_idx)
    throw std::runtime_error("ERROR: Incompatible PySYCL device.");

  if (C.get_platform_index() != A.get_platform_index()) {
    throw std::runtime_error("ERROR: Incompatible PySYCL device.");
  }

  if (C.get_device_index() != A.get_device_index()) {
    throw std::runtime_error("ERROR: Incompatible PySYCL device.");
  }

  const auto M = A.num_rows();
  const auto N = A.num_cols();
  const auto P = B.num_cols();

  auto Q = sycl::queue(sycl::platform::get_platforms()[C.get_platform_index()]
                           .get_devices()[C.get_device_index()]);

  Q.submit([&](sycl::handler &h) {
     const size_t global_size_M = ((M + wg_size - 1) / wg_size) * wg_size;
     const size_t global_size_P = ((P + wg_size - 1) / wg_size) * wg_size;

     sycl::range<2> global{global_size_M, global_size_P};
     sycl::range<2> local{wg_size, wg_size};

     sycl::local_accessor<float, 2> A_block({wg_size, wg_size}, h);
     sycl::local_accessor<float, 2> B_block({wg_size, wg_size}, h);

     float *A_ptr = A.get_device_data_ptr();
     float *B_ptr = B.get_device_data_ptr();
     float *C_ptr = C.get_device_data_ptr();

     const auto tile_size = (N - 1) / wg_size + 1;

     h.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> it) {
       const auto iG = it.get_global_id(0);
       const auto jG = it.get_global_id(1);

       const auto iL = it.get_local_id(0);
       const auto jL = it.get_local_id(1);

       if (iG >= M || jG >= P)
         return;

       float c_ij = 0.0;

       for (int tile_idx = 0; tile_idx < tile_size; ++tile_idx) {
         // load A and B into local shared memory
         if (iG < M && jL + tile_idx * wg_size < N)
           A_block[iL][jL] = A_ptr[iG * N + tile_idx * wg_size + jL];
         else
           A_block[iL][jL] = 0;

         if (iL + tile_idx * wg_size < N && jG < P)
           B_block[iL][jL] = B_ptr[(iL + tile_idx * wg_size) * P + jG];
         else
           B_block[iL][jL] = 0;

         // sync threads
         it.barrier(sycl::access::fence_space::local_space);

         // vector dot product
         for (int n = 0; n < wg_size; ++n)
           c_ij += A_block[iL][n] * B_block[n][jL];

         // sync threads
         it.barrier(sycl::access::fence_space::local_space);
       }

       C_ptr[iG * P + jG] = c_ij;
     });
   }).wait();
}

/// @} // end "Linear_Algebra" doxygen group

} // namespace pysycl

#endif // MATRIX_MULTIPLICATION_H