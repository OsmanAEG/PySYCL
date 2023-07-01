#ifndef ARRAY2D_OBJECT_H
#define ARRAY2D_OBJECT_H

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
/// \brief Array2D in PySYCL.
///////////////////////////////////////////////////////////////////////

#include <CL/sycl.hpp>
#include <vector>
//#include <random>

#include "../Device/Device_Object.h"

///////////////////////////////////////////////////////////////////////
/// \addtogroup Array2D
/// @{

namespace pysycl {
///////////////////////////////////////////////////////////////////////
/// \brief Class defining a 2D array for use in PySYCL.
class Array2D_Object {
public:
  /////////////////////////////////////////////////////////////////////
  /// \brief Default constructor, use compiler generated version.
  Array2D_Object() = default;

  /////////////////////////////////////////////////////////////////////
  /// \brief Copy constructor, use compiler generated version.

  /////////////////////////////////////////////////////////////////////
  /// \brief Move constructor, use compiler generated version.
  Array2D_Object(Array2D_Object &&) = default;

  /////////////////////////////////////////////////////////////////////
  /// \brief Copy assignment operator, use compiler generated version.
  /// \return Reference to the assigned object.
  Array2D_Object &operator=(const Array2D_Object &) = default;

  /////////////////////////////////////////////////////////////////////
  /// \brief Move assignment operator, use compiler generated version.
  /// \return Reference to the assigned object.
  Array2D_Object &operator=(Array2D_Object &&) = default;

  /////////////////////////////////////////////////////////////////////
  /// \brief Copy data from device to host.
  void copy_device_to_host();

  /////////////////////////////////////////////////////////////////////
  /// \brief Copy data from host to device.
  void copy_host_to_device();

  /////////////////////////////////////////////////////////////////////
  /// \brief Get host data.
  /// \return Return the host data as a 2D array.
  std::vector<std::vector<float>> get_host_data();

  /////////////////////////////////////////////////////////////////////
  /// \brief Fill the array on device with a given value.
  /// \param[in] value Value to fill the array with.
  void fill(float value);

  /////////////////////////////////////////////////////////////////////
  /// \brief Fill the array on device with random values.
  void fill_random(float min_val = 0.0, float max_val = 1.0);

  /////////////////////////////////////////////////////////////////////
  /// \brief Fill an element of the array on host with a given value.
  /// \param[in] i Row index of the element to fill.
  /// \param[in] j Column index of the element to fill.
  /// \param[in] value Value to fill the array with.
  void fill_element_host(size_t i, size_t j, float value);

  ///////////////////////////////////////////////////////////////////////
  /// \brief Sum reduction
  /// \return Sum of all elements in the array.
  float sum_reduction();

  /////////////////////////////////////////////////////////////////////
  /// \brief Min reduction
  /// \return Minimum value of all elements in the array.
  float min_reduction();

  /////////////////////////////////////////////////////////////////////
  /// \brief Max reduction
  /// \return Maximum value of all elements in the array.
  float max_reduction();

  /////////////////////////////////////////////////////////////////////
  /// \brief Constructor for Array2D.
  /// \param[in] M Number of rows in the array.
  /// \param[in] N Number of columns in the array.
  /// \param[in] init Initial value for all elements in the array.
  /// \param[in] device SYCL device to use for allocation.
  Array2D_Object(size_t M_in, size_t N_in, pysycl::Device_Object device_in) :
          M(M_in), N(N_in), Q(device_in.queue()) {
    // Allocate the array.
    data_host.resize(M*N);
    data_device = sycl::malloc_device<float>(M*N, Q);
  }

private:
  /////////////////////////////////////////////////////////////////////
  /// \brief Number of rows in the array.
  size_t M;

  /////////////////////////////////////////////////////////////////////
  /// \brief Number of columns in the array.
  size_t N;

  /////////////////////////////////////////////////////////////////////
  /// \brief Device queue to use for allocation and computation.
  sycl::queue Q;

  /////////////////////////////////////////////////////////////////////
  /// \brief Array data on host
  std::vector<float> data_host;

  /////////////////////////////////////////////////////////////////////
  /// \brief Array data on device
  float* data_device;

}; // class Array2D_Object

///////////////////////////////////////////////////////////////////////
void Array2D_Object::copy_device_to_host() {
  Q.memcpy(data_host.data(), data_device, M*N*sizeof(float)).wait();
}

///////////////////////////////////////////////////////////////////////
void Array2D_Object::copy_host_to_device() {
  Q.memcpy(data_device, data_host.data(), M*N*sizeof(float)).wait();
}

///////////////////////////////////////////////////////////////////////
std::vector<std::vector<float>> Array2D_Object::get_host_data() {
  std::vector<std::vector<float>> data_host_2d(M, std::vector<float>(N));
  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      data_host_2d[i][j] = data_host[i*N + j];
    }
  }
  return data_host_2d;
}

///////////////////////////////////////////////////////////////////////
void Array2D_Object::fill(float value) {
  Q.submit([&](sycl::handler &h){
    const auto data_device_ptr = data_device;
    h.parallel_for(M*N, [=](sycl::id<1> idx){
      data_device_ptr[idx] = value;
    });
  }).wait();
}

///////////////////////////////////////////////////////////////////////
void Array2D_Object::fill_random(float min_val = 0.0, float max_val = 1.0) {
  Q.submit([&](sycl::handler &h){
    float range = max_val - min_val;
    const auto data_device_ptr = data_device;
    h.parallel_for(M*N, [=](sycl::id<1> idx){
      float random_value = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
      data_device_ptr[idx] = (random_value * range) + min_val;
    });
  }).wait();
}

///////////////////////////////////////////////////////////////////////
void Array2D_Object::fill_element_host(size_t i, size_t j, float value) {
  data_host[i*N + j] = value;
}

///////////////////////////////////////////////////////////////////////
float Array2D_Object::sum_reduction() {
  float array_sum = 0.0;

  {
    sycl::buffer<float> buffer_array_sum(&array_sum, 1);
    Q.submit([&](sycl::handler& h){
      const auto data_device_ptr = data_device;
      auto sum_reduce = sycl::reduction(buffer_array_sum, h, sycl::plus<float>());

      h.parallel_for(sycl::range<1>(M*N), sum_reduce, [=](sycl::id<1> idx, auto& sum){
        sum.combine(data_device_ptr[idx]);
      });
    }).wait();
  }

  return array_sum;
}

///////////////////////////////////////////////////////////////////////
float Array2D_Object::min_reduction() {
  float array_min = std::numeric_limits<float>::max();

  {
    sycl::buffer<float> buffer_array_min(&array_min, 1);
    Q.submit([&](sycl::handler& h){
      const auto data_device_ptr = data_device;
      auto min_reduce = sycl::reduction(buffer_array_min, h, sycl::minimum<float>());

      h.parallel_for(sycl::range<1>(M*N), min_reduce, [=](sycl::id<1> idx, auto& min){
        min.combine(data_device_ptr[idx]);
      });
    }).wait();
  }

  return array_min;
}

///////////////////////////////////////////////////////////////////////
float Array2D_Object::max_reduction() {
  float array_max = std::numeric_limits<float>::lowest();

  {
    sycl::buffer<float> buffer_array_max(&array_max, 1);
    Q.submit([&](sycl::handler& h){
      const auto data_device_ptr = data_device;
      auto max_reduce = sycl::reduction(buffer_array_max, h, sycl::maximum<float>());

      h.parallel_for(sycl::range<1>(M*N), max_reduce, [=](sycl::id<1> idx, auto& max){
        max.combine(data_device_ptr[idx]);
      });
    }).wait();
  }

  return array_max;
}

}// namespace pysycl

#endif // #ifndef ARRAY2D_OBJECT_H