#ifndef ARRAY1D_H
#define ARRAY1D_H

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
/// \brief Array1D in PySYCL.
///////////////////////////////////////////////////////////////////////
#include <CL/sycl.hpp>
#include <vector>
#include <cmath>

#include "../Device/Device_Instance.h"

///////////////////////////////////////////////////////////////////////
/// \addtogroup Array2D
/// @{
namespace pysycl {
///////////////////////////////////////////////////////////////////////
/// \brief Array1D class for PySYCL
class Array1D {
public:
  ///////////////////////////////////////////////////////////////////////
  /// \brief Defining the device type.
  using Device_T = pysycl::Device_Instance;

  ///////////////////////////////////////////////////////////////////////
  /// \brief Default constructor.
  Array1D() = delete;

  ///////////////////////////////////////////////////////////////////////
  /// \brief Copy constructor.
  Array1D(const Array1D& og) :
      size(og.size),
      data_host(og.data_host),
      device(og.device),
      Q(og.Q){
        data_device = sycl::malloc_device<float>(size, Q);
        Q.memcpy(data_device, og.data_device, size*sizeof(float)).wait();
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Move constructor.
  Array1D(Array1D&& og) noexcept :
      size(og.size),
      data_host(std::move(og.data_host)),
      data_device(og.data_device),
      Q(std::move(og.Q)),
      device(std::move(og.device)){
        og.data_device = nullptr;
        og.size = 0;
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Copy assignment operator.
  Array1D& operator=(const Array1D& og){
    if(this != &og){
      data_host = og.data_host;
      sycl::free(data_device, Q);
      size = og.size;
      device = og.device;
      Q = og.Q;
      data_device = sycl::malloc_device<float>(size, Q);
      Q.memcpy(data_device, og.data_device, size*sizeof(float)).wait();
    }

    return *this;
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Move assignment operator.
  Array1D& operator=(Array1D&& og) noexcept{
    if(this != &og){
      data_host = std::move(og.data_host);
      sycl::free(data_device, Q);
      data_device = og.data_device;
      og.data_device = nullptr;
      size = og.size;
      Q = std::move(og.Q);
      device = std::move(og.device);
      size = 0;
    }

    return *this;
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Destructor.
  ~Array1D(){
    if(data_device){
      sycl::free(data_device, Q);
      data_device = nullptr;
    }
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Constructor for the Array1D class.
  /// \param[in] size Number of elements in the array.
  Array1D(int size_in, Device_T device_in = Device_T(0, 0)) :
      size(size_in),
      data_host(size),
      device(device_in),
    Q(device_in.get_queue()){
    if(size <= 0){
      throw std::runtime_error("ERROR IN ARRAY1D: number of elements must be > 0.");
    }

    data_device = sycl::malloc_device<float>(size, Q);
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Overloaded operator() direct element access
  float& operator()(int i) {
    if(i < 0 || i >= size) throw std::out_of_range("Array1D access out of range");
    return data_host[i];
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Overloaded operator() read-only element access
  const float& operator()(int i) const {
    if(i < 0 || i >= size) throw std::out_of_range("Array1D access out of range");
    return data_host[i];
  }

private:
  ///////////////////////////////////////////////////////////////////////
  /// \brief Number of elements in the array.
  int size;

  ///////////////////////////////////////////////////////////////////////
  /// \brief Vector for data stored in host memory.
  std::vector<float> data_host;

  ///////////////////////////////////////////////////////////////////////
  /// \brief Pointer to data stored in device memory.
  float* data_device;

  ///////////////////////////////////////////////////////////////////////
  /// \brief Device that will store and handle Array2D memory and operations
  Device_T device;

  ///////////////////////////////////////////////////////////////////////
  /// \brief Device SYCL queue.
  sycl::queue Q;
}; // class Array1D

/// @} // end "Array1D" doxygen group

} // namespace pysycl

#endif // ARRAY1D_H