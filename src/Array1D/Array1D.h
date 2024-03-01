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
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "../Device/Device_Instance.h"

namespace py = pybind11;

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
      Q(og.Q)
  {
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
      device(std::move(og.device))
  {
        og.data_device = nullptr;
        og.size = 0;
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Copy assignment operator.
  Array1D& operator=(const Array1D& og){
    data_host = og.data_host;
    sycl::free(data_device, Q);
    size = og.size;
    device = og.device;
    Q = og.Q;
    data_device = sycl::malloc_device<float>(size, Q);
    Q.memcpy(data_device, og.data_device, size*sizeof(float)).wait();

    return *this;
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Move assignment operator.
  Array1D& operator=(Array1D&& og) noexcept{
    data_host = std::move(og.data_host);
    sycl::free(data_device, Q);
    data_device = og.data_device;
    og.data_device = nullptr;
    size = og.size;
    Q = std::move(og.Q);
    device = std::move(og.device);
    size = 0;

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
  /// \brief Basic constructor that takes in the size and device of
  ///        the array.
  /// \param[in] size_in Number of elements in the array.
  /// \param[in] device_in Number of elements in the array (Optional).
  Array1D(int size_in, Device_T device_in = Device_T(0, 0)) :
    size(size_in),
    data_host(size),
    device(device_in),
    Q(device_in.get_queue())
  {
      if(size <= 0) throw std::runtime_error("ERROR IN ARRAY1D: number of elements must be > 0.");
      data_device = sycl::malloc_device<float>(size, Q);
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Constructor that takes in a numpy array.
  /// \param[in] np_array_in Number of elements in the array.
  /// \param[in] device_in Number of elements in the array (Optional).
  Array1D(py::array_t<float> np_array_in, Device_T device_in = Device_T(0, 0)) :
    device(device_in),
    Q(device_in.get_queue())
  {
      if(np_array_in.ndim() != 1) throw std::runtime_error("The input numpy array must be 1D.");

      auto unchecked = np_array_in.unchecked<1>();
      size = unchecked.shape(0);
      data_host.resize(size);

      if(size <= 0) throw std::runtime_error("ERROR IN ARRAY1D: number of elements must be > 0.");

      for(int i = 0; i < size; ++i){
        data_host[i] = unchecked(i);
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

  ///////////////////////////////////////////////////////////////////////
  /// \brief Overloaded operator+ for vector addition (creating a new array).
  /// \param[in] B The Array1D that is being added.
  /// \return Array1D representing the sum of the addition.
  Array1D operator+(Array1D& B){return binary_vector_operations(B, binary_operations::ADD, false);}

  ///////////////////////////////////////////////////////////////////////
  /// \brief Overloaded operator+= for vector addition (edits self).
  /// \param[in] B The Array1D that is being added to self.
  Array1D& operator+=(Array1D& B){
    binary_vector_operations(B, binary_operations::ADD, true);
    return *this;
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Overloaded operator- for vector subtraction (creating a new array).
  /// \param[in] B The Array1D that is being subtracted.
  /// \return Array1D representing the difference of the subtraction.
  Array1D operator-(Array1D& B){return binary_vector_operations(B, binary_operations::SUBTRACT, false);}

  ///////////////////////////////////////////////////////////////////////
  /// \brief Overloaded operator-= for vector subtraction (edits self).
  /// \param[in] B The Array2D that is being subtracted.
  Array1D& operator-=(Array1D& B){
    binary_vector_operations(B, binary_operations::SUBTRACT, true);
    return *this;
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Overloaded operator* for element-wise multiplication (creating a new array).
  /// \param[in] B The Array1D that is being multiplied.
  /// \return Array1D representing the product of the multiplication.
  Array1D operator*(Array1D& B){return binary_vector_operations(B, binary_operations::MULTIPLY, false);}

  ///////////////////////////////////////////////////////////////////////
  /// \brief Overloaded operator*= for element-wise multiplication (edits self).
  /// \param[in] B The Array1D that is being multiplied.
  Array1D& operator*=(Array1D& B){
    binary_vector_operations(B, binary_operations::MULTIPLY, true);
    return *this;
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Overloaded operator/ for element-wise division (creating a new array).
  /// \param[in] B The Array1D that is being divided.
  /// \return Array1D representing the result of the division.
  Array1D operator/(Array1D& B){return binary_vector_operations(B, binary_operations::DIVIDE, false);}

  ///////////////////////////////////////////////////////////////////////
  /// \brief Overloaded operator/= for element-wise division (edits self).
  /// \param[in] B The Array2D that is being divided.
  Array1D& operator/=(Array1D& B){
    binary_vector_operations(B, binary_operations::DIVIDE, true);
    return *this;
  }

  ///////////////////////////////////////////////////////////////////////
  /// \brief Get the number of elements in the Array1D.
  /// \return Number of elements in the Array1D.
  int get_size() const;

  ///////////////////////////////////////////////////////////////////////
  /// \brief Get the data pointer from Array1D.
  /// \return Pointer to Array1D host data.
  std::vector<float> get_host_data_vector();

  ///////////////////////////////////////////////////////////////////////
  /// \brief Get the data pointer from Array1D.
  /// \return Pointer to Array1D device data.
  float* get_device_data_ptr();

  ///////////////////////////////////////////////////////////////////////
  /// \brief Copy memory from the CPU to the GPU.
  void mem_to_gpu();

  ///////////////////////////////////////////////////////////////////////
  /// \brief Copy memory from the GPU to the CPU
  void mem_to_cpu();

  ///////////////////////////////////////////////////////////////////////
  /// \brief Get the device from Array1D.
  /// \return The Array1D device instance.
  pysycl::Device_Instance& get_device();

  ///////////////////////////////////////////////////////////////////////
  /// \brief Fill the device with a specific value
  void fill(const float C);

  ///////////////////////////////////////////////////////////////////////
  /// \brief Function that finds the maximum value in the array
  /// \return Maximum value in the array
  auto max();

  ///////////////////////////////////////////////////////////////////////
  /// \brief Function that finds the minimum value in the array
  /// \return Minimum value in the array
  auto min();

  ///////////////////////////////////////////////////////////////////////
  /// \brief Function that finds the sum of all value in the array
  /// \return Sum of all values in the array
  auto sum();

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

  ///////////////////////////////////////////////////////////////////////
  /// \brief Defining enumerations of binary operations
  enum class binary_operations{ADD, SUBTRACT, MULTIPLY, DIVIDE};

  ///////////////////////////////////////////////////////////////////////
  /// \brief Function to perform binary matrix operations
  Array1D binary_vector_operations(Array1D& B, binary_operations op, bool edit_self);

  ///////////////////////////////////////////////////////////////////////
  /// \brief Function to perform reduction operations
  template<typename Operation_T>
  auto reductions(Operation_T op, float val = 0.0);
}; // class Array1D

/// @} // end "Array1D" doxygen group

} // namespace pysycl

/////////////////////////////////////////////////////////////////////////
int pysycl::Array1D::get_size() const {
  return size;
}

/////////////////////////////////////////////////////////////////////////
std::vector<float> pysycl::Array1D::get_host_data_vector(){
  return data_host;
}

/////////////////////////////////////////////////////////////////////////
float* pysycl::Array1D::get_device_data_ptr(){
  return data_device;
}

/////////////////////////////////////////////////////////////////////////
/// \brief Copy memory from the CPU to the GPU.
void pysycl::Array1D::mem_to_gpu(){
  Q.memcpy(data_device, &data_host[0], size*sizeof(float)).wait();
}

/////////////////////////////////////////////////////////////////////////
/// \brief Copy memory from the GPU to the CPU
void pysycl::Array1D::mem_to_cpu(){
  Q.memcpy(&data_host[0], data_device, size*sizeof(float)).wait();
}

/////////////////////////////////////////////////////////////////////////
pysycl::Device_Instance& pysycl::Array1D::get_device(){
  return device;
}

/////////////////////////////////////////////////////////////////////////
void pysycl::Array1D::fill(const float C){
  const size_t B = device.get_max_workgroup_size();

  Q.submit([&](sycl::handler& h){
    const size_t global_size = ((size + B - 1)/B)*B;
    const auto N = size;
    const auto A = data_device;

    sycl::range<1> global{global_size};
    sycl::range<1> local{B};

    h.parallel_for(sycl::nd_range<1>(global, local), [=](sycl::nd_item<1> it){
      const auto i = it.get_global_id();

      if(i >= N) return;

      A[i] = C;
    });
  }).wait();
}

/////////////////////////////////////////////////////////////////////////
pysycl::Array1D pysycl::Array1D::binary_vector_operations(Array1D& B,
                                                          binary_operations op,
                                                          bool edit_self){
  const auto rows = this->get_size();

  if(size != B.get_size()){
    throw std::runtime_error("ERROR: Incompatible Array1D dimensions.");
  }

  const auto platform_idx = this->device.get_platform_index();
  const auto device_idx = this->device.get_device_index();

  if(platform_idx != B.device.get_platform_index() || device_idx != B.device.get_device_index()){
    throw std::runtime_error("ERROR: Incompatible PySYCL device.");
  }

  const size_t wg_size = this->device.get_max_workgroup_size();

  Array1D C = edit_self ? *this : Array1D(size, this->device);

  Q.submit([&](sycl::handler& h){
    const size_t global_size = ((size + wg_size - 1)/wg_size)*wg_size;
    const auto N = size;

    sycl::range<1> global{global_size};
    sycl::range<1> local{wg_size};

    auto data_1   = this->get_device_data_ptr();
    auto data_2   = B.get_device_data_ptr();
    auto data_new = C.get_device_data_ptr();

    h.parallel_for(sycl::nd_range<1>(global, local), [=](sycl::nd_item<1> it){
      const auto i = it.get_global_id();

      if(i >= N) return;

      switch(op){
        case binary_operations::ADD:
          data_new[i] = data_1[i] + data_2[i];
          break;

        case binary_operations::SUBTRACT:
          data_new[i] = data_1[i] - data_2[i];
          break;

        case binary_operations::MULTIPLY:
          data_new[i] = data_1[i] * data_2[i];
          break;

        case binary_operations::DIVIDE:
          data_new[i] = data_1[i] / data_2[i];
          break;
      }
    });
  }).wait();

  return C;
}

/////////////////////////////////////////////////////////////////////////
template<typename Operation_T>
auto pysycl::Array1D::reductions(Operation_T op, float val){
  sycl::buffer<float> buf{&val, 1};

  const size_t wg_size = device.get_max_workgroup_size();

  Q.submit([&](sycl::handler& h){
    const auto reduction_func = sycl::reduction(buf, h, op);

    const size_t global_size = ((size + wg_size - 1)/wg_size)*wg_size;
    sycl::range<1> global{global_size};
    sycl::range<1> local{wg_size};

    auto data_device_ptr = data_device;
    const size_t N = size;

    h.parallel_for(sycl::nd_range<1>(global, local), reduction_func, [=](sycl::nd_item<1> it, auto& el){
      const auto idx = it.get_global_id();
      if(idx >= N) return;

      el.combine(data_device_ptr[idx]);
    });
  }).wait();

  sycl::host_accessor val_host{buf, sycl::read_only};
  val = val_host[0];

  return val;
}

/////////////////////////////////////////////////////////////////////////
auto pysycl::Array1D::max(){
  return reductions(sycl::maximum<float>(), std::numeric_limits<float>::lowest());
}

/////////////////////////////////////////////////////////////////////////
auto pysycl::Array1D::min(){
  return reductions(sycl::minimum<float>(), std::numeric_limits<float>::max());
}

/////////////////////////////////////////////////////////////////////////
auto pysycl::Array1D::sum(){
  return reductions(sycl::plus<float>());
}

#endif // ARRAY1D_H