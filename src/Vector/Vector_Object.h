#ifndef VECTOR_OBJECT_H
#define VECTOR_OBJECT_H

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
/// \brief Vector Object in PySYCL.
///////////////////////////////////////////////////////////////////////

#include <CL/sycl.hpp>
#include <vector>

#include "../Device/SYCL_Device_Inquiry.h"
#include "../Math/Math_Functions.h"

///////////////////////////////////////////////////////////////////////
/// \addtogroup Vector
/// @{

namespace pysycl{
///////////////////////////////////////////////////////////////////////
/// \brief Class defining a Vector Object for use in PySYCL.
class Vector_Object {
public:
  /////////////////////////////////////////////////////////////////////
  /// \brief Default constructor, use compiler generated version.
  Vector_Object() = default;

  /////////////////////////////////////////////////////////////////////
  /// \brief Copy constructor, use compiler generated version.
  Vector_Object(const Vector_Object&) = default;

  /////////////////////////////////////////////////////////////////////
  /// \brief Move constructor, use compiler generated version.
  Vector_Object(Vector_Object&&) = default;

  /////////////////////////////////////////////////////////////////////
  /// \brief Copy assignment, use compiler generated version.
  /// \return reference to the assigned object.
  Vector_Object& operator=(const Vector_Object&) = default;

  /////////////////////////////////////////////////////////////////////
  /// \brief Move assignment, use compiler generated version.
  /// \return reference to the assigned object.
  Vector_Object& operator=(Vector_Object&&) = default;

  /////////////////////////////////////////////////////////////////////
  /// \brief Sets the vector to input values
  void set_data(std::vector<double> data_in);

  /////////////////////////////////////////////////////////////////////
  /// \brief Resets all vector elements to 0
  void reset_data();

  /////////////////////////////////////////////////////////////////////
  /// \brief Returns the current vector state
  /// \return Vector state.
  std::vector<double> get_data();

  /////////////////////////////////////////////////////////////////////
  /// \brief Element-wise vector-vector operations
  template<typename Function_type>
  void element_vector_operation(Function_type function, std::vector<double> data_in);

  /////////////////////////////////////////////////////////////////////
  /// \brief Element-wise vector-constant operations
  template<typename Function_type>
  void element_vector_operation(Function_type function, double C);

  /////////////////////////////////////////////////////////////////////
  /// \brief Vector Element Addition (Vector-Vector)
  /// \param[in] vector_in Vector to be added.
  void add_vector(std::vector<double> vector_in){
    auto function = pysycl::add_function<double>();
    element_vector_operation(function, vector_in);
  }

  /////////////////////////////////////////////////////////////////////
  /// \brief Vector Element Addition (Vector-Constant)
  /// \param[in] C Constant to be added.
  void add_constant(double C){
    auto function = pysycl::add_function<double>();
    element_vector_operation(function, C);
  }

  /////////////////////////////////////////////////////////////////////
  /// \brief Vector Element Subtraction (Vector-Vector)
  /// \param[in] vector_in Vector to be subtracted.
  void subtract_vector(std::vector<double> vector_in){
    auto function = pysycl::subtract_function<double>();
    element_vector_operation(function, vector_in);
  }

  /////////////////////////////////////////////////////////////////////
  /// \brief Vector Element Subtraction (Vector-Constant)
  /// \param[in] C Constant to be subtracted.
  void subtract_constant(double C){
    auto function = pysycl::subtract_function<double>();
    element_vector_operation(function, C);
  }

  /////////////////////////////////////////////////////////////////////
  /// \brief Vector Element Multiplication (Vector-Vector)
  /// \param[in] vector_in Vector to be multiplied.
  void multiply_vector(std::vector<double> vector_in){
    auto function = pysycl::multiply_function<double>();
    element_vector_operation(function, vector_in);
  }

  /////////////////////////////////////////////////////////////////////
  /// \brief Vector Element Multiplication (Vector-Constant)
  /// \param[in] C Constant to be multiplied.
  void multiply_constant(double C){
    auto function = pysycl::multiply_function<double>();
    element_vector_operation(function, C);
  }

  /////////////////////////////////////////////////////////////////////
  /// \brief Vector Element Division (Vector-Vector)
  /// \param[in] vector_in Vector to be divided.
  void divide_vector(std::vector<double> vector_in){
    auto function = pysycl::divide_function<double>();
    element_vector_operation(function, vector_in);
  }

  /////////////////////////////////////////////////////////////////////
  /// \brief Vector Element Division (Vector-Constant)
  /// \param[in] C Constant to be divided.
  void divide_constant(double C){
    auto function = pysycl::divide_function<double>();
    element_vector_operation(function, C);
  }

  /////////////////////////////////////////////////////////////////////
  /// \brief Constructor for Vector_Object.
  /// \param[in] N Vector size.
  /// \param[in] platform_index Index of the sycl platform to select.
  /// \param[in] device_index Index of the sycl device to select.
  Vector_Object(size_t N, int platform_index = 0, int device_index = 0){
    if (N < 0) {
      throw std::runtime_error("Vector size must be greater than 0.");
    }

    device_queue = pysycl::get_queue(platform_index, device_index);
    data_device = sycl::malloc_device<double>(N, device_queue);
  }

private:
  /////////////////////////////////////////////////////////////////////
  /// \brief Vector size.
  size_t N;

  /////////////////////////////////////////////////////////////////////
  /// \brief Vector data input data on device.
  double* data_device_in;

  /////////////////////////////////////////////////////////////////////
  /// \brief Vector data on device.
  double* data_device;

  /////////////////////////////////////////////////////////////////////
  /// \brief The selected device.
  sycl::queue device_queue;
};

///////////////////////////////////////////////////////////////////////
void Vector_Object::set_data(std::vector<double> data_in){
  if (data_in.size() != N) {
    throw std::runtime_error("Input vector size does not match vector size.");
  }

  device_queue.submit([&](sycl::handler& h){
    h.memcpy(data_device, &data_in[0], N*sizeof(double));
  });
}

///////////////////////////////////////////////////////////////////////
void Vector_Object::reset_data(){
  device_queue.submit([&](sycl::handler& h){
    const auto data_device_ptr = data_device;
    h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx){
      const int i = idx[0];
      data_device_ptr[i] = 0.0;
    });
  });
}

///////////////////////////////////////////////////////////////////////
std::vector<double> Vector_Object::get_data(){
  std::vector<double> data_out(N);

  device_queue.submit([&](sycl::handler& h){
    h.memcpy(&data_out[0], data_device, N*sizeof(double));
  });

  return data_out;
}

///////////////////////////////////////////////////////////////////////
template<typename Function_type>
void Vector_Object::element_vector_operation(Function_type function, std::vector<double> data_in){
  if (data_in.size() != N) {
    throw std::runtime_error("Input vector size does not match vector size.");
  }

  device_queue.submit([&](sycl::handler& h){
    h.memcpy(data_device_in, &data_in[0], N*sizeof(double));
  });

  device_queue.submit([&](sycl::handler& h){
    const auto data_device_ptr = data_device;
    const auto data_device_in_ptr = data_device_in;
    h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx){
      const int i = idx[0];
      data_device_ptr[i] = function(data_device_ptr[i], data_device_in_ptr[i]);
    });
  });
}

///////////////////////////////////////////////////////////////////////
template<typename Function_type>
void Vector_Object::element_vector_operation(Function_type function, double C){
  device_queue.submit([&](sycl::handler& h){
    const auto data_device_ptr = data_device;
    h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx){
      const int i = idx[0];
      data_device_ptr[i] = function(data_device_ptr[i], C);
    });
  });
}

} // namespace pysycl

#endif // #ifndef VECTOR_OBJECT_H