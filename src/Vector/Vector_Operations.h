#ifndef VECTOR_OPERATIONS_H
#define VECTOR_OPERATIONS_H

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
/// \brief Vector operations in PySYCL. These operations receive
///        two individual vectors and return a single vector.
///////////////////////////////////////////////////////////////////////

#include <iostream>
#include <CL/sycl.hpp>
#include "../Device/SYCL_Device_Inquiry.h"
#include "../Math/Math_Functions.h"

///////////////////////////////////////////////////////////////////////
/// \addtogroup Vector
/// @{

namespace pysycl{

///////////////////////////////////////////////////////////////////////
/// \brief Function for element-wise vector operations in SYCL.
///        This function receives two vectors and returns a single
///        vector.
/// \tparam Function_T Type of operation to be performed.
/// \param vector_a First vector.
/// \param vector_b Second vector.
/// \param Q SYCL queue.
/// \param function Operation to be performed.
/// \return Vector resulting from the operation.
template <typename Function_T>
std::vector<double> Vector_Operation(std::vector<double> vector_a,
                                     std::vector<double> vector_b,
                                     sycl::queue Q,
                                     Function_T function){
  if(vector_a.size() != vector_b.size()){
    throw std::runtime_error("Error: vectors must have the same size.");
  }

  const size_t N = vector_a.size();

  std::vector<double> vector_c(vector_a.size());

  {
    sycl::buffer<double> buffer_a(vector_a);
    sycl::buffer<double> buffer_b(vector_b);
    sycl::buffer<double> buffer_c(vector_c);

    Q.submit([&](sycl::handler& h){
      sycl::accessor acc_a(buffer_a, h, sycl::read_only);
      sycl::accessor acc_b(buffer_b, h, sycl::read_only);
      sycl::accessor acc_c(buffer_c, h, sycl::write_only);

      h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx){
        acc_c[idx] = function(acc_a[idx], acc_b[idx]);
      });
    });
  }

  return vector_c;
};

///////////////////////////////////////////////////////////////////////
/// \brief Function for vector addition.
/// \param vector_a First vector.
/// \param vector_b Second vector.
/// \param platform_index Index of the sycl platform to select.
/// \param device_index Index of the sycl device to select.
std::vector<double> Vector_Addition(std::vector<double> vector_a,
                                    std::vector<double> vector_b,
                                    int platform_index = 0,
                                    int device_index = 0){
  auto Q = pysycl::get_queue(platform_index, device_index);
  auto function = pysycl::add_function<double>();
  return Vector_Operation(vector_a, vector_b, Q, function);
}

///////////////////////////////////////////////////////////////////////
/// \brief Function for vector subtraction.
/// \param vector_a First vector.
/// \param vector_b Second vector.
/// \param platform_index Index of the sycl platform to select.
/// \param device_index Index of the sycl device to select.
std::vector<double> Vector_Subtraction(std::vector<double> vector_a,
                                       std::vector<double> vector_b,
                                       int platform_index = 0,
                                       int device_index = 0){
  auto Q = pysycl::get_queue(platform_index, device_index);
  auto function = pysycl::subtract_function<double>();
  return Vector_Operation(vector_a, vector_b, Q, function);
}

///////////////////////////////////////////////////////////////////////
/// \brief Function for vector element multiplication.
/// \param vector_a First vector.
/// \param vector_b Second vector.
/// \param platform_index Index of the sycl platform to select.
/// \param device_index Index of the sycl device to select.
std::vector<double> Vector_Element_Multiplication(std::vector<double> vector_a,
                                                  std::vector<double> vector_b,
                                                  int platform_index = 0,
                                                  int device_index = 0){
  auto Q = pysycl::get_queue(platform_index, device_index);
  auto function = pysycl::multiply_function<double>();
  return Vector_Operation(vector_a, vector_b, Q, function);
}

///////////////////////////////////////////////////////////////////////
/// \brief Function for vector element division.
/// \param vector_a First vector.
/// \param vector_b Second vector.
/// \param platform_index Index of the sycl platform to select.
/// \param device_index Index of the sycl device to select.
std::vector<double> Vector_Element_Division(std::vector<double> vector_a,
                                            std::vector<double> vector_b,
                                            int platform_index = 0,
                                            int device_index = 0){
  auto Q = pysycl::get_queue(platform_index, device_index);
  auto function = pysycl::divide_function<double>();
  return Vector_Operation(vector_a, vector_b, Q, function);
}

///////////////////////////////////////////////////////////////////////
/// \brief Function for vector sum reduction.
/// \param vector_a Input vector.
/// \param platform_index Index of the sycl platform to select.
/// \param device_index Index of the sycl device to select.
double Vector_Sum_Reduction(std::vector<double> vector_a,
                            int platform_index = 0,
                            int device_index = 0){

  const size_t N = vector_a.size();

  auto Q = pysycl::get_queue(platform_index, device_index);

  double vector_sum = 0.0;

  {
    sycl::buffer<double> buffer_a(vector_a);
    sycl::buffer<double> buffer_vector_sum(&vector_sum, 1);

    Q.submit([&](sycl::handler& h){
      sycl::accessor acc_a(buffer_a, h, sycl::read_only);

      auto sum_reduction = sycl::reduction(buffer_vector_sum, h, sycl::plus<double>());

      h.parallel_for(sycl::range<1>(N), sum_reduction, [=](sycl::id<1> idx, auto& sum){
        sum.combine(acc_a[idx]);
      });
    });
  }

  return vector_sum;
}

///////////////////////////////////////////////////////////////////////
/// \brief Function for vector minimum reduction.
/// \param vector_a Input vector.
/// \param platform_index Index of the sycl platform to select.
/// \param device_index Index of the sycl device to select.
double Vector_Min_Reduction(std::vector<double> vector_a,
                            int platform_index = 0,
                            int device_index = 0){

  const size_t N = vector_a.size();

  auto Q = pysycl::get_queue(platform_index, device_index);

  double vector_min = 0.0;

  {
    sycl::buffer<double> buffer_a(vector_a);
    sycl::buffer<double> buffer_vector_min(&vector_min, 1);

    Q.submit([&](sycl::handler& h){
      sycl::accessor acc_a(buffer_a, h, sycl::read_only);

      auto min_reduction = sycl::reduction(buffer_vector_min, h, sycl::minimum<double>());

      h.parallel_for(sycl::range<1>(N), min_reduction, [=](sycl::id<1> idx, auto& min){
        min.combine(acc_a[idx]);
      });
    });
  }

  return vector_min;
}

///////////////////////////////////////////////////////////////////////
/// \brief Function for vector maximum reduction.
/// \param vector_a Input vector.
/// \param platform_index Index of the sycl platform to select.
/// \param device_index Index of the sycl device to select.
double Vector_Max_Reduction(std::vector<double> vector_a,
                            int platform_index = 0,
                            int device_index = 0){

  const size_t N = vector_a.size();

  auto Q = pysycl::get_queue(platform_index, device_index);

  double vector_max = 0.0;

  {
    sycl::buffer<double> buffer_a(vector_a);
    sycl::buffer<double> buffer_vector_max(&vector_max, 1);

    Q.submit([&](sycl::handler& h){
      sycl::accessor acc_a(buffer_a, h, sycl::read_only);

      auto max_reduction = sycl::reduction(buffer_vector_max, h, sycl::maximum<double>());

      h.parallel_for(sycl::range<1>(N), max_reduction, [=](sycl::id<1> idx, auto& max){
        max.combine(acc_a[idx]);
      });
    });
  }

  return vector_max;
}

///////////////////////////////////////////////////////////////////////
/// \brief Vector Dot Product.
/// \param vector_a First vector.
/// \param vector_b Second vector.
/// \param platform_index Index of the sycl platform to select.
/// \param device_index Index of the sycl device to select.
double Vector_Dot_Product(std::vector<double> vector_a,
                          std::vector<double> vector_b,
                          int platform_index = 0,
                          int device_index = 0){
  const auto vector_c = Vector_Element_Multiplication(vector_a, vector_b, platform_index, device_index);
  return Vector_Sum_Reduction(vector_c, platform_index, device_index);
}

} // namespace pysycl

#endif // #ifndef VECTOR_OPERATIONS_H