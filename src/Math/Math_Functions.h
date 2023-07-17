#ifndef MATH_FUNCTIONS_H
#define MATH_FUNCTIONS_H

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
/// \brief Math functions returned in the form of lambda functions.
///////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////
/// \addtogroup Math
/// @{

namespace pysycl{

///////////////////////////////////////////////////////////////////////
/// \brief Lambda function that returns the sum of two numbers.
/// \tparam T Type of the numbers to be added.
template <typename T>
auto add_function(){
  return [](T a, T b) { return a + b; };
}

///////////////////////////////////////////////////////////////////////
/// \brief Lambda function that returns the difference of two numbers.
/// \tparam T Type of the numbers to be added.
template <typename T>
auto subtract_function(){
  return [](T a, T b) { return a - b; };
}

///////////////////////////////////////////////////////////////////////
/// \brief Lambda function that returns the product of two numbers.
/// \tparam T Type of the numbers to be added.
template <typename T>
auto multiply_function(){
  return [](T a, T b) { return a * b; };
}

///////////////////////////////////////////////////////////////////////
/// \brief Lambda function that returns the quotient of two numbers.
/// \tparam T Type of the numbers to be added.
template <typename T>
auto divide_function(){
  return [](T a, T b) { return a / b; };
}

} // namespace pysycl

/// @}
// end "Math" doxygen group

#endif // MATH_FUNCTIONS_H