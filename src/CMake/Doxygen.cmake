#######################################################################
#######################################################################
##                  src/CMake/Doxygen.cmake
#######################################################################
#######################################################################
## This file is part of the PySYCL software for SYCL development in
## Python.  It is licensed under the MIT licence.  A copy of
## this license, in a file named LICENSE.md, should have been
## distributed with this file.  A copy of this license is also
## currently available at "http://opensource.org/licenses/MIT".
##
## Unless explicitly stated, all contributions intentionally submitted
## to this project shall also be under the terms and conditions of this
## license, without any additional terms or conditions.
#######################################################################
#######################################################################

set(PYSYCL_DOC_DIR "PySYCL_doc_html"
  CACHE STRING "Name of the documentation directory")

set(PYSYCL_XML_DIR "PySYCL_doc_xml"
  CACHE STRING "Name of the xml documentation directory")

if("$PYSYCL_DOC_DIR}" STREQUAL "")
  message(FATAL_ERROR "PYSYCL_DOC_DIR cannot be blank.")
endif()

if("$PYSYCL_XML_DIR}" STREQUAL "")
  message(FATAL_ERROR "PYSYCL_XML_DIR cannot be blank.")
endif()

add_custom_target(doc COMMAND env DOXYGEN_OUTPUT_DIRECTORY=${CMAKE_BINARY_DIR}/../sphinx/_build/
                              env DOXYGEN_HTML_OUTPUT_DIRECTORY=${PYSYCL_DOC_DIR}
                              env DOXYGEN_XML_OUTPUT_DIRECTORY=${PYSYCL_XML_DIR}
                                  doxygen Doxyfile > ${CMAKE_BINARY_DIR}/doxygen.log 2> ${CMAKE_BINARY_DIR}/doxygen.err
                      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/../doc/
                      COMMENT
"Build Doxygen documentation.
     HTML:     ${CMAKE_BINARY_DIR}/../sphinx/_build/${PYSYCL_DOC_DIR}
     XML:      ${CMAKE_BINARY_DIR}/../sphinx/_build/${PYSYCL_XML_DIR}
     Output:   ${CMAKE_BINARY_DIR}/doxygen.log
     Warnings: ${CMAKE_BINARY_DIR}/doxygen.err")