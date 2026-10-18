cmake_minimum_required(VERSION 3.15)
project(coxdev_c LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

set(COXDEV_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
# Eigen: coxdev's eigen submodule by default; override with -DEIGEN_INCLUDE_DIR=...
set(EIGEN_INCLUDE_DIR ${COXDEV_ROOT}/eigen CACHE PATH "Eigen include directory")

add_library(coxdev_c SHARED coxdev_c.cpp)
target_include_directories(coxdev_c PUBLIC ${COXDEV_ROOT}/cpp/include PRIVATE ${EIGEN_INCLUDE_DIR})

option(COXDEV_C_TESTS "Build the plain C test program" ON)
if(COXDEV_C_TESTS)
  add_executable(test_c_api test_c_api.c)
  set_target_properties(test_c_api PROPERTIES C_STANDARD 99 C_STANDARD_REQUIRED ON)
  target_link_libraries(test_c_api coxdev_c m)
endif()
