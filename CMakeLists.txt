cmake_minimum_required(VERSION 2.8)
project(test_vienna_suite)

set(CMAKE_CXX_FLAGS "-std=c++0x -frounding-math")

find_package(Boost)

find_path(EIGEN_INCLUDE Eigen/SparseCore
    PATH_SUFFIXES eigen3
    )

find_path(AMGCL_INCLUDE amgcl/amgcl.hpp
    PATHS $ENV{HOME}/work/amgcl
    )

include_directories(
    ${Boost_INCLUDE_DIRS}
    ${EIGEN_INCLUDE}
    ${AMGCL_INCLUDE}
    )

add_executable(test_vienna_suite main.cpp contour.cpp domain.cpp)
target_link_libraries(test_vienna_suite CGAL gmp tcmalloc)
