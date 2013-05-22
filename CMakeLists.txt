cmake_minimum_required(VERSION 2.8)
project(test_vienna_suite)

set(CMAKE_CXX_FLAGS "-std=c++0x -frounding-math")

find_package(Boost)

find_path(AMGCL_INCLUDE amgcl/amgcl.hpp
    PATHS $ENV{HOME}/work/amgcl
    )

find_path(VIENNA_SUITE  viennafem/fem.hpp
    PATHS $ENV{HOME}/work/vienna/fem
    )

include_directories(
    ${Boost_INCLUDE_DIRS}
    ${AMGCL_INCLUDE}
    ${VIENNA_SUITE}
    )

add_executable(test_vienna_suite main.cpp contour.cpp mesher.cpp)
target_link_libraries(test_vienna_suite CGAL gmp tcmalloc)
