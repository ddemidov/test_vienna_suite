#ifndef MESHER_HPP
#define MESHER_HPP

#include <chrono>

#include <viennagrid/element.hpp>
#include <viennagrid/point.hpp>
#include <viennagrid/element.hpp>
#include <viennagrid/domain.hpp>
#include <viennagrid/segment.hpp>
#include <viennagrid/config/simplex.hpp>

#include <amgcl/profiler.hpp>

#include "contour.h"

namespace mesher {

typedef viennagrid::config::triangular_2d                                 Config;
typedef viennagrid::result_of::domain<Config>::type                       Domain;
typedef viennagrid::result_of::point<Config>::type                        Point;
typedef viennagrid::result_of::ncell<Config, Config::cell_tag::dim>::type Triangle;
typedef viennagrid::result_of::ncell<Config, 0>::type                     Vertex;

Domain get(const contour &cnt, double resolution,
        amgcl::profiler<std::chrono::high_resolution_clock> &prof
        );

}

#endif
