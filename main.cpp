#include <chrono>
#include <algorithm>

#include <viennagrid/algorithm/boundary.hpp>
#include <viennafem/fem.hpp>
#include <viennafem/io/vtk_writer.hpp>
#include <viennamath/expression.hpp>

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>

#include <amgcl/amgcl.hpp>
#include <amgcl/interp_smoothed_aggr.hpp>
#include <amgcl/aggr_plain.hpp>
#include <amgcl/level_cpu.hpp>
#include <amgcl/operations_ublas.hpp>
#include <amgcl/bicgstab.hpp>
#include <amgcl/profiler.hpp>

#include "contour.h"
#include "mesher.h"

typedef boost::numeric::ublas::compressed_matrix<double> ublas_matrix;
typedef boost::numeric::ublas::vector<double> ublas_vector;

namespace grid = viennagrid;
namespace data = viennadata;
namespace math = viennamath;
namespace fem  = viennafem;

// Solve equation laplace(u) = 1 on a semicircle.
int main(int argc, char *argv[]) {
    amgcl::profiler<std::chrono::high_resolution_clock> prof;



    prof.tic("generate mesh");
    contour c(256, 0.4);
    auto domain = mesher::get(c, argc > 1 ? std::stod(argv[1]) : 3e-2, prof);
    prof.toc("generate mesh");



    prof.tic("boundary conditions");
    auto vertices = grid::ncells<0>(domain);
    for(auto v = vertices.begin(); v != vertices.end(); ++v) {
        if (data::find<fem::boundary_key, bool>(fem::boundary_key(0))(*v))
            fem::set_dirichlet_boundary(*v, 0.0);
    }
    prof.toc("boundary conditions");



    prof.tic("assemble");
    ublas_matrix A;
    ublas_vector f;

    {
        math::function_symbol u(0, math::unknown_tag<>());
        math::function_symbol v(0, math::test_tag<>());

        auto weak_poisson = math::make_equation(
                math::integral(math::symbolic_interval(), math::grad(u) * math::grad(v)),
                math::integral(math::symbolic_interval(), -1 * v)
                );

        fem::pde_assembler()(fem::make_linear_pde_system(weak_poisson, u), domain, A, f);
    }
    prof.toc("assemble");



    prof.tic("solve");
    typedef amgcl::solver<
        double, ptrdiff_t,
        amgcl::interp::smoothed_aggregation<amgcl::aggr::plain>,
        amgcl::level::cpu<amgcl::relax::spai0>
        > AMG;

    prof.tic("setup");
    AMG amg( amgcl::sparse::map(A), AMG::params() );
    prof.toc("setup");

    std::cout << amg << std::endl;

    ublas_vector x(f.size(), 0.0);
    amgcl::solve(A, f, amg, x, amgcl::bicg_tag());
    prof.toc("solve");



    prof.tic("save result");
    fem::io::write_solution_to_VTK_file(x, "test", domain, 0);
    prof.toc("save result");



    std::cout << prof << std::endl;
}
