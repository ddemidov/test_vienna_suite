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

// Solve equation laplace(u) = 1 on a semicircle.
int main(int argc, char *argv[]) {
    amgcl::profiler<std::chrono::high_resolution_clock> prof;

    // Generate domain for a semicircle.
    prof.tic("domain");
    contour c(256, 0.4);
    auto domain = mesher::get(c, argc > 1 ? std::stod(argv[1]) : 3e-2);
    prof.toc("domain");

    // Assemble linear system for the PDE.
    prof.tic("assemble");
    viennamath::function_symbol u;
    auto poisson = viennamath::make_equation( viennamath::laplace(u), 1 );

    // Boundary condition on the outer boundary.
    auto vertices = viennagrid::ncells<0>(domain);
    for(auto v = vertices.begin(); v != vertices.end(); ++v) {
        using viennafem::boundary_key;

        if (viennadata::find<boundary_key, bool>(boundary_key(0))(*v))
            viennafem::set_dirichlet_boundary(*v, 0.0);
    }

    ublas_matrix A;
    ublas_vector f;

    viennafem::pde_assembler fem_assembler;

    fem_assembler(viennafem::make_linear_pde_system(poisson, u), domain, A, f);
    prof.toc("assemble");

    ublas_vector x(f.size(), 0.0);

    // Solve the assembled system with amgcl
    typedef amgcl::solver<
        double, ptrdiff_t,
        amgcl::interp::smoothed_aggregation<amgcl::aggr::plain>,
        amgcl::level::cpu<amgcl::relax::spai0>
        > AMG;

    prof.tic("solve");
    AMG amg( amgcl::sparse::map(A), AMG::params() );

    std::cout << amg << std::endl;

    auto cnv = amgcl::solve(A, f, amg, x, amgcl::bicg_tag());
    prof.toc("solve");

    // Save the result.
    prof.tic("write");
    viennafem::io::write_solution_to_VTK_file(x, "test", domain, 0);
    prof.toc("write");

    std::cout << prof << std::endl;
}
