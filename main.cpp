#include <chrono>
#include <algorithm>

#include <viennagrid/algorithm/boundary.hpp>
#include <viennafem/fem.hpp>
#include <viennafem/io/vtk_writer.hpp>
#include <viennamath/expression.hpp>
#include <viennacl/tools/adapter.hpp>

#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <boost/numeric/ublas/vector.hpp>

#include <amgcl/amgcl.hpp>
#include <amgcl/interp_smoothed_aggr.hpp>
#include <amgcl/aggr_plain.hpp>
#include <amgcl/level_cpu.hpp>
#include <amgcl/operations_eigen.hpp>
#include <amgcl/bicgstab.hpp>
#include <amgcl/profiler.hpp>

#include "contour.h"
#include "mesher.h"

typedef boost::numeric::ublas::vector<double> ublas_vector;

// Solve equation laplace(u) = 1 on a semicircle.
int main(int argc, char *argv[]) {
    amgcl::profiler<std::chrono::high_resolution_clock> prof;

    // Generate domain for a semicircle.
    prof.tic("generate mesh");
    contour c(256, 0.4);
    auto domain = mesher::get(c, argc > 1 ? std::stod(argv[1]) : 3e-2, prof);
    prof.toc("generate mesh");

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


    amgcl::sparse::matrix<double, int> A;
    viennafem::pde_assembler fem_assembler;
    Eigen::VectorXd f;

    {
        std::vector< std::map<unsigned, double> > A_tmp;
        viennacl::tools::sparse_matrix_adapter<double> A_proxy(A_tmp);
        ublas_vector f_tmp;

        fem_assembler(viennafem::make_linear_pde_system(poisson, u), domain,
                A_proxy, f_tmp);

        // Copy matrix to CSR format, wrap it into Eigen type.
        size_t n   = A_tmp.size();
        size_t nnz = std::accumulate(A_tmp.begin(), A_tmp.end(), 0UL,
                [](size_t sum, const std::map<unsigned, double> &r) {
                    return sum + r.size();
                });

        A.rows = A.cols = n;
        A.row.reserve(A_tmp.size() + 1);
        A.col.reserve(nnz);
        A.val.reserve(nnz);

        A.row.push_back(0);
        for(auto r = A_tmp.begin(); r != A_tmp.end(); ++r) {
            for(auto v = r->begin(); v != r->end(); ++v) {
                A.col.push_back(std::get<0>(*v));
                A.val.push_back(std::get<1>(*v));
            }
            A.row.push_back(A.col.size());
        }

        f.resize(n);
        std::copy(f_tmp.begin(), f_tmp.end(), &f[0]);
    }

    prof.toc("assemble");

    // Solve the assembled system with amgcl
    typedef amgcl::solver<
        double, int,
        amgcl::interp::smoothed_aggregation<amgcl::aggr::plain>,
        amgcl::level::cpu<amgcl::relax::spai0>
        > AMG;

    prof.tic("solve");
    prof.tic("setup");
    AMG amg( A, AMG::params() );
    prof.toc("setup");

    std::cout << amg << std::endl;

    Eigen::MappedSparseMatrix<double, Eigen::RowMajor, int> A_eigen(
            A.rows, A.cols, A.row.back(), A.row.data(), A.col.data(), A.val.data()
            );
    Eigen::VectorXd x = Eigen::VectorXd::Zero(A.rows), rhs(A.rows);

    amgcl::solve(A_eigen, f, amg, x, amgcl::bicg_tag());
    prof.toc("solve");

    // Save the result.
    prof.tic("write");
    viennafem::io::write_solution_to_VTK_file(x, "test", domain, 0);
    prof.toc("write");

    std::cout << prof << std::endl;
}
