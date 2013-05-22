#include <chrono>
#include <algorithm>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <amgcl/amgcl.hpp>
#include <amgcl/interp_smoothed_aggr.hpp>
#include <amgcl/aggr_plain.hpp>
#include <amgcl/level_cpu.hpp>
#include <amgcl/operations_eigen.hpp>
#include <amgcl/bicgstab.hpp>
#include <amgcl/profiler.hpp>

#include "contour.h"
#include "domain.h"

//---------------------------------------------------------------------------
// Solve equation div(M * grad(u)) = 1 on a semicircle.
//---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    amgcl::profiler<std::chrono::high_resolution_clock> prof;



    prof.tic("generate mesh");
    contour c(256, 0.4);
    domain domain(c, argc > 1 ? std::stod(argv[1]) : 3e-2);
    prof.toc("generate mesh");



    prof.tic("assemble");
    domain::Matrix A;
    domain::Vector f;
    domain.assemble(A, f);
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

    domain::Vector x = domain::Vector::Zero(A.rows());
    amgcl::solve(A, f, amg, x, amgcl::bicg_tag());
    prof.toc("solve");


    std::cout << prof << std::endl;
}

