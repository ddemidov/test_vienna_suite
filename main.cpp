#include <chrono>
#include <algorithm>

#include <viennagrid/algorithm/boundary.hpp>
#include <viennafem/fem.hpp>
#include <viennafem/io/vtk_writer.hpp>
#include <viennamath/expression.hpp>
#include <viennacl/tools/adapter.hpp>

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
#include "eigen_adapter.hpp"

typedef Eigen::SparseMatrix<double, Eigen::RowMajor> eigen_matrix;
typedef boost::numeric::ublas::vector<double> ublas_vector;

namespace grid = viennagrid;
namespace data = viennadata;
namespace math = viennamath;
namespace fem  = viennafem;

struct viscosity_tag {
    bool operator<(viscosity_tag) const { return false; }
};

amgcl::sparse::matrix<double, int> convert_to_csr(
        const std::vector< std::map<unsigned, double> > &A
        );

//---------------------------------------------------------------------------
// Solve equation div(M * grad(u)) = 1 on a semicircle.
//---------------------------------------------------------------------------
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



    prof.tic("domain properties");
    for(size_t s_id = 0; s_id < 2; ++s_id) {
        auto elements = grid::ncells<mesher::CellTag::dim>(domain.segments()[s_id]);
        for(auto e = elements.begin(); e != elements.end(); ++e)
            data::access<viscosity_tag, double>(viscosity_tag())(*e) = s_id ? 10.0 : 1.0;
    }
    prof.toc("domain properties");



    prof.tic("assemble");
    eigen_matrix A;
    Eigen::VectorXd f;

    {
        eigen_sparse_matrix_adaptor<eigen_matrix> A_proxy(A);
        ublas_vector f_tmp;

        math::function_symbol u(0, math::unknown_tag<>());
        math::function_symbol v(0, math::test_tag<>());

        fem::cell_quan<mesher::Triangle, math::expr::interface_type> viscosity;
        viscosity.wrap_constant(viscosity_tag());

        auto weak_poisson = math::make_equation(
                math::integral(math::symbolic_interval(),
                    viscosity * (math::grad(u) * math::grad(v))),
                math::integral(math::symbolic_interval(), -1 * v)
                );

        fem::pde_assembler()(
                fem::make_linear_pde_system(weak_poisson, u),
                domain, A_proxy, f_tmp
                );

        A.makeCompressed();
        f = Eigen::Map<Eigen::VectorXd>(&f_tmp[0], f_tmp.size());
    }
    prof.toc("assemble");

    std::cout << A.outerSize() << " x " << A.innerSize() << " : " << A.nonZeros() << std::endl;

    /*
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

    Eigen::VectorXd x = Eigen::VectorXd::Zero(f.size());
    amgcl::solve(A, f, amg, x, amgcl::bicg_tag());
    prof.toc("solve");



    prof.tic("save result");
    fem::io::write_solution_to_VTK_file(x, "test", domain, 0);
    prof.toc("save result");
    */



    std::cout << prof << std::endl;
}

//---------------------------------------------------------------------------
amgcl::sparse::matrix<double, int> convert_to_csr(
        const std::vector< std::map<unsigned, double> > &a
        )
{
    size_t n   = a.size();
    size_t nnz = std::accumulate(a.begin(), a.end(), 0UL,
            [](size_t sum, const std::map<unsigned, double> &r) {
                return sum + r.size();
            });

    amgcl::sparse::matrix<double, int> A;

    A.rows = A.cols = n;
    A.row.reserve(n+1);
    A.col.reserve(nnz);
    A.val.reserve(nnz);

    A.row.push_back(0);
    for(auto r = a.begin(); r != a.end(); ++r) {
        for(auto v = r->begin(); v != r->end(); ++v) {
            A.col.push_back(std::get<0>(*v));
            A.val.push_back(std::get<1>(*v));
        }
        A.row.push_back(A.col.size());
    }

    return A;
}

