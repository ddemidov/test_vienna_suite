#ifndef DOMAIN_HPP
#define DOMAIN_HPP

#include <memory>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "contour.h"

struct mesh;

class domain {
    public:
        typedef Eigen::SparseMatrix<double, Eigen::RowMajor> Matrix;
        typedef Eigen::VectorXd                              Vector;

        domain(const contour &cnt, double resolution);
        ~domain();

        void assemble(Matrix &A, Vector &f) const;
    private:
        std::unique_ptr<mesh> m;
};

#endif
