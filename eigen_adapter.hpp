#ifndef EIGEN_ADAPTER_HPP
#define EIGEN_ADAPTER_HPP

#include <Eigen/SparseCore>

template <class EigenSparseMatrix>
class eigen_sparse_matrix_adaptor {
    public:
        typedef typename EigenSparseMatrix::Scalar value_type;
        typedef typename EigenSparseMatrix::Index  size_type;

        class iterator2 {
            public:
                iterator2(EigenSparseMatrix &A, size_type row, size_type ptr)
                    : col(A.innerIndexPtr()), val(A.valuePtr()), row(row), ptr(ptr)
                {}

                size_type index1() const { return row;                    }
                size_type index2() const { return col[ptr]; }

                value_type  operator*() const { return val[ptr]; }
                value_type& operator*()       { return val[ptr]; }

                iterator2& operator++() {
                    ++ptr;
                    return *this;
                }

                bool operator!=(const iterator2 &other) const {
                    return ptr != other.ptr;
                }
            private:
                size_type  *col;
                value_type *val; 
                size_type row, ptr;
        };

        class iterator1 {
            public:
                iterator1(EigenSparseMatrix &A, size_t row) : A(&A), row(row) { }

                iterator2 begin() {
                    return iterator2(*A, row, A->outerIndexPtr()[row]);
                }

                iterator2 end()   {
                    if (A->isCompressed())
                        return iterator2(*A, row, A->outerIndexPtr()[row + 1]);
                    else
                        return iterator2(*A, row, A->outerIndexPtr()[row] + A->innerNonZeroPtr()[row]);
                }

                bool operator!=(const iterator1 &other) const {
                    return row != other.row;
                }

                iterator1& operator++() {
                    ++row;
                    return *this;
                }
            private:
                EigenSparseMatrix *A;
                size_type row;
        };

        eigen_sparse_matrix_adaptor(EigenSparseMatrix &A) : A(A) {}

        iterator1 begin1() { return iterator1(A, 0); }
        iterator1 end1  () { return iterator1(A, A.outerSize()); }

        size_type size1() const { return A.outerSize(); }
        size_type size2() const { return A.innerSize(); }

        value_type  operator()(size_type i, size_type j) const {
            //return A.coeff(i,j);
            return 0;
        }
        value_type& operator()(size_type i, size_type j) {
            //return A.coeffRef(i,j);
            static value_type dummy = 0;
            return dummy;
        }

        void resize(size_type rows, size_type cols, bool) {
            A.resize(rows, cols);
            A.reserve(Eigen::VectorXi::Constant(rows, 16));
        }

        void clear() {
            A.setZero();
            A.reserve(Eigen::VectorXi::Constant(A.outerSize(), 16));
        }
    private:
        EigenSparseMatrix &A;
};


#endif
