#define CGAL_DISABLE_ROUNDING_MATH_CHECK

#include <vector>
#include <map>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Constrained_triangulation_plus_2.h>
#include <CGAL/Delaunay_mesher_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
#include <CGAL/Triangulation_face_base_with_info_2.h>
#include <CGAL/Delaunay_mesh_face_base_2.h>
#include <CGAL/Delaunay_mesh_size_criteria_2.h>

#include "domain.h"

struct mesh {
    typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;

    struct vertex_info {
        long id;
        bool dirichlet;

        vertex_info() : id(-1), dirichlet(false) {}
    };

    typedef CGAL::Triangulation_vertex_base_with_info_2<vertex_info, Kernel> Vb;

    struct face_info {
        long                           id;
        double                         detJ;
        std::array<Eigen::Vector2d, 3> grad;
    };

    typedef CGAL::Delaunay_mesh_face_base_2<
        Kernel,
        CGAL::Constrained_triangulation_face_base_2<
            Kernel, CGAL::Triangulation_face_base_with_info_2<face_info, Kernel>
            >
        > Fb;

    typedef CGAL::Triangulation_data_structure_2<Vb, Fb>            Tds;
    typedef CGAL::Constrained_Delaunay_triangulation_2<Kernel, Tds> CDT;
    typedef CGAL::Constrained_triangulation_plus_2<CDT>             CDTPlus;
    typedef CGAL::Delaunay_mesh_size_criteria_2<CDTPlus>            Criteria;
    typedef CDTPlus::Point                                          Point;
    typedef CDTPlus::Vertex_handle                                  Vertex;

    CDTPlus             cdt;
    std::vector<Vertex> boundary;
    Vertex              wmark_left, wmark_right;
    double              watermark;

    mesh(const contour &cnt, double resolution) {
        // Generate mesh.
        boundary.reserve(cnt.size() + 1);

        double x = 0, y = 0;
        boundary.push_back(cdt.insert(Point(x, y)));

        auto add_segment = [&](const contour::segment &p) {
            x += std::get<1>(p) * cos(std::get<0>(p));
            y += std::get<1>(p) * sin(std::get<0>(p));

            auto last = boundary.back();

            boundary.push_back(cdt.insert(Point(x, y)));

            cdt.insert_constraint(last, boundary.back());
        };

        std::for_each(cnt.begin(contour::wet), cnt.end(contour::wet), add_segment);

        wmark_right = boundary.back();
        watermark   = wmark_right->point().y();

        std::for_each(cnt.begin(contour::dry), cnt.end(contour::dry), add_segment);

        wmark_left = cdt.insert(Point(0, watermark));

        cdt.insert_constraint(boundary.back(), wmark_left);
        cdt.insert_constraint(wmark_left, boundary.front());
        cdt.insert_constraint(wmark_left, wmark_right);

        CGAL::refine_Delaunay_mesh_2(cdt, Criteria(0.2, resolution));

        // Enumerate mesh elements.
        long v_id = 0;
        for(auto v = cdt.finite_vertices_begin(), e = cdt.finite_vertices_end(); v != e; ++v)
            v->info().id = v_id++;

        // Mark dirichlet points.
        for(auto v1 = boundary.begin(), v2 = v1 + 1; v2 != boundary.end(); ++v1, ++v2) {
            for(auto v = cdt.vertices_in_constraint_begin(*v1, *v2),
                     e = cdt.vertices_in_constraint_end  (*v1, *v2);
                     v != e; ++v
               )
            {
                (*v)->info().dirichlet = true;
            }
        }

        // Compute element properties.
        long nfaces = 0;
        for(auto f = cdt.finite_faces_begin(), e = cdt.finite_faces_end(); f != e; ++f) {
            if (!f->is_in_domain()) continue;

            f->info().id = nfaces++;

            double J00 = f->vertex(1)->point().x() - f->vertex(0)->point().x();
            double J10 = f->vertex(1)->point().y() - f->vertex(0)->point().y();
            double J01 = f->vertex(2)->point().x() - f->vertex(0)->point().x();
            double J11 = f->vertex(2)->point().y() - f->vertex(0)->point().y();

            double detJ = f->info().detJ = fabs(J00 * J11 - J01 * J10);

            double Jit00 =  J11 / detJ;
            double Jit01 = -J10 / detJ;
            double Jit10 = -J01 / detJ;
            double Jit11 =  J00 / detJ;

            f->info().grad[0] = Eigen::Vector2d(-Jit00 - Jit01, -Jit10 - Jit11);
            f->info().grad[1] = Eigen::Vector2d(     Jit00    ,     Jit10     );
            f->info().grad[2] = Eigen::Vector2d(     Jit01    ,     Jit11     );
        }
    }
};

//---------------------------------------------------------------------------
domain::domain(const contour &cnt, double resolution)
    : m(new mesh(cnt, resolution)) { }

//---------------------------------------------------------------------------
domain::~domain() { }

//---------------------------------------------------------------------------
void domain::assemble(Matrix &A, Vector &rhs) const {
    const auto &cdt = m->cdt;
    size_t n = cdt.number_of_vertices();

    rhs = Vector::Zero(n);

    double wm = 3 * m->watermark;

    std::vector< std::map<long, double> > A_tmp(n);

    for(auto face = cdt.finite_faces_begin(); face != cdt.finite_faces_end(); ++face) {
	if (!face->is_in_domain()) continue;

        std::array<long, 3> id = {
		face->vertex(0)->info().id,
		face->vertex(1)->info().id,
		face->vertex(2)->info().id
                };

        std::array<bool, 3> dirichlet = {
		face->vertex(0)->info().dirichlet,
		face->vertex(1)->info().dirichlet,
		face->vertex(2)->info().dirichlet
                };

        std::array<double, 3> y = {
		face->vertex(0)->point().y(),
		face->vertex(1)->point().y(),
		face->vertex(2)->point().y()
                };

	double M    = (y[0] + y[1] + y[2]) > wm ? 10.0 : 1.0;
        double detJ = face->info().detJ;

	for(int i = 0; i < 3; i++) {
	    if (dirichlet[i]) continue;

	    for(int j = 0; j < 3; j++) {
		rhs[id[i]] += detJ / 24 * (1 + (i == j));

		if (dirichlet[j]) continue;

		A_tmp[id[i]][id[j]] += 0.5 * M * detJ
		    * (face->info().grad[i].dot(face->info().grad[j]));
	    }
	}
    }

    // Точки с условием Дирихле.
    for(auto vtx = cdt.finite_vertices_begin(); vtx != cdt.finite_vertices_end(); ++vtx) {
	if (vtx->info().dirichlet) {
            long row = vtx->info().id;

	    A_tmp[row][row] = 1.0;
	    rhs[row] = 0.0;
	}
    }

    size_t nnz = std::accumulate(A_tmp.begin(), A_tmp.end(), 0UL,
            [](size_t sum, const std::map<long, double> &r) {
                return sum + r.size();
            });

    A.resize(n, n);
    A.reserve(nnz);

    for(long i = 0; i < n; ++i)
        for(auto v = A_tmp[i].begin(), e = A_tmp[i].end(); v != e; ++v)
            A.insert(i, v->first) = v->second;

    A.makeCompressed();
}
