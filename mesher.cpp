#include <viennafem/fem.hpp>
#include <viennadata/api.hpp>

#define CGAL_DISABLE_ROUNDING_MATH_CHECK

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Constrained_triangulation_plus_2.h>
#include <CGAL/Delaunay_mesher_2.h>
#include <CGAL/Triangulation_vertex_base_2.h>
#include <CGAL/Delaunay_mesh_face_base_2.h>
#include <CGAL/Delaunay_mesh_size_criteria_2.h>

#include "mesher.h"

namespace mesher {

namespace cgal {
    typedef CGAL::Exact_predicates_inexact_constructions_kernel     Kernel;
    typedef Kernel::Vector_2                                        Vector;
    typedef CGAL::Triangulation_vertex_base_2<Kernel>               Vb;
    typedef CGAL::Delaunay_mesh_face_base_2<Kernel>                 Fb;
    typedef CGAL::Triangulation_data_structure_2<Vb, Fb>            Tds;
    typedef CGAL::Constrained_Delaunay_triangulation_2<Kernel, Tds> CDT;
    typedef CGAL::Constrained_triangulation_plus_2<CDT>             CDTPlus;
    typedef CGAL::Delaunay_mesh_size_criteria_2<CDTPlus>            Criteria;
    typedef CDTPlus::Point                                          Point;
    typedef CDTPlus::Triangle                                       Triangle;
    typedef CDTPlus::Vertex_handle                                  Vertex;
}

Domain get(const contour &cnt, double resolution) {
    // Generate mesh with CGAL.
    cgal::CDTPlus cdt;

    std::vector<cgal::Vertex> boundary;
    boundary.reserve(cnt.size() + 1);

    double x = 0, y = 0;
    boundary.push_back(cdt.insert(cgal::Point(x, y)));

    auto add_segment = [&](const contour::segment &p) {
        x += std::get<1>(p) * cos(std::get<0>(p));
        y += std::get<1>(p) * sin(std::get<0>(p));

        auto last = boundary.back();

        boundary.push_back(cdt.insert(cgal::Point(x, y)));

        cdt.insert_constraint(last, boundary.back());
    };

    std::for_each(cnt.begin(contour::wet), cnt.end(contour::wet), add_segment);

    auto wmark_right = boundary.back();
    double watermark = wmark_right->point().y();

    std::for_each(cnt.begin(contour::dry), cnt.end(contour::dry), add_segment);

    auto wmark_left = cdt.insert(cgal::Point(0, wmark_right->point().y()));

    cdt.insert_constraint(boundary.back(), wmark_left);
    cdt.insert_constraint(wmark_left, boundary.front());
    cdt.insert_constraint(wmark_left, wmark_right);

    CGAL::refine_Delaunay_mesh_2(cdt, cgal::Criteria(0.2, resolution));

    // Populate ViennaGrid domain.
    Domain domain;
    domain.segments().resize(2);

    std::map<cgal::Point, Vertex*> points;

    Triangle cell;

    for(auto e = cdt.finite_faces_begin(); e != cdt.finite_faces_end(); ++e) {
        if (!e->is_in_domain()) continue;

        auto tri = cdt.triangle(e);
        Vertex *v[3];
        double center = 0;

        for(int i = 0; i < 3; ++i) {
            auto p = points.find( tri[i] );
            if (p == points.end()) {
                domain.push_back(Point(tri[i].x(), tri[i].y()));
                size_t j = viennagrid::ncells<0>(domain).size();
                v[i] = &( viennagrid::ncells<0>(domain)[j-1] );
                points.insert(std::make_pair(tri[i], v[i]));
            } else {
                v[i] = p->second;
            }

            center += (*v[i])[1];
        }
        center /= 3;

        cell.vertices(v);
        domain.segments()[center > watermark].push_back(cell);
    }


    // Mark dirichlet boundaries.
    for(auto v1 = boundary.begin(), v2 = v1 + 1; v2 != boundary.end(); ++v1, ++v2) {
        for(auto v = cdt.vertices_in_constraint_begin(*v1, *v2),
                 e = cdt.vertices_in_constraint_end  (*v1, *v2);
            v != e; ++v
           )
        {
            using viennafem::boundary_key;

            viennadata::access<boundary_key, bool>(boundary_key(0))(*points[(*v)->point()]) = true;
        }
    }

    return domain;
}

}

