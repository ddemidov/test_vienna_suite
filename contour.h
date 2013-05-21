#ifndef CONTOUR_HPP
#define CONTOUR_HPP

#include <cstddef>
#include <vector>
#include <tuple>
#include <cmath>

/// A dicretization of a tube shape.
class contour {
    public:
        typedef std::tuple<double, double> segment;
        typedef typename std::vector< segment >::const_iterator const_iterator;
        typedef typename std::vector< segment >::iterator iterator;

        enum perimeter_part {
            wet = 0,
            dry = 1
        };

        /// Constructor.
        /**
         * Creates ideal semi-circle.
         * \param n     number of segments in the discretiztion.
         * \param gamma wetted fraction of tube perimeter.
         */
        contour(size_t n, double gamma);

        segment& operator[](size_t i) {
            return shape[i];
        }

        const segment operator[](size_t i) const {
            return shape[i];
        }

        size_t size() const {
            return shape.size();
        }

        const_iterator begin() const { return shape.begin(); }
        const_iterator end()   const { return shape.end();   }

        iterator begin() { return shape.begin(); }
        iterator end()   { return shape.end();   }

        const_iterator begin(perimeter_part part) const {
            switch (part) {
                case wet:
                    return shape.begin();
                case dry:
                    return shape.begin() + watermark;
            }
        }

        const_iterator end(perimeter_part part) const {
            switch (part) {
                case wet:
                    return shape.begin() + watermark;
                case dry:
                    return shape.end();
            }
        }

        iterator begin(perimeter_part part) {
            switch (part) {
                case wet:
                    return shape.begin();
                case dry:
                    return shape.begin() + watermark;
            }
        }

        iterator end(perimeter_part part) {
            switch (part) {
                case wet:
                    return shape.begin() + watermark;
                case dry:
                    return shape.end();
            }
        }

    private:
        size_t watermark;

        // Each segment is represented as a pair of angle and length.
        std::vector< std::tuple<double, double> > shape;
};

#endif
