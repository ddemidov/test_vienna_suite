#include "contour.h"

contour::contour(size_t n, double gamma) {
    watermark = n * gamma;

    if (watermark == 0 && gamma > 0) watermark = 1;

    double dw1 = M_PI * gamma / watermark;
    double dw2 = M_PI * (1 - gamma) / (n - watermark);

    double dl1 = 2 * sin(dw1 / 2);
    double dl2 = 2 * sin(dw2 / 2);

    shape.resize(n);

    {
        double omega = dw1 / 2;
        for(int i = 0; i < watermark; ++i, omega += dw1)
            shape[i] = std::make_tuple(omega, dl1);
    }

    {
        double omega = M_PI - dw2 / 2;
        for(int i = n - 1; i >= watermark; --i, omega -= dw2)
            shape[i] = std::make_tuple(omega, dl2);
    }
}
