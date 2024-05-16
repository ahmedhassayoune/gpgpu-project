#include "rgb_to_lab.hh"

void rgbToXyz(double r, double g, double b, double& x, double& y, double& z) {
    r = r / 255.0;
    g = g / 255.0;
    b = b / 255.0;

    r = (r > 0.04045) ? pow((r + 0.055) / 1.055, 2.4) : r / 12.92;
    g = (g > 0.04045) ? pow((g + 0.055) / 1.055, 2.4) : g / 12.92;
    b = (b > 0.04045) ? pow((b + 0.055) / 1.055, 2.4) : b / 12.92;

    r *= 100.0;
    g *= 100.0;
    b *= 100.0;

    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375;
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750;
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041;

    std::cout << "RGB -> XYZ: (" << r << ", " << g << ", " << b << ") -> (" << x << ", " << y << ", " << z << ")\n";
}

void xyzToLab(double x, double y, double z, LAB& lab) {
    x /= 95.047;
    y /= 100.000;
    z /= 108.883;

    x = (x > 0.008856) ? pow(x, 1.0/3.0) : (7.787 * x) + (16.0 / 116.0);
    y = (y > 0.008856) ? pow(y, 1.0/3.0) : (7.787 * y) + (16.0 / 116.0);
    z = (z > 0.008856) ? pow(z, 1.0/3.0) : (7.787 * z) + (16.0 / 116.0);

    lab.l = (116 * y) - 16;
    lab.a = 500 * (x - y);
    lab.b = 200 * (y - z);

    std::cout << "XYZ -> LAB: (" << x << ", " << y << ", " << z << ") -> (" << lab.l << ", " << lab.a << ", " << lab.b << ")\n";
}

double labDistance(const LAB& lab1, const LAB& lab2) {
    return sqrt(pow(lab1.l - lab2.l, 2) + pow(lab1.a - lab2.a, 2) + pow(lab1.b - lab2.b, 2));
}
