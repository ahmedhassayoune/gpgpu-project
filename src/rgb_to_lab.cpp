#include "rgb_to_lab.hh"
#include <cmath>

const float D65_XYZ[9] = {
    0.412453f, 0.357580f, 0.180423f,
    0.212671f, 0.715160f, 0.072169f,
    0.019334f, 0.119193f, 0.950227f
};

float gammaCorrect(float channel) {
    return (channel > 0.04045f) ? powf((channel + 0.055f) / 1.055f, 2.4f) : channel / 12.92f;
}

void rgbToXyz(float r, float g, float b, float& x, float& y, float& z) {
    r = r / 255.0f;
    g = g / 255.0f;
    b = b / 255.0f;

    r = gammaCorrect(r);
    g = gammaCorrect(g);
    b = gammaCorrect(b);

    x = r * D65_XYZ[0] + g * D65_XYZ[1] + b * D65_XYZ[2];
    y = r * D65_XYZ[3] + g * D65_XYZ[4] + b * D65_XYZ[5];
    z = r * D65_XYZ[6] + g * D65_XYZ[7] + b * D65_XYZ[8];
}
const float D65_Xn = 0.95047f;
const float D65_Yn = 1.00000f;
const float D65_Zn = 1.08883f;

void xyzToLab(float x, float y, float z, LAB& lab) {
    x /= D65_Xn;
    y /= D65_Yn;
    z /= D65_Zn;

    float fx = (x > 0.008856f) ? powf(x, 1.0f / 3.0f) : (7.787f * x + 16.0f / 116.0f);
    float fy = (y > 0.008856f) ? powf(y, 1.0f / 3.0f) : (7.787f * y + 16.0f / 116.0f);
    float fz = (z > 0.008856f) ? powf(z, 1.0f / 3.0f) : (7.787f * z + 16.0f / 116.0f);

    lab.l = (116.0f * fy) - 16.0f;
    lab.a = 500.0f * (fx - fy);
    lab.b = 200.0f * (fy - fz);
}

float labDistance(const LAB& lab1, const LAB& lab2) {
    return sqrtf(powf(lab1.l - lab2.l, 2) + powf(lab1.a - lab2.a, 2) + powf(lab1.b - lab2.b, 2));
}
