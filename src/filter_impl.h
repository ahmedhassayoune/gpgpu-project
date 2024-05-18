#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  void filter_impl(uint8_t* buffer,
                   int width,
                   int height,
                   int plane_stride,
                   int pixel_stride);

#ifdef __cplusplus
}
#endif

struct LAB
{
  float l, a, b;
};

float gamma_correct(float channel);
void rgbToXyz(float r, float g, float b, float& x, float& y, float& z);
void xyzToLab(float x, float y, float z, float& l, float& a, float& b);
float labDistance(const LAB& lab1, const LAB& lab2);
void rgb_to_lab(uint8_t* reference_buffer,
                uint8_t* buffer,
                int width,
                int height,
                int stride,
                int pixel_stride);