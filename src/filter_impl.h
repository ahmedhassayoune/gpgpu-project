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

#ifdef __cplusplus

/** Background Estimation **/
#  define _BE_FSIGN                                                            \
    uint8_t **buffers, int buffers_amount, int width, int height, int stride,  \
      int pixel_stride, uint8_t *out

void estimate_background_mean(_BE_FSIGN);
void estimate_background_median(_BE_FSIGN);

#  undef _BE_FSIGN

/** Conversion from RGB to LAB **/
void rgb_to_lab(uint8_t* reference_buffer,
                uint8_t* buffer,
                int width,
                int height,
                int stride,
                int pixel_stride);

/** 
 * applies a morphological opening (3-disk) 
 * on each rgb channel independently 
**/
void opening_impl_inplace(uint8_t* buffer,
                          int width,
                          int height,
                          int stride,
                          int pixel_stride);

/** Hysteresis Threshold **/
void apply_hysteresis_threshold(uint8_t* buffer,
                                int width,
                                int height,
                                int stride,
                                int pixel_stride,
                                uint8_t low_threshold,
                                uint8_t high_threshold);

/** Masking **/
void apply_masking(uint8_t* buffer,
                   int width,
                   int height,
                   int stride,
                   int pixel_stride,
                   uint8_t* mask);

#endif