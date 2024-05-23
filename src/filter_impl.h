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
/** Conversion from RGB to LAB **/
struct LAB
{
  float l, a, b;
};

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
struct pos
{
  int x, y;
};

struct Queue
{
  pos* buffer_pos;
  int front = 0;
  int rear = 0;
};

void enqueue(Queue& q, int x, int y);
pos dequeue(Queue& q);
bool is_empty(Queue& q);
void apply_hysteresis_threshold(uint8_t* buffer,
                                int width,
                                int height,
                                int stride,
                                int pixel_stride,
                                uint8_t low_threshold,
                                uint8_t high_threshold);

#  define _BE_FSIGN                                                            \
    uint8_t **buffers, int buffers_amount, int width, int height, int stride,  \
      int pixel_stride, uint8_t *out

void estimate_background_mean(_BE_FSIGN);
void estimate_background_median(_BE_FSIGN);

#  undef _BE_FSIGN

void apply_masking(uint8_t* buffer,
                   int width,
                   int height,
                   int stride,
                   int pixel_stride,
                   uint8_t* mask);

#endif