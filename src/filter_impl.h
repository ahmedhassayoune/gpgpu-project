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
#endif
