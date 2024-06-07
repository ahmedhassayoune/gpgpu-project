#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

#define BG_SAMPLING_RATE 500 // sampling rate in ms
#define BG_NUMBER_FRAMES 10  // number of frames to sample

  struct rgb
  {
    uint8_t r, g, b;
  };

  constexpr size_t N_CHANNELS = sizeof(rgb);

  struct frame_info
  {
    int width;
    int height;
    int stride;
    int pixel_stride;
    double timestamp;
  };

  void filter_impl(uint8_t* buffer,
                   const struct frame_info* buffer_info,
                   int th_low,
                   int th_high);

#ifdef __cplusplus
}
#endif