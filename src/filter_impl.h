#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

#define BG_SAMPLING_RATE 500 // sampling rate in ms
#define BG_NUMBER_FRAMES 10  // number of frames to sample
#define MAX_LAB_DISTANCE                                                       \
  292.0f // maximum distance in LAB space for sRGB pixels [0, 255]

  struct rgb
  {
    uint8_t r, g, b;
  };

  struct LAB
  {
    float l, a, b;
  };

#ifdef __cplusplus
  constexpr size_t N_CHANNELS = sizeof(rgb);
#endif

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