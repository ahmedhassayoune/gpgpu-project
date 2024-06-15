#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

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

  struct filter_params
  {
    uint8_t* bg;
    int th_low;
    int th_high;
    int bg_sampling_rate;
    int bg_number_frames;
  };

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
                   const struct filter_params* params);

#ifdef __cplusplus
}
#endif