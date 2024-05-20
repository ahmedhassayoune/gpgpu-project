#include "filter_impl.h"

#include <chrono>
#include <thread>
#include "logo.h"

#define N_CHANNELS 3

struct rgb
{
  uint8_t r, g, b;
};

extern "C"
{
  void filter_impl(uint8_t* buffer,
                   int width,
                   int height,
                   int stride,
                   int pixel_stride)
  {
    for (int y = 0; y < height; ++y)
      {
        uint8_t* lineptr = (uint8_t*)(buffer + y * stride);
        for (int x = 0; x < width; ++x)
          {
            rgb pxl = *(rgb*)(lineptr + x * pixel_stride);
            pxl.r = 0; // Back out red component

            if (x < logo_width && y < logo_height)
              {
                float alpha = logo_data[y * logo_width + x] / 255.f;
                pxl.g = uint8_t(alpha * pxl.g + (1 - alpha) * 255);
                pxl.b = uint8_t(alpha * pxl.b + (1 - alpha) * 255);
              }
          }
      }

    // You can fake a long-time process with sleep
    {
      using namespace std::chrono_literals;
      //std::this_thread::sleep_for(100ms);
    }
  }

  void estimate_background_mean(uint8_t** buffers,
                                int buffers_amount,
                                int width,
                                int height,
                                int stride,
                                int pixel_stride,
                                uint8_t* out)
  {
    // It is expected `out` has same stride values

    if (!buffers || !out)
      return;

    for (int ii = 0; ii < buffers_amount; ++ii)
      if (!buffers[ii])
        return;

    int sums[N_CHANNELS] = {0};

    for (int yy = 0; yy < height; ++yy)
      {
        for (int xx = 0; xx < width; ++xx)
          {
            for (int ii = 0; ii < N_CHANNELS; ++ii)
              sums[ii] = 0;

            // compute sums for each channel
            for (int ii = 0; ii < buffers_amount; ++ii)
              {
                uint8_t* ptr = buffers[ii] + yy * stride + xx * pixel_stride;
                for (int jj = 0; jj < N_CHANNELS; ++jj)
                  sums[jj] += ptr[jj];
              }

            // compute mean for each channel
            uint8_t* outptr = out + yy * stride + xx * pixel_stride;
            for (int ii = 0; ii < N_CHANNELS; ++ii)
              outptr[ii] = (uint8_t)(sums[ii] / buffers_amount);
          }
      }
  }

  void _selection_sort(uint8_t* bytes, int start, int end, int step)
  {
    for (int ii = start; ii + step < end; ii += step)
      {
        int jj = ii;

        for (int kk = jj + step; kk < end; kk += step)
          if (bytes[jj] > bytes[kk])
            jj = kk;

        if (jj != ii)
          {
            uint8_t tmp = bytes[jj];
            bytes[jj] = bytes[ii];
            bytes[ii] = tmp;
          }
      }
  }

  void estimate_background_median(uint8_t** buffers,
                                  int buffers_amount,
                                  int width,
                                  int height,
                                  int stride,
                                  int pixel_stride,
                                  uint8_t* out)
  {
    if (!buffers || !out)
      return;

    for (int ii = 0; ii < buffers_amount; ++ii)
      if (!buffers[ii])
        return;

    uint8_t B[512];
    int batch_size = buffers_amount * pixel_stride;

    for (int yy = 0; yy < height; ++yy)
      {
        for (int xx = 0; xx < width; ++xx)
          {
            // for each buffer store pixel at (yy, xx)
            for (int ii = 0; ii < buffers_amount; ++ii)
              {
                uint8_t* ptr = buffers[ii] + yy * stride + xx * pixel_stride;
                int jj = ii * pixel_stride;
                for (int kk = 0; kk < pixel_stride; ++kk)
                  B[jj + kk] = ptr[kk];
              }

            // the median is computed for each channel separately

            // each channel is sorted in order to find its mid value
            for (int ii = 0; ii < pixel_stride; ++ii)
              _selection_sort(B, ii, batch_size, pixel_stride);

            // select mid
            uint8_t* outptr = out + yy * stride + xx * pixel_stride;
            if (buffers_amount % 2 == 0)
              {
                for (int ii = 0; ii < pixel_stride; ++ii)
                  {
                    int a = B[(buffers_amount / 2) * pixel_stride + ii];
                    int b = B[(buffers_amount / 2 - 1) * pixel_stride + ii];
                    outptr[ii] = (uint8_t)((a + b) / 2);
                  }
              }
            else
              {
                for (int ii = 0; ii < pixel_stride; ++ii)
                  outptr[ii] = B[(buffers_amount / 2) * pixel_stride + ii];
              }
          }
      }
  }
}
