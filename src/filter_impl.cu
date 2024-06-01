#include "filter_impl.h"

#include <cassert>
#include <chrono>
#include <cstdio>
#include <thread>
#include "logo.h"

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err,
           const char* const func,
           const char* const file,
           const int line)
{
  if (err != cudaSuccess)
    {
      std::fprintf(stderr, "CUDA Runtime Error at: %s: %d\n", file, line);
      std::fprintf(stderr, "%s %s\n", cudaGetErrorString(err), func);
      // We don't exit when we encounter CUDA errors in this example.
      std::exit(EXIT_FAILURE);
    }
}

__constant__ uint8_t* logo;

/// @brief Black out the red channel from the video and add EPITA's logo
/// @param buffer
/// @param width
/// @param height
/// @param stride
/// @param pixel_stride
/// @return
__global__ void
remove_red_channel_inp(std::byte* buffer, int width, int height, int stride)
{
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;

  if (x >= width || y >= height)
    return;

  rgb* lineptr = (rgb*)(buffer + y * stride);
  if (y < logo_height && x < logo_width)
    {
      float alpha = logo[y * logo_width + x] / 255.f;
      lineptr[x].r = 0;
      lineptr[x].g = uint8_t(alpha * lineptr[x].g + (1 - alpha) * 255);
      lineptr[x].b = uint8_t(alpha * lineptr[x].b + (1 - alpha) * 255);
    }
  else
    {
      lineptr[x].r = 0;
    }
}



//******************************************************
//**                                                  **
//**             Morphological Opening                **
//**                                                  **
//******************************************************



__global__ void morphological_opening_inplace(uint8_t* buffer,
                                         int width,
                                         int height,
                                         int stride,
                                         int pixel_stride)
{
  int yy = blockIdx.y * blockDim.y + threadIdx.y;
  int xx = (blockIdx.x * blockDim.x + threadIdx.x) / 3;
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) % 3

  if (xx >= width || yy >= height)
    return;
  
  uint8_t res = 255;

  if (yy >= 3) {
    res = buffer[(yy - 3) * stride + xx * pixel_stride + idx];
  }
  for (int i = y - 2; i < y; ++i) {
    if (i >= 0) {
      for (int j = xx - 2; j <= xx +2; j++) {
        if (j >= 0 && j < width) {
          res = min(res, buffer[i * stride + j * pixel_stride + idx]);
        }
      }
    }
  }
  for (int j = xx - 3; j <= xx + 3; j++) {
    if (j >= 0 && j < width) {
        res = min(res, buffer[yy * stride + j * pixel_stride + idx]);
    }
  }
  for (int i = y + 1; i <= y + 2; ++i) {
    if (i < width) {
      for (int j = xx - 2; j <= xx +2; j++) {
        if (j >= 0 && j < width) {
          res = min(res, buffer[i * stride + j * pixel_stride + idx]);
        }
      }
    }
  }
  if (yy + 3 < width) {
    res = min(res, buffer[(yy - 3) * stride + xx * pixel_stride + idx]);
  }

  //__syncthreads();
  buffer[yy * stride + xx * pixel_stride + idx] = res;
  //__syncthreads();

  res = 0;

  if (yy >= 3) {
    res = buffer[(yy - 3) * stride + xx * pixel_stride + idx];
  }
  for (int i = y - 2; i < y; ++i) {
    if (i >= 0) {
      for (int j = xx - 2; j <= xx +2; j++) {
        if (j >= 0 && j < width) {
          res = max(res, buffer[i * stride + j * pixel_stride + idx]);
        }
      }
    }
  }
  for (int j = xx - 3; j <= xx + 3; j++) {
    if (j >= 0 && j < width) {
        res = max(res, buffer[yy * stride + j * pixel_stride + idx]);
    }
  }
  for (int i = y + 1; i <= y + 2; ++i) {
    if (i < width) {
      for (int j = xx - 2; j <= xx +2; j++) {
        if (j >= 0 && j < width) {
          res = max(res, buffer[i * stride + j * pixel_stride + idx]);
        }
      }
    }
  }
  if (yy + 3 < width) {
    res = max(res, buffer[(yy - 3) * stride + xx * pixel_stride + idx]);
  }
}

namespace
{
  void load_logo()
  {
    static auto buffer =
      std::unique_ptr<std::byte, decltype(&cudaFree)>{nullptr, &cudaFree};

    if (buffer == nullptr)
      {
        cudaError_t err;
        std::byte* ptr;
        err = cudaMalloc(&ptr, logo_width * logo_height);
        CHECK_CUDA_ERROR(err);

        err = cudaMemcpy(ptr, logo_data, logo_width * logo_height,
                         cudaMemcpyHostToDevice);
        CHECK_CUDA_ERROR(err);

        err = cudaMemcpyToSymbol(logo, &ptr, sizeof(ptr));
        CHECK_CUDA_ERROR(err);

        buffer.reset(ptr);
      }
  }
} // namespace

extern "C"
{
  void filter_impl(uint8_t* src_buffer,
                   const frame_info* buffer_info,
                   int th_low,
                   int th_high)
  {
    int width = buffer_info->width;
    int height = buffer_info->height;
    int src_stride = buffer_info->stride;

    load_logo();

    assert(sizeof(rgb) == buffer_info->pixel_stride);
    std::byte* dBuffer;
    size_t pitch;

    cudaError_t err;

    err = cudaMallocPitch(&dBuffer, &pitch, width * sizeof(rgb), height);
    CHECK_CUDA_ERROR(err);

    err = cudaMemcpy2D(dBuffer, pitch, src_buffer, src_stride,
                       width * sizeof(rgb), height, cudaMemcpyDefault);
    CHECK_CUDA_ERROR(err);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + (blockSize.x - 1)) / blockSize.x,
                  (height + (blockSize.y - 1)) / blockSize.y);

    remove_red_channel_inp<<<gridSize, blockSize>>>(dBuffer, width, height,
                                                    pitch);

    err = cudaMemcpy2D(src_buffer, src_stride, dBuffer, pitch,
                       width * sizeof(rgb), height, cudaMemcpyDefault);
    CHECK_CUDA_ERROR(err);

    cudaFree(dBuffer);

    err = cudaDeviceSynchronize();
    CHECK_CUDA_ERROR(err);

    {
      using namespace std::chrono_literals;
      //std::this_thread::sleep_for(100ms);
    }
  }



}
