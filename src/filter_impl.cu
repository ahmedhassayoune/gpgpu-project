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
//**               Background Estimation              **
//**                                                  **
//******************************************************

#define _BE_FSIGN                                                              \
  std::byte **buffers, size_t *bpitches, int buffers_amount, std::byte *out,   \
    size_t opitch, int width, int height

__global__ void estimate_background_mean(_BE_FSIGN)
{
#define _BACKGROUND_ESTIMATION_MEAN_SPST // single position single thread

  int yy = blockIdx.y * blockDim.y + threadIdx.y;
  int xx = blockIdx.x * blockDim.x + threadIdx.x;

  if (xx >= width || yy >= height)
    return;

  constexpr size_t PIXEL_STRIDE = N_CHANNELS;

#ifdef _BACKGROUND_ESTIMATION_MEAN_SPST
  // compute sum per channel
  int sums[N_CHANNELS] = {0};
  std::byte* ptr;
  for (int ii = 0; ii < buffers_amount; ++ii)
    {
      ptr = buffers[ii] + yy * bpitches[ii] + xx * PIXEL_STRIDE;
      for (int jj = 0; jj < N_CHANNELS; ++jj)
        sums[jj] += (int)ptr[jj];
    }

  // compute mean per channel
  ptr = out + yy * opitch + xx * PIXEL_STRIDE;
  for (int ii = 0; ii < N_CHANNELS; ++ii)
    ptr[ii] = (std::byte)(sums[ii] / buffers_amount);
#else
#endif

#undef _BACKGROUND_ESTIMATION_MEAN_SPST
}

__device__ void _insertion_sort(std::byte* arr, int start, int end, int step)
{
  for (int ii = start + step; ii < end; ii += step)
    {
      int jj = ii;

      while (jj > start && arr[jj - step] > arr[jj])
        {
          std::byte tmp = arr[jj - step];
          arr[jj - step] = arr[jj];
          arr[jj] = tmp;
          jj -= step;
        }
    }
}

__global__ void estimate_background_median(_BE_FSIGN)
{
#define _BACKGROUND_ESTIMATION_MEDIAN_SPST // single position single thread

  int yy = blockIdx.y * blockDim.y + threadIdx.y;
  int xx = blockIdx.x * blockDim.x + threadIdx.x;

  if (xx >= width || yy >= height)
    return;

  constexpr size_t PIXEL_STRIDE = N_CHANNELS;

#ifdef _BACKGROUND_ESTIMATION_MEDIAN_SPST
  // 3 channels, at most 42 buffers
  // 4 channels, at most 32 buffers
  std::byte B[128];

  // for each buffer, store pixel at (yy, xx)
  for (int ii = 0; ii < buffers_amount; ++ii)
    {
      std::byte* ptr = buffers[ii] + yy * bpitches[ii] + xx * PIXEL_STRIDE;
      int jj = ii * N_CHANNELS;
      for (int kk = 0; kk < N_CHANNELS; ++kk)
        B[jj + kk] = ptr[kk];
    }

  // the median is computed for each channel
  for (int ii = 0; ii < N_CHANNELS; ++ii)
    _insertion_sort(B, ii, buffers_amount * N_CHANNELS, N_CHANNELS);

  // select mid
  // not treating differently even and odd `buffers_amount`
  // in order to avoid if clause inside a kernel
  std::byte* ptr = out + yy * opitch + xx * PIXEL_STRIDE;
  for (int ii = 0; ii < N_CHANNELS; ++ii)
    ptr[ii] = B[(buffers_amount / 2) * N_CHANNELS + ii];
#else
#endif

#undef _BACKGROUND_ESTIMATION_MEDIAN_SPST
}

#undef _BE_FSIGN

//******************************************************
//**                                                  **
//**                Apply Masking                     **
//**                                                  **
//******************************************************

__global__ void apply_masking(std::byte* buffer,
                              size_t bpitch,
                              std::byte* mask,
                              size_t mpitch,
                              int width,
                              int height)
{
  int xx = blockIdx.x * blockDim.x + threadIdx.x;
  int yy = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (xx >= width || yy >= height)
    return;

  constexpr size_t PIXEL_STRIDE = N_CHANNELS;
  rgb* ipptr = (rgb*)(buffer + yy * bpitch + xx * PIXEL_STRIDE);
  rgb* mpptr = (rgb*)(mask + yy * mpitch + xx * PIXEL_STRIDE);
  int red = (int)ipptr->r + (mpptr->r > 0) * ipptr->r / 2;
  ipptr->r = (uint8_t)(red > 0xff ? 0xff : red);
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
