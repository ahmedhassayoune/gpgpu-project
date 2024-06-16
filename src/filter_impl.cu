#include "filter_impl.h"

#include <cassert>
#include <chrono>
#include <cstdio>
#include <thread>

#define BLOCK_SIZE 16
#define HYSTERESIS_ITER 5

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

__global__ void resize_pixels(std::byte* in,
                              size_t ipxsize,
                              size_t ipitch,
                              std::byte* out,
                              size_t opxsize,
                              size_t opitch,
                              int width,
                              int height)
{
  int yy = blockIdx.y * blockDim.y + threadIdx.y;
  int xx = blockIdx.x * blockDim.x + threadIdx.x;
  if (xx < width && yy < height)
    {
      size_t min_px_size = min(ipxsize, opxsize);
      for (size_t i = 0; i < min_px_size; ++i)
        {
          out[yy * opitch + xx * opxsize + i] =
            in[yy * ipitch + xx * ipxsize + i];
        }
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

#ifdef _BACKGROUND_ESTIMATION_MEAN_SPST
  // compute sum per channel
  int sums[N_CHANNELS] = {0};
  std::byte* ptr;
  for (int ii = 0; ii < buffers_amount; ++ii)
    {
      ptr = buffers[ii] + yy * bpitches[ii] + xx * N_CHANNELS;
      for (int jj = 0; jj < N_CHANNELS; ++jj)
        sums[jj] += (int)ptr[jj];
    }

  // compute mean per channel
  ptr = out + yy * opitch + xx * N_CHANNELS;
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
#define _BACKGROUND_ESTIMATION_MEDIAN_ISORT
  // #define _BACKGROUND_ESTIMATION_MEDIAN_LOCAL_HIST
  // #define _BACKGROUND_ESTIMATION_MEDIAN_SHARED_HIST

  int yy = blockIdx.y * blockDim.y + threadIdx.y;
  int xx = blockIdx.x * blockDim.x + threadIdx.x;

  if (xx >= width || yy >= height)
    return;

  constexpr size_t PIXEL_STRIDE = N_CHANNELS;

#ifdef _BACKGROUND_ESTIMATION_MEDIAN_ISORT
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
#  ifdef _BACKGROUND_ESTIMATION_MEDIAN_LOCAL_HIST
  // a histogram per channel
  uint8_t H[N_CHANNELS * 256] = {0};

  // for each buffer, for each channel, compute histogram
  for (int ii = 0; ii < buffers_amount; ++ii)
    {
      uint8_t* ptr =
        (uint8_t*)buffers[ii] + yy * bpitches[ii] + xx * PIXEL_STRIDE;
      for (int jj = 0; jj < N_CHANNELS; ++jj)
        ++H[jj * 256 + ptr[jj]];
    }

  // for each histogram, compute cumulative sum
  uint8_t* outptr = (uint8_t*)out + yy * opitch + xx * PIXEL_STRIDE;
  for (int ii = 0; ii < N_CHANNELS; ++ii)
    {
      uint8_t cumsum = 0;
      int jj = 0;
      for (; jj < 256 && cumsum < buffers_amount / 2 + 1; ++jj)
        cumsum += H[ii * 256 + jj];
      outptr[ii] = (uint8_t)(jj - 1);
    }
#  else
#    ifdef _BACKGROUND_ESTIMATION_MEDIAN_SHARED_HIST
  // each thread has `N_CHANNELS` histograms
  extern __shared__ uint8_t H[];

  constexpr size_t THREAD_STRIDE = 256 * N_CHANNELS;
  size_t offset =
    ((yy % blockDim.y) * blockDim.x + (xx % blockDim.x)) * THREAD_STRIDE;

  // clean histograms
  for (size_t ii = 0; ii < THREAD_STRIDE; ++ii)
    H[offset + ii] = 0x00;

  // for each buffer, compute histogram
  for (int ii = 0; ii < buffers_amount; ++ii)
    {
      uint8_t* ptr =
        (uint8_t*)buffers[ii] + yy * bpitches[ii] + xx * PIXEL_STRIDE;
      for (int jj = 0; jj < N_CHANNELS; ++jj)
        ++H[offset + jj * 256 + ptr[jj]];
    }

  // for each histogram, compute cumulative sum
  uint8_t* outptr = (uint8_t*)out + yy * opitch + xx * PIXEL_STRIDE;
  for (int ii = 0; ii < N_CHANNELS; ++ii)
    {
      uint8_t cumsum = 0;
      int jj = 0;
      for (; jj < 256 && cumsum < buffers_amount / 2 + 1; ++jj)
        cumsum += H[offset + ii * 256 + jj];
      outptr[ii] = (uint8_t)(jj - 1);
    }
#    else
#    endif
#  endif
#endif

  // #undef _BACKGROUND_ESTIMATION_MEDIAN_ISORT
  // #undef _BACKGROUND_ESTIMATION_MEDIAN_LOCAL_HIST
  // #undef _BACKGROUND_ESTIMATION_MEDIAN_SHARED_HIST
}

#undef _BE_FSIGN

//******************************************************
//**                                                  **
//**           Conversion from RGB to LAB (GPU)       **
//**                                                  **
//******************************************************

// Declare all constant variable outsite the kernel for better access time
__constant__ float D65_XYZ[9] = {0.412453f, 0.357580f, 0.180423f,
                                 0.212671f, 0.715160f, 0.072169f,
                                 0.019334f, 0.119193f, 0.950227f};

__constant__ float D65_XN = 0.95047f;
__constant__ float D65_YN = 1.00000f;
__constant__ float D65_ZN = 1.08883f;

__constant__ float EPSILON = 0.008856f;
__constant__ float KAPPA = 903.3f;

__device__ void
rgbToLab(float r, float g, float b, float& l_, float& a_, float& b_)
{
  r = r / 255.0f;
  g = g / 255.0f;
  b = b / 255.0f;

#define GAMMA_CORRECT(C)                                                       \
  ((C) > 0.04045f ? __powf(((C) + 0.055f) / 1.055f, 2.4f) : (C) / 12.92f)
  r = GAMMA_CORRECT(r);
  g = GAMMA_CORRECT(g);
  b = GAMMA_CORRECT(b);
#undef GAMMA_CORRECT

  r = (r * D65_XYZ[0] + g * D65_XYZ[1] + b * D65_XYZ[2]) / D65_XN;
  g = (r * D65_XYZ[3] + g * D65_XYZ[4] + b * D65_XYZ[5]) / D65_YN;
  b = (r * D65_XYZ[6] + g * D65_XYZ[7] + b * D65_XYZ[8]) / D65_ZN;

#define NONLINEAR(C)                                                           \
  ((C) > EPSILON ? __powf((C), 1.0f / 3.0f) : ((KAPPA * (C) + 16.0f) / 116.0f))
  float fx = NONLINEAR(r);
  float fy = NONLINEAR(g);
  float fz = NONLINEAR(b);
#undef NONLINEAR

  l_ = (116.0f * fy) - 16.0f;
  a_ = 500.0f * (fx - fy);
  b_ = 200.0f * (fy - fz);
}

__global__ void rgbToLabDistanceKernel(std::byte* referenceBuffer,
                                       size_t rpitch,
                                       std::byte* buffer,
                                       size_t bpitch,
                                       const int width,
                                       const int height)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  float LRef, ARef, BRef;
  rgb ref_pixel = ((rgb*)(referenceBuffer + y * rpitch))[x];
  rgbToLab(ref_pixel.r, ref_pixel.g, ref_pixel.b, LRef, ARef, BRef);

  float L, A, B;
  rgb buf_pixel = ((rgb*)(buffer + y * bpitch))[x];
  rgbToLab(buf_pixel.r, buf_pixel.g, buf_pixel.b, L, A, B);

  float distance = sqrtf((L - LRef) * (L - LRef) + (A - ARef) * (A - ARef)
                         + (B - BRef) * (B - BRef));
  uint8_t distance8bit =
    static_cast<uint8_t>(fminf(distance / MAX_LAB_DISTANCE * 255.0f, 255.0f));

  ((rgb*)(buffer + y * bpitch))[x] = {distance8bit, distance8bit, distance8bit};
}

//******************************************************
//**                                                  **
//**             Morphological Opening                **
//**                                                  **
//******************************************************

__global__ void morphological_erosion(std::byte* buffer,
                                      size_t bpitch,
                                      std::byte* output_buffer,
                                      size_t opitch,
                                      const int width,
                                      const int height)
{
  int yy = blockIdx.y * blockDim.y + threadIdx.y;
  int xx = blockIdx.x * blockDim.x + threadIdx.x;

  if (xx >= width || yy >= height)
    return;
  unsigned int res = 0xffffffff;

  // Compute the minimum value in the 5x5 neighborhood
  for (int j = yy - 2; j <= yy + 2; j++)
    {
      for (int i = xx - 2; i <= xx + 2; i++)
        {
          if (i >= 0 && i < width && j >= 0 && j < height)
            {
              res = __vminu4(
                res, *((unsigned int*)&buffer[j * bpitch + i * N_CHANNELS]));
            }
        }
    }

  // Compute the minimum value in the extremities
  if (xx - 3 >= 0)
    {
      res = __vminu4(
        res, *((unsigned int*)&buffer[yy * bpitch + (xx - 3) * N_CHANNELS]));
    }
  if (xx + 3 < width)
    {
      res = __vminu4(
        res, *((unsigned int*)&buffer[yy * bpitch + (xx + 3) * N_CHANNELS]));
    }
  if (yy - 3 >= 0)
    {
      res = __vminu4(
        res, *((unsigned int*)&buffer[(yy - 3) * bpitch + xx * N_CHANNELS]));
    }
  if (yy + 3 < height)
    {
      res = __vminu4(
        res, *((unsigned int*)&buffer[(yy + 3) * bpitch + xx * N_CHANNELS]));
    }

  *((unsigned int*)&buffer[yy * opitch + xx * N_CHANNELS]) = res;
}

__global__ void morphological_dilation(std::byte* buffer,
                                       size_t bpitch,
                                       std::byte* output_buffer,
                                       size_t opitch,
                                       const int width,
                                       const int height)
{
  int yy = blockIdx.y * blockDim.y + threadIdx.y;
  int xx = blockIdx.x * blockDim.x + threadIdx.x;

  if (xx >= width || yy >= height)
    return;
  unsigned int res = 0x00000000;

  // Compute the minimum value in the 5x5 neighborhood
  for (int j = yy - 2; j <= yy + 2; j++)
    {
      for (int i = xx - 2; i <= xx + 2; i++)
        {
          if (i >= 0 && i < width && j >= 0 && j < height)
            {
              res = __vmaxu4(
                res, *((unsigned int*)&buffer[j * bpitch + i * N_CHANNELS]));
            }
        }
    }

  // Compute the minimum value in the extremities
  if (xx - 3 >= 0)
    {
      res = __vmaxu4(
        res, *((unsigned int*)&buffer[yy * bpitch + (xx - 3) * N_CHANNELS]));
    }
  if (xx + 3 < width)
    {
      res = __vmaxu4(
        res, *((unsigned int*)&buffer[yy * bpitch + (xx + 3) * N_CHANNELS]));
    }
  if (yy - 3 >= 0)
    {
      res = __vmaxu4(
        res, *((unsigned int*)&buffer[(yy - 3) * bpitch + xx * N_CHANNELS]));
    }
  if (yy + 3 < height)
    {
      res = __vmaxu4(
        res, *((unsigned int*)&buffer[(yy + 3) * bpitch + xx * N_CHANNELS]));
    }

  *((unsigned int*)&buffer[yy * opitch + xx * N_CHANNELS]) = res;
}

//******************************************************
//**                                                  **
//**               Hysteresis Threshold               **
//**                                                  **
//******************************************************

/// @brief Apply a threshold on the buffer and store the result in the marker
/// @param buffer The input buffer
/// @param bpitch The pitch of the input buffer
/// @param marker The marker buffer
/// @param mpitch The pitch of the marker buffer
/// @param width The width of the image
/// @param height The height of the image
/// @param high_threshold The high threshold
/// @return
__global__ void apply_threshold_on_marker(std::byte* buffer,
                                          size_t bpitch,
                                          std::byte* marker,
                                          size_t mpitch,
                                          const int width,
                                          const int height,
                                          int high_threshold)
{
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;

  if (x >= width || y >= height)
    return;

  rgb* buffer_line = (rgb*)(buffer + y * bpitch);

  marker[y * mpitch + x] =
    buffer_line[x].r > high_threshold ? std::byte{1} : std::byte{0};
}

__global__ void hysteresis_opti(std::byte* buffer,
                                size_t bpitch,
                                std::byte* marker,
                                size_t mpitch,
                                const int width,
                                const int height,
                                int low_th,
                                bool end)
{
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ unsigned char changed;

  rgb* buf_ptr = (rgb*)(buffer + y * bpitch);
  do
    {
      __syncthreads();
      changed = 0;
      __syncthreads();

      if ((x < width && y < height)
          && (marker[y * mpitch + x] != std::byte{0} && (buf_ptr[x].r != 255)))
        {
          buf_ptr[x].r = 255;
          changed = 1;

// Check 8-neighbors
#define C8N(cond, x, y)                                                        \
  if ((cond) && ((rgb*)(buffer + (y)*bpitch))[(x)].r > low_th)                 \
    {                                                                          \
      marker[(y) * (mpitch) + (x)] = std::byte{1};                             \
    }

          C8N(x > 0 && y > 0, x - 1, y - 1)
          C8N(y > 0, x, y - 1)
          C8N(x < width - 1 && y > 0, x + 1, y - 1)
          C8N(x < width - 1, x + 1, y)
          C8N(x < width - 1 && y < height - 1, x + 1, y + 1)
          C8N(y < height - 1, x, y + 1)
          C8N(x > 0 && y < height - 1, x - 1, y + 1)
          C8N(x > 0, x - 1, y)

#undef C8N
        }
      __syncthreads();
  } while (changed);

  // set all non max values to 0
  if (end && x < width && y < height && buf_ptr[x].r != 255)
    {
      buf_ptr[x].r = 0;
    }
}

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

__global__ void copy_buffer_kernel(std::byte* dbuffer,
                                   size_t bpitch,
                                   std::byte* cpy_buffer,
                                   size_t cpitch,
                                   std::byte* mask,
                                   size_t mpitch,
                                   std::byte* fallback_dbuffer,
                                   size_t fpitch,
                                   const int width,
                                   const int height)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  rgb* lineptr = (rgb*)(dbuffer + y * bpitch);
  rgb* cpy_lineptr = (rgb*)(cpy_buffer + y * cpitch);
  rgb* fallback_lineptr = (rgb*)(fallback_dbuffer + y * fpitch);
  rgb* mask_lineptr = (rgb*)(mask + y * mpitch);

  if (mask_lineptr[x].r == 0)
    {
      cpy_lineptr[x].r = lineptr[x].r;
      cpy_lineptr[x].g = lineptr[x].g;
      cpy_lineptr[x].b = lineptr[x].b;
    }
  else
    {
      cpy_lineptr[x].r = fallback_lineptr[x].r;
      cpy_lineptr[x].g = fallback_lineptr[x].g;
      cpy_lineptr[x].b = fallback_lineptr[x].b;
    }
}

namespace
{
  //******************************************************
  //**                                                  **
  //**           Conversion from RGB to LAB (GPU)       **
  //**                                                  **
  //******************************************************

  void rgb_to_lab_cuda(std::byte* referenceBuffer,
                       size_t rpitch,
                       std::byte* buffer,
                       size_t bpitch,
                       const frame_info* buffer_info)
  {
    int width = buffer_info->width;
    int height = buffer_info->height;

    cudaError_t err;

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    rgbToLabDistanceKernel<<<gridSize, blockSize>>>(
      referenceBuffer, rpitch, buffer, bpitch, width, height);

    err = cudaDeviceSynchronize();
    CHECK_CUDA_ERROR(err);
  }

  //******************************************************
  //**                                                  **
  //**             Morphological Opening                **
  //**                                                  **
  //******************************************************

  void opening_impl_inplace(std::byte* buffer,
                            size_t bpitch,
                            const frame_info* buffer_info)
  {
    int width = buffer_info->width;
    int height = buffer_info->height;

    std::byte* gpu_image;
    size_t gpu_pitch;
    cudaError_t err =
      cudaMallocPitch(&gpu_image, &gpu_pitch, width * N_CHANNELS, height);
    CHECK_CUDA_ERROR(err);

    err = cudaMemcpy2D(gpu_image, gpu_pitch, buffer, bpitch, width * N_CHANNELS,
                       height, cudaMemcpyDeviceToHost);
    CHECK_CUDA_ERROR(err);

    std::byte* gpu_intermediate_image;
    size_t gpu_intermediate_pitch;
    err = cudaMallocPitch(&gpu_intermediate_image, &gpu_intermediate_pitch,
                          width * N_CHANNELS, height);
    CHECK_CUDA_ERROR(err);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + (blockSize.x - 1)) / blockSize.x,
                  (height + (blockSize.y - 1)) / blockSize.y);

    morphological_erosion<<<gridSize, blockSize>>>(
      gpu_image, gpu_pitch, gpu_intermediate_image, gpu_intermediate_pitch,
      width, height);
    err = cudaDeviceSynchronize();
    CHECK_CUDA_ERROR(err);

    morphological_dilation<<<gridSize, blockSize>>>(
      gpu_intermediate_image, gpu_intermediate_pitch, gpu_image, gpu_pitch,
      width, height);
    err = cudaDeviceSynchronize();
    CHECK_CUDA_ERROR(err);

    err = cudaMemcpy2D(buffer, bpitch, gpu_image, gpu_pitch, width * N_CHANNELS,
                       height, cudaMemcpyHostToDevice);
    CHECK_CUDA_ERROR(err);

    cudaFree(gpu_image);
    cudaFree(gpu_intermediate_image);
  }

  //******************************************************
  //**                                                  **
  //**               Hysteresis Threshold               **
  //**                                                  **
  //******************************************************

  /// @brief Apply hysteresis thresholding on the buffer
  /// @param buffer The input buffer
  /// @param bpitch The pitch of the input buffer
  /// @param buffer_info The buffer info
  /// @param low_threshold The low threshold
  /// @param high_threshold The high threshold
  void apply_hysteresis_threshold(std::byte** buffer,
                                  size_t* bpitch,
                                  const frame_info* buffer_info,
                                  int low_th,
                                  int high_th)
  {
    int width = buffer_info->width;
    int height = buffer_info->height;

    // Ensure low threshold is less than high threshold
    if (low_th > high_th)
      {
        low_th = high_th;
      }

    // Create a marker buffer to store the pixels that are above the high threshold
    std::byte* marker;
    size_t mpitch;
    cudaError_t err;
    err = cudaMallocPitch(&marker, &mpitch, width, height);
    CHECK_CUDA_ERROR(err);

    dim3 blockSize(2 * BLOCK_SIZE, 2 * BLOCK_SIZE);
    dim3 gridSize((width + (blockSize.x - 1)) / blockSize.x,
                  (height + (blockSize.y - 1)) / blockSize.y);
    apply_threshold_on_marker<<<gridSize, blockSize>>>(
      *buffer, *bpitch, marker, mpitch, width, height, high_th);

    err = cudaDeviceSynchronize();
    CHECK_CUDA_ERROR(err);

    // Apply hysteresis thresholding
    for (int i = 0; i < HYSTERESIS_ITER; i++)
      {
        hysteresis_opti<<<gridSize, blockSize>>>(*buffer, *bpitch, marker,
                                                 mpitch, width, height, low_th,
                                                 i == (HYSTERESIS_ITER - 1));
        err = cudaDeviceSynchronize();
        CHECK_CUDA_ERROR(err);
      }

    cudaFree(marker);
  }

  //******************************************************
  //**                                                  **
  //**             Background Model Update              **
  //**                                                  **
  //******************************************************

  void copy_buffer(std::byte* dbuffer,
                   size_t bpitch,
                   std::byte** cpy_dbuffer,
                   size_t* cpitch,
                   std::byte* dmask,
                   size_t mpitch,
                   std::byte* fallback_dbuffer,
                   size_t fpitch,
                   const frame_info* buffer_info)
  {
    int width = buffer_info->width;
    int height = buffer_info->height;

    cudaError_t err;

    if (*cpy_dbuffer == nullptr)
      {
        // Allocate memory for the copy buffer
        err = cudaMallocPitch(cpy_dbuffer, cpitch, width * N_CHANNELS, height);
        CHECK_CUDA_ERROR(err);
      }

    if (dmask == nullptr || fallback_dbuffer == nullptr)
      {
        // Copy dbuffer to cpy_buffer
        err =
          cudaMemcpy2D(*cpy_dbuffer, *cpitch, dbuffer, bpitch,
                       width * N_CHANNELS, height, cudaMemcpyDeviceToDevice);
        CHECK_CUDA_ERROR(err);
      }
    else
      {
        // Copy dbuffer where mask is false and fallback_dbuffer where mask is true
        dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                      (height + blockSize.y - 1) / blockSize.y);
        copy_buffer_kernel<<<gridSize, blockSize>>>(
          dbuffer, bpitch, *cpy_dbuffer, *cpitch, dmask, mpitch,
          fallback_dbuffer, fpitch, width, height);

        err = cudaDeviceSynchronize();
        CHECK_CUDA_ERROR(err);
      }
  }

  void update_bg_model(std::byte* dbuffer,
                       size_t bpitch,
                       std::byte** bg_model,
                       size_t* bg_pitch,
                       std::byte* dmask,
                       size_t mpitch,
                       const frame_info* buffer_info,
                       const filter_params* params,
                       bool is_median)
  {
    static std::byte** dbuffer_samples = nullptr;
    static size_t* pitches = nullptr;
    static int dbuffers_amount = 0;
    static double last_timestamp = 0.0;

    int height = buffer_info->height;
    int width = buffer_info->width;

    // First frame is set to the background model
    if (dbuffers_amount == 0)
      {
        // Use unified memory to store the buffer samples
        cudaError_t err;
        err = cudaMallocManaged(&dbuffer_samples,
                                params->bg_number_frames * sizeof(std::byte*));
        CHECK_CUDA_ERROR(err);
        err = cudaMallocManaged(&pitches,
                                params->bg_number_frames * sizeof(size_t));
        CHECK_CUDA_ERROR(err);

        // Copy buffer
        std::byte* cpy_buffer = nullptr;
        size_t cpy_pitch;
        copy_buffer(dbuffer, bpitch, &cpy_buffer, &cpy_pitch, nullptr, 0,
                    nullptr, 0, buffer_info);

        dbuffer_samples[0] = cpy_buffer;
        pitches[0] = cpy_pitch;
        dbuffers_amount = 1;
        last_timestamp = buffer_info->timestamp;

        // First bg_model is set to the buffer pointer
        // so we set it to null to reallocate new memory after
        *bg_model = nullptr;
      }
    else if (buffer_info->timestamp - last_timestamp
             >= params->bg_sampling_rate)
      {
        if (dbuffers_amount < params->bg_number_frames)
          {
            // Copy buffer and apply mask to remove foreground
            std::byte* cpy_buffer = nullptr;
            size_t cpy_pitch;
            copy_buffer(dbuffer, bpitch, &cpy_buffer, &cpy_pitch, dmask, mpitch,
                        *bg_model, *bg_pitch, buffer_info);

            dbuffer_samples[dbuffers_amount] = cpy_buffer;
            pitches[dbuffers_amount] = cpy_pitch;
            dbuffers_amount += 1;
            last_timestamp = buffer_info->timestamp;
          }
        else
          {
            // Copy buffer on the oldest frame w/o reallocating
            // And apply mask to remove foreground
            std::byte* cpy_buffer = dbuffer_samples[0];
            size_t cpy_pitch = pitches[0];
            copy_buffer(dbuffer, bpitch, &cpy_buffer, &cpy_pitch, dmask, mpitch,
                        *bg_model, *bg_pitch, buffer_info);

            // Shift frame samples
            for (int i = 0; i < params->bg_number_frames - 1; ++i)
              {
                dbuffer_samples[i] = dbuffer_samples[i + 1];
                pitches[i] = pitches[i + 1];
              }

            dbuffer_samples[params->bg_number_frames - 1] = cpy_buffer;
            pitches[params->bg_number_frames - 1] = cpy_pitch;
            last_timestamp = buffer_info->timestamp;
          }
      }
    else
      {
        return;
      }

    // Allocate device memory for background model if not already allocated
    if (*bg_model == nullptr)
      {
        cudaError_t err;
        err = cudaMallocPitch(bg_model, bg_pitch, width * N_CHANNELS, height);
        CHECK_CUDA_ERROR(err);
      }

    size_t _BE_BLOCK_SIZE = BLOCK_SIZE;
#ifdef _BACKGROUND_ESTIMATION_MEDIAN_SHARED_HIST
    _BE_BLOCK_SIZE = is_median ? (N_CHANNELS > 3 ? 6 : 8) : BLOCK_SIZE;
#endif
    dim3 blockSize(_BE_BLOCK_SIZE, _BE_BLOCK_SIZE);
    dim3 gridSize((width + (blockSize.x - 1)) / blockSize.x,
                  (height + (blockSize.y - 1)) / blockSize.y);

// Estimate background
#define _BE_FPARAMS                                                            \
  dbuffer_samples, pitches, dbuffers_amount, *bg_model, *bg_pitch, width, height

    if (is_median)
      {
#ifdef _BACKGROUND_ESTIMATION_MEDIAN_SHARED_HIST
#  define _SHARED_MEM_SIZE                                                     \
    _BE_BLOCK_SIZE* _BE_BLOCK_SIZE* N_CHANNELS * 256 * sizeof(uint8_t)
        estimate_background_median<<<gridSize, blockSize, _SHARED_MEM_SIZE>>>(
          _BE_FPARAMS);
#  undef _SHARED_MEM_SIZE
#else
        estimate_background_median<<<gridSize, blockSize>>>(_BE_FPARAMS);
#endif
      }
    else
      estimate_background_mean<<<gridSize, blockSize>>>(_BE_FPARAMS);

#undef _BE_FPARAMS

    cudaError_t err = cudaDeviceSynchronize();
    CHECK_CUDA_ERROR(err);
  }
} // namespace

extern "C"
{
  void filter_impl(uint8_t* src_buffer,
                   const frame_info* buffer_info,
                   const filter_params* params)
  {
    int width = buffer_info->width;
    int height = buffer_info->height;
    int src_stride = buffer_info->stride;

    std::byte *dmask, *dbuffer, *intermediate_buffer;
    size_t mpitch, bpitch, ipitch;

    cudaError_t err;

    // Allocate memory on the device
    err = cudaMallocPitch(&intermediate_buffer, &ipitch,
                          width * buffer_info->pixel_stride, height);
    CHECK_CUDA_ERROR(err);
    err = cudaMallocPitch(&dmask, &mpitch, width * N_CHANNELS, height);
    CHECK_CUDA_ERROR(err);
    err = cudaMallocPitch(&dbuffer, &bpitch, width * N_CHANNELS, height);
    CHECK_CUDA_ERROR(err);

    // Copy the input buffer to the device
    err = cudaMemcpy2D(intermediate_buffer, ipitch, src_buffer, src_stride,
                       width * buffer_info->pixel_stride, height,
                       cudaMemcpyHostToDevice);
    CHECK_CUDA_ERROR(err);

    // Set thread block and grid dimensions
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    resize_pixels<<<gridSize, blockSize>>>(
      intermediate_buffer, buffer_info->pixel_stride, ipitch, dmask, N_CHANNELS,
      mpitch, width, height);

    resize_pixels<<<gridSize, blockSize>>>(
      intermediate_buffer, buffer_info->pixel_stride, ipitch, dbuffer,
      N_CHANNELS, bpitch, width, height);

    err = cudaDeviceSynchronize();
    CHECK_CUDA_ERROR(err);

    // Set first frame as bg model or set it to user provided bg
    static std::byte* bg_buffer = params->bg != nullptr ? nullptr : dbuffer;
    static size_t bg_pitch = params->bg != nullptr ? 0 : bpitch;

    if (bg_buffer == nullptr && params->bg != nullptr)
      {
        err =
          cudaMallocPitch(&bg_buffer, &bg_pitch, width * N_CHANNELS, height);
        CHECK_CUDA_ERROR(err);
        err = cudaMemcpy2D(intermediate_buffer, ipitch, params->bg, src_stride,
                           width * buffer_info->pixel_stride, height,
                           cudaMemcpyDefault);
        CHECK_CUDA_ERROR(err);
        resize_pixels<<<gridSize, blockSize>>>(
          intermediate_buffer, buffer_info->pixel_stride, ipitch, bg_buffer,
          N_CHANNELS, bg_pitch, width, height);

        err = cudaDeviceSynchronize();
        CHECK_CUDA_ERROR(err);
      }

    // Convert RGB to LAB
    rgb_to_lab_cuda(bg_buffer, bg_pitch, dmask, mpitch, buffer_info);

    // Apply morphological opening
    opening_impl_inplace(dmask, mpitch, buffer_info);

    // Apply hysteresis thresholding
    apply_hysteresis_threshold(&dmask, &mpitch, buffer_info, params->th_low,
                               params->th_high);

    // Apply masking
    apply_masking<<<gridSize, blockSize>>>(dbuffer, bpitch, dmask, mpitch,
                                           width, height);
    err = cudaDeviceSynchronize();
    CHECK_CUDA_ERROR(err);

    // Update background model if no user background is provided
    if (params->bg == nullptr)
      {
        update_bg_model(dbuffer, bpitch, &bg_buffer, &bg_pitch, dmask, mpitch,
                        buffer_info, params, true);
      }

    resize_pixels<<<gridSize, blockSize>>>(
      dbuffer, N_CHANNELS, bpitch, intermediate_buffer,
      buffer_info->pixel_stride, ipitch, width, height);

    err = cudaDeviceSynchronize();
    CHECK_CUDA_ERROR(err);

    // Copy the result back to the host
    err = cudaMemcpy2D(src_buffer, src_stride, intermediate_buffer, ipitch,
                       width * buffer_info->pixel_stride, height,
                       cudaMemcpyDeviceToHost);
    CHECK_CUDA_ERROR(err);

    cudaFree(intermediate_buffer);
    cudaFree(dmask);
    cudaFree(dbuffer);
  }
}
