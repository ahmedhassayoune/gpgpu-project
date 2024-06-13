#include "filter_impl.h"

#include <cassert>
#include <chrono>
#include <cstdio>
#include <thread>

#define BLOCK_SIZE 16

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

template <typename T>
__device__ inline T* eltPtr(T* baseAddress, int col, int row, size_t pitch)
{
  return (T*)((char*)baseAddress + row * pitch + col * sizeof(T)); // FIXME
}

__device__ bool hysteresis_has_changed;

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
//**           Conversion from RGB to LAB (GPU)       **
//**                                                  **
//******************************************************

__device__ void
rgbToXyz(float r, float g, float b, float& x, float& y, float& z)
{
  const float D65_XYZ[9] = {0.412453f, 0.357580f, 0.180423f,
                            0.212671f, 0.715160f, 0.072169f,
                            0.019334f, 0.119193f, 0.950227f};

  r = r / 255.0f;
  g = g / 255.0f;
  b = b / 255.0f;

#define GAMMA_CORRECT(C)                                                       \
  ((C) > 0.04045f ? powf(((C) + 0.055f) / 1.055f, 2.4f) : (C) / 12.92f)
  r = GAMMA_CORRECT(r);
  g = GAMMA_CORRECT(g);
  b = GAMMA_CORRECT(b);
#undef GAMMA_CORRECT

  x = r * D65_XYZ[0] + g * D65_XYZ[1] + b * D65_XYZ[2];
  y = r * D65_XYZ[3] + g * D65_XYZ[4] + b * D65_XYZ[5];
  z = r * D65_XYZ[6] + g * D65_XYZ[7] + b * D65_XYZ[8];
}

__device__ void
xyzToLab(float x, float y, float z, float& l, float& a, float& b)
{
  const float D65_Xn = 0.95047f;
  const float D65_Yn = 1.00000f;
  const float D65_Zn = 1.08883f;

  x /= D65_Xn;
  y /= D65_Yn;
  z /= D65_Zn;

  const float epsilon = 0.008856f;
  const float kappa = 903.3f;

#define NONLINEAR(C)                                                           \
  ((C) > epsilon ? powf((C), 1.0f / 3.0f) : ((kappa * (C) + 16.0f) / 116.0f))
  float fx = NONLINEAR(x);
  float fy = NONLINEAR(y);
  float fz = NONLINEAR(z);
#undef NONLINEAR

  l = (116.0f * fy) - 16.0f;
  a = 500.0f * (fx - fy);
  b = 200.0f * (fy - fz);
}

__global__ void rgbToLabDistanceKernel(std::byte* referenceBuffer,
                                       size_t rpitch,
                                       std::byte* buffer,
                                       size_t bpitch,
                                       float* distanceArray,
                                       size_t dpitch,
                                       const int width,
                                       const int height)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  rgb* lineptrReference = (rgb*)(referenceBuffer + y * rpitch);
  float rRef = lineptrReference[x].r;
  float gRef = lineptrReference[x].g;
  float bRef = lineptrReference[x].b;

  float XRef, YRef, ZRef;
  rgbToXyz(rRef, gRef, bRef, XRef, YRef, ZRef);

  float LRef, ARef, BRef;
  xyzToLab(XRef, YRef, ZRef, LRef, ARef, BRef);

  LAB referenceLab = {LRef, ARef, BRef};

  rgb* lineptr = (rgb*)(buffer + y * bpitch);
  float r = lineptr[x].r;
  float g = lineptr[x].g;
  float b = lineptr[x].b;

  float X, Y, Z;
  rgbToXyz(r, g, b, X, Y, Z);

  float L, A, B;
  xyzToLab(X, Y, Z, L, A, B);

  LAB currentLab = {L, A, B};
#define LAB_DISTANCE(lab1, lab2)                                               \
  (sqrtf(powf((lab1).l - (lab2).l, 2) + powf((lab1).a - (lab2).a, 2)           \
         + powf((lab1).b - (lab2).b, 2)))
  float distance = LAB_DISTANCE(currentLab, referenceLab);
#undef LAB_DISTANCE
  float* distancePtr = (float*)((char*)distanceArray + y * dpitch);
  distancePtr[x] = distance;
}

__global__ void normalizeAndConvertTo8bitKernel(std::byte* buffer,
                                                size_t bpitch,
                                                float* distanceArray,
                                                size_t dpitch,
                                                float max_distance,
                                                const int width,
                                                const int height)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  float* distancePtr = (float*)((char*)distanceArray + y * dpitch);
  float distance = distancePtr[x];

  uint8_t distance8bit =
    static_cast<uint8_t>(fminf(distance / max_distance * 255.0f, 255.0f));

  rgb* lineptr = (rgb*)(buffer + y * bpitch);
  lineptr[x].r = distance8bit;
  lineptr[x].g = distance8bit;
  lineptr[x].b = distance8bit;
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
  unsigned int min_red = 0xff;
  unsigned int min_green = 0xff;
  unsigned int min_blue = 0xff;

  // Compute the minimum value in the 5x5 neighborhood
  for (int j = yy - 2; j <= yy + 2; j++)
    {
      for (int i = xx - 2; i <= xx + 2; i++)
        {
          if (i >= 0 && i < width && j >= 0 && j < height)
            {
              min_red =
                min(min_red, (unsigned int)buffer[j * bpitch + i * N_CHANNELS]);
              min_green =
                min(min_green,
                    (unsigned int)buffer[j * bpitch + i * N_CHANNELS + 1]);
              min_blue =
                min(min_blue,
                    (unsigned int)buffer[j * bpitch + i * N_CHANNELS + 2]);
            }
        }
    }

  // Compute the minimum value in the extremities
  if (xx - 3 >= 0)
    {
      int i = xx - 3;
      min_red =
        min(min_red, (unsigned int)buffer[yy * bpitch + i * N_CHANNELS]);
      min_green =
        min(min_green, (unsigned int)buffer[yy * bpitch + i * N_CHANNELS + 1]);
      min_blue =
        min(min_blue, (unsigned int)buffer[yy * bpitch + i * N_CHANNELS + 2]);
    }
  if (xx + 3 < width)
    {
      int i = xx + 3;
      min_red =
        min(min_red, (unsigned int)buffer[yy * bpitch + i * N_CHANNELS]);
      min_green =
        min(min_green, (unsigned int)buffer[yy * bpitch + i * N_CHANNELS + 1]);
      min_blue =
        min(min_blue, (unsigned int)buffer[yy * bpitch + i * N_CHANNELS + 2]);
    }
  if (yy - 3 >= 0)
    {
      int j = yy - 3;
      min_red =
        min(min_red, (unsigned int)buffer[j * bpitch + xx * N_CHANNELS]);
      min_green =
        min(min_green, (unsigned int)buffer[j * bpitch + xx * N_CHANNELS + 1]);
      min_blue =
        min(min_blue, (unsigned int)buffer[j * bpitch + xx * N_CHANNELS + 2]);
    }
  if (yy + 3 < height)
    {
      int j = yy + 3;
      min_red =
        min(min_red, (unsigned int)buffer[j * bpitch + xx * N_CHANNELS]);
      min_green =
        min(min_green, (unsigned int)buffer[j * bpitch + xx * N_CHANNELS + 1]);
      min_blue =
        min(min_blue, (unsigned int)buffer[j * bpitch + xx * N_CHANNELS + 2]);
    }

  size_t out_index = yy * opitch + xx * N_CHANNELS;
  output_buffer[out_index] = (std::byte)min_red;
  output_buffer[out_index + 1] = (std::byte)min_green;
  output_buffer[out_index + 2] = (std::byte)min_blue;
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

  unsigned int max_red = 0;
  unsigned int max_green = 0;
  unsigned int max_blue = 0;

  // Compute the maximum value in the 5x5 neighborhood
  for (int j = yy - 2; j <= yy + 2; j++)
    {
      for (int i = xx - 2; i <= xx + 2; i++)
        {
          if (i >= 0 && i < width && j >= 0 && j < height)
            {
              max_red =
                max(max_red, (unsigned int)buffer[j * bpitch + i * N_CHANNELS]);
              max_green =
                max(max_green,
                    (unsigned int)buffer[j * bpitch + i * N_CHANNELS + 1]);
              max_blue =
                max(max_blue,
                    (unsigned int)buffer[j * bpitch + i * N_CHANNELS + 2]);
            }
        }
    }

  // Compute the maximum value in the extremities
  if (xx - 3 >= 0)
    {
      int i = xx - 3;
      max_red =
        max(max_red, (unsigned int)buffer[yy * bpitch + i * N_CHANNELS]);
      max_green =
        max(max_green, (unsigned int)buffer[yy * bpitch + i * N_CHANNELS + 1]);
      max_blue =
        max(max_blue, (unsigned int)buffer[yy * bpitch + i * N_CHANNELS + 2]);
    }
  if (xx + 3 < width)
    {
      int i = xx + 3;
      max_red =
        max(max_red, (unsigned int)buffer[yy * bpitch + i * N_CHANNELS]);
      max_green =
        max(max_green, (unsigned int)buffer[yy * bpitch + i * N_CHANNELS + 1]);
      max_blue =
        max(max_blue, (unsigned int)buffer[yy * bpitch + i * N_CHANNELS + 2]);
    }
  if (yy - 3 >= 0)
    {
      int j = yy - 3;
      max_red =
        max(max_red, (unsigned int)buffer[j * bpitch + xx * N_CHANNELS]);
      max_green =
        max(max_green, (unsigned int)buffer[j * bpitch + xx * N_CHANNELS + 1]);
      max_blue =
        max(max_blue, (unsigned int)buffer[j * bpitch + xx * N_CHANNELS + 2]);
    }
  if (yy + 3 < height)
    {
      int j = yy + 3;
      max_red =
        max(max_red, (unsigned int)buffer[j * bpitch + xx * N_CHANNELS]);
      max_green =
        max(max_green, (unsigned int)buffer[j * bpitch + xx * N_CHANNELS + 1]);
      max_blue =
        max(max_blue, (unsigned int)buffer[j * bpitch + xx * N_CHANNELS + 2]);
    }

  int out_index = yy * opitch + xx * N_CHANNELS;
  output_buffer[out_index] = (std::byte)max_red;
  output_buffer[out_index + 1] = (std::byte)max_green;
  output_buffer[out_index + 2] = (std::byte)max_blue;
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
                                          bool* marker,
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
  bool* marker_line = (bool*)((std::byte*)marker + y * mpitch);

  marker_line[x] = buffer_line[x].r > high_threshold;
}

/// @brief Reconstruct the hysteresis thresholding image from the marker
/// @param buffer The input buffer
/// @param bpitch The pitch of the input buffer
/// @param out The output buffer
/// @param opitch The pitch of the output buffer
/// @param marker The marker buffer
/// @param mpitch The pitch of the marker buffer
/// @param buffer_info The buffer info
/// @param low_threshold The low threshold
/// @return
__global__ void reconstruct_image(std::byte* buffer,
                                  size_t bpitch,
                                  std::byte* out,
                                  size_t opitch,
                                  bool* marker,
                                  size_t mpitch,
                                  const int width,
                                  const int height,
                                  int low_threshold)
{
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;

  if (x >= width || y >= height)
    return;

  rgb* out_line = (rgb*)(out + y * opitch);
  bool* marker_line = (bool*)((std::byte*)marker + y * mpitch);

  if (!marker_line[x] || out_line[x].r != 0)
    {
      return;
    }

  // Set the pixel to white
  out_line[x].r = 255;
  out_line[x].g = 255;
  out_line[x].b = 255;

  // Mark the 8-connected neighbors if they are above the low threshold
  for (int i = -1; i <= 1; i++)
    {
      for (int j = -1; j <= 1; j++)
        {
          int ny = y + j;
          int nx = x + i;
          // Skip the current pixel
          if ((i == 0 && j == 0) || nx < 0 || nx >= width || ny < 0
              || ny >= height)
            {
              continue;
            }

          // Check if the pixel is within the image boundaries
          rgb* buffer_line = (rgb*)(buffer + ny * bpitch);
          bool* neighbor_marker_line =
            (bool*)((std::byte*)marker + ny * mpitch);
          if (!neighbor_marker_line[nx] && buffer_line[nx].r > low_threshold)
            {
              neighbor_marker_line[nx] = true;
              hysteresis_has_changed = true;
            }
        }
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
    float* distanceArray;
    size_t dpitch;

    err =
      cudaMallocPitch(&distanceArray, &dpitch, width * sizeof(float), height);
    CHECK_CUDA_ERROR(err);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    rgbToLabDistanceKernel<<<gridSize, blockSize>>>(
      referenceBuffer, rpitch, buffer, bpitch, distanceArray, dpitch, width,
      height);
    err = cudaDeviceSynchronize();
    CHECK_CUDA_ERROR(err);

    float* h_distanceArray = new float[width * height];
    err = cudaMemcpy2D(h_distanceArray, width * sizeof(float), distanceArray,
                       dpitch, width * sizeof(float), height,
                       cudaMemcpyDeviceToHost);
    CHECK_CUDA_ERROR(err);

    float maxDistance = 0.0f;
    for (int i = 0; i < width * height; ++i)
      {
        maxDistance = fmaxf(maxDistance, h_distanceArray[i]);
      }
    delete[] h_distanceArray;

    normalizeAndConvertTo8bitKernel<<<gridSize, blockSize>>>(
      buffer, bpitch, distanceArray, dpitch, maxDistance, width, height);
    err = cudaDeviceSynchronize();
    CHECK_CUDA_ERROR(err);

    err = cudaFree(distanceArray);
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
  void apply_hysteresis_threshold(std::byte* buffer,
                                  size_t bpitch,
                                  const frame_info* buffer_info,
                                  int low_threshold,
                                  int high_threshold)
  {
    int width = buffer_info->width;
    int height = buffer_info->height;

    // Ensure low threshold is less than high threshold
    if (low_threshold > high_threshold)
      {
        low_threshold = high_threshold;
      }

    // Create a marker buffer to store the pixels that are above the high threshold
    bool* marker;
    size_t mpitch;
    cudaError_t err;
    err = cudaMallocPitch(&marker, &mpitch, width * sizeof(bool), height);
    CHECK_CUDA_ERROR(err);

    // And set it to false
    err = cudaMemset2D(marker, mpitch, 0, width * sizeof(bool), height);
    CHECK_CUDA_ERROR(err);

    // Create an out buffer to store the final image
    std::byte* out_buffer;
    size_t opitch;
    err = cudaMallocPitch(&out_buffer, &opitch, width * N_CHANNELS, height);
    CHECK_CUDA_ERROR(err);

    // And set it to black
    err = cudaMemset2D(out_buffer, opitch, 0, width * N_CHANNELS, height);
    CHECK_CUDA_ERROR(err);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + (blockSize.x - 1)) / blockSize.x,
                  (height + (blockSize.y - 1)) / blockSize.y);
    apply_threshold_on_marker<<<gridSize, blockSize>>>(
      buffer, bpitch, marker, mpitch, width, height, high_threshold);

    err = cudaDeviceSynchronize();
    CHECK_CUDA_ERROR(err);

    // Apply hysteresis thresholding
    bool h_hysteresis_has_changed = true;
    while (h_hysteresis_has_changed)
      {
        // Copy the value of hysteresis_has_changed to device
        h_hysteresis_has_changed = false;
        err =
          cudaMemcpyToSymbol(hysteresis_has_changed, &h_hysteresis_has_changed,
                             sizeof(h_hysteresis_has_changed));
        CHECK_CUDA_ERROR(err);

        reconstruct_image<<<gridSize, blockSize>>>(
          buffer, bpitch, out_buffer, opitch, marker, mpitch, width, height,
          low_threshold);

        err = cudaDeviceSynchronize();
        CHECK_CUDA_ERROR(err);

        // Retrieve the value of hysteresis_has_changed from device
        err = cudaMemcpyFromSymbol(&h_hysteresis_has_changed,
                                   hysteresis_has_changed,
                                   sizeof(hysteresis_has_changed));
        CHECK_CUDA_ERROR(err);
      }

    // Copy the final image to the buffer
    err = cudaMemcpy2D(buffer, bpitch, out_buffer, opitch, width * N_CHANNELS,
                       height, cudaMemcpyDefault);
    CHECK_CUDA_ERROR(err);

    cudaFree(marker);
    cudaFree(out_buffer);
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
                       bool is_median)
  {
    static std::byte* dbuffer_samples[BG_NUMBER_FRAMES];
    static size_t pitches[BG_NUMBER_FRAMES];
    static int dbuffers_amount = 0;
    static double last_timestamp = 0.0;

    int height = buffer_info->height;
    int width = buffer_info->width;

    // First frame is set to the background model
    if (dbuffers_amount == 0)
      {
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
    else if (buffer_info->timestamp - last_timestamp >= BG_SAMPLING_RATE)
      {
        if (dbuffers_amount < BG_NUMBER_FRAMES)
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
            for (int i = 0; i < BG_NUMBER_FRAMES - 1; ++i)
              {
                dbuffer_samples[i] = dbuffer_samples[i + 1];
                pitches[i] = pitches[i + 1];
              }

            dbuffer_samples[BG_NUMBER_FRAMES - 1] = cpy_buffer;
            pitches[BG_NUMBER_FRAMES - 1] = cpy_pitch;
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

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + (blockSize.x - 1)) / blockSize.x,
                  (height + (blockSize.y - 1)) / blockSize.y);

    // Convert samples and pitches to device memory
    std::byte** dbuffers;
    cudaMalloc(&dbuffers, dbuffers_amount * sizeof(std::byte*));
    cudaMemcpy(dbuffers, dbuffer_samples, dbuffers_amount * sizeof(std::byte*),
               cudaMemcpyHostToDevice);
    size_t* dpitches;
    cudaMalloc(&dpitches, dbuffers_amount * sizeof(size_t));
    cudaMemcpy(dpitches, pitches, dbuffers_amount * sizeof(size_t),
               cudaMemcpyHostToDevice);

// Estimate background
#define _BE_FPARAMS                                                            \
  dbuffers, dpitches, dbuffers_amount, *bg_model, *bg_pitch, width, height
    is_median ? estimate_background_median<<<gridSize, blockSize>>>(_BE_FPARAMS)
              : estimate_background_mean<<<gridSize, blockSize>>>(_BE_FPARAMS);
#undef _BE_FPARAMS

    cudaError_t err = cudaDeviceSynchronize();
    CHECK_CUDA_ERROR(err);

    cudaFree(dbuffers);
    cudaFree(dpitches);
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

    assert(N_CHANNELS == buffer_info->pixel_stride);
    std::byte *dmask, *dbuffer;
    size_t mpitch, bpitch;

    cudaError_t err;

    // Allocate memory on the device
    err = cudaMallocPitch(&dmask, &mpitch, width * N_CHANNELS, height);
    CHECK_CUDA_ERROR(err);
    err = cudaMallocPitch(&dbuffer, &bpitch, width * N_CHANNELS, height);
    CHECK_CUDA_ERROR(err);

    // Copy the input buffer to the device
    err = cudaMemcpy2D(dmask, mpitch, src_buffer, src_stride,
                       width * N_CHANNELS, height, cudaMemcpyDefault);
    CHECK_CUDA_ERROR(err);
    err = cudaMemcpy2D(dbuffer, bpitch, src_buffer, src_stride,
                       width * N_CHANNELS, height, cudaMemcpyDefault);
    CHECK_CUDA_ERROR(err);

    // Set thread block and grid dimensions
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    // Set first frame as bg model
    static std::byte* bg_buffer = dbuffer;
    static size_t bg_pitch = bpitch;

    // Convert RGB to LAB
    rgb_to_lab_cuda(bg_buffer, bg_pitch, dmask, mpitch, buffer_info);

    // Apply morphological opening
    opening_impl_inplace(dmask, mpitch, buffer_info);

    // Apply hysteresis thresholding
    apply_hysteresis_threshold(dmask, mpitch, buffer_info, th_low, th_high);

    // Apply masking
    apply_masking<<<gridSize, blockSize>>>(dbuffer, bpitch, dmask, mpitch,
                                           width, height);
    err = cudaDeviceSynchronize();
    CHECK_CUDA_ERROR(err);

    // Update background model
    update_bg_model(dbuffer, bpitch, &bg_buffer, &bg_pitch, dmask, mpitch,
                    buffer_info, true);

    // Copy the result back to the host
    err = cudaMemcpy2D(src_buffer, src_stride, dbuffer, bpitch,
                       width * N_CHANNELS, height, cudaMemcpyDefault);
    CHECK_CUDA_ERROR(err);

    cudaFree(dmask);
    cudaFree(dbuffer);
  }
}
