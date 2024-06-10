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

__device__ bool hysteresis_has_changed;

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
                                          const frame_info* buffer_info,
                                          int high_threshold)
{
  int width = buffer_info->width;
  int height = buffer_info->height;

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
                                  const frame_info* buffer_info,
                                  int low_threshold)
{
  int width = buffer_info->width;
  int height = buffer_info->height;

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
          if ((i == 0 && j == 0) || nx < 0 || nx >= width || ny < 0 || ny >= height)
            {
              continue;
            }

          // Check if the pixel is within the image boundaries
          rgb* buffer_line = (rgb*)(buffer + ny * bpitch);
          bool* neighbor_marker_line =
            (bool*)((std::byte*)marker + ny * mpitch);
          if (!neighbor_marker_line[nx]
              && buffer_line[nx].r > low_threshold)
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
    err = cudaMallocPitch(&out_buffer, &opitch, width * sizeof(rgb), height);
    CHECK_CUDA_ERROR(err);

    // And set it to black
    err = cudaMemset2D(out_buffer, opitch, 0, width * sizeof(rgb), height);
    CHECK_CUDA_ERROR(err);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + (blockSize.x - 1)) / blockSize.x,
                  (height + (blockSize.y - 1)) / blockSize.y);
    apply_threshold_on_marker<<<gridSize, blockSize>>>(
      buffer, bpitch, marker, mpitch, buffer_info, high_threshold);

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
          buffer, bpitch, out_buffer, opitch, marker, mpitch, buffer_info,
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
    err = cudaMemcpy2D(buffer, bpitch, out_buffer, opitch, width * sizeof(rgb),
                       height, cudaMemcpyDefault);
    CHECK_CUDA_ERROR(err);

    cudaFree(marker);
    cudaFree(out_buffer);
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
