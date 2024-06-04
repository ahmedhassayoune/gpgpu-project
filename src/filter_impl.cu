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

struct rgb
{
  uint8_t r, g, b;
};

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
                                          int width,
                                          int height,
                                          int high_threshold)
{
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;

  if (x >= width || y >= height)
    return;

  rgb* buffer_line = (rgb*)(buffer + y * bpitch);
  bool* marker_line = (bool*)((std::byte*)marker + y * mpitch);

  if (buffer_line[x].r > high_threshold)
    {
      marker_line[x] = true;
    }
}

/// @brief Reconstruct the hysteresis thresholding image from the marker
/// @param buffer The input buffer
/// @param bpitch The pitch of the input buffer
/// @param out The output buffer
/// @param opitch The pitch of the output buffer
/// @param marker The marker buffer
/// @param mpitch The pitch of the marker buffer
/// @param width The width of the image
/// @param height The height of the image
/// @param low_threshold The low threshold
/// @return
__global__ void reconstruct_image(std::byte* buffer,
                                  size_t bpitch,
                                  std::byte* out,
                                  size_t opitch,
                                  bool* marker,
                                  size_t mpitch,
                                  int width,
                                  int height,
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
          // Skip the current pixel
          if (i == 0 && j == 0)
            {
              continue;
            }

          int ny = y + j;
          int nx = x + i;

          // Check if the pixel is within the image boundaries
          if (nx >= 0 && nx < width && ny >= 0 && ny < height)
            {
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

  /// @brief Apply hysteresis thresholding on the buffer
  /// @param buffer The input buffer
  /// @param width The width of the image
  /// @param height The height of the image
  /// @param bpitch The pitch of the input buffer
  /// @param low_threshold The low threshold
  /// @param high_threshold The high threshold
  void apply_hysteresis_threshold(std::byte* buffer,
                                  int width,
                                  int height,
                                  size_t bpitch,
                                  int low_threshold,
                                  int high_threshold)
  {
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

    err = cudaDeviceSynchronize();
    CHECK_CUDA_ERROR(err);

    dim3 blockSize(16, 16);
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
