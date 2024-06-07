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



__global__ void morphological_erosion(uint8_t* buffer,
                                         uint8_t* output_buffer,
                                         int width,
                                         int height,
                                         int stride,
                                         int output_stride,
                                         int pixel_stride)
{
  int yy = blockIdx.y * blockDim.y + threadIdx.y;
  int xx = (blockIdx.x * blockDim.x + threadIdx.x);

  if (xx >= width || yy >= height)
    return;

  uint8_t res0 = 0xFF;
  uint8_t res1 = 0xFF;
  uint8_t res2 = 0xFF;

  if (yy >= 3) {
    res0 = buffer[(yy - 3) * stride + xx * pixel_stride];
    res1 = buffer[(yy - 3) * stride + xx * pixel_stride + 1];
    res2 = buffer[(yy - 3) * stride + xx * pixel_stride + 2];
  }
  for (int i = yy - 2; i < yy; ++i) {
    if (i >= 0) {
      for (int j = xx - 2; j <= xx +2; j++) {
        if (j >= 0 && j < width) {
          res0 = min(res0, buffer[i * stride + j * pixel_stride ]);
          res1 = min(res1, buffer[i * stride + j * pixel_stride + 1]);
          res2 = min(res2, buffer[i * stride + j * pixel_stride + 2]);
        }
      }
    }
  }
  for (int j = xx - 3; j <= xx + 3; j++) {
    if (j >= 0 && j < width) {
        res0 = min(res0, buffer[yy * stride + j * pixel_stride ]);
        res1 = min(res1, buffer[yy * stride + j * pixel_stride + 1]);
        res2 = min(res2, buffer[yy * stride + j * pixel_stride + 2]);
    }
  }
  for (int i = yy + 1; i <= yy + 2; ++i) {
    if (i < width) {
      for (int j = xx - 2; j <= xx +2; j++) {
        if (j >= 0 && j < width) {
          res0 = min(res0, buffer[i * stride + j * pixel_stride ]);
          res1 = min(res1, buffer[i * stride + j * pixel_stride + 1]);
          res2 = min(res2, buffer[i * stride + j * pixel_stride + 2]);
        }
      }
    }
  }
  if (yy + 3 < width) {
    res0 = min(res0, buffer[(yy - 3) * stride + xx * pixel_stride ]);
    res1 = min(res1, buffer[(yy - 3) * stride + xx * pixel_stride + 1]);
    res2 = min(res2, buffer[(yy - 3) * stride + xx * pixel_stride + 2]);
  }

  output_buffer[yy * output_stride + xx * pixel_stride] = res0;
  output_buffer[yy * output_stride + xx * pixel_stride + 1] = res1;
  output_buffer[yy * output_stride + xx * pixel_stride + 2] = res2;
}

__global__ void morphological_dilation(uint8_t* buffer,
                                         uint8_t* output_buffer,
                                         int width,
                                         int height,
                                         int stride,
                                         int output_stride,
                                         int pixel_stride)
{
  int yy = blockIdx.y * blockDim.y + threadIdx.y;
  int xx = (blockIdx.x * blockDim.x + threadIdx.x);

  if (xx >= width || yy >= height)
    return;

  uint8_t res0 = 0x00;
  uint8_t res1 = 0x00;
  uint8_t res2 = 0x00;

  if (yy >= 3) {
    res0 = buffer[(yy - 3) * stride + xx * pixel_stride];
    res1 = buffer[(yy - 3) * stride + xx * pixel_stride + 1];
    res2 = buffer[(yy - 3) * stride + xx * pixel_stride + 2];
  }
  for (int i = yy - 2; i < yy; ++i) {
    if (i >= 0) {
      for (int j = xx - 2; j <= xx +2; j++) {
        if (j >= 0 && j < width) {
          res0 = max(res0, buffer[i * stride + j * pixel_stride ]);
          res1 = max(res1, buffer[i * stride + j * pixel_stride + 1]);
          res2 = max(res2, buffer[i * stride + j * pixel_stride + 2]);
        }
      }
    }
  }
  for (int j = xx - 3; j <= xx + 3; j++) {
    if (j >= 0 && j < width) {
        res0 = max(res0, buffer[yy * stride + j * pixel_stride ]);
        res1 = max(res1, buffer[yy * stride + j * pixel_stride + 1]);
        res2 = max(res2, buffer[yy * stride + j * pixel_stride + 2]);
    }
  }
  for (int i = yy + 1; i <= yy + 2; ++i) {
    if (i < width) {
      for (int j = xx - 2; j <= xx +2; j++) {
        if (j >= 0 && j < width) {
          res0 = max(res0, buffer[i * stride + j * pixel_stride ]);
          res1 = max(res1, buffer[i * stride + j * pixel_stride + 1]);
          res2 = max(res2, buffer[i * stride + j * pixel_stride + 2]);
        }
      }
    }
  }
  if (yy + 3 < width) {
    res0 = max(res0, buffer[(yy - 3) * stride + xx * pixel_stride ]);
    res1 = max(res1, buffer[(yy - 3) * stride + xx * pixel_stride + 1]);
    res2 = max(res2, buffer[(yy - 3) * stride + xx * pixel_stride + 2]);
  }

  output_buffer[yy * output_stride + xx * pixel_stride] = res0;
  output_buffer[yy * output_stride + xx * pixel_stride + 1] = res1;
  output_buffer[yy * output_stride + xx * pixel_stride + 2] = res2;
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

  void opening_impl_inplace(uint8_t* buffer,
                          int width,
                          int height,
                          int stride,
                          int pixel_stride)
  {
      uint8_t *gpu_image;
      size_t gpu_pitch;
      cudaError_t err = cudaMallocPitch(&gpu_image, &gpu_pitch, width * pixel_stride * sizeof(uint8_t), height);
      CHECK_CUDA_ERROR(err);

      err = cudaMemcpy2D(gpu_image, gpu_pitch, buffer, stride,
                        width * pixel_stride * sizeof(uint8_t), height, cudaMemcpyDeviceToHost );
      CHECK_CUDA_ERROR(err);

      uint8_t *gpu_intermediate_image;
      size_t gpu_intermediate_pitch;
      err = cudaMallocPitch(&gpu_intermediate_image, &gpu_intermediate_pitch, width * pixel_stride * sizeof(uint8_t), height);
      CHECK_CUDA_ERROR(err);

      dim3 blockSize(16, 16);
      dim3 gridSize((width + (blockSize.x - 1)) / blockSize.x,
                  (height + (blockSize.y - 1)) / blockSize.y);
      
      morphological_erosion<<<gridSize, blockSize>>>(gpu_image,
                            gpu_intermediate_image,
                            width,
                            height,
                            gpu_pitch,
                            gpu_intermediate_pitch,
                            pixel_stride);
      err = cudaDeviceSynchronize();
      CHECK_CUDA_ERROR(err);
      
      morphological_dilation<<<gridSize, blockSize>>>(gpu_intermediate_image,
                            gpu_image,
                            width,
                            height,
                            gpu_intermediate_pitch,
                            gpu_pitch,
                            pixel_stride);
      err = cudaDeviceSynchronize();
      CHECK_CUDA_ERROR(err);

      err = cudaMemcpy2D(buffer, stride, gpu_image, gpu_pitch,
                        width * pixel_stride * sizeof(uint8_t), height, cudaMemcpyHostToDevice );
      CHECK_CUDA_ERROR(err);

  }



}
