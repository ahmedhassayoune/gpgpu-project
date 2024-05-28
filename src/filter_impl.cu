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
                   int width,
                   int height,
                   int src_stride,
                   int pixel_stride)
  {
    load_logo();

    assert(sizeof(rgb) == pixel_stride);
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

//******************************************************
//**                                                  **
//**           Conversion from RGB to LAB (GPU)       **
//**                                                  **
//******************************************************

#include <cuda_runtime.h> 
#include <iostream>
#include <cmath>
#include <algorithm>

struct rgb {
    uint8_t r, g, b;
};

struct LAB {
    float l, a, b;
};

__device__ void rgbToXyz(float r, float g, float b, float& x, float& y, float& z) {
    const float D65_XYZ[9] = {0.412453f, 0.357580f, 0.180423f,
                              0.212671f, 0.715160f, 0.072169f,
                              0.019334f, 0.119193f, 0.950227f};

    r = r / 255.0f;
    g = g / 255.0f;
    b = b / 255.0f;

    r = (r > 0.04045f) ? powf((r + 0.055f) / 1.055f, 2.4f) : (r / 12.92f);
    g = (g > 0.04045f) ? powf((g + 0.055f) / 1.055f, 2.4f) : (g / 12.92f);
    b = (b > 0.04045f) ? powf((b + 0.055f) / 1.055f, 2.4f) : (b / 12.92f);

    x = r * D65_XYZ[0] + g * D65_XYZ[1] + b * D65_XYZ[2];
    y = r * D65_XYZ[3] + g * D65_XYZ[4] + b * D65_XYZ[5];
    z = r * D65_XYZ[6] + g * D65_XYZ[7] + b * D65_XYZ[8];
}

__device__ void xyzToLab(float x, float y, float z, float& l, float& a, float& b) {
    const float D65_Xn = 0.95047f;
    const float D65_Yn = 1.00000f;
    const float D65_Zn = 1.08883f;

    x /= D65_Xn;
    y /= D65_Yn;
    z /= D65_Zn;

    float fx = (x > 0.008856f) ? powf(x, 1.0f / 3.0f) : (7.787f * x + 16.0f / 116.0f);
    float fy = (y > 0.008856f) ? powf(y, 1.0f / 3.0f) : (7.787f * y + 16.0f / 116.0f);
    float fz = (z > 0.008856f) ? powf(z, 1.0f / 3.0f) : (7.787f * z + 16.0f / 116.0f);

    l = (116.0f * fy) - 16.0f;
    a = 500.0f * (fx - fy);
    b = 200.0f * (fy - fz);
}

__device__ float labDistance(const LAB& lab1, const LAB& lab2) {
    return sqrtf(powf(lab1.l - lab2.l, 2) + powf(lab1.a - lab2.a, 2) + powf(lab1.b - lab2.b, 2));
}

__global__ void rgbToLabDistanceKernel(uint8_t* referenceBuffer, uint8_t* buffer, float* distanceArray,
                                       int width, int height, int stride, int pixelStride) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    uint8_t* lineptrReference = referenceBuffer + y * stride;
    uint8_t* lineptr = buffer + y * stride;

    rgb* pxlRef = (rgb*)(lineptrReference + x * pixelStride);
    float rRef = pxlRef->r;
    float gRef = pxlRef->g;
    float bRef = pxlRef->b;

    float XRef, YRef, ZRef;
    rgbToXyz(rRef, gRef, bRef, XRef, YRef, ZRef);

    float LRef, ARef, BRef;
    xyzToLab(XRef, YRef, ZRef, LRef, ARef, BRef);

    LAB referenceLab = {LRef, ARef, BRef};

    rgb* pxl = (rgb*)(lineptr + x * pixelStride);
    float r = pxl->r;
    float g = pxl->g;
    float b = pxl->b;

    float X, Y, Z;
    rgbToXyz(r, g, b, X, Y, Z);

    float L, A, B;
    xyzToLab(X, Y, Z, L, A, B);

    LAB currentLab = {L, A, B};

    float distance = labDistance(currentLab, referenceLab);
    distanceArray[y * width + x] = distance;
}

__global__ void normalizeAndConvertTo8bitKernel(uint8_t* buffer, float* distanceArray, float maxDistance,
                                                int width, int height, int stride, int pixelStride) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    uint8_t* lineptr = buffer + y * stride;
    rgb* pxl = (rgb*)(lineptr + x * pixelStride);

    float distance = distanceArray[y * width + x];
    uint8_t distance8bit = static_cast<uint8_t>(fminf(distance / maxDistance * 255.0f, 255.0f));

    pxl->r = distance8bit;
    pxl->g = distance8bit;
    pxl->b = distance8bit;
}

void rgb_to_lab_cuda(uint8_t* referenceBuffer, uint8_t* buffer, int width, int height, int stride, int pixelStride) {
    float* distanceArray;
    size_t distanceArraySize = width * height * sizeof(float);
    cudaMalloc(&distanceArray, distanceArraySize);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    rgbToLabDistanceKernel<<<gridSize, blockSize>>>(referenceBuffer, buffer, distanceArray, width, height, stride, pixelStride);
    cudaDeviceSynchronize();

    float* h_distanceArray = new float[width * height];
    cudaMemcpy(h_distanceArray, distanceArray, distanceArraySize, cudaMemcpyDeviceToHost);
    float maxDistance = 0.0f;
    for (int i = 0; i < width * height; ++i) {
        maxDistance = fmaxf(maxDistance, h_distanceArray[i]);
    }
    delete[] h_distanceArray;

    normalizeAndConvertTo8bitKernel<<<gridSize, blockSize>>>(buffer, distanceArray, maxDistance, width, height, stride, pixelStride);
    cudaDeviceSynchronize();

    cudaFree(distanceArray);
}
