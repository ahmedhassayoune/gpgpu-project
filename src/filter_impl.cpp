#include "filter_impl.h"

#include <chrono>
#include <iostream>
#include <thread>
#include <math.h>
#include "logo.h"

#define N_CHANNELS 3

struct rgb
{
  uint8_t r, g, b;
};

struct LAB
{
  float l, a, b;
};

struct pos
{
  int x, y;
};

struct Queue
{
  pos* buffer_pos;
  int front = 0;
  int rear = 0;
};

/* prototypes */
static void _selection_sort(uint8_t* bytes, int start, int end, int step);

static void rgbToXyz(float r, float g, float b, float& x, float& y, float& z);
static void xyzToLab(float x, float y, float z, float& l, float& a, float& b);

static void enqueue(Queue& q, int x, int y);
static pos dequeue(Queue& q);
static bool is_empty(Queue& q);

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
}

//******************************************************
//**                                                  **
//**               Background Estimation              **
//**                                                  **
//******************************************************

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

static void _selection_sort(uint8_t* bytes, int start, int end, int step)
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

  for (int yy = 0; yy < height; ++yy)
    {
      for (int xx = 0; xx < width; ++xx)
        {
          // for each buffer store pixel at (yy, xx)
          for (int ii = 0; ii < buffers_amount; ++ii)
            {
              uint8_t* ptr = buffers[ii] + yy * stride + xx * pixel_stride;
              int jj = ii * N_CHANNELS;
              for (int kk = 0; kk < N_CHANNELS; ++kk)
                B[jj + kk] = ptr[kk];
            }

          // the median is computed for each channel separately

          // each channel is sorted in order to find its mid value
          for (int ii = 0; ii < N_CHANNELS; ++ii)
            _selection_sort(B, ii, buffers_amount * N_CHANNELS, N_CHANNELS);

          // select mid
          uint8_t* outptr = out + yy * stride + xx * pixel_stride;
          if (buffers_amount % 2 == 0)
            {
              for (int ii = 0; ii < N_CHANNELS; ++ii)
                {
                  int a = B[(buffers_amount / 2) * N_CHANNELS + ii];
                  int b = B[(buffers_amount / 2 - 1) * N_CHANNELS + ii];
                  outptr[ii] = (uint8_t)((a + b) / 2);
                }
            }
          else
            {
              for (int ii = 0; ii < N_CHANNELS; ++ii)
                outptr[ii] = B[(buffers_amount / 2) * N_CHANNELS + ii];
            }
        }
    }
}

//******************************************************
//**                                                  **
//**           Conversion from RGB to LAB             **
//**                                                  **
//******************************************************

static void rgbToXyz(float r, float g, float b, float& x, float& y, float& z)
{
  const float D65_XYZ[9] = {0.412453f, 0.357580f, 0.180423f,
                            0.212671f, 0.715160f, 0.072169f,
                            0.019334f, 0.119193f, 0.950227f};

  // Convert from 0-255 range to 0-1 range
  r = r / 255.0f;
  g = g / 255.0f;
  b = b / 255.0f;

// Apply sRGB gamma correction
#define GAMMA_CORRECT(x)                                                       \
  ((x) > 0.04045f ? powf(((x) + 0.055f) / 1.055f, 2.4f) : (x) / 12.92f)
  r = GAMMA_CORRECT(r);
  g = GAMMA_CORRECT(g);
  b = GAMMA_CORRECT(b);
#undef GAMMA_CORRECT

  // Convert to XYZ using the D65 illuminant
  x = r * D65_XYZ[0] + g * D65_XYZ[1] + b * D65_XYZ[2];
  y = r * D65_XYZ[3] + g * D65_XYZ[4] + b * D65_XYZ[5];
  z = r * D65_XYZ[6] + g * D65_XYZ[7] + b * D65_XYZ[8];
}

static void xyzToLab(float x, float y, float z, float& l, float& a, float& b)
{
  const float D65_Xn = 0.95047f;
  const float D65_Yn = 1.00000f;
  const float D65_Zn = 1.08883f;

  // Normalize by the reference white
  x /= D65_Xn;
  y /= D65_Yn;
  z /= D65_Zn;

// Apply the nonlinear transformation
#define NONLINEAR(x)                                                           \
  ((x) > 0.008856f ? powf(x, 1.0f / 3.0f) : (7.787f * x + 16.0f / 116.0f))
  float fx = NONLINEAR(x);
  float fy = NONLINEAR(x);
  float fz = NONLINEAR(x);
#undef NONLINEAR

  // Convert to Lab
  l = (116.0f * fy) - 16.0f;
  a = 500.0f * (fx - fy);
  b = 200.0f * (fy - fz);
}

void rgb_to_lab(uint8_t* reference_buffer,
                uint8_t* buffer,
                int width,
                int height,
                int stride,
                int pixel_stride)
{
  float* array_distance = new float[width * height];

  // Step 1 - Fill distance array
  float max_distance = 0.0f;
  for (int y = 0; y < height; ++y)
    {
      uint8_t* lineptr_reference = reference_buffer + y * stride;
      uint8_t* lineptr = buffer + y * stride;
      for (int x = 0; x < width; ++x)
        {
          // Compupte pixel on reference frame
          rgb* pxl = (rgb*)(lineptr_reference + x * pixel_stride);
          float r = pxl->r;
          float g = pxl->g;
          float b = pxl->b;

          float X, Y, Z;
          rgbToXyz(r, g, b, X, Y, Z);

          float L, A, B;
          xyzToLab(X, Y, Z, L, A, B);

          LAB referenceLab = {L, A, B};

          // Compupte pixel on current frame
          rgb* pxl_ = (rgb*)(lineptr + x * pixel_stride);
          float r_ = pxl_->r;
          float g_ = pxl_->g;
          float b_ = pxl_->b;

          float X_, Y_, Z_;
          rgbToXyz(r_, g_, b_, X_, Y_, Z_);

          float L_, A_, B_;
          xyzToLab(X_, Y_, Z_, L_, A_, B_);

          LAB currentLab = {L_, A_, B_};

// Compute distance between both pixels
#define LAB_DISTANCE(lab1, lab2)                                               \
  (sqrtf(powf((lab1).l - (lab2).l, 2) + powf((lab1).a - (lab2).a, 2)           \
         + powf((lab1).b - (lab2).b, 2)))
          float distance = LAB_DISTANCE(currentLab, referenceLab);
#undef LAB_DISTANCE
          array_distance[y * width + x] = distance;
          max_distance = std::max(max_distance, distance);
        }
    }

  // Step 2 - Convert Distance array to uint8 buffer

  for (int y = 0; y < height; ++y)
    {
      uint8_t* lineptr = buffer + y * stride;
      for (int x = 0; x < width; ++x)
        {
          rgb* pxl = (rgb*)(lineptr + x * pixel_stride);
          float distance = array_distance[y * width + x];
          uint8_t distance8bit = static_cast<uint8_t>(
            std::min(distance / max_distance * 255.0f, 255.0f));

          pxl->r = distance8bit;
          pxl->g = distance8bit;
          pxl->b = distance8bit;
        }
    }

  delete[] array_distance;
}

//******************************************************
//**                                                  **
//**                Hysteresis Threshold              **
//**                                                  **
//******************************************************

/**
 * Enqueue a position (x, y) into the queue.
 * 
 * @param q Reference to the Queue structure.
 * @param x The x-coordinate of the position to enqueue.
 * @param y The y-coordinate of the position to enqueue.
 */
static void enqueue(Queue& q, int x, int y)
{
  q.buffer_pos[q.rear].x = x;
  q.buffer_pos[q.rear].y = y;
  q.rear += 1;
}

/**
 * Dequeue a position from the queue.
 * 
 * @param q Reference to the Queue structure.
 * @return The dequeued position.
 */
static pos dequeue(Queue& q)
{
  if (q.front == q.rear)
    {
      std::cout << "Warning: Queue is empty" << std::endl;
    }
  pos p = q.buffer_pos[q.front];
  q.front += 1;
  return p;
}

/**
 * Check if the queue is empty.
 * 
 * @param q Reference to the Queue structure.
 * @return True if the queue is empty, otherwise false.
 */
static bool is_empty(Queue& q) { return q.front == q.rear; }

/**
 * Apply hysteresis thresholding to the input image buffer.
 * 
 * This algorithm finds regions where the pixel intensity is greater than the 
 * high threshold OR the pixel intensity is greater than the low threshold and 
 * that region is connected to a region greater than the high threshold.
 * 
 * @param buffer The pointer to the buffer containing the image data.
 * @param width The width of the image.
 * @param height The height of the image.
 * @param stride The stride of the image buffer (typically width * pixel size).
 * @param pixel_stride The stride of each pixel in the buffer (typically 3 for RGB).
 * @param low_threshold The lower threshold.
 * @param high_threshold The higher threshold.
 */
void apply_hysteresis_threshold(uint8_t* buffer,
                                int width,
                                int height,
                                int stride,
                                int pixel_stride,
                                uint8_t low_threshold,
                                uint8_t high_threshold)
{
  // Ensure low threshold is less than high threshold
  if (low_threshold > high_threshold)
    {
      low_threshold = high_threshold;
    }

  // Init queue
  Queue q = {nullptr, 0, 0};
  q.buffer_pos = (pos*)malloc(width * height * sizeof(pos));

  // Enqueue pixels with intensity greater than high threshold
  for (int y = 0; y < height; ++y)
    {
      uint8_t* lineptr = (uint8_t*)(buffer + y * stride);
      for (int x = 0; x < width; ++x)
        {
          rgb pxl = *(rgb*)(lineptr + x * pixel_stride);
          if (pxl.r > high_threshold)
            {
              enqueue(q, x, y);
              pxl.r = 255;
              pxl.g = 255;
              pxl.b = 255;
              *(rgb*)(lineptr + x * pixel_stride) = pxl;
            }
        }
    }

  // Apply hysteresis threshold
  while (!is_empty(q))
    {
      pos p = dequeue(q);

      // Go through all 8 neighbors
      for (int dy = -1; dy <= 1; ++dy)
        {
          for (int dx = -1; dx <= 1; ++dx)
            {
              // Skip the center pixel
              if (dx == 0 && dy == 0)
                {
                  continue;
                }

              int nx = p.x + dx;
              int ny = p.y + dy;

              if (nx < 0 || nx >= width || ny < 0 || ny >= height)
                {
                  continue;
                }

              uint8_t* lineptr = (uint8_t*)(buffer + ny * stride);
              rgb pxl = *(rgb*)(lineptr + nx * pixel_stride);

              // If pixel is already visited or below low threshold, skip
              if (pxl.r <= low_threshold || pxl.r == 255)
                {
                  continue;
                }

              pxl.r = 255;
              pxl.g = 255;
              pxl.b = 255;

              *(rgb*)(lineptr + nx * pixel_stride) = pxl;
              enqueue(q, nx, ny);
            }
        }
    }

  // Keep only pixels with intensity 255
  for (int y = 0; y < height; ++y)
    {
      uint8_t* lineptr = (uint8_t*)(buffer + y * stride);
      for (int x = 0; x < width; ++x)
        {
          rgb pxl = *(rgb*)(lineptr + x * pixel_stride);
          if (pxl.r != 255)
            {
              pxl.r = 0;
              pxl.g = 0;
              pxl.b = 0;
            }
          *(rgb*)(lineptr + x * pixel_stride) = pxl;
        }
    }

  free(q.buffer_pos);
}

//******************************************************
//**                                                  **
//**                Apply Masking                     **
//**                                                  **
//******************************************************

void apply_masking(uint8_t* buffer,
                   int width,
                   int height,
                   int stride,
                   int pixel_stride,
                   uint8_t* mask)
{
  if (!buffer || !mask)
    return;

  for (int yy = 0; yy < height; ++yy)
    {
      uint8_t* in_lineptr = buffer + yy * stride;
      uint8_t* mask_lineptr = mask + yy * stride;

      for (int xx = 0; xx < width; ++xx)
        {
          rgb* in_pixelptr = (rgb*)(in_lineptr + xx * pixel_stride);
          rgb* mask_pixelptr = (rgb*)(mask_lineptr + xx * pixel_stride);
          int red = in_pixelptr->r;
          red = red + (mask_pixelptr->r > 0) * red / 2;
          red = red > 0xff ? 0xff : red;
          in_pixelptr->r = (uint8_t)(red);
        }
    }
}
