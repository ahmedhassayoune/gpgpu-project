#include "filter_impl.h"

#include <chrono>
#include <iostream>
#include <thread>
#include <math.h>
#include "logo.h"

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
}

//******************************************************
//**                                                  **
//**           Conversion from RGB to LAB             **
//**                                                  **
//******************************************************

static float gamma_correct(float channel)
{
  return (channel > 0.04045f) ? powf((channel + 0.055f) / 1.055f, 2.4f)
                              : channel / 12.92f;
}

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
  r = gamma_correct(r);
  g = gamma_correct(g);
  b = gamma_correct(b);

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
  float fx =
    (x > 0.008856f) ? powf(x, 1.0f / 3.0f) : (7.787f * x + 16.0f / 116.0f);
  float fy =
    (y > 0.008856f) ? powf(y, 1.0f / 3.0f) : (7.787f * y + 16.0f / 116.0f);
  float fz =
    (z > 0.008856f) ? powf(z, 1.0f / 3.0f) : (7.787f * z + 16.0f / 116.0f);

  // Convert to Lab
  l = (116.0f * fy) - 16.0f;
  a = 500.0f * (fx - fy);
  b = 200.0f * (fy - fz);
}

static float labDistance(const LAB& lab1, const LAB& lab2)
{
  return sqrtf(powf(lab1.l - lab2.l, 2) + powf(lab1.a - lab2.a, 2)
               + powf(lab1.b - lab2.b, 2));
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
          float distance = labDistance(currentLab, referenceLab);
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
void enqueue(Queue& q, int x, int y)
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
pos dequeue(Queue& q)
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
bool is_empty(Queue& q) { return q.front == q.rear; }

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
  Queue q;
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