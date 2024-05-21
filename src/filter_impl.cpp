#include "filter_impl.h"

#include <chrono>
#include <iostream>
#include <thread>
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
