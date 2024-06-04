#include "filter_impl.h"

#include <chrono>
#include <cstring>
#include <iostream>
#include <thread>
#include <math.h>
#include "logo.h"

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

enum op_type
{
  EROSION,
  DILATION
};

/* prototypes */
static void update_bg_model(uint8_t* buffer,
                            const frame_info* buffer_info,
                            uint8_t** bg_model,
                            uint8_t* mask,
                            bool is_median);

static void _selection_sort(uint8_t* bytes, int start, int end, int step);
#define _BE_FSIGN                                                              \
  uint8_t **buffers, int buffers_amount, const frame_info *buffer_info,        \
    uint8_t *out
static void estimate_background_mean(_BE_FSIGN);
static void estimate_background_median(_BE_FSIGN);
#undef _BE_FSIGN

static void rgbToXyz(float r, float g, float b, float& x, float& y, float& z);
static void xyzToLab(float x, float y, float z, float& l, float& a, float& b);
static void rgb_to_lab(uint8_t* reference_buffer,
                       uint8_t* buffer,
                       const frame_info* buffer_info);

static inline void min_assign(rgb* lhs, rgb* rhs);
static inline void max_assign(rgb* lhs, rgb* rhs);
static void morphology_impl(uint8_t* buffer,
                            const frame_info* buffer_info,
                            uint8_t* output,
                            op_type op);
static void opening_impl_inplace(uint8_t* buffer,
                                 const frame_info* buffer_info);

static void enqueue(Queue& q, int x, int y);
static pos dequeue(Queue& q);
static bool is_empty(Queue& q);
static void apply_hysteresis_threshold(uint8_t* buffer,
                                       const frame_info* buffer_info,
                                       uint8_t low_threshold,
                                       uint8_t high_threshold);

static void
apply_masking(uint8_t* buffer, const frame_info* buffer_info, uint8_t* mask);

static void copy_buffer(uint8_t* buffer,
                        uint8_t** cpy_buffer,
                        const frame_info* buffer_info,
                        uint8_t* mask,
                        uint8_t* fallback_buffer);

extern "C"
{
  void filter_impl(uint8_t* buffer,
                   const frame_info* buffer_info,
                   int th_low,
                   int th_high)
  {
    // Set first frame as background model
    static uint8_t* bg_model = buffer;

    // Copy frame buffer
    uint8_t* cpy_buffer = nullptr;
    copy_buffer(buffer, &cpy_buffer, buffer_info, nullptr, nullptr);

    // Convert frame to LAB
    rgb_to_lab(bg_model, cpy_buffer, buffer_info);

    // Apply morphological opening
    opening_impl_inplace(cpy_buffer, buffer_info);

    // Apply hysteresis threshold
    apply_hysteresis_threshold(cpy_buffer, buffer_info, th_low, th_high);

    // Apply masking
    apply_masking(buffer, buffer_info, cpy_buffer);

    // Update background model
    update_bg_model(buffer, buffer_info, &bg_model, cpy_buffer, true);

    free(cpy_buffer);
  }
}
//******************************************************
//**                                                  **
//**               Background Estimation              **
//**                                                  **
//******************************************************

static void update_bg_model(uint8_t* buffer,
                            const frame_info* buffer_info,
                            uint8_t** bg_model,
                            uint8_t* mask,
                            bool is_median)
{
  static uint8_t* frame_samples[BG_NUMBER_FRAMES];
  static int frame_samples_count = 0;
  static double last_timestamp = 0.0;

  int height = buffer_info->height;
  int stride = buffer_info->stride;

  // First frame is set to the background model
  if (frame_samples_count == 0)
    {
      // Copy buffer
      uint8_t* cpy_buffer = nullptr;
      copy_buffer(buffer, &cpy_buffer, buffer_info, nullptr, nullptr);

      frame_samples[0] = cpy_buffer;
      frame_samples_count = 1;
      last_timestamp = buffer_info->timestamp;

      // First bg_model is set to the buffer pointer
      // so we set it to null to reallocate new memory after
      *bg_model = nullptr;
    }
  else if (buffer_info->timestamp - last_timestamp >= BG_SAMPLING_RATE)
    {
      if (frame_samples_count < BG_NUMBER_FRAMES)
        {
          // Copy buffer and apply mask to remove foreground
          uint8_t* cpy_buffer = nullptr;
          copy_buffer(buffer, &cpy_buffer, buffer_info, mask, *bg_model);

          frame_samples[frame_samples_count] = cpy_buffer;
          frame_samples_count += 1;
          last_timestamp = buffer_info->timestamp;
        }
      else
        {
          // Copy buffer on the oldest frame w/o reallocating
          // And apply mask to remove foreground
          uint8_t* cpy_buffer = frame_samples[0];
          copy_buffer(buffer, &cpy_buffer, buffer_info, mask, *bg_model);

          // Shift frame samples
          for (int i = 0; i < BG_NUMBER_FRAMES - 1; ++i)
            {
              frame_samples[i] = frame_samples[i + 1];
            }

          frame_samples[BG_NUMBER_FRAMES - 1] = cpy_buffer;
          last_timestamp = buffer_info->timestamp;
        }
    }
  else
    {
      return;
    }

  // Allocate memory for background model if not already allocated
  if (*bg_model == nullptr)
    {
      *bg_model = (uint8_t*)malloc(height * stride);
    }

// Estimate background
#define _BE_FPARAMS frame_samples, frame_samples_count, buffer_info, *bg_model
  is_median ? estimate_background_median(_BE_FPARAMS)
            : estimate_background_mean(_BE_FPARAMS);
#undef _BE_FPARAMS
}

void estimate_background_mean(uint8_t** buffers,
                              int buffers_amount,
                              const frame_info* buffer_info,
                              uint8_t* out)
{
  // It is expected `out` has same stride values

  if (!buffers || !out)
    return;

  for (int ii = 0; ii < buffers_amount; ++ii)
    if (!buffers[ii])
      return;

  int width = buffer_info->width;
  int height = buffer_info->height;
  int stride = buffer_info->stride;
  int pixel_stride = buffer_info->pixel_stride;

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
                                const frame_info* buffer_info,
                                uint8_t* out)
{
  if (!buffers || !out)
    return;

  for (int ii = 0; ii < buffers_amount; ++ii)
    if (!buffers[ii])
      return;

  int width = buffer_info->width;
  int height = buffer_info->height;
  int stride = buffer_info->stride;
  int pixel_stride = buffer_info->pixel_stride;

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
                const frame_info* buffer_info)
{
  int width = buffer_info->width;
  int height = buffer_info->height;
  int stride = buffer_info->stride;
  int pixel_stride = buffer_info->pixel_stride;

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
//**             Morphological Opening                **
//**                                                  **
//******************************************************
static inline void min_assign(rgb* lhs, rgb* rhs)
{
  lhs->r = std::min(lhs->r, rhs->r);
  lhs->g = std::min(lhs->g, rhs->g);
  lhs->b = std::min(lhs->b, rhs->b);
}

static inline void max_assign(rgb* lhs, rgb* rhs)
{
  lhs->r = std::max(lhs->r, rhs->r);
  lhs->g = std::max(lhs->g, rhs->g);
  lhs->b = std::max(lhs->b, rhs->b);
}

static void morphology_impl(uint8_t* input,
                            const frame_info* buffer_info,
                            uint8_t* output,
                            op_type op)
{
  int width = buffer_info->width;
  int height = buffer_info->height;
  int stride = buffer_info->stride;
  int pixel_stride = buffer_info->pixel_stride;

  // top line (1/7)
  for (int y = 0; y < height && y < 3; ++y)
    {
      uint8_t* output_lineptr = (uint8_t*)(output + y * stride);
      for (int x = 0; x < width; ++x)
        {
          *(rgb*)(output_lineptr + x * pixel_stride) = (op == EROSION)
            ? rgb{.r = 255, .g = 255, .b = 255}
            : rgb{.r = 0, .g = 0, .b = 0};
        }
    }

  for (int y = 3; y < height; ++y)
    {
      uint8_t* input_lineptr = (uint8_t*)(input + (y - 3) * stride);
      uint8_t* output_lineptr = (uint8_t*)(output + y * stride);
      for (int x = 0; x < width; ++x)
        {
          *(rgb*)(output_lineptr + x * pixel_stride) =
            *(rgb*)(input_lineptr + x * pixel_stride);
        }
    }

  //second and third lines (3/7)
  for (int i = 2; i > 0; --i)
    {
      for (int y = i; y < height; ++y)
        {
          uint8_t* input_lineptr = (uint8_t*)(input + (y - i) * stride);
          uint8_t* output_lineptr = (uint8_t*)(output + y * stride);
          for (int x = 0; x < width; ++x)
            {
              int min = x < 2 ? 0 : x - 2;
              int max = std::min(x + 3, width);
              for (int j = min; j < max; j++)
                {
                  if (op == EROSION)
                    {
                      min_assign((rgb*)(output_lineptr + x * pixel_stride),
                                 (rgb*)(input_lineptr + j * pixel_stride));
                    }
                  else
                    {
                      max_assign((rgb*)(output_lineptr + x * pixel_stride),
                                 (rgb*)(input_lineptr + j * pixel_stride));
                    }
                }
            }
        }
    }

  //middle line (4/7)
  for (int y = 0; y < height; ++y)
    {
      uint8_t* input_lineptr = (uint8_t*)(input + y * stride);
      uint8_t* output_lineptr = (uint8_t*)(output + y * stride);
      for (int x = 0; x < width; ++x)
        {
          int min = x < 3 ? 0 : x - 3;
          int max = std::min(x + 4, width);
          for (int j = min; j < max; j++)
            {
              if (op == EROSION)
                {
                  min_assign((rgb*)(output_lineptr + x * pixel_stride),
                             (rgb*)(input_lineptr + j * pixel_stride));
                }
              else
                {
                  max_assign((rgb*)(output_lineptr + x * pixel_stride),
                             (rgb*)(input_lineptr + j * pixel_stride));
                }
            }
        }
    }

  //fifth and sixth lines (3/7)
  for (int i = 1; i < 3; ++i)
    {
      for (int y = 0; y + i < height; ++y)
        {
          uint8_t* input_lineptr = (uint8_t*)(input + (y + i) * stride);
          uint8_t* output_lineptr = (uint8_t*)(output + y * stride);
          for (int x = 0; x < width; ++x)
            {
              int min = x < 2 ? 0 : x - 2;
              int max = std::min(x + 3, width);
              for (int j = min; j < max; j++)
                {
                  if (op == EROSION)
                    {
                      min_assign((rgb*)(output_lineptr + x * pixel_stride),
                                 (rgb*)(input_lineptr + j * pixel_stride));
                    }
                  else
                    {
                      max_assign((rgb*)(output_lineptr + x * pixel_stride),
                                 (rgb*)(input_lineptr + j * pixel_stride));
                    }
                }
            }
        }
    }

  // bottom line (7/7)
  for (int y = 0; y + 3 < height; ++y)
    {
      uint8_t* input_lineptr = (uint8_t*)(input + (y + 3) * stride);
      uint8_t* output_lineptr = (uint8_t*)(output + y * stride);
      for (int x = 0; x < width; ++x)
        {
          if (op == EROSION)
            {
              min_assign((rgb*)(output_lineptr + x * pixel_stride),
                         (rgb*)(input_lineptr + x * pixel_stride));
            }
          else
            {
              max_assign((rgb*)(output_lineptr + x * pixel_stride),
                         (rgb*)(input_lineptr + x * pixel_stride));
            }
        }
    }
}

void opening_impl_inplace(uint8_t* buffer, const frame_info* buffer_info)
{
  uint8_t* other_buffer =
    (uint8_t*)malloc(buffer_info->height * buffer_info->stride);
  // i ignore potential malloc failure because i can't be bothered

  morphology_impl(buffer, buffer_info, other_buffer, EROSION);

  morphology_impl(other_buffer, buffer_info, buffer, DILATION);

  free(other_buffer);
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
 * @param frame The input frame.
 * @param low_threshold The lower threshold.
 * @param high_threshold The higher threshold.
 */
void apply_hysteresis_threshold(uint8_t* buffer,
                                const frame_info* buffer_info,
                                uint8_t low_threshold,
                                uint8_t high_threshold)
{
  int width = buffer_info->width;
  int height = buffer_info->height;
  int stride = buffer_info->stride;
  int pixel_stride = buffer_info->pixel_stride;

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
                   const frame_info* buffer_info,
                   uint8_t* mask)
{
  int width = buffer_info->width;
  int height = buffer_info->height;
  int stride = buffer_info->stride;
  int pixel_stride = buffer_info->pixel_stride;

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

//******************************************************
//**                                                  **
//**                    UTILS                         **
//**                                                  **
//******************************************************

/**
 * Copy the content of a buffer to another buffer.
 * If the destination buffer is null, it will be allocated.
 * 
 * @param buffer The buffer to copy.
 * @param cpy_buffer The buffer to copy to.
 * @param buffer_info The buffer information.
 * @param mask The mask to apply.
 * @param fallback_buffer The buffer to apply for the empty pixels.

*/
static void copy_buffer(uint8_t* buffer,
                        uint8_t** cpy_buffer,
                        const frame_info* buffer_info,
                        uint8_t* mask,
                        uint8_t* fallback_buffer)
{
  int width = buffer_info->width;
  int height = buffer_info->height;
  int stride = buffer_info->stride;
  int pixel_stride = buffer_info->pixel_stride;

  if (*cpy_buffer == nullptr)
    {
      *cpy_buffer = (uint8_t*)malloc(height * stride);
    }

  if (mask == nullptr || fallback_buffer == nullptr)
    {
      // Copy buffer directly
      std::memcpy(*cpy_buffer, buffer, height * stride);
    }
  else
    {
      // Copy buffer where mask is not false, otherwise copy fallback buffer
      for (int y = 0; y < height; ++y)
        {
          for (int x = 0; x < width; ++x)
            {
              int step = y * stride + x * pixel_stride;
#define PXL_POINTER(ptr) ((rgb*)(ptr + step))
              std::memcpy(PXL_POINTER(*cpy_buffer),
                          (PXL_POINTER(mask)->r ? PXL_POINTER(buffer)
                                                : PXL_POINTER(fallback_buffer)),
                          pixel_stride);
#undef PXL_POINTER
            }
        }
    }
}