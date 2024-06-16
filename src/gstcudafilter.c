/* GStreamer
 * Copyright (C) 2023 FIXME <fixme@example.com>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 51 Franklin Street, Suite 500,
 * Boston, MA 02110-1335, USA.
 */
/**
 * SECTION:element-gstcudafilter
 *
 * The cudafilter element does FIXME stuff.
 *
 * <refsect2>
 * <title>Example launch line</title>
 * |[
 * gst-launch-1.0 -v fakesrc ! cudafilter ! FIXME ! fakesink
 * ]|
 * FIXME Describe what the pipeline does.
 * </refsect2>
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <gst/gst.h>
#include <gst/video/gstvideofilter.h>
#include <gst/video/video.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "filter_impl.h"
#include "gstcudafilter.h"

GST_DEBUG_CATEGORY_STATIC(gst_cuda_filter_debug_category);
#define GST_CAT_DEFAULT gst_cuda_filter_debug_category

/* prototypes */
static uint8_t* load_ppm_image(const char* filename,
                               const struct frame_info* f_info);

static void gst_cuda_filter_set_property(GObject* object,
                                         guint property_id,
                                         const GValue* value,
                                         GParamSpec* pspec);
static void gst_cuda_filter_get_property(GObject* object,
                                         guint property_id,
                                         GValue* value,
                                         GParamSpec* pspec);
static void gst_cuda_filter_dispose(GObject* object);
static void gst_cuda_filter_finalize(GObject* object);

static gboolean gst_cuda_filter_start(GstBaseTransform* trans);
static gboolean gst_cuda_filter_stop(GstBaseTransform* trans);
static gboolean gst_cuda_filter_set_info(GstVideoFilter* filter,
                                         GstCaps* incaps,
                                         GstVideoInfo* in_info,
                                         GstCaps* outcaps,
                                         GstVideoInfo* out_info);

//static GstFlowReturn gst_cuda_filter_transform_frame (GstVideoFilter * filter, GstVideoFrame * inframe, GstVideoFrame * outframe);
static GstFlowReturn gst_cuda_filter_transform_frame_ip(GstVideoFilter* filter,
                                                        GstVideoFrame* frame);

enum
{
  PROP_0,
  PROP_BG,
  PROP_TH_LOW,
  PROP_TH_HIGH,
  PROP_BG_SAMPLING_RATE,
  PROP_BG_NUMBER_FRAMES
};

static uint8_t* load_ppm_image(const char* filename,
                               const struct frame_info* f_info)
{
  FILE* fp = fopen(filename, "rb");
  if (!fp)
    {
      fprintf(stderr, "Failed to open file: %s\n", filename);
      return NULL;
    }

  char header[256];
  if (fgets(header, sizeof(header), fp) == NULL)
    {
      fprintf(stderr, "Error reading header from file\n");
      fclose(fp);
      return NULL;
    }
  if (strncmp(header, "P6", 2) != 0)
    {
      fclose(fp);
      fprintf(stderr, "Not a valid PPM file\n");
      return NULL;
    }

  // Ignore comments
  char c;
  while ((c = fgetc(fp)) == '#')
    {
      while ((c = fgetc(fp)) != '\n')
        ;
    }
  ungetc(c, fp);

  int max_val, width, height;
  int num_items = fscanf(fp, "%d %d\n%d\n", &width, &height, &max_val);
  if (num_items != 3)
    {
      fprintf(stderr, "Error reading width, height, and max value from file\n");
      fclose(fp);
      return NULL;
    }
  if (max_val != 255)
    {
      fclose(fp);
      fprintf(stderr, "Only 8-bit PPM images are supported\n");
      return NULL;
    }

  if (width != f_info->width || height != f_info->height)
    {
      fclose(fp);
      fprintf(stderr, "Image dimensions do not match frame dimensions\n");
      return NULL;
    }

  uint8_t* image_data = (uint8_t*)malloc(f_info->stride * (height));
  if (!image_data)
    {
      fclose(fp);
      fprintf(stderr, "Failed to allocate memory for image data\n");
      return NULL;
    }

  // Real image is not strided but we need to copy it to a strided buffer
  for (int i = 0; i < height; i++)
    {
      size_t bytes_to_read = width * 3;
      if (fread(image_data + i * f_info->stride, 1, width * 3, fp)
          != bytes_to_read)
        {
          fclose(fp);
          free(image_data);
          fprintf(stderr, "Error reading image data from file\n");
          return NULL;
        }
    }

  fclose(fp);

  return image_data;
}

/* pad templates */

/* FIXME: add/remove formats you can handle */
#define VIDEO_SRC_CAPS GST_VIDEO_CAPS_MAKE("{ RGB }")

/* FIXME: add/remove formats you can handle */
#define VIDEO_SINK_CAPS GST_VIDEO_CAPS_MAKE("{ RGB }")

/* class initialization */

G_DEFINE_TYPE_WITH_CODE(
  GstCudaFilter,
  gst_cuda_filter,
  GST_TYPE_VIDEO_FILTER,
  GST_DEBUG_CATEGORY_INIT(gst_cuda_filter_debug_category,
                          "cudafilter",
                          0,
                          "debug category for cudafilter element"));

static void gst_cuda_filter_class_init(GstCudaFilterClass* klass)
{
  GObjectClass* gobject_class = G_OBJECT_CLASS(klass);
  GstBaseTransformClass* base_transform_class = GST_BASE_TRANSFORM_CLASS(klass);
  GstVideoFilterClass* video_filter_class = GST_VIDEO_FILTER_CLASS(klass);

  /* Setting up pads and setting metadata should be moved to
     base_class_init if you intend to subclass this class. */
  gst_element_class_add_pad_template(
    GST_ELEMENT_CLASS(klass),
    gst_pad_template_new("src", GST_PAD_SRC, GST_PAD_ALWAYS,
                         gst_caps_from_string(VIDEO_SRC_CAPS)));
  gst_element_class_add_pad_template(
    GST_ELEMENT_CLASS(klass),
    gst_pad_template_new("sink", GST_PAD_SINK, GST_PAD_ALWAYS,
                         gst_caps_from_string(VIDEO_SINK_CAPS)));

  gst_element_class_set_static_metadata(
    GST_ELEMENT_CLASS(klass), "FIXME Long name", "Generic", "FIXME Description",
    "FIXME <fixme@example.com>");

  gobject_class->set_property = gst_cuda_filter_set_property;
  gobject_class->get_property = gst_cuda_filter_get_property;
  gobject_class->dispose = gst_cuda_filter_dispose;
  gobject_class->finalize = gst_cuda_filter_finalize;
  base_transform_class->start = GST_DEBUG_FUNCPTR(gst_cuda_filter_start);
  base_transform_class->stop = GST_DEBUG_FUNCPTR(gst_cuda_filter_stop);
  video_filter_class->set_info = GST_DEBUG_FUNCPTR(gst_cuda_filter_set_info);
  //video_filter_class->transform_frame = GST_DEBUG_FUNCPTR (gst_cuda_filter_transform_frame);
  video_filter_class->transform_frame_ip =
    GST_DEBUG_FUNCPTR(gst_cuda_filter_transform_frame_ip);

  g_object_class_install_property(
    gobject_class, PROP_BG,
    g_param_spec_string("bg", "Background Image", "URI to background image", "",
                        G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property(
    gobject_class, PROP_TH_LOW,
    g_param_spec_int("th-low", "Threshold Low", "Low threshold value", 0, 255,
                     3, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property(
    gobject_class, PROP_TH_HIGH,
    g_param_spec_int("th-high", "Threshold High", "High threshold value", 0,
                     255, 30, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property(
    gobject_class, PROP_BG_SAMPLING_RATE,
    g_param_spec_int("bg-sampling-rate", "Background Sampling Rate",
                     "Sampling rate for background estimation (in ms)", 0,
                     10000, 500, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property(
    gobject_class, PROP_BG_NUMBER_FRAMES,
    g_param_spec_int("bg-number-frames", "Background Number Frames",
                     "Number of frames used for background estimation", 0, 100,
                     10, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
}

static void gst_cuda_filter_init(GstCudaFilter* cudafilter)
{
  cudafilter->bg = g_strdup("");
  cudafilter->th_low = 3;
  cudafilter->th_high = 30;
  cudafilter->bg_sampling_rate = 500;
  cudafilter->bg_number_frames = 10;
}

void gst_cuda_filter_set_property(GObject* object,
                                  guint property_id,
                                  const GValue* value,
                                  GParamSpec* pspec)
{
  GstCudaFilter* cudafilter = GST_CUDA_FILTER(object);

  GST_DEBUG_OBJECT(cudafilter, "set_property");

  switch (property_id)
    {
    case PROP_BG:
      g_free(cudafilter->bg);
      cudafilter->bg = g_value_dup_string(value);
      break;
    case PROP_TH_LOW:
      cudafilter->th_low = g_value_get_int(value);
      break;
    case PROP_TH_HIGH:
      cudafilter->th_high = g_value_get_int(value);
      break;
    case PROP_BG_SAMPLING_RATE:
      cudafilter->bg_sampling_rate = g_value_get_int(value);
      break;
    case PROP_BG_NUMBER_FRAMES:
      cudafilter->bg_number_frames = g_value_get_int(value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID(object, property_id, pspec);
      break;
    }
}

void gst_cuda_filter_get_property(GObject* object,
                                  guint property_id,
                                  GValue* value,
                                  GParamSpec* pspec)
{
  GstCudaFilter* cudafilter = GST_CUDA_FILTER(object);

  GST_DEBUG_OBJECT(cudafilter, "get_property");

  switch (property_id)
    {
    case PROP_BG:
      g_value_set_string(value, cudafilter->bg);
      break;
    case PROP_TH_LOW:
      g_value_set_int(value, cudafilter->th_low);
      break;
    case PROP_TH_HIGH:
      g_value_set_int(value, cudafilter->th_high);
      break;
    case PROP_BG_SAMPLING_RATE:
      g_value_set_int(value, cudafilter->bg_sampling_rate);
      break;
    case PROP_BG_NUMBER_FRAMES:
      g_value_set_int(value, cudafilter->bg_number_frames);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID(object, property_id, pspec);
      break;
    }
}

void gst_cuda_filter_dispose(GObject* object)
{
  GstCudaFilter* cudafilter = GST_CUDA_FILTER(object);

  GST_DEBUG_OBJECT(cudafilter, "dispose");

  /* clean up as possible.  may be called multiple times */

  G_OBJECT_CLASS(gst_cuda_filter_parent_class)->dispose(object);
}

void gst_cuda_filter_finalize(GObject* object)
{
  GstCudaFilter* cudafilter = GST_CUDA_FILTER(object);

  GST_DEBUG_OBJECT(cudafilter, "finalize");

  g_free(cudafilter->bg);

  G_OBJECT_CLASS(gst_cuda_filter_parent_class)->finalize(object);
}

static gboolean gst_cuda_filter_start(GstBaseTransform* trans)
{
  GstCudaFilter* cudafilter = GST_CUDA_FILTER(trans);

  GST_DEBUG_OBJECT(cudafilter, "start");

  return TRUE;
}

static gboolean gst_cuda_filter_stop(GstBaseTransform* trans)
{
  GstCudaFilter* cudafilter = GST_CUDA_FILTER(trans);

  GST_DEBUG_OBJECT(cudafilter, "stop");

  return TRUE;
}

static gboolean gst_cuda_filter_set_info(GstVideoFilter* filter,
                                         GstCaps* incaps,
                                         GstVideoInfo* in_info,
                                         GstCaps* outcaps,
                                         GstVideoInfo* out_info)
{
  GstCudaFilter* cudafilter = GST_CUDA_FILTER(filter);

  GST_DEBUG_OBJECT(cudafilter, "set_info");

  return TRUE;
}

/* transform */
/* Uncomment if you want a transform not inplace

static GstFlowReturn
gst_cuda_filter_transform_frame (GstVideoFilter * filter, GstVideoFrame * inframe,
    GstVideoFrame * outframe)
{
  GstCudaFilter *cudafilter = GST_CUDA_FILTER (filter);

  GST_DEBUG_OBJECT (cudafilter, "transform_frame");

  return GST_FLOW_OK;
}
*/

static GstFlowReturn gst_cuda_filter_transform_frame_ip(GstVideoFilter* filter,
                                                        GstVideoFrame* frame)
{
  GstCudaFilter* cudafilter = GST_CUDA_FILTER(filter);

  GST_DEBUG_OBJECT(cudafilter, "transform_frame_ip");

  int width = GST_VIDEO_FRAME_COMP_WIDTH(frame, 0);
  int height = GST_VIDEO_FRAME_COMP_HEIGHT(frame, 0);

  uint8_t* pixels = GST_VIDEO_FRAME_PLANE_DATA(frame, 0);
  int plane_stride = GST_VIDEO_FRAME_PLANE_STRIDE(frame, 0);
  int pixel_stride = GST_VIDEO_FRAME_COMP_PSTRIDE(frame, 0);

  // Get the frame timestamp
  double frame_timestamp = 0.0;
  GstBuffer* buffer = frame->buffer;
  if (buffer)
    {
      GstClockTime timestamp = GST_BUFFER_PTS(buffer);
      frame_timestamp = (double)timestamp / GST_MSECOND;
    }

  // Set frame info
  struct frame_info f_info = {width, height, plane_stride, pixel_stride,
                              frame_timestamp};

  // Load background image
  static uint8_t* user_bg = NULL;
  if (cudafilter->bg[0] != '\0' && user_bg == NULL)
    {
      user_bg = load_ppm_image(cudafilter->bg, &f_info);
      if (!user_bg)
        {
          return GST_FLOW_ERROR;
        }
    }

  // Set filter params
  struct filter_params f_params = {
    user_bg, cudafilter->th_low, cudafilter->th_high,
    cudafilter->bg_sampling_rate, cudafilter->bg_number_frames};

  // Apply filter
  filter_impl(pixels, &f_info, &f_params);

  return GST_FLOW_OK;
}

static gboolean plugin_init(GstPlugin* plugin)
{
  /* FIXME Remember to set the rank if it's an element that is meant
     to be autoplugged by decodebin. */
  return gst_element_register(plugin, "cudafilter", GST_RANK_NONE,
                              GST_TYPE_CUDA_FILTER);
}

/* FIXME: these are normally defined by the GStreamer build system.
   If you are creating an element to be included in gst-plugins-*,
   remove these, as they're always defined.  Otherwise, edit as
   appropriate for your external plugin package. */
#ifndef VERSION
#  define VERSION "0.0.FIXME"
#endif
#ifndef PACKAGE
#  define PACKAGE "FIXME_package"
#endif
#ifndef PACKAGE_NAME
#  define PACKAGE_NAME "FIXME_package_name"
#endif
#ifndef GST_PACKAGE_ORIGIN
#  define GST_PACKAGE_ORIGIN "http://FIXME.org/"
#endif

GST_PLUGIN_DEFINE(GST_VERSION_MAJOR,
                  GST_VERSION_MINOR,
                  cudafilter,
                  "FIXME plugin description",
                  plugin_init,
                  VERSION,
                  "LGPL",
                  PACKAGE_NAME,
                  GST_PACKAGE_ORIGIN)
