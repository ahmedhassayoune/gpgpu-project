#include "morphology_impl.h"
#include <stdlib.h>
#include <algorithm>

struct rgb {
    uint8_t r, g, b;
};

extern "C" {

    static void min_assign(rgb *lhs, rgb * rhs)
    {
        *lhs = rgb { .r = std::min(lhs->r, lhs->r), .g = std::min(lhs->g, lhs->g), .b = std::min(lhs->b, lhs->b)};
    }

    static void max_assign(rgb *lhs, rgb * rhs)
    {
        *lhs = rgb { .r = std::max(lhs->r, lhs->r), .g = std::max(lhs->g, lhs->g), .b = std::max(lhs->b, lhs->b)};
    }


    void opening_impl(uint8_t* buffer, int width, int height, int stride, int pixel_stride)
    {
        uint8_t* other_buffer = (uint8_t*) malloc(width * stride);
        // i ignore potential malloc failure because i can't be bothered


        /*
        ############################################################
        #########################  EROSION #########################
        ############################################################

        other_buffer becomes the erosion of buffer
        */

        uint8_t* input = buffer;
        uint8_t* output = other_buffer;

       // top line (1/7)
        for (int y = 0; y < height && y < 3; ++y)
        {
            uint8_t* output_lineptr = (uint8_t*) (output + y * stride);
            for (int x = 0; x < width; ++x)
            {
                *(rgb *) (output_lineptr + x * pixel_stride) = rgb { .r = 255, .g = 255, .b = 255 };
            }
        }

        for (int y = 3; y < height; ++y)
        {
            uint8_t* input_lineptr = (uint8_t*) (input + (y - 3) * stride);
            uint8_t* output_lineptr = (uint8_t*) (output + y * stride);
            for (int x = 0; x < width; ++x)
            {
                *(rgb *) (output_lineptr + x * pixel_stride) = *(rgb *) (input_lineptr + x * pixel_stride);
            }
        }

        //second and third lines (3/7)
        for (int i = 2; i > 0; --i)
        {
            for (int y = i; y < height; ++y)
            {
                uint8_t* input_lineptr = (uint8_t*) (input + (y - i) * stride);
                uint8_t* output_lineptr = (uint8_t*) (output + y * stride);
                for (int x = 0; x < width; ++x)
                {
                    int min = x < 2 ? 0 : x -2;
                    int max = std::min(x + 3, width);
                    for (int j = min; j < max; j++)
                    {
                        min_assign(
                            (rgb *) (output_lineptr + x * pixel_stride), 
                            (rgb *) (input_lineptr + j * pixel_stride)
                            );
                    }
                }
            }
        }

        //middle line (4/7)
        for (int y = 0; y < height; ++y)
        {
            uint8_t* input_lineptr = (uint8_t*) (input + y * stride);
            uint8_t* output_lineptr = (uint8_t*) (output + y * stride);
            for (int x = 0; x < width; ++x)
            {
                int min = x < 3 ? 0 : x -3;
                int max = std::min(x + 4, width);
                for (int j = min; j < max; j++)
                {
                    min_assign(
                        (rgb *) (output_lineptr + x * pixel_stride), 
                        (rgb *) (input_lineptr + j * pixel_stride)
                        );
                }
            }
        }

        //fifth and sixth lines (3/7)
        for (int i = 1; i < 3; ++i)
        {
            for (int y = i; y < height; ++y)
            {
                uint8_t* input_lineptr = (uint8_t*) (input + (y + i) * stride);
                uint8_t* output_lineptr = (uint8_t*) (output + y * stride);
                for (int x = 0; x < width; ++x)
                {
                    int min = x < 2 ? 0 : x -2;
                    int max = std::min(x + 3, width);
                    for (int j = min; j < max; j++)
                    {
                        min_assign(
                            (rgb *) (output_lineptr + x * pixel_stride), 
                            (rgb *) (input_lineptr + j * pixel_stride)
                            );
                    }
                }
            }
        }

        // bottom line (7/7)
        for (int y = 0; y + 3 < height; ++y)
        {
            uint8_t* input_lineptr = (uint8_t*) (input + (y + 3) * stride);
            uint8_t* output_lineptr = (uint8_t*) (output + y * stride);
            for (int x = 0; x < width; ++x)
            {
                min_assign(
                            (rgb *) (output_lineptr + x * pixel_stride), 
                            (rgb *) (input_lineptr + x * pixel_stride)
                            );
            }
        }

        

        /*
        ##############################################################
        #########################  DILATION  #########################
        ##############################################################

        buffer becomes the dilation of other_buffer
        */

        input = other_buffer;
        output = buffer;

       // top line (1/7)
        for (int y = 0; y < height && y < 3; ++y)
        {
            uint8_t* output_lineptr = (uint8_t*) (output + y * stride);
            for (int x = 0; x < width; ++x)
            {
                *(rgb *) (output_lineptr + x * pixel_stride) = rgb { .r = 0, .g = 0, .b = 0 };
            }
        }

        for (int y = 3; y < height; ++y)
        {
            uint8_t* input_lineptr = (uint8_t*) (input + (y - 3) * stride);
            uint8_t* output_lineptr = (uint8_t*) (output + y * stride);
            for (int x = 0; x < width; ++x)
            {
                *(rgb *) (output_lineptr + x * pixel_stride) = *(rgb *) (input_lineptr + x * pixel_stride);
            }
        }

        //second and third lines (3/7)
        for (int i = 2; i > 0; --i)
        {
            for (int y = i; y < height; ++y)
            {
                uint8_t* input_lineptr = (uint8_t*) (input + (y - i) * stride);
                uint8_t* output_lineptr = (uint8_t*) (output + y * stride);
                for (int x = 0; x < width; ++x)
                {
                    int min = x < 2 ? 0 : x -2;
                    int max = std::min(x + 3, width);
                    for (int j = min; j < max; j++)
                    {
                        max_assign(
                            (rgb *) (output_lineptr + x * pixel_stride), 
                            (rgb *) (input_lineptr + j * pixel_stride)
                            );
                    }
                }
            }
        }

        //middle line (4/7)
        for (int y = 0; y < height; ++y)
        {
            uint8_t* input_lineptr = (uint8_t*) (input + y * stride);
            uint8_t* output_lineptr = (uint8_t*) (output + y * stride);
            for (int x = 0; x < width; ++x)
            {
                int min = x < 3 ? 0 : x -3;
                int max = std::min(x + 4, width);
                for (int j = min; j < max; j++)
                {
                    max_assign(
                        (rgb *) (output_lineptr + x * pixel_stride), 
                        (rgb *) (input_lineptr + j * pixel_stride)
                        );
                }
            }
        }

        //fifth and sixth lines (3/7)
        for (int i = 1; i < 3; ++i)
        {
            for (int y = i; y < height; ++y)
            {
                uint8_t* input_lineptr = (uint8_t*) (input + (y + i) * stride);
                uint8_t* output_lineptr = (uint8_t*) (output + y * stride);
                for (int x = 0; x < width; ++x)
                {
                    int min = x < 2 ? 0 : x -2;
                    int max = std::min(x + 3, width);
                    for (int j = min; j < max; j++)
                    {
                        max_assign(
                            (rgb *) (output_lineptr + x * pixel_stride), 
                            (rgb *) (input_lineptr + j * pixel_stride)
                            );
                    }
                }
            }
        }

        // bottom line (7/7)
        for (int y = 0; y + 3 < height; ++y)
        {
            uint8_t* input_lineptr = (uint8_t*) (input + (y + 3) * stride);
            uint8_t* output_lineptr = (uint8_t*) (output + y * stride);
            for (int x = 0; x < width; ++x)
            {
                max_assign(
                            (rgb *) (output_lineptr + x * pixel_stride), 
                            (rgb *) (input_lineptr + x * pixel_stride)
                            );
            }
        }
        free(other_buffer);

    }   
}