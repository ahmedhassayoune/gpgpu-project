#include "filter_impl.h"

#include <chrono>
#include <thread>
#include "logo.h"

struct rgb {
    uint8_t r, g, b;
};

extern "C" {
    void filter_impl(uint8_t* buffer, int width, int height, int stride, int pixel_stride)
    {
        for (int y = 0; y < height; ++y)
        {
            uint8_t* lineptr = (uint8_t*) (buffer + y * stride);
            for (int x = 0; x < width; ++x)
            {
                rgb pxl = *(rgb *) (lineptr + x * pixel_stride);
                pxl.r = 0; // Back out red component

                if (x < logo_width && y < logo_height)
                {
                    float alpha = logo_data[y * logo_width + x] / 255.f;
                    pxl.g = uint8_t(alpha * pxl.g + (1-alpha) * 255);
                    pxl.b = uint8_t(alpha * pxl.b + (1-alpha) * 255);

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
