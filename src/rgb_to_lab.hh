#pragma once 

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>



struct Pixel {
    double r, g, b;
};

struct LAB {
    double l, a, b;
};


void rgbToXyz(float r, float g, float b, float& x, float& y, float& z);
void xyzToLab(float x, float y, float z, LAB& lab);
float labDistance(const LAB& lab1, const LAB& lab2);
