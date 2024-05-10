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

void rgbToXyz(double r, double g, double b, double& x, double& y, double& z);
void xyzToLab(double x, double y, double z, LAB& lab);
double labDistance(const LAB& lab1, const LAB& lab2);
