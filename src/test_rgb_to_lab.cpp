#include <iostream>
#include <cmath>
#include <cassert>
#include <opencv2/opencv.hpp>
#include "rgb_to_lab.hh"

void convertRgbToLabWithOpenCV(float r, float g, float b, LAB& lab) {
    cv::Mat rgb(1, 1, CV_8UC3, cv::Scalar(b, g, r)); // OpenCV uses BGR format
    cv::Mat labMat;
    cv::cvtColor(rgb, labMat, cv::COLOR_BGR2Lab);

    cv::Vec3b labVec = labMat.at<cv::Vec3b>(0, 0);
    lab.l = labVec[0] * 100.0f / 255.0f; // Scale L from [0, 255] to [0, 100]
    lab.a = labVec[1] - 128.0f;
    lab.b = labVec[2] - 128.0f;
}

void testRgbToLabConversion() {
    float r = 255.0f, g = 0.0f, b = 0.0f;
    float x, y, z;
    LAB myLab, opencvLab;

    rgbToXyz(r, g, b, x, y, z);
    std::cout << "My XYZ: X = " << x << ", Y = " << y << ", Z = " << z << std::endl;

    cv::Mat rgb(1, 1, CV_32FC3, cv::Scalar(b / 255.0f, g / 255.0f, r / 255.0f)); // OpenCV uses BGR format
    cv::Mat xyzMat;
    cv::cvtColor(rgb, xyzMat, cv::COLOR_BGR2XYZ);

    cv::Vec3f xyzVec = xyzMat.at<cv::Vec3f>(0, 0);
    float opencvX = xyzVec[0];
    float opencvY = xyzVec[1];
    float opencvZ = xyzVec[2];

    std::cout << "OpenCV XYZ: X = " << opencvX << ", Y = " << opencvY << ", Z = " << opencvZ << std::endl;

    xyzToLab(x, y, z, myLab);

    convertRgbToLabWithOpenCV(r, g, b, opencvLab);

    std::cout << "My Lab: L = " << myLab.l << ", a = " << myLab.a << ", b = " << myLab.b << std::endl;
    std::cout << "OpenCV Lab: L = " << opencvLab.l << ", a = " << opencvLab.a << ", b = " << opencvLab.b << std::endl;

    const float tolerance = 0.1f;
    assert(fabs(myLab.l - opencvLab.l) < tolerance);
    assert(fabs(myLab.a - opencvLab.a) < tolerance);
    assert(fabs(myLab.b - opencvLab.b) < tolerance);

    std::cout << "RGB to Lab conversion test passed!" << std::endl;
}

int main() {
    testRgbToLabConversion();
    std::cout << "All tests passed!" << std::endl;
    return 0;
}
