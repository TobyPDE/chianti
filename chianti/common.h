#pragma once

#include <opencv2/opencv.hpp>

namespace Chianti {
    /**
     * This struct holds a pair of image and label image.
     */
    struct ImageLabelPair {
        cv::Mat img;
        cv::Mat target;
    };
}