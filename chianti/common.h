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

    /**
     * This exception is thrown when an error in the library occurs
     */
    class RuntimeException : public std::exception {
    public:
        RuntimeException(const char* message) : message(message) {}

        const char* what() const throw()
        {
            return message;
        }

    private:
        /**
         * A pointer to an error message.
         */
        const char* message;
    };

}