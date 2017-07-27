/* Copyright (C) 2017 Google Inc.
 * 
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms of the MIT 
 * license.  See the LICENSE file for details.
 */

#include "chianti/loaders.h"

#include <exception>
#include <sstream>

namespace chianti {

    cv::Mat BaseLoader::_load(const std::string& filename, bool color) const {
        cv::Mat result = cv::imread(filename, color);

        // Was the image loaded correctly?
        if (!result.data) {
            // Nope
            std::stringstream error;
            error << "Could not load image '" << filename << "'.";
            throw std::runtime_error(error.str());
        }

        return result;
    }

    cv::Mat RGBLoader::load(const std::string& filename) const {
        cv::Mat image = _load(filename, true);

        // Convert image to [0, 1] floating point
        cv::Mat result;
        image.convertTo(result, CV_32FC3);
        result /= 255.0f;
        
        // Convert to RGB
        cv::cvtColor(result, result, CV_BGR2RGB);

        return result;
    }

    cv::Mat LabelLoader::load(const std::string& filename) const {
        return _load(filename, false);
    }

    cv::Mat ValueMapperLoader::load(const std::string& filename) const {
        // Load the image
        cv::Mat result = _load(filename, false);

        // Map the values
        for (int i = 0; i < result.rows; i++) {
            for (int j = 0; j < result.cols; j++) {
                result.at<uchar>(i, j) = valueMap[result.at<uchar>(i, j)];
            }
        }

        return result;
    }

    cv::Mat ColorMapperLoader::load(const std::string& filename) const {
        // Load the image
        cv::Mat image = _load(filename, true);
        cv::Mat result(image.rows, image.cols, CV_8UC1);

        // Map the colors
        for (int i = 0; i < result.rows; i++) {
            for (int j = 0; j < result.cols; j++) {
                const cv::Vec3b value = image.at<cv::Vec3b>(i, j);

                // Throw an error if this is an unknown color
                if (colorMap.find(value) == colorMap.end()) {
                    std::stringstream error;
                    error << "Unknown color (" << value[0] << ", " << value[1]
                            << ", " << value[2] << ") in image '" << filename
                            << "'.";
                    throw std::runtime_error(error.str());
                }

                result.at<uchar>(i, j) = colorMap.at(value);
            }
        }

        return result;
    }

    ImageTargetPair ImageTargetPairLoader::load(
            IteratorInterface::ElementIterator filenames) const {
        ImageTargetPair result = {
            imageLoader->load(filenames->image),
            targetLoader->load(filenames->target)
        };
        return result;
    }

} // namespace chianti