/* Copyright (C) 2017 Google Inc.
 * 
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms of the MIT 
 * license.  See the LICENSE file for details.
 */

#include "chianti/augmentors.h"

#include "fastlog.h"

#include <cstring>
#include <exception>
#include <limits>
#include <string.h>

namespace chianti {

    void CombinedAugmentor::augment(ImageTargetPair& pair) {
        for (auto i = augmentors.begin(); i != augmentors.end(); i++) {
            (*i)->augment(pair);
        }
    }

    void SubsampleAugmentor::resizeImage(cv::Mat& image) {
        auto newSize = cv::Size(image.cols / factor, image.rows / factor);
        cv::resize(image, image, newSize, 0, 0, CV_INTER_LANCZOS4);
    }

    void SubsampleAugmentor::resizeTarget(cv::Mat& target) {
        auto newSize = cv::Size(target.cols / factor, target.rows / factor);

        // Allocate the new image
        cv::Mat tNew(newSize.height, newSize.width, CV_8UC1);

        // This is the size of the region in the original image that corresponds
        // to one pixel in the subsampled image
        const int halfRegionSize = factor * factor / 2;

        // For each pixel in the resized image
        for (int i = 0; i < tNew.rows; i++) {
            for (int j = 0; j < tNew.cols; j++) {

                // Create a histogram of the target labels in the original image
                int histogram[256] = {0};

                for (int _i = i * factor; _i < (i + 1) * factor; _i++) {
                    for (int _j = j * factor; _j < (j + 1) * factor; _j++) {
                        histogram[target.at<uchar>(_i, _j)]++;
                    }
                }

                // Determine the mode of the histogram
                int mode = 0;
                for (int k = 0; k < 256; k++) {
                    if (histogram[k] > histogram[mode]) {
                        mode = k;
                    }
                }

                // Is the label sufficiently distinct?
                if (histogram[mode] > halfRegionSize) {
                    tNew.at<uchar>(i, j) = static_cast<uchar> (mode);
                } else {
                    tNew.at<uchar>(i, j) = 255;
                }
            }
        }
        tNew.copyTo(target);
    }

    void GammaAugmentor::augment(ImageTargetPair& pair) {
        // Sample the gamma value
        double gamma;
        {
            std::lock_guard<std::mutex> lock(rngMutex);
            gamma = d(g);
        }

        // Apply the non-linear transformation
        const double inv_sqrt_2 = 1.0 / std::sqrt(2.0);

        gamma = std::log(0.5 + inv_sqrt_2 * gamma) /
                std::log(0.5 - inv_sqrt_2 * gamma);

        auto float_gamma = static_cast<float> (gamma);

        // Apply the transformation to each pixel individually
        cv::pow(pair.image, float_gamma, pair.image);
    }

    void TranslationAugmentor::augment(ImageTargetPair& pair) {
        // Sample the translation offset in each direction
        int translation_x, translation_y;
        {
            std::lock_guard<std::mutex> lock(rngMutex);
            translation_x = d(g);
            translation_y = d(g);
        }

        cv::Mat iNew(pair.image.rows, pair.image.cols, CV_32FC3);
        cv::Mat tNew(pair.target.rows, pair.target.cols, CV_8UC1);

        // This only works if the two images are of the same size
        if (pair.image.rows != pair.target.rows ||
                pair.image.cols != pair.target.cols) {
            throw std::runtime_error("Image and target must be of the same "
                    "size when using translation "
                    "augmentation.");
        }

        for (int i = 0; i < pair.image.rows; i++) {
            for (int j = 0; j < pair.image.cols; j++) {

                // Compute the offset position
                int _i = i + translation_x;
                int _j = j + translation_y;
                bool outOfScore = false;

                if (_i < 0) {
                    _i = std::abs(_i);
                    outOfScore = true;
                } else if (_i >= pair.image.rows) {
                    _i = 2 * pair.image.rows - _i - 1;
                    outOfScore = true;
                }

                if (_j < 0) {
                    _j = std::abs(_j);
                    outOfScore = true;
                } else if (_j >= pair.image.cols) {
                    _j = 2 * pair.image.cols - _j - 1;
                    outOfScore = true;
                }

                iNew.at<cv::Vec3f>(i, j) = pair.image.at<cv::Vec3f>(_i, _j);
                if (!outOfScore) {
                    tNew.at<uchar>(i, j) = pair.target.at<uchar>(_i, _j);
                } else {
                    tNew.at<uchar>(i, j) = 255;
                }
            }
        }

        pair.image = iNew;
        pair.target = tNew;
    }

    void ZoomingAugmentor::augment(ImageTargetPair& pair) {
        // Sample the zooming factor
        double factor;
        {
            std::lock_guard<std::mutex> lock(rngMutex);
            factor = d(g);
        }

        // Compute the new image size
        const int rows = static_cast<int> (pair.image.rows * factor);
        const int cols = static_cast<int> (pair.image.cols * factor);

        // Resize the data
        cv::Mat iNew, tNew;
        cv::resize(pair.image,
                iNew, cv::Size(cols, rows), 0, 0, CV_INTER_LANCZOS4);
        cv::resize(pair.target, tNew, cv::Size(cols, rows), 0, 0, CV_INTER_NN);

        pair.image = cv::Mat::zeros(pair.image.rows, pair.image.cols, CV_32FC3);
        pair.target = 255 * cv::Mat::ones(
                pair.target.rows, pair.target.cols, CV_8UC1);

        // Create the final images
        // If the images are up-sampled, they have to be cropped
        // If the images are down-sampled, they have to be embedded into a 
        // bigger image
        if (factor > 1.0) {
            // Up-sample -> crop
            // Calculate the offset on both sides
            const int rowOffset = (rows - pair.image.rows) / 2;
            const int colOffset = (cols - pair.image.cols) / 2;

            for (int i = 0; i < pair.image.rows; i++) {
                for (int j = 0; j < pair.image.cols; j++) {
                    pair.image.at<cv::Vec3f>(i, j) =
                            iNew.at<cv::Vec3f>(i + rowOffset, j + colOffset);
                    pair.target.at<uchar>(i, j) =
                            tNew.at<uchar>(i + rowOffset, j + colOffset);
                }
            }
        } else {
            // Down-sample
            // Calculate the offset on both sides
            const int rowOffset = (pair.image.rows - rows) / 2;
            const int colOffset = (pair.image.cols - cols) / 2;

            for (int i = 0; i < iNew.rows; i++) {
                for (int j = 0; j < iNew.cols; j++) {
                    pair.image.at<cv::Vec3f>(i + rowOffset, j + colOffset) =
                            iNew.at<cv::Vec3f>(i, j);
                    pair.target.at<uchar>(i + rowOffset, j + colOffset) =
                            tNew.at<uchar>(i, j);
                }
            }
        }
    }

    void RotationAugmentor::augment(ImageTargetPair& pair) {
        // Sample the rotation angle
        double factor;
        {
            std::lock_guard<std::mutex> lock(rngMutex);
            factor = d(g);
        }
        
        if (factor < 0) {
            factor += 360;
        }

        const int rows = pair.image.rows;
        const int cols = pair.image.cols;

        // Create the rotation matrix
        cv::Mat img, target;
        cv::Mat M = cv::getRotationMatrix2D(
                cv::Point2f(cols / 2, rows / 2), factor, 1);

        // Rotate the image
        cv::warpAffine(pair.image, pair.image, M, cv::Size(cols, rows));
        cv::warpAffine(pair.target, pair.target, M, cv::Size(cols, rows),
                CV_INTER_NN, cv::BORDER_CONSTANT, 255);
    }

    void SaturationAugmentor::augment(ImageTargetPair& pair) {
        float offset;
        {
            std::lock_guard<std::mutex> lock(rngMutex);
            offset = static_cast<float> (d(g));
        }
        
        cv::Mat iNew;
        cv::cvtColor(pair.image, iNew, CV_RGB2HSV);

        // Adjust the saturation channel
        for (int i = 0; i < pair.image.rows; i++) {
            for (int j = 0; j < pair.image.cols; j++) {
                iNew.at<cv::Vec3f>(i, j)[1] *= offset;
                const auto value = iNew.at<cv::Vec3f>(i, j)[1];
                iNew.at<cv::Vec3f>(i, j)[1] =
                        std::max(0.0f, std::min(1.0f, value));
            }
        }

        cv::cvtColor(iNew, pair.image, CV_HSV2RGB);
    }

    void HueAugmentor::augment(ImageTargetPair& pair) {
        float offset;
        {
            std::lock_guard<std::mutex> lock(rngMutex);
            offset = static_cast<float> (d(g));
        }
        
        cv::Mat iNew;
        cv::cvtColor(pair.image, iNew, CV_RGB2HSV);

        // Adjust the hue channel
        for (int i = 0; i < pair.image.rows; i++) {
            for (int j = 0; j < pair.image.cols; j++) {
                auto value = iNew.at<cv::Vec3f>(i, j)[0];
                value += offset;

                if (value > 360) {
                    value -= 360;
                } else if (value < 0) {
                    value += 360;
                }
                iNew.at<cv::Vec3f>(i, j)[0] = value;
            }
        }

        cv::cvtColor(iNew, pair.image, CV_HSV2RGB);
    }

    inline static void updateHistgramCell(
            cv::Mat & histograms, int i, int j, int c, int _c) {
        if (_c != 255) {
            histograms.at<int>(i, j, _c)--;
        }
        if (c != 255) {
            histograms.at<int>(i, j, c)++;
        }
    }

    void CropAugmentor::computeHistogram0(
            const cv::Mat& target,
            int i,
            int j,
            cv::Mat& histograms) const {

        for (int _i = 0; _i < size; _i++) {
            for (int _j = 0; _j < size; _j++) {
                const uchar c = target.at<uchar>(_i, _j);
                if (c != 255) {
                    histograms.at<int>(i, j, c)++;
                }
            }
        }
    }

    void CropAugmentor::computeHistogram1Left(
            const cv::Mat& target,
            int i,
            int j,
            cv::Mat& histograms) const {
        // Copy the old histogram
        for (int c = 0; c < numClasses; c++) {
            histograms.at<int>(i, j, c) = histograms.at<int>(i, j - 1, c);
        }

        uchar c, _c;
        // Adjust the values using dynamic programming
        for (int __i = i; __i < i + size; __i++) {
            _c = target.at<uchar>(__i, j - 1);
            c = target.at<uchar>(__i, j + size - 1);
            updateHistgramCell(histograms, i, j, c, _c);
        }
    }

    void CropAugmentor::computeHistogram1Up(
            const cv::Mat& target,
            int i,
            int j,
            cv::Mat& histograms) const {
        // Copy the old histogram
        for (int c = 0; c < numClasses; c++) {
            histograms.at<int>(i, j, c) = histograms.at<int>(i - 1, j, c);
        }

        uchar c, _c;
        // Adjust the values using dynamic programming
        for (int __j = j; __j < j + size; __j++) {
            _c = target.at<uchar>(i - 1, __j);
            c = target.at<uchar>(i + size - 1, __j);
            updateHistgramCell(histograms, i, j, c, _c);
        }
    }
    
    void CropAugmentor::computeHistogram3(
            const cv::Mat& target, 
            int i, 
            int j, 
            cv::Mat& histograms) const {
        // Copy the old histograms
        for (int c = 0; c < numClasses; c++) {
            histograms.at<int>(i, j, c) = 
                    histograms.at<int>(i - 1, j, c) + 
                    histograms.at<int>(i, j - 1, c) - 
                    histograms.at<int>(i - 1, j - 1, c);
        }
        
        // Fix the corners of the areas
        uchar c;
        c = target.at<uchar>(i - 1, j - 1);
        if (c != 255) {
            histograms.at<int>(i, j, c)++;
        }
        
        c = target.at<uchar>(i - 1, j + size - 1);
        if (c != 255) {
            histograms.at<int>(i, j, c)--;
        }
        
        c = target.at<uchar>(i + size - 1, j - 1);
        if (c != 255) {
            histograms.at<int>(i, j, c)--;
        }
        
        c = target.at<uchar>(i + size - 1, j + size - 1);
        if (c != 255) {
            histograms.at<int>(i, j, c)++;
        }
    }

    void CropAugmentor::computeClassHistograms(
            const cv::Mat& target, cv::Mat& histograms) const {
        // Allocate memory for the class histograms
        int sizes[] = {target.rows - size, target.cols - size, numClasses};
        histograms = cv::Mat(3, sizes, CV_32S, cv::Scalar(0));

        // Initialize the histogram of the pixel in the top left corner of the
        // image
        computeHistogram0(target, 0, 0, histograms);
        
        // Compute the histograms for the first row in the image
        for (int j = 1; j < target.cols - size; j++) {
            computeHistogram1Left(target, 0, j, histograms);
        }
        
        // Compute the histograms for the remaining pixels more efficiently 
        for (int i = 1; i < target.rows - size; i++) {
            // Compute the histogram for the first pixel in the row
            computeHistogram1Up(target, i, 0, histograms);
            
            // Compute the histograms for the remaining pixels in the row
            // based on three previously computed histograms
            for (int j = 1; j < target.cols - size; j++) {
                computeHistogram3(target, i, j, histograms);
            }
        }
    }
    
    void CropAugmentor::computeCumulativeDistribution(
            const cv::Mat& histograms, cv::Mat& distribution) const {
        // Compute the entropy per image pixel
        cv::Mat entropies(
                histograms.size[0], 
                histograms.size[1], 
                CV_32F, 
                cv::Scalar(0.0f));
        
        float sum = 0.0f;
        const float n = size * size;
        for (int i = 0; i < entropies.rows; i++) {
            for (int j = 0; j < entropies.cols; j++) {
                float entropy = 0.0f;
                float m = 0.0f;
                for (int c = 0; c < numClasses; c++) {
                    const float value = histograms.at<int>(i, j, c);
                    m += value;
                    if (value > 0) 
                        entropy -= value * fastlog2(value);
                }
                if (m > 0) {
                    entropy += m * fastlog2(m);
                    // Rescale all pixels by the same factor. This ensures that
                    // Really interesting regions get a high score.
                    entropy /= n;
                }
                entropies.at<float>(i, j) = entropy;
                sum += entropy;
            }
        }
        
        // Compute the cumulative distribution
        distribution = entropies.reshape(1, 1);
        float previous = 0.0f;
        for (int k = 0; k < distribution.cols; k++) {
            distribution.at<float>(k) /= sum;
            distribution.at<float>(k) += previous;
            previous = distribution.at<float>(k);
        }
    }
    
    std::array<int, 2> CropAugmentor::samplePosition(
            const cv::Mat& target, 
            const cv::Mat& distribution) {
        // Sample a random number
        float u;
        {
            std::lock_guard<std::mutex> lock(rngMutex);
            u = d(g);
        }
        
        // Get the index of the cumulative distribution that corresponds to the
        // drawn number
        float previous = 0.0f;
        int k;
        for (k = 0; k < distribution.cols; k++) {
            if (previous <= u && u < distribution.at<float>(k)) {
                break;
            } else {
                previous = distribution.at<float>(k);
            }
        }
        
        // Reconstruct the row and column index from the distribution index
        int row = k / (target.cols - size);
        int col = k - row * (target.cols - size);
        
        return {row, col};
    }
    
    void CropAugmentor::augment(ImageTargetPair& pair) {
        // Compute the pixel-wise class histograms
        cv::Mat histograms;
        computeClassHistograms(pair.target, histograms);
        
        // Compute the cumulative pixel distribution based on the the class
        // entropies
        cv::Mat distribution;
        computeCumulativeDistribution(histograms, distribution);
        
        // Sample a crop position from the distribution
        auto position = samplePosition(pair.target, distribution);
        
        // Extract the crop
        cv::Mat iNew(size, size, CV_32FC3);
        cv::Mat tNew(size, size, CV_8UC1);
        for (int i = position[0]; i < position[0] + size; i++) {
            for (int j = position[1]; j < position[1] + size; j++) {
                iNew.at<cv::Vec3f>(i - position[0], j - position[1]) = 
                        pair.image.at<cv::Vec3f>(i, j);
                tNew.at<uchar>(i - position[0], j - position[1]) = 
                        pair.target.at<uchar>(i, j);
            }
        }
        
        iNew.copyTo(pair.image);
        tNew.copyTo(pair.target);
    }
    
} // namespace chianti