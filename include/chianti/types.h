/* Copyright (C) 2017 Google Inc.
 * 
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms of the MIT 
 * license.  See the LICENSE file for details.
 */

#ifndef CHIANTI_TYPES_H
#define CHIANTI_TYPES_H

#include <cstring>
#include <cstdlib>
#include <array>
#include <memory>

#include <opencv2/opencv.hpp>

namespace chianti {

    /**
     * Holds the filename of the image and the target image.
     */
    struct FilenamePair {
        std::string image;
        std::string target;
    };

    /**
     * Holds the source image and the target image. The source image is an RGB
     * image and the target is a 1-channel 8-bit image.
     */
    struct ImageTargetPair {
        cv::Mat image;
        cv::Mat target;
    };

    /**
     * A tensor that stores its values in a row major ordering.
     */
    template<typename T, int Rank>
    class Tensor {
    public:
        /**
         * Initializes a new instance of the Tensor class.
         */
        Tensor() : Tensor(std::array<int, Rank>()) {
        }

        /**
         * Initializes a new instance of the Tensor class.
         * 
         * @param _shape The shape of the tensor.
         */
        Tensor(const std::array<int, Rank> & _shape) : shape(_shape) {
            data.resize(getSize());
        }

        /**
         * Copies the tensor.
         */
        Tensor(const Tensor<T, Rank> & other) {
            shape = other.shape;
            data.resize(getSize());
            data.insert(data.begin(), other.data.begin(), other.data.end());
        }

        /**
         * Assignment operator.
         */
        Tensor & operator=(const Tensor & other) {
            if (this != &other) {
                reshape(other.shape);
                data.insert(data.begin(), other.data.begin(), other.data.end());
            }
            return *this;
        }

        /**
         * Returns the size of the tensor.
         * 
         * @return The total size of the tensor.
         */
        int getSize() const {
            int size = 1;
            for (int r = 0; r < Rank; r++) {
                size *= shape[r];
            }
            return size;
        }
        
        /**
         * Reshapes the tensor. Data may get corrupted.
         * 
         * @param newShape The new shape of the tensor.
         */
        void reshape(const std::array<int, Rank> & newShape) {
            shape = newShape;
            data.resize(getSize());
        }

        /**
         * Fills the tensor with a constant value.
         * 
         * @param val The value to fill the tensor with.
         */
        void fill(T val) {
            std::fill(data.begin(), data.end(), val);
        }
        
        std::array<int, Rank> shape;
        std::vector<T> data;
    };

    /**
     * A batch of images and targets. 
     */
    class Batch {
    public:
        /**
         * Initializes a new instance of the Batch class.
         */
        Batch() {}
        
        /**
         * Initializes a new instance of the Batch class.
         * 
         * @param imagesShape The shape of the images tensor.
         * @param targetsShape The shape of the targets tensor.
         */
        Batch(
                const std::array<int, 4> & imagesShape,
                const std::array<int, 4> & targetsShape) :
        images(imagesShape),
        targets(targetsShape) {}
        
        Tensor<float, 4> images;
        Tensor<float, 4> targets;
    };

} // namespace chianti

#endif

