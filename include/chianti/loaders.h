/* Copyright (C) 2017 Google Inc.
 * 
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms of the MIT 
 * license.  See the LICENSE file for details.
 */

#ifndef CHIANTI_LOADERS_H
#define CHIANTI_LOADERS_H

#include <opencv2/opencv.hpp>

#include <array>
#include <memory>
#include <string>
#include <unordered_map>

#include "types.h"
#include "iterators.h"

namespace std {

    template <>
    struct hash<cv::Vec3b> {
        /**
         * In order to use cv::Vec3b as a dictionary key, we have to implement
         * a hash function.
         * 
         * @param value A cv::Vec3b instance.
         * @return The computed hash code.
         */
        std::size_t operator()(const cv::Vec3b & value) const {
            std::size_t hash = 0;
            for (int i = 0; i < 3; i++) {
                hash = value[i] ^ 31 * hash;
            }
            return hash;
        }
    };
}

namespace chianti {

    /**
     * This class takes a filename and produces and returns an opencv image. 
     */
    class LoaderInterface {
    public:
        /**
         * Takes a filename and returns an image.
         * 
         * @param filename The image file to load.
         * @return The loaded image.
         */
        virtual cv::Mat load(const std::string & filename) const = 0;
    };

    /**
     * Base class for all loaders.
     */
    class BaseLoader : public LoaderInterface {
    protected:
        /**
         * Loads the given image from disk and throws an error if it canont
         * be loaded.
         * 
         * @param filename The filename of the image to load.
         * @param color True if this shall be loaded as color image.
         * @return The loaded image.
         */
        cv::Mat _load(const std::string & filename, bool color) const;
    };

    /**
     * Loads a simple RGB image. This is usually used in order to load the 
     * source image.
     */
    class RGBLoader : public BaseLoader {
    public:
        /**
         * Takes a filename and returns an image.
         * 
         * @param filename The image file to load.
         * @return The loaded image.
         */
        cv::Mat load(const std::string & filename) const;
    };

    /**
     * Loads a simple 8-bit label image.
     */
    class LabelLoader : public BaseLoader {
    public:
        /**
         * Takes a filename and returns an image.
         * 
         * @param filename The image file to load.
         * @return The loaded image.
         */
        cv::Mat load(const std::string & filename) const;
    };

    /**
     * Loads a 1-channel 8-bit image and re-maps the values according to the 
     * given mapping table.
     */
    class ValueMapperLoader : public BaseLoader {
    public:

        /**
         * Initializes a new instance of the ValueMapperClass.
         */
        explicit ValueMapperLoader() {}

        /**
         * Initializes a new instance of the ValueMapperClass.
         * 
         * @param The value map.
         */
        explicit ValueMapperLoader(const std::array<uchar, 256> & _valueMap) :
        valueMap(_valueMap) {
        }

        /**
         * Takes a filename and returns an image.
         * 
         * @param filename The image file to load.
         * @return The loaded image.
         */
        cv::Mat load(const std::string & filename) const;

    private:
        /**
         * This is the underlying value map. 
         */
        std::array<uchar, 256> valueMap;
    };

    /**
     * Loads an RGB color image and maps the color values to 8-bit unsigned 
     * integer values.
     */
    class ColorMapperLoader : public BaseLoader {
    public:

        /**
         * Initializes a new instance of the ColorMapperLoader.
         */
        explicit ColorMapperLoader() {}

        /**
         * Initializes a new instance of the ColorMapperLoader.
         * 
         * @param The color map.
         */
        explicit ColorMapperLoader(
                const std::unordered_map<cv::Vec3b, uchar> & _colorMap) :
        colorMap(_colorMap) {
        }

        /**
         * Takes a filename and returns an image.
         * 
         * @param filename The image file to load.
         * @return The loaded image.
         */
        cv::Mat load(const std::string & filename) const;

    private:
        /**
         * This is the underlying value map. 
         */
        std::unordered_map<cv::Vec3b, uchar> colorMap;
    };

    /**
     * Loads an image and a target image from disk.
     */
    class ImageTargetPairLoader {
    public:
        /**
         * Initializes a new instance of the ImageTargetPairLoader class.
         * @param _imageLoader The image loader.
         * @param _targetLoader The target loader
         */
        ImageTargetPairLoader(
                std::shared_ptr<LoaderInterface> _imageLoader,
                std::shared_ptr<LoaderInterface> _targetLoader) :
        imageLoader(_imageLoader),
        targetLoader(_targetLoader) {
        }
        
        /**
         * Loads the image and the target image from disk.
         * 
         * @param filenames The filenames to load.
         * @return The loaded images
         */
        ImageTargetPair load(
                IteratorInterface::ElementIterator filenames) const;
        
    private:
        /**
         * The image loader.
         */
        std::shared_ptr<LoaderInterface> imageLoader;
        /**
         * The target loader.
         */
        std::shared_ptr<LoaderInterface> targetLoader;
    };
} // namespace chianti
#endif