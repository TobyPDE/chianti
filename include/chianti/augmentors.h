/* Copyright (C) 2017 Google Inc.
 * 
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms of the MIT 
 * license.  See the LICENSE file for details.
 */

#ifndef CHIANTI_AUGMENTORS_H
#define CHIANTI_AUGMENTORS_H

#include <array>
#include <random>
#include <memory>
#include <mutex>

#include <opencv2/opencv.hpp>

#include "types.h"

namespace chianti {

    /**
     * This interface defines an augmentor.
     */
    class AugmentorInterface {
    public:
        /**
         * Augments an image/label pair.
         * 
         * @param pair the pair to augment.
         */
        virtual void augment(ImageTargetPair & pair) = 0;
    };

    /**
     * The augmentor combines several augmentors into one.
     */
    class CombinedAugmentor : public AugmentorInterface {
    public:

        /**
         * Adds a new augmentor.
         */
        void addAugmentor(std::shared_ptr<AugmentorInterface> augmentor) {
            this->augmentors.push_back(augmentor);
        }

        /**
         * Augments an image/label pair.
         * 
         * @param pair the pair to augment.
         */
        void augment(ImageTargetPair & pair);

    private:
        /**
         * The individual augmentation steps.
         */
        std::vector<std::shared_ptr<AugmentorInterface>> augmentors;
    };

    /**
     * Subsamples the pair by a given factor.
     */
    class SubsampleAugmentor : public AugmentorInterface {
    public:

        SubsampleAugmentor(int _factor) : factor(_factor) {
        }

        /**
         * Augments an image/label pair.
         * 
         * @param pair the pair to augment.
         */
        void augment(ImageTargetPair & pair) {
            resizeImage(pair.image);
            resizeTarget(pair.target);
        }

    private:
        /**
         * Resizes the image.
         */
        void resizeImage(cv::Mat & image);

        /**
         * Resizes the target.
         */
        void resizeTarget(cv::Mat & target);

        /**
         * The resize factor. 
         */
        int factor;
    };

    /**
     * Performs random gamma augmentation on the image.
     */
    class GammaAugmentor : public AugmentorInterface {
    public:

        /**
         * Initializes a new instance of the GammAugmentor class.
         */
        GammaAugmentor() : GammaAugmentor(0.05) {
        }

        /**
         * Initializes a new instance of the GammAugmentor class.
         * 
         * @param strength Parameter in [0, 0.5] that determines the strength
         *                 of the augmentation.
         */
        GammaAugmentor(double strength) :
        GammaAugmentor(strength, std::random_device()()) {
        }

        /**
         * Initializes a new instance of the GammAugmentor class.
         * 
         * @param strength Parameter in [0, 0.5] that determines the strength
         *                 of the augmentation.
         * @param seed The random seed
         */
        GammaAugmentor(double strength, int seed) :
        g(seed),
        d(std::max(-0.5, -strength), std::min(0.5, strength)) {
        }

        /**
         * Augments an image/label pair.
         * 
         * @param pair the pair to augment.
         */
        void augment(ImageTargetPair & pair);

    private:
        /**
         * Mutex for access to the RNG
         */
        std::mutex rngMutex;
        /**
         * Random number generator
         */
        std::mt19937 g;
        /**
         * Source distribution
         */
        std::uniform_real_distribution<double> d;
    };

    /**
     * Performs random translation augmentation on the image.
     */
    class TranslationAugmentor : public AugmentorInterface {
    public:

        /**
         * Initializes a new instance of the TranslationAugmentor class.
         */
        TranslationAugmentor() : TranslationAugmentor(120) {
        }

        /**
         * Initializes a new instance of the TranslationAugmentor class.
         * 
         * @param offset The offset by which the image is translated.
         */
        TranslationAugmentor(int offset) :
        TranslationAugmentor(offset, std::random_device()()) {
        }

        /**
         * Initializes a new instance of the TranslationAugmentor class.
         * 
         * @param offset The offset by which the image is translated.
         * @param seed The random seed
         */
        TranslationAugmentor(int offset, int seed) :
        g(seed),
        d(-std::abs(offset), std::abs(offset)) {
        }

        /**
         * Augments an image/label pair.
         * 
         * @param pair the pair to augment.
         */
        void augment(ImageTargetPair & pair);

    private:
        /**
         * Mutex for access to the RNG
         */
        std::mutex rngMutex;
        /**
         * Random number generator
         */
        std::mt19937 g;
        /**
         * Source distribution
         */
        std::uniform_int_distribution<int> d;
    };

    /**
     * Randomly zooms into/out of the image.
     */
    class ZoomingAugmentor : public AugmentorInterface {
    public:

        /**
         * Initializes a new instance of the ZoomingAugmentor class.
         */
        ZoomingAugmentor() : ZoomingAugmentor(0.3) {
        }

        /**
         * Initializes a new instance of the ZoomingAugmentor class.
         * 
         * @param factor The maximum factor by which we zoom in/out.
         */
        ZoomingAugmentor(double factor) :
        ZoomingAugmentor(factor, std::random_device()()) {
        }

        /**
         * Initializes a new instance of the ZoomingAugmentor class.
         * 
         * @param factor The maximum factor by which we zoom in/out.
         * @param seed The random seed
         */
        ZoomingAugmentor(double factor, int seed) :
        g(seed),
        d(1 - factor, 1 + factor) {
        }

        /**
         * Augments an image/label pair.
         * 
         * @param pair the pair to augment.
         */
        void augment(ImageTargetPair & pair);

    private:
        /**
         * Mutex for access to the RNG
         */
        std::mutex rngMutex;
        /**
         * Random number generator
         */
        std::mt19937 g;
        /**
         * Source distribution
         */
        std::uniform_real_distribution<double> d;
    };

    /**
     * Randomly rotates the image.
     */
    class RotationAugmentor : public AugmentorInterface {
    public:

        /**
         * Initializes a new instance of the RotationAugmentor class.
         */
        RotationAugmentor() : RotationAugmentor(10) {
        }

        /**
         * Initializes a new instance of the RotationAugmentor class.
         * 
         * @param Angel The maximum angel by which an image is rotated.
         */
        RotationAugmentor(double maxAngel) :
        RotationAugmentor(maxAngel, std::random_device()()) {
        }

        /**
         * Initializes a new instance of the RotationAugmentor class.
         * 
         * @param maxAngel The maximum angel by which an image is rotated.
         * @param seed The random seed
         */
        RotationAugmentor(double maxAngel, int seed) :
        g(seed),
        d(-maxAngel, maxAngel) {
        }

        /**
         * Augments an image/label pair.
         * 
         * @param pair the pair to augment.
         */
        void augment(ImageTargetPair & pair);

    private:
        /**
         * Mutex for access to the RNG
         */
        std::mutex rngMutex;
        /**
         * Random number generator
         */
        std::mt19937 g;
        /**
         * Source distribution
         */
        std::uniform_real_distribution<double> d;
    };

    /**
     * Randomly adjusts the image Saturation.
     */
    class SaturationAugmentor : public AugmentorInterface {
    public:

        /**
         * Initializes a new instance of the SaturationAugmentor class.
         */
        SaturationAugmentor() : SaturationAugmentor(0.5, 1.5) {
        }

        /**
         * Initializes a new instance of the SaturationAugmentor class.
         * 
         * @param delta_min Smallest saturation rescale factor
         * @param delta_max Largest saturation rescale factor
         */
        SaturationAugmentor(double delta_min, double delta_max) :
        SaturationAugmentor(delta_min, delta_max, std::random_device()()) {
        }

        /**
         * Initializes a new instance of the SaturationAugmentor class.
         * 
         * @param delta_min Smallest saturation rescale factor
         * @param delta_max Largest saturation rescale factor
         * @param seed The random seed
         */
        SaturationAugmentor(double delta_min, double delta_max, int seed) :
        g(seed),
        d(delta_min, delta_max) {
        }

        /**
         * Augments an image/label pair.
         * 
         * @param pair the pair to augment.
         */
        void augment(ImageTargetPair & pair);

    private:
        /**
         * Mutex for access to the RNG
         */
        std::mutex rngMutex;
        /**
         * Random number generator
         */
        std::mt19937 g;
        /**
         * Source distribution
         */
        std::uniform_real_distribution<double> d;
    };

    /**
     * Randomly adjusts the image Hue.
     */
    class HueAugmentor : public AugmentorInterface {
    public:

        /**
         * Initializes a new instance of the HueAugmentor class.
         */
        HueAugmentor() : HueAugmentor(-30, 30) {
        }

        /**
         * Initializes a new instance of the HueAugmentor class.
         * 
         * @param delta_min Smallest hue rescale factor
         * @param delta_max Largest huerescale factor
         */
        HueAugmentor(double delta_min, double delta_max) :
        HueAugmentor(delta_min, delta_max, std::random_device()()) {
        }

        /**
         * Initializes a new instance of the HueAugmentor class.
         * 
         * @param delta_min Smallest hue rescale factor
         * @param delta_max Largest hue rescale factor
         * @param seed The random seed
         */
        HueAugmentor(double delta_min, double delta_max, int seed) :
        g(seed),
        d(delta_min, delta_max) {
        }

        /**
         * Augments an image/label pair.
         * 
         * @param pair the pair to augment.
         */
        void augment(ImageTargetPair & pair);

    private:
        /**
         * Mutex for access to the RNG
         */
        std::mutex rngMutex;
        /**
         * Random number generator
         */
        std::mt19937 g;
        /**
         * Source distribution
         */
        std::uniform_real_distribution<double> d;
    };

    /**
     * Randomly extracts quadratic crops from the image. The crops are sampled
     * with a probability proportional to the entropy of their class 
     * distributions.
     */
    class CropAugmentor : public AugmentorInterface {
    public:

        /**
         * Initializes a new instance of the CropAugmentor class.
         * 
         * @param size The size of the crop to extract
         * @param numClasses the number of possible classes in the target image
         */
        CropAugmentor(int size, int numClasses) :
        CropAugmentor(size, numClasses, std::random_device()()) {
        }

        /**
         * Initializes a new instance of the CropAugmentor class.
         * 
         * @param size The size of the crop to extract
         * @param numClasses the number of possible classes in the target image
         * @param seed The random seed
         */
        CropAugmentor(int size, int numClasses, int seed) :
        g(seed),
        d(0, 1), 
        size(size),
        numClasses(numClasses) {
        }

        /**
         * Augments an image/label pair.
         * 
         * @param pair the pair to augment.
         */
        void augment(ImageTargetPair & pair);

    private:
        /**
         * Computes the class histogram for an entry in the image without any
         * reference histograms.
         * Runtime: O(size^2)
         */
        void computeHistogram0(const cv::Mat & target, int i, int j, 
                cv::Mat & histograms) const;
        
        /**
         * Computes the class histogram for an entry with a single reference
         * point above the entry.
         * Runtime: O(size)
         */
        void computeHistogram1Up(const cv::Mat & target, int i, int j, 
                cv::Mat & histograms) const;
        
        /**
         * Computes the class histogram for an entry with a single reference
         * point left of the entry.
         * runtime: O(size)
         */
        void computeHistogram1Left(const cv::Mat & target, int i, int j, 
                cv::Mat & histograms) const;
        
        /**
         * Computes the class histogram for an entry with three reference 
         * points (left, above, and above left).
         * Runtime: O(1)
         */
        void computeHistogram3(const cv::Mat & target, int i, int j, 
                cv::Mat & histograms) const;
        
        /**
         * Computes the class histogram for all pixels via dynamic programming.
         */
        void computeClassHistograms(const cv::Mat & target, 
                cv::Mat & histograms) const;
        
        /**
         * Computes the cumulative probability distribution over all pixels.
         */
        void computeCumulativeDistribution(const cv::Mat & histograms, 
                cv::Mat & distribution) const;
        
        /**
         * Samples the position of a crop based on the cumulative distribution.
         */
        std::array<int, 2> samplePosition(const cv::Mat & target, 
                const cv::Mat & distribution);
        
        /**
         * Mutex for access to the RNG
         */
        std::mutex rngMutex;
        /**
         * Random number generator
         */
        std::mt19937 g;
        /**
         * Source distribution
         */
        std::uniform_real_distribution<float> d;
        /**
         * The size of the crops to extract.
         */
        int size;
        /**
         * The number of classes in the target image.
         */
        int numClasses;
    };

} // namespace chianti

#endif