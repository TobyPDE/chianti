/* Copyright (C) 2017 Google Inc.
 * 
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms of the MIT 
 * license.  See the LICENSE file for details.
 */

#ifndef CHIANTI_PROVIDERS_H
#define CHIANTI_PROVIDERS_H

#include <array>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>

#include <opencv2/opencv.hpp>

#include "augmentors.h"
#include "iterators.h"
#include "loaders.h"
#include "types.h"

namespace chianti {
    /**
     * A threaded data provider that loads images from disk asynchronously.
     */
    class DataProvider {
    public:
        /**
         * Initializes a new instance of the ThreadedDataProvider class.
         * 
         * @param _loader The image loader.
         * @param _iterator The file iterator.
         * @param _batchSize The batch size.
         */
        DataProvider(
                std::shared_ptr<AugmentorInterface> _augmentor, 
                std::shared_ptr<ImageTargetPairLoader> _loader,
                std::shared_ptr<IteratorInterface> _iterator,
                int _batchSize, 
                int _numClasses) :
        augmentor(_augmentor), 
        loader(_loader),
        iterator(_iterator),
        batchSize(_batchSize), 
        numClasses(_numClasses), 
        terminateThread(false) {
        }
        
        /**
         * Destructor.
         */
        ~DataProvider();
        
        /**
         * Returns the next batch of images.
         * 
         * @return The next batch of images.
         */
        std::unique_ptr<Batch> next();
        
        /**
         * Initializes the provider.
         */
        void init();
        
        /**
         * Resets the provider.
         */
        void reset() {
            iterator->reset();
        }
        
        /**
         * Returns the number of batches.
         * 
         * @return The number of batches.
         */
        int getNumBatches() const {
            return iterator->getNumElements() / batchSize;
        }
        
    private:
        /**
         * Loads the next batch.
         */
        void loadBatch();
        
        /**
         * Loads a single image.
         * 
         * @return Image target pair
         */
        ImageTargetPair load(IteratorInterface::ElementIterator filenames);
        
        /**
         * Fills the target tensor with one-hot encoded values.
         * 
         * @param img
         * @param size
         */
        void encode_onehot(const cv::Mat & targetImg, 
                           Tensor<float, 4> & targetTensor, 
                           int offset);
        
        /**
         * Throws a runtime exception if image is not of the given size.
         */
        static void assertSize(const cv::Mat & img, 
                               const std::array<int, 2> & size);
        
        /**
         * Throws a runtime exception if image is not of the given type.
         */
        static void assertType(const cv::Mat & img, int type);
        
        /**
         * Data augmentor.
         */
        std::shared_ptr<AugmentorInterface> augmentor;
        /**
         * The image loader.
         */
        std::shared_ptr<ImageTargetPairLoader> loader;
        /**
         * The filename iterator.
         */
        std::shared_ptr<IteratorInterface> iterator;
        /**
         * The size of the images.
         */
        std::array<int, 2> imageSize;
        /**
         * The size of the target images.
         */
        std::array<int, 2> targetSize;
        /**
         * The batch size.
         */
        int batchSize;
        /**
         * The number of classes.
         */
        int numClasses;
        /**
         * The next batch of images.
         */
        std::unique_ptr<Batch> batch;
        /**
         * Batch access mutex
         */
        std::mutex batchAccessMutex;
        /**
         * Conditional variable for waiting for the next batch to be computed.
         */
        std::condition_variable cv;
        /**
         * Whether the prefill thread shall be terminated.
         */
        bool terminateThread;
        /**
         * The prefill thread.
         */
        std::thread prefillThread;
        
    };
    
} // namespace chianti

#endif
