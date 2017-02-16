#pragma once

#include "common.h"
#include "augmentation.h"
#include "iterators.h"

#include <string>

namespace Chianti {
    /**
     * This class loads a batch of images and potentially applies data augmentation operations.
     */
    class DataProvider {
    public:
        /**
         * Initializes a new instance of the DataProvider class.
         */
        DataProvider(std::shared_ptr<IIterator<std::pair<std::string, std::string>>> imageNames, size_t batchSize, std::shared_ptr<IAugmentor> augmentor) :
                imageNames(imageNames),
                nextBatch(nullptr),
                batchSize(batchSize),
                augmentor(augmentor),
                batchReady(false) {}

        /**
         * Returns the next batch.
         */
        std::shared_ptr<std::vector<ImageLabelPair>> next()
        {
            batchMutex.lock();
            if (!batchReady) {
                // Compute the next batch
                computeNextBatch();
            }

            auto result = nextBatch;
            batchReady = false;
            nextBatch = nullptr;
            batchMutex.unlock();

            // Compute the next batch
            std::thread t([this]() {
                std::lock_guard<std::mutex> lock(batchMutex);
                computeNextBatch();
            });
            t.detach();

            return result;
        }

        /**
         * Returns the batch size.
         */
        size_t getBatchSize() const
        {
            return batchSize;
        }

        /**
         * Returns the number of batches
         */
        size_t getNumBatches() const
        {
            return imageNames->size() / batchSize;
        }

        /**
         * Resets the provider
         */
        void reset()
        {
            imageNames->reset();
        }
    private:
        /**
         * Computes the next batch of images.
         */
        void computeNextBatch()
        {
            nextBatch = std::make_shared<std::vector<ImageLabelPair>>(batchSize);

            #pragma omp parallel for
            for (size_t i = 0; i < batchSize; i++)
            {
                loadImageLabelPair(*imageNames->next(), nextBatch->at(i));
            }

            batchReady = true;
        }

        /**
         * Loads an image/label pair from the hard drive.
         */
        void loadImageLabelPair(const std::pair<std::string, std::string> & imageName, ImageLabelPair & pair)
        {
            pair.img = cv::imread(imageName.first, 1);
            pair.target = cv::imread(imageName.second, 0);

            // Apply the augmentation steps if necessary
            if (augmentor != nullptr)
            {
                augmentor->augment(pair);
            }
        }


        /**
         * This is a list of images to cycle through.
         */
        std::shared_ptr<IIterator<std::pair<std::string, std::string>> > imageNames;
        /**
         * This is the buffer of loaded images.
         */
        std::shared_ptr<std::vector<ImageLabelPair>> nextBatch;
        /**
         * This is the batch size = The number of images per batch
         */
        size_t batchSize;
        /**
         * A data augmentor.
         */
        std::shared_ptr<IAugmentor> augmentor;
        /**
         * The batch computation mutex.
         */
        std::mutex batchMutex;
        /**
         * Whether or not the current batch is ready.
         */
        bool batchReady;
    };
}