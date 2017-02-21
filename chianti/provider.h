#pragma once

#include "common.h"
#include "augmentation.h"
#include "iterators.h"

#include <string>
#include <array>

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

    typedef struct {
        float* imgs;
        std::array<int, 4> imgsShape;
        int* targets;
        std::array<int, 3> targetsShape;
    } Batch;

    /**
     * This data provider converts the batch to a pair of tensors.
     */
    class TensorDataProvider {
    public:
        /**
         * Initializes a new instance of the DataProvider class.
         */
        TensorDataProvider(std::shared_ptr<DataProvider> provider) :
                provider(provider),
                nextBatch(nullptr),
                batchReady(false) {}

        /**
         * Returns the next batch.
         */
        std::shared_ptr<Batch> next()
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
            return provider->getBatchSize();
        }

        /**
         * Returns the number of batches
         */
        size_t getNumBatches() const
        {
            return provider->getNumBatches();
        }

        /**
         * Resets the provider
         */
        void reset()
        {
            provider->reset();
        }

    private:
        /**
         * Computes the next batch of images.
         */
        void computeNextBatch()
        {
            // Get the next batch
            auto batch = this->provider->next();

            // Determine the size of the result tensors
            std::array<int, 4> imgsShape;
            imgsShape[0] = static_cast<int>(this->provider->getBatchSize());
            imgsShape[1] = 3;
            imgsShape[2] = batch->at(0).img.rows;
            imgsShape[3] = batch->at(0).img.cols;

            std::array<int, 3> targetsShape;
            targetsShape[0] = static_cast<int>(this->provider->getBatchSize());
            targetsShape[1] = batch->at(0).img.rows;
            targetsShape[2] = batch->at(0).img.cols;

            // Determine the tensor strides
            std::array<int, 4> imgsStrides;
            imgsStrides[3] = 1;
            for (int i = 2; i >= 0; i--)
            {
                imgsStrides[i] = imgsStrides[i + 1] * imgsShape[i + 1];
            }

            std::array<int, 3> targetsStrides;
            targetsStrides[2] = 1;
            for (int i = 1; i >= 0; i--)
            {
                targetsStrides[i] = targetsStrides[i + 1] * targetsShape[i + 1];
            }

            auto imgs = new float[prod(imgsShape)];
            auto targets = new int[prod(targetsShape)];

            for (int i0 = 0; i0 < imgsShape[0]; i0++) {
                for (int i2 = 0; i2 < imgsShape[2]; i2++) {
                    for (int i3 = 0; i3 < imgsShape[3]; i3++) {
                        const uchar label = batch->at(i0).target.at<uchar>(i2, i3);;
                        targets[i0 * targetsStrides[0] + i2 * targetsStrides[1] + i3 * targetsStrides[2]] = label == 255 ? -1 : label;

                        const cv::Vec3f & img = batch->at(i0).img.at<cv::Vec3f>(i2, i3);
                        for (int i1 = 0; i1 < imgsShape[1]; i1++) {
                            // Convert it to RGB
                            const int _i1 = 2 - i1;
                            imgs[i0 * imgsStrides[0] + _i1 * imgsStrides[1] + i2 * imgsStrides[2] + i3 * imgsStrides[3]] = img[i1];
                        }
                    }
                }
            }

            nextBatch = std::make_shared<Batch>();
            nextBatch->imgs = imgs;
            nextBatch->imgsShape = imgsShape;
            nextBatch->targets = targets;
            nextBatch->targetsShape = targetsShape;

            batchReady = true;
        }

        template<size_t N>
        int prod(const std::array<int, N> & a) {
            int result = 1;
            for (size_t i = 0; i < a.size(); i++) {
                result *= a[i];
            }
            return result;
        }

        /**
         * The underlying data provider.
         */
        std::shared_ptr<DataProvider> provider;
        /**
         * This is the buffer of loaded images.
         */
        std::shared_ptr<Batch> nextBatch;
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