/* Copyright (C) 2017 Google Inc.
 * 
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms of the MIT 
 * license.  See the LICENSE file for details.
 */

#include "chianti/providers.h"
#include "chianti/memory.h"

#include <cmath>
#include <iostream>
#include <cstring>
#include <sstream>
#include <thread>

namespace chianti {

    std::unique_ptr<Batch> DataProvider::next() {
        // Wait until a new batch is available
        std::unique_lock<std::mutex> lock(batchAccessMutex);
        cv.wait(lock, [this]() {
            return batch != nullptr;
        });

        // Copy the batch 
        auto result = std::move(batch);

        // Compute the next batch
        batch = nullptr;

        lock.unlock();
        cv.notify_one();

        return result;
    }

    void DataProvider::init() {
        // Load an image/target pair in order to get the size of the images
        auto pair = load(iterator->next());

        // Make sure that next() corresponds to the ordering of iterator
        iterator->reset();

        imageSize = {pair.image.rows, pair.image.cols};
        targetSize = {pair.target.rows, pair.target.cols};

        // Launch the prefill thread
        prefillThread = std::thread(&DataProvider::loadBatch, this);
    }

    void DataProvider::assertSize(const cv::Mat& img,
            const std::array<int, 2>& size) {
        if (img.rows != size[0] || img.cols != size[1]) {
            std::stringstream error;
            error << "Expected image to be of size " << size[0] << "x"
                    << size[1] << ". Image was of size " << img.rows << "x"
                    << img.cols << ".";
            throw std::runtime_error(error.str());
        }
    }

    void DataProvider::assertType(const cv::Mat& img, int type) {
        if (img.type() != type) {
            std::stringstream error;
            error << "Expected image to be of type " << type << ". Image was"
                    << " of type " << img.type() << ".";
            throw std::runtime_error(error.str());
        }
    }

    static void filterNans(ImageTargetPair & pair) {
        // Replace Nan values by 0
        auto data = reinterpret_cast<float*>(pair.image.data);
        const auto size = 3 * pair.image.total();
        
        for (size_t i = 0; i < size; i++) {
            if (std::isnan(data[i])) {
                data[i] = 0.0f;
            }
        }
    }
    
    ImageTargetPair DataProvider::load(
            IteratorInterface::ElementIterator filenames) {
        auto result = loader->load(filenames);
        if (augmentor != nullptr) {
            augmentor->augment(result);
        }
        
        // Remove NaN
        filterNans(result);
        
        return result;
    }

    void DataProvider::encode_onehot(
            const cv::Mat & target, 
            Tensor<float, 4> & tensor, 
            int offset) {
        const auto imgSize = target.rows * target.cols;
        for (int i = 0; i < target.rows; i++) {
            for (int j = 0; j < target.cols; j++) {
                const auto val = static_cast<int>(target.at<uchar>(i, j));
                if (val != 255) {
                    int index = offset + val * imgSize + i * target.cols + j;
                    tensor.data[index] = 1.0f;
                }
            }
        }
    }
    
    void DataProvider::loadBatch() {
        while (!terminateThread) {
            // Wait until a new batch must be computed
            std::unique_lock<std::mutex> lock(batchAccessMutex);
            cv.wait(lock, [this]() {
                return batch == nullptr;
            });

            batch = make_unique<Batch>(
                std::array<int, 4>{batchSize, 3, imageSize[0], imageSize[1]},
                std::array<int, 4>{batchSize, numClasses, targetSize[0], 
                        targetSize[1]});
            batch->targets.fill(0.0f);
            
            const int imageOffset = 3 * imageSize[0] * imageSize[1];
            const int targetOffset = numClasses * targetSize[0] * targetSize[1];

#pragma omp parallel for
            for (int i = 0; i < batchSize; i++) {
                // Load the image/label pair
                auto pair = load(iterator->next());

                // Make sure all images are of the right size and type
                assertSize(pair.image, imageSize);
                assertSize(pair.target, targetSize);
                assertType(pair.image, CV_32FC3);
                assertType(pair.target, CV_8UC1);

                // Convert the targets to a one-hot encoding.
                this->encode_onehot(
                        pair.target, batch->targets, i * targetOffset);

                // Copy the images to the right destination
                // While we can use memcpy for the targets, we have to shuffle
                // The array dimensions for the images
                cv::Mat rgb[3];
                cv::split(pair.image, rgb);
                auto dest = batch->images.data.data();
                std::memcpy(dest + imageOffset * i,
                        rgb[0].data,
                        imageOffset / 3 * sizeof (float));
                std::memcpy(dest + imageOffset * i + imageOffset / 3,
                        rgb[1].data,
                        imageOffset / 3 * sizeof (float));
                std::memcpy(dest + imageOffset * i + 2 * imageOffset / 3,
                        rgb[2].data,
                        imageOffset / 3 * sizeof (float));
            }

            lock.unlock();
            cv.notify_one();
        }
    }

    DataProvider::~DataProvider() {
        // Terminate the prefill thread
        std::unique_lock<std::mutex> lock(batchAccessMutex);
        cv.wait(lock, [this]() {
            return batch != nullptr;
        });

        batch = nullptr;
        terminateThread = true;

        lock.unlock();
        cv.notify_one();

        prefillThread.join();
    }

} // namespace chianti