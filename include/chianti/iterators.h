/* Copyright (C) 2017 Google Inc.
 * 
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms of the MIT 
 * license.  See the LICENSE file for details.
 */

#ifndef CHIANTI_ITERATORS_H
#define CHIANTI_ITERATORS_H

#include "types.h"

#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace chianti {

    /**
     * This class defines the interface for iterating over a collection of 
     * FilenamePair values.
     */
    class IteratorInterface {
    public:
        typedef std::unique_ptr<std::vector<FilenamePair>> ContainerPtr;
        typedef std::vector<FilenamePair>::iterator ElementIterator;

        /**
         * Returns the next FilenamePair in the sequence.
         * 
         * @return The next FilenamePair in the sequence over which we 
         *         iterate.
         */
        virtual ElementIterator next() = 0;

        /**
         * Resets the iterator to the beginning if that's possible.
         */
        virtual void reset() = 0;

        /**
         * Returns the number of elements in the underlying structure. May be 
         * infinite.
         * 
         * @return Number of elements in the universe.
         */
        virtual size_t getNumElements() const = 0;
    };

    /**
     * The base class for all iterators.
     */
    class BaseIterator : public IteratorInterface {
    public:

        /**
         * Returns the number of elements in the underlying structure. May be 
         * infinite.
         * 
         * @return Number of elements in the universe.
         */
        size_t getNumElements() const {
            return container->size();
        }

    protected:

        /**
         * Initializes a new instance of the BaseIterator class.
         * 
         * @param _container The container of filenames.
         */
        BaseIterator(ContainerPtr _container) :
        container(std::move(_container)) {
        }

        /**
         * The access mutex. It may be the case that multiple threads try to 
         * access the iterator simultaneously.
         */
        std::mutex accessMutex;
        /**
         * The underlying container over which we iterate.
         */
        ContainerPtr container;
    };

    /**
     * This class sequentially iterates over a given list of FilenamePairs.
     */
    class SequentialIterator : public BaseIterator {
    public:

        /**
         * Initializes a new instance of the SequentialIterator class.
         * 
         * @param _container The container of FilenamePairs to iterate over.
         */
        SequentialIterator(ContainerPtr _container) :
        BaseIterator(std::move(_container)) {
            nextElement = container->begin();
        }

        /**
         * Returns the next FilenamePair in the sequence.
         * 
         * @return The next FilenamePair in the sequence over which we 
         *         iterate.
         */
        ElementIterator next();

        /**
         * Resets the iterator to the beginning if that's possible.
         */
        void reset() {
            nextElement = container->begin();
        }

    private:
        /**
         * A pointer to the next element in the container.
         */
        std::vector<FilenamePair>::iterator nextElement;
    };

    /**
     * This class randomly returns elements from an underlying container.
     */
    class RandomIterator : public BaseIterator {
    public:

        /**
         * Initializes a new instance of the RandomIterator class.
         * 
         * @param _container The container of FilenamePairs to iterate over.
         * @param _seed The random seed.
         */
        RandomIterator(ContainerPtr _container, unsigned int _seed) :
        BaseIterator(std::move(_container)),
        g(_seed),
        seed(_seed) {
            keys.resize(container->size());
            std::iota(keys.begin(), keys.end(), 0);
            shuffle();
        }

        /**
         * Initializes a new instance of the RandomIterator class.
         * 
         * @param _container The container of FilenamePairs to iterate over.
         */
        RandomIterator(ContainerPtr _container) :
        RandomIterator(std::move(_container), std::random_device()()) {
        }
        
        /**
         * Returns the next FilenamePair in the sequence.
         * 
         * @return The next FilenamePair in the sequence over which we 
         *         iterate.
         */
        ElementIterator next();

        /**
         * Resets the iterator to the beginning if that's possible.
         */
        void reset() {
            // Reset the RNG
            g = std::mt19937(seed);
            shuffle();
        }

    private:
        /**
         * Shuffles the dataset.
         */
        void shuffle() {
            std::shuffle(keys.begin(), keys.end(), g);
            nextKey = keys.begin();
        }
        
        /**
         * The random number generator.
         */
        std::mt19937 g;
        /**
         * The next element in the key sequence.
         */
        std::vector<size_t>::iterator nextKey;
        /**
         * The current dataset shuffle.
         */
        std::vector<size_t> keys;
        /**
         * The random seed.
         */
        unsigned int seed;
    };

    /**
     * This class randomly samples elements from the underlying container based
     * on given weights.
     */
    class WeightedRandomIterator : public BaseIterator {
    public:
        typedef std::unique_ptr<std::vector<double>> WeightPtr;

        /**
         * Initializes a new instance of the WeightedRandomIterator class.
         * 
         * @param _container The container of FilenamePairs to iterate over.
         * @param _weights The weights associated with each element of the 
         *                 container.
         */
        WeightedRandomIterator(
                ContainerPtr _container,
                WeightPtr _weights) :
        WeightedRandomIterator(
                std::move(_container), 
                std::move(_weights), 
                std::random_device()()) {}

        /**
         * Initializes a new instance of the WeightedRandomIterator class.
         * 
         * @param _container The container of FilenamePairs to iterate over.
         * @param _weights The weights associated with each element of the 
         *                 container.
         * @param _seed The random seed.
         */
        WeightedRandomIterator(
                ContainerPtr _container,
                WeightPtr _weights,
                unsigned int _seed) :
        BaseIterator(std::move(_container)),
        weights(std::move(_weights)),
        g(_seed),
        uniformDistribution(0.0, 1.0),
        seed(_seed) {
            normalizeWeights();
        }

        /**
         * Returns the next FilenamePair in the sequence.
         * 
         * @return The next FilenamePair in the sequence over which we 
         *         iterate.
         */
        ElementIterator next();

        /**
         * Resets the iterator to the beginning if that's possible.
         */
        void reset() {
            // Reset the RNG
            g = std::mt19937(seed);
        }

    private:
        /**
         * Computes the cumulative weight distribution.
         */
        void normalizeWeights();

        /**
         * The underlying weights according to which we sample.
         */
        WeightPtr weights;
        /**
         * The random number generator.
         */
        std::mt19937 g;
        /**
         * The sampling distribution.
         */
        std::uniform_real_distribution<double> uniformDistribution;
        /**
         * The random seed.
         */
        unsigned int seed;
    };

} // namespace chianti
#endif