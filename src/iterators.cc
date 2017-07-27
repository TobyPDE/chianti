/* Copyright (C) 2017 Google Inc.
 * 
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms of the MIT 
 * license.  See the LICENSE file for details.
 */

#include "chianti/iterators.h"

#include <exception>
#include <limits>

namespace chianti
{
    IteratorInterface::ElementIterator SequentialIterator::next() {
        std::lock_guard<std::mutex> lock(accessMutex);
        
        // If we have reached the end of the container, go back to the beginning
        if (nextElement == container->end()) {
            nextElement = container->begin();
        }
        
        // If there are not elements in the container, throw an exception
        if (container->empty()) {
            throw std::runtime_error("Container is empty.");
        }
        
        // Advance the iterator and return the current element
        auto retVal = nextElement;
        nextElement++;
        return retVal;
    }
    
    IteratorInterface::ElementIterator RandomIterator::next() {
        std::lock_guard<std::mutex> lock(accessMutex);
        
        // If we have reached the end of the key list, shuffle
        if (nextKey == keys.end()) {
            shuffle();
        }
        
        // If there are not elements in the container, throw an exception
        if (container->empty()) {
            throw std::runtime_error("Container is empty.");
        }
        
        auto current = *nextKey;
        nextKey++;
        
        // Sample an index and return the element
        return container->begin() + current;
    }
    
    void WeightedRandomIterator::normalizeWeights() {
        // If the number of weights is different from the number of container
        // elements, throw an exception
        if (container->size() != weights->size()) {
            throw std::runtime_error("Number of weights differs from number of "
                                     "elements in container.");
        }
        
        // Make sure that all weights are non-negative and sum to one
        double sum = 0.0f;
        for(auto i = weights->begin(); i != weights->end(); i++) {
            *i = std::abs(*i);
            sum += *i;
        }
        
        // Normalize the weights
        for(auto i = weights->begin(); i != weights->end(); i++) {
            *i /= sum;
            if (i != weights->begin()) {
                *i += *(i - 1);
            }
        }
    }
    
    IteratorInterface::ElementIterator WeightedRandomIterator::next() {
        std::lock_guard<std::mutex> lock(accessMutex);
        
        // If there are not elements in the container, throw an exception
        if (container->empty()) {
            throw std::runtime_error("Container is empty.");
        }
        
        // Sample from the given weight distribution
        const double u = uniformDistribution(g);
        double previous = 0.0f;
        for(auto i = weights->begin(); i != weights->end(); i++) {
            if (previous <= u && u < *i) {
                const auto offset = std::distance(weights->begin(), i);
                return container->begin() + offset;
            } else {
                previous = *i;
            }
        }
        
        return container->end() - 1;
    }
} // namespace chianti