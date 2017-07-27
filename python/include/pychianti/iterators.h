/* Copyright (C) 2017 Google Inc.
 * 
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms of the MIT 
 * license.  See the LICENSE file for details.
 */

#ifndef CHIANTI_PYCHIANTI_ITERATORS_H
#define CHIANTI_PYCHIANTI_ITERATORS_H

#include <boost/python.hpp>

#include <memory>

#include "chianti/iterators.h"

namespace pychianti {
    
    /**
     * The base class for all adapters.
     */
    class IteratorAdapter {
    public:
        IteratorAdapter(std::shared_ptr<chianti::IteratorInterface> iterator) : 
        iterator(iterator) {}
        
        /**
         * Returns the next element from the iterator.
         * 
         * @return A string tuple.
         */
        boost::python::tuple next();
        
        /**
         * Resets the iterator.
         */
        void reset() {
            iterator->reset();
        }
        
        /**
         * Returns the number of elements in the container.
         * 
         * @return The number of elements to iterate over.
         */
        int getNumElements() const {
            return iterator->getNumElements();
        }
        
        /**
         * Exposes the iterator.
         * 
         * @return The underlying iterator.
         */
        std::shared_ptr<chianti::IteratorInterface> getIterator() const {
            return iterator;
        }
        
        /**
         * This is a wrapper class for chianti::SequentialIterator. It allows us
         * to expose its API to python.
         */
        static IteratorAdapter createSequentialIterator(
                const boost::python::object & elementList);

        /**
         * This is a wrapper class for chianti::RandomIterator. It allows us
         * to expose its API to python.
         */
        static IteratorAdapter createRandomIterator(
                const boost::python::object & elementList);

        /**
         * This is a wrapper class for chianti::WeightedRandomIterator. It 
         * allows us to expose its API to python.
         */
        static IteratorAdapter createWeightedRandomIterator(
                    const boost::python::object & elementList, 
                    const boost::python::object & weights);

    protected:
        /**
         * The underlying iterator instance.
         */
        std::shared_ptr<chianti::IteratorInterface> iterator;
    };
    
} // namespace pychianti
#endif