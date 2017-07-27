/* Copyright (C) 2017 Google Inc.
 * 
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms of the MIT 
 * license.  See the LICENSE file for details.
 */

#ifndef CHIANTI_PYCHIANTI_PROVIDERS_H
#define CHIANTI_PYCHIANTI_PROVIDERS_H

#include <boost/python.hpp>

#include <memory>

#include "chianti/providers.h"
#include "pychianti/augmentors.h"
#include "pychianti/iterators.h"
#include "pychianti/loaders.h"

namespace pychianti {
    
#if PY_MAJOR_VERSION == 2
    void wrap_imports();
#else
    PyObject * wrap_imports();
#endif

    /**
     * An adapter class for exposing chianti::DataProvider to python.
     */
    class DataProviderAdapter {
    public:
        /**
         * Initializes a new instance of the DataProviderAdapter class.
         * 
         * @param augmentor A python object that wraps a data augmentor.
         * @param loader A python object that wraps a loader.
         * @param iterator A python object that wraps an iterator.
         * @param batchSize The batch size.
         * @param numClasses The number of classes.
         */
        DataProviderAdapter(
                const AugmentorAdapter & augmentor, 
                const LoaderAdapter & imageLoader, 
                const LoaderAdapter & targetLoader, 
                const IteratorAdapter & iterator, 
                int batchSize, 
                int numClasses);
        
        /**
         * Returns the next batch of images. 
         * 
         * @return A tuple of two numpy arrays
         */
        boost::python::tuple  next();
        
        /**
         * Resets the provider.
         */
        void reset() {
            provider->reset();
        }
        
        /**
         * Returns the number of batches. 
         * 
         * @return The number of batches.
         */
        int getNumBatches() const {
            return provider->getNumBatches();
        }
        
    private:
        /**
         * The underlying reference to the data provider.
         */
        std::shared_ptr<chianti::DataProvider> provider;
    };
    
} // namespace pychianti

#endif