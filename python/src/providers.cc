/* Copyright (C) 2017 Google Inc.
 * 
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms of the MIT 
 * license.  See the LICENSE file for details.
 */

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "pychianti/providers.h"

#include <boost/python/stl_iterator.hpp>
#include <numpy/arrayobject.h>

#include <array>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>

namespace pychianti {

#if PY_MAJOR_VERSION == 2
void wrap_imports() {
    import_array();
}
#else
PyObject * wrap_imports() {
    import_array();
    Py_RETURN_NONE;
}
#endif

    DataProviderAdapter::DataProviderAdapter(
            const AugmentorAdapter & augmentor,
            const LoaderAdapter & imageLoader,
            const LoaderAdapter & targetLoader,
            const IteratorAdapter & iterator,
            int batchSize, 
            int numClasses) {

        provider = std::make_shared<chianti::DataProvider>(
                augmentor.getAugmentor(),
                std::make_shared<chianti::ImageTargetPairLoader>(
                imageLoader.getLoader(),
                targetLoader.getLoader()
                ),
                iterator.getIterator(),
                batchSize, 
                numClasses);
        provider->init();
    }

    /**
     * Converts a C++ type to a numpy type.
     */
    template<class T> struct GetNumpyType;
    template<> struct GetNumpyType<float> { 
        static constexpr NPY_TYPES type = NPY_FLOAT32;
    };
    template<> struct GetNumpyType<uchar> {
        static constexpr NPY_TYPES type = NPY_UINT8;
    };
    template<> struct GetNumpyType<int> {
        static constexpr NPY_TYPES type = NPY_INT32;
    };
    
    /**
     * Converts a tensor to a numpy array.
     */
    template<typename T, int Rank>
    boost::python::object convertTensor(
            const chianti::Tensor<T, Rank> & tensor) {

        PyArrayObject* result = (PyArrayObject*) PyArray_FromDims(
                static_cast<int>(tensor.shape.size()), 
                const_cast<int*>(tensor.shape.data()), 
                GetNumpyType<T>::type);
        
        // Copy the data 
        std::memcpy(PyArray_BYTES(result),
                tensor.data.data(),
                tensor.getSize() * sizeof (T));
        
        // Wrap the pointer in a boost python object
        boost::python::handle<> handle(reinterpret_cast<PyObject*>(result));
        boost::python::object object(handle);
        return object;
    }

    boost::python::tuple DataProviderAdapter::next() {
        // Get the next batch
        auto batch = provider->next();

        return boost::python::make_tuple(convertTensor(batch->images), 
                convertTensor(batch->targets));
    }

} // namespace pychianti