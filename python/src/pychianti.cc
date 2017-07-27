/* Copyright (C) 2017 Google Inc.
 * 
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms of the MIT 
 * license.  See the LICENSE file for details.
 */

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <memory>

#include <boost/python.hpp>

#include "chianti/augmentors.h"
#include "chianti/iterators.h"
#include "chianti/loaders.h"
#include "chianti/providers.h"

#include "pychianti/augmentors.h"
#include "pychianti/iterators.h"
#include "pychianti/loaders.h"
#include "pychianti/providers.h"

BOOST_PYTHON_MODULE(pychianti) {
    pychianti::wrap_imports();
    
    // AUGMENTORS
    boost::python::class_<pychianti::AugmentorAdapter> (
            "Augmentor", boost::python::no_init)
            .def("Subsample", 
                    &pychianti::AugmentorAdapter::createSubsampleAugmentor)
            .def("Gamma", &pychianti::AugmentorAdapter::createGammaAugmentor)
            .def("Translation", 
                    &pychianti::AugmentorAdapter::createTranslationAugmentor)
            .def("Zooming", 
                    &pychianti::AugmentorAdapter::createZoomingAugmentor)
            .def("Rotation", 
                    &pychianti::AugmentorAdapter::createRotationAugmentor)
            .def("Saturation", 
                    &pychianti::AugmentorAdapter::createSaturationAugmentor)
            .def("Hue", &pychianti::AugmentorAdapter::createHueAugmentor)
            .def("Crop", &pychianti::AugmentorAdapter::createCropAugmentor)
            .def("Combined", 
                    &pychianti::AugmentorAdapter::createCombinedAugmentor)
            .staticmethod("Subsample")
            .staticmethod("Gamma")
            .staticmethod("Translation")
            .staticmethod("Zooming")
            .staticmethod("Rotation")
            .staticmethod("Saturation")
            .staticmethod("Hue")
            .staticmethod("Crop")
            .staticmethod("Combined");
    
    // ITERATORS
    boost::python::class_<pychianti::IteratorAdapter>(
            "Iterator", boost::python::no_init)
            .def("next", &pychianti::IteratorAdapter::next)
            .def("reset", &pychianti::IteratorAdapter::reset)
            .def("get_num_elements", &pychianti::IteratorAdapter::getNumElements)
            .def("Sequential", 
                    &pychianti::IteratorAdapter::createSequentialIterator)
            .def("Random", 
                    &pychianti::IteratorAdapter::createRandomIterator)
            .def("WeightedRandom", 
                    &pychianti::IteratorAdapter::createWeightedRandomIterator)
            .staticmethod("Sequential")
            .staticmethod("Random")
            .staticmethod("WeightedRandom");

    // LOADERS
    boost::python::class_<pychianti::LoaderAdapter> (
            "Loader", boost::python::no_init)
            .def("RGB", &pychianti::LoaderAdapter::createRGBLoader)
            .def("Label", &pychianti::LoaderAdapter::createLabelLoader)
            .def("ValueMapper", 
                    &pychianti::LoaderAdapter::createValueMapperLoader)
            .def("ColorMapper", 
                    &pychianti::LoaderAdapter::createColorMapperLoader)
            .staticmethod("RGB")
            .staticmethod("Label")
            .staticmethod("ValueMapper")
            .staticmethod("ColorMapper");
            
   // DATA PROVIDER
    boost::python::class_<
            pychianti::DataProviderAdapter> (
            "DataProvider", boost::python::init<pychianti::AugmentorAdapter,
            pychianti::LoaderAdapter, pychianti::LoaderAdapter, 
            pychianti::IteratorAdapter, int, int>())
            .def("next", &pychianti::DataProviderAdapter::next)
            .def("reset", &pychianti::DataProviderAdapter::reset)
            .def("get_num_batches", &pychianti::DataProviderAdapter::getNumBatches);
}
