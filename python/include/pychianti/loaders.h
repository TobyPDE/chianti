/* Copyright (C) 2017 Google Inc.
 * 
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms of the MIT 
 * license.  See the LICENSE file for details.
 */

#ifndef CHIANTI_PYCHIANTI_LOADERS_H
#define CHIANTI_PYCHIANTI_LOADERS_H

#include <boost/python.hpp>

#include <memory>

#include "chianti/loaders.h"

namespace pychianti {
    
    /**
     * This is template adapter can be used in order to expose some loaders
     * to python.
     */
    class LoaderAdapter {
    public:
        /**
         * Initializes a new instance of the LoaderAdapter class.
         */
        LoaderAdapter(std::shared_ptr<chianti::LoaderInterface> loader) : 
        loader(loader) {}
        
        /**
         * Returns the underlying loader.
         * 
         * @return The underlying loader.
         */
        std::shared_ptr<chianti::LoaderInterface> getLoader() const {
            return loader;
        }
        
        /**
         * Creates a RGBLoader.
         */
        static LoaderAdapter createRGBLoader() {
            return LoaderAdapter(
                    std::make_shared<chianti::RGBLoader>());
        }
        
        /**
         * Creates a LabelLoader.
         */
        static LoaderAdapter createLabelLoader() {
            return LoaderAdapter(
                    std::make_shared<chianti::LabelLoader>());
        }
        
        /**
         * Creates a ValueMapperLoader.
         */
        static LoaderAdapter createValueMapperLoader(
                const boost::python::object& list);
        
        /**
         * Creates a ColorMapperLoader.
         */
        static LoaderAdapter createColorMapperLoader(
                const boost::python::dict& colorDict);
        
    protected:
        /**
         * Pointer to the underlying loader.
         */
        std::shared_ptr<chianti::LoaderInterface> loader;
    };
    
} // namespace pychianti

#endif