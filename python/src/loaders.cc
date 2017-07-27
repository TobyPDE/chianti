/* Copyright (C) 2017 Google Inc.
 * 
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms of the MIT 
 * license.  See the LICENSE file for details.
 */

#include "pychianti/loaders.h"

#include <boost/python/stl_iterator.hpp>
#include <opencv2/opencv.hpp>

#include <array>
#include <exception>
#include <memory>
#include <string>
#include <unordered_map>

namespace pychianti {

    typedef boost::python::stl_input_iterator<int> list_iterator;
    LoaderAdapter LoaderAdapter::createValueMapperLoader(
            const boost::python::object& list) {
        std::array<uchar, 256> valueMap;
        list_iterator begin(list), end;
        
        // There must be exactly 256 elements in the list
        if (std::distance(list_iterator(list), list_iterator()) != 256) {
            throw std::runtime_error("Expected 256 elements in value map.");
        }

        int k = 0;
        for(auto i = begin; i != end; i++) {
            valueMap[k++] = static_cast<uchar>(*i);
        }
        
        return LoaderAdapter(
                std::make_shared<chianti::ValueMapperLoader>(valueMap));
    }

    LoaderAdapter LoaderAdapter::createColorMapperLoader(
            const boost::python::dict& colorDict) {
        
        std::unordered_map<cv::Vec3b, uchar> map;
        
        auto items = colorDict.iteritems();
        boost::python::stl_input_iterator<boost::python::tuple> begin(items), end;
        
        for(auto i = begin; i != end; i++) {
            // The key must be a tuple of length 3
            boost::python::tuple key = 
                    boost::python::extract<boost::python::tuple>((*i)[0]);
            
            cv::Vec3b color;
            for (int c = 0; c < 3; c++) {
                color[c] = static_cast<uchar>(
                        boost::python::extract<int>(key[c]));
            }
            
            uchar value = static_cast<uchar>(
                    boost::python::extract<int>((*i)[1]));
            
            map[color] = value;
        }
        
        return LoaderAdapter(std::make_shared<chianti::ColorMapperLoader>(map));
    }

} // namespace pychianti