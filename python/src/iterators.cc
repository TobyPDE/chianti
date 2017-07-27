/* Copyright (C) 2017 Google Inc.
 * 
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms of the MIT 
 * license.  See the LICENSE file for details.
 */

#include "pychianti/iterators.h"

#include <boost/python/stl_iterator.hpp>

#include <memory>
#include <string>

namespace pychianti {

    static chianti::IteratorInterface::ContainerPtr
    pythonTupleListToVector(const boost::python::object & list) {
        chianti::IteratorInterface::ContainerPtr container =
                chianti::IteratorInterface::ContainerPtr(
                new std::vector<chianti::FilenamePair>());

        boost::python::stl_input_iterator<boost::python::tuple> begin(list), end;

        for(auto i = begin; i != end; i++) {
            chianti::FilenamePair pair = {
                boost::python::extract<std::string>((*i)[0])(),
                boost::python::extract<std::string>((*i)[1])()
            };
            container->push_back(pair);
        }

        return container;
    }
    
    static chianti::WeightedRandomIterator::WeightPtr
    pythonDoubleListToVector(const boost::python::object & list) {
         chianti::WeightedRandomIterator::WeightPtr container =
                chianti::WeightedRandomIterator::WeightPtr(
                new std::vector<double>());

        boost::python::stl_input_iterator<double> begin(list), end;
        container->insert(container->begin(), begin, end);

        return container;
    }
    
    boost::python::tuple IteratorAdapter::next() {
        auto element = iterator->next();
        return boost::python::make_tuple(
                boost::python::object(element->image), 
                boost::python::object(element->target));
    }

    IteratorAdapter IteratorAdapter::createSequentialIterator(
            const boost::python::object& elementList) {
        return IteratorAdapter(std::make_shared<chianti::SequentialIterator>(
                pythonTupleListToVector(elementList)));
    }
    
    IteratorAdapter IteratorAdapter::createRandomIterator(
            const boost::python::object& elementList) {
        return IteratorAdapter(std::make_shared<chianti::RandomIterator>(
                pythonTupleListToVector(elementList)));
    }
    
    IteratorAdapter IteratorAdapter::createWeightedRandomIterator(
            const boost::python::object& elementList,
            const boost::python::object& weights) {
        return IteratorAdapter(
                std::make_shared<chianti::WeightedRandomIterator>(
                pythonTupleListToVector(elementList), 
                pythonDoubleListToVector(weights)));
    }

} // namespace pychianti