/* Copyright (C) 2017 Google Inc.
 * 
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms of the MIT 
 * license.  See the LICENSE file for details.
 */

#include "pychianti/augmentors.h"

#include <boost/python/stl_iterator.hpp>

#include <memory>
#include <string>

namespace pychianti {

    AugmentorAdapter AugmentorAdapter::createCombinedAugmentor(
            const boost::python::object & augmentors) {
        auto augmentor = std::make_shared<chianti::CombinedAugmentor>();

        boost::python::stl_input_iterator<AugmentorAdapter> begin(augmentors), end;

        for (auto i = begin; i != end; i++) {
            AugmentorAdapter current = *i;
            augmentor->addAugmentor(i->getAugmentor());
        }

        return AugmentorAdapter(augmentor);
    }

} // namespace pychianti