/* Copyright (C) 2017 Google Inc.
 * 
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms of the MIT 
 * license.  See the LICENSE file for details.
 */

#ifndef CHIANTI_PYCHIANTI_AUGMENTORS_H
#define CHIANTI_PYCHIANTI_AUGMENTORS_H

#include <boost/python.hpp>

#include <memory>

#include "chianti/augmentors.h"

namespace pychianti {
    
    /**
     * The base class for all augmentation adapters.
     */
    class AugmentorAdapter {
    public:
        AugmentorAdapter(std::shared_ptr<chianti::AugmentorInterface> augmentor)
        : augmentor(augmentor) {}
        
        /**
         * Returns the underlying augmentor.
         * 
         * @return The underlying augmentor.
         */
        std::shared_ptr<chianti::AugmentorInterface> getAugmentor() const {
            return augmentor;
        }
        
        /**
         * Creates a SubsampleAugmentor.
         */
        static AugmentorAdapter createSubsampleAugmentor(int factor) {
            return AugmentorAdapter(
                    std::make_shared<chianti::SubsampleAugmentor>(factor));
        }
        
        /**
         * Creates a GammaAugmentor.
         */
        static AugmentorAdapter createGammaAugmentor(double strength) {
            return AugmentorAdapter(
                    std::make_shared<chianti::GammaAugmentor>(strength));
        }
        
        /**
         * Creates a TranslationAugmentor.
         */
        static AugmentorAdapter createTranslationAugmentor(int offset) {
            return AugmentorAdapter(
                    std::make_shared<chianti::TranslationAugmentor>(offset));
        }
        
        /**
         * Creates a ZoomingAugmentor.
         */
        static AugmentorAdapter createZoomingAugmentor(double factor) {
            return AugmentorAdapter(
                    std::make_shared<chianti::ZoomingAugmentor>(factor));
        }
        
        /**
         * Creates a RotationAugmentor.
         */
        static AugmentorAdapter createRotationAugmentor(double maxAngel) {
            return AugmentorAdapter(
                    std::make_shared<chianti::RotationAugmentor>(maxAngel));
        }
        
        /**
         * Creates a SaturationAugmentor.
         */
        static AugmentorAdapter createSaturationAugmentor(double delta_min, 
                double delta_max) {
            return AugmentorAdapter(
                    std::make_shared<chianti::SaturationAugmentor>(
                    delta_min, delta_max));
        }
        
        /**
         * Creates a HueAugmentor.
         */
        static AugmentorAdapter createHueAugmentor(double delta_min, 
                double delta_max) {
            return AugmentorAdapter(
                    std::make_shared<chianti::HueAugmentor>(
                    delta_min, delta_max));
        }
        
        /**
         * Creates a CropAugmentor.
         */
        static AugmentorAdapter createCropAugmentor(int size, int numClasses) {
            return AugmentorAdapter(
                    std::make_shared<chianti::CropAugmentor>(size, numClasses));
        }
        
        /**
         * Creates a combined Augmentor.
         */
        static AugmentorAdapter createCombinedAugmentor(
                const boost::python::object & augmentors);
        
    protected:
        /**
         * Pointer to the underlying augmentor.
         */
        std::shared_ptr<chianti::AugmentorInterface> augmentor;
    };
    
} // namespace pychianti

#endif