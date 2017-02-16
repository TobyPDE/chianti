#pragma once

#include <memory>
#include <opencv2/opencv.hpp>

#include "common.h"

namespace Chianti {
    /**
     * This is the interface implemented by augmentation operations.
     */
    class IAugmentor {
    public:
        /**
         * Augments the given ImageLabelPair.
         */
        virtual void augment(ImageLabelPair&) const = 0;
    };

    /**
     * The augmentor combines several augmentors into one.
     */
    class CombinedAugmentor : public IAugmentor {
    public:
        /**
         * Adds a new augmentor.
         */
        void addAugmentor(std::shared_ptr<IAugmentor> augmentor)
        {
            this->augmentors.push_back(augmentor);
        }

        /**
         * Augments the given ImageLabelPair.
         */
        void augment(ImageLabelPair& data) const
        {
            for(size_t i = 0; i < this->augmentors.size(); i++)
            {
                this->augmentors[i]->augment(data);
            }
        }

    private:
        /**
         * The individual augmentation steps.
         */
        std::vector<std::shared_ptr<IAugmentor>> augmentors;
    };

    /**
     * This augmentor transforms the image to the float32 format and scales the intensities to [0, 1].
     */
    class CastToFloatAugmentor : public IAugmentor {
    public:
        /**
         * Augments the given ImageLabelPair.
         */
        void augment(ImageLabelPair& data) const
        {
            // Cast to float
            cv::Mat floatImg;
            data.img.convertTo(floatImg, CV_32FC3);

            // Scale to [0, 1]
            floatImg /= 255.0f;

            data.img = floatImg;
        }
    };

    /**
     * This augmentor subsamples the image and the label image by given factor.
     */
    class SubsampleAugmentor : public IAugmentor {
    public:
        /**
         * Initializes a new instance of the SubsampleAugmentor class.
         */
        SubsampleAugmentor(int samplingFactor) : samplingFactor(samplingFactor) {}

        /**
         * Augments the given ImageLabelPair.
         */
        void augment(ImageLabelPair& data) const
        {
            auto newSize = cv::Size(data.img.cols / samplingFactor, data.img.rows / samplingFactor);
            cv::resize(data.img, data.img, newSize, 0, 0, CV_INTER_LANCZOS4);
            cv::resize(data.target, data.target, newSize, 0, 0, CV_INTER_NN);
        }

    private:
        /**
         * The subsampling factor
         */
        int samplingFactor;
    };

    /**
     * Applies a transformation from label ids to training ids. This is specific to the cityscapes dataset.
     */
    class CSLabelTransformationAugmentation : public IAugmentor {
    public:
        /**
         * Initializes a new instance of the CSLabelTransformationAugmentation class.
         */
        CSLabelTransformationAugmentation()
        {
            fillTransformationTable();
        }

        /**
         * Augments the given ImageLabelPair.
         */
        void augment(ImageLabelPair& data) const
        {
            for (int i = 0; i < data.target.rows; i++)
            {
                for (int j = 0; j < data.target.cols; j++)
                {
                    data.target.at<uchar>(i, j) = static_cast<uchar>(transformationTable[data.target.at<uchar>(i, j)]);
                }
            }
        }

    private:
        /**
         * Fills the transformation table.
         */
        void fillTransformationTable()
        {
            transformationTable.resize(34);
            transformationTable[0] = 255;

            transformationTable[0] = 255;
            transformationTable[1] = 255;
            transformationTable[2] = 255;
            transformationTable[3] = 255;
            transformationTable[4] = 255;
            transformationTable[5] = 255;
            transformationTable[6] = 255;
            transformationTable[7] = 0;
            transformationTable[8] = 1;
            transformationTable[9] = 255;
            transformationTable[10] = 255;
            transformationTable[11] = 2;
            transformationTable[12] = 3;
            transformationTable[13] = 4;
            transformationTable[14] = 255;
            transformationTable[15] = 255;
            transformationTable[16] = 255;
            transformationTable[17] = 5;
            transformationTable[18] = 255;
            transformationTable[19] = 6;
            transformationTable[20] = 7;
            transformationTable[21] = 8;
            transformationTable[22] = 9;
            transformationTable[23] = 10;
            transformationTable[24] = 11;
            transformationTable[25] = 12;
            transformationTable[26] = 13;
            transformationTable[27] = 14;
            transformationTable[28] = 15;
            transformationTable[29] = 255;
            transformationTable[30] = 255;
            transformationTable[31] = 16;
            transformationTable[32] = 17;
            transformationTable[33] = 18;
        }

        /**
         * This table defines which label ids are mapped to which training ids.
         */
        std::vector<int> transformationTable;
    };

    /**
     * Augments the image brithness using random gamma augmentation.
     */
    class GammaAugmentor : public IAugmentor {
    public:
        /**
         * Initializes a new instance of the CastToFloatAugmentor class.
         */
        GammaAugmentor(double a) : a(a) {}

        /**
         * Augments the given ImageLabelPair.
         */
        void augment(ImageLabelPair& data) const
        {
            std::random_device rd;
            std::mt19937 g(rd());
            std::uniform_real_distribution<float> d(std::max(-0.5f, -static_cast<float>(a)), std::min(0.5f, static_cast<float>(a)));

            // Sample the gamma value
            float gamma = d(g);

            // Apply the non-linear transformation
            gamma = std::log(0.5f + 1.0f / std::sqrt(2.0f) * gamma) / std::log(0.5f - 1.0f / std::sqrt(2.0f) * gamma);

            for (int i = 0; i < data.img.rows; i++)
            {
                for (int j = 0; j < data.img.cols; j++)
                {
                    for (int c = 0; c < 3; c++)
                    {
                        data.img.at<cv::Vec3f>(i, j)[c] = std::pow(data.img.at<cv::Vec3f>(i, j)[c], gamma);
                    }
                }
            }
        }

    private:
        /**
         * The augmentation parameter
         */
        double a;
    };

    /**
     * Augments the image by randomly translating it.
     */
    class TranslationAugmentor : public IAugmentor {
    public:
        /**
         * Initializes a new instance of the TranslationAugmentor class.
         */
        TranslationAugmentor(int offset) : offset(offset) {}

        /**
         * Augments the given ImageLabelPair.
         */
        void augment(ImageLabelPair& data) const
        {
            std::random_device rd;
            std::mt19937 g(rd());
            std::uniform_int_distribution<int> d(-offset, offset);

            // Sample the translation offset
            int translation_x = d(g);
            int translation_y = d(g);

            cv::Mat newImage(data.img.rows, data.img.cols, CV_32FC3);
            cv::Mat newTarget(data.target.rows, data.target.cols, CV_8UC1);

            for (int i = 0; i < data.img.rows; i++)
            {
                for (int j = 0; j < data.img.cols; j++)
                {
                    // Compute the offset position
                    int i_translation = i + translation_x;
                    int j_translation = j + translation_y;
                    bool outOfScore = false;

                    if (i_translation < 0)
                    {
                        i_translation = std::abs(i_translation);
                        outOfScore = true;
                    }
                    else if (i_translation >= data.img.rows)
                    {
                        i_translation = 2 * data.img.rows - i_translation - 1;
                        outOfScore = true;
                    }

                    if (j_translation < 0)
                    {
                        j_translation = std::abs(j_translation);
                        outOfScore = true;
                    }
                    else if (j_translation >= data.img.cols)
                    {
                        j_translation = 2 * data.img.cols - j_translation - 1;
                        outOfScore = true;
                    }

                    newImage.at<cv::Vec3f>(i, j) = data.img.at<cv::Vec3f>(i_translation, j_translation);
                    if (!outOfScore)
                    {
                        newTarget.at<uchar>(i, j) = data.target.at<uchar>(i_translation, j_translation);
                    }
                    else
                    {
                        newTarget.at<uchar>(i, j) = 255;
                    }
                }
            }

            data.img = newImage;
            data.target = newTarget;
        }

    private:
        /**
         * The maximum pixel offset
         */
        int offset;
    };
}