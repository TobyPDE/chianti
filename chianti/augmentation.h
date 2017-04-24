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
            resizeLabels(data.target);
        }

    private:
        /**
         * Resizes a label image.
         */
        void resizeLabels(cv::Mat & t) const
        {
            // Allocate the new image
            cv::Mat tNew(t.rows / samplingFactor, t.cols / samplingFactor, CV_8UC1);

            for (int i = 0; i < tNew.rows; i++)
            {
                for (int j = 0; j < tNew.cols; j++)
                {
                    // Create a histogram of labels
                    int histogram[256] = {0};

                    for (int blockI = i * samplingFactor; blockI < (i + 1) * samplingFactor; blockI++)
                    {
                        for (int blockJ = j * samplingFactor; blockJ < (j + 1) * samplingFactor; blockJ++)
                        {
                            histogram[t.at<uchar>(blockI, blockJ)]++;
                        }
                    }

                    // Determine the best label
                    int bestLabel = 0;
                    int bestLabelFrequency = 0;
                    for (int k = 0; k < 256; k++)
                    {
                        if (histogram[k] > bestLabelFrequency)
                        {
                            bestLabelFrequency = histogram[k];
                            bestLabel = k;
                        }
                    }

                    // Is the label sufficiently distinct?
                    if (bestLabelFrequency > 0.5 * samplingFactor * samplingFactor)
                    {
                        tNew.at<uchar>(i, j) = static_cast<uchar>(bestLabel);
                    }
                    else
                    {
                        tNew.at<uchar>(i, j) = 255;
                    }
                }
            }

            tNew.copyTo(t);
        }

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

    /**
     * Augments the image by randomly zooming in and out.
     */
    class ZoomingAugmentor : public IAugmentor {
    public:
        /**
         * Initializes a new instance of the TranslationAugmentor class.
         */
        ZoomingAugmentor(double range) : range(range) {}

        /**
         * Augments the given ImageLabelPair.
         */
        void augment(ImageLabelPair& data) const
        {
            std::random_device rd;
            std::mt19937 g(rd());
            std::uniform_real_distribution<double> d(1 - range, 1 + range);

            // Sample the zooming factor
            const double factor = d(g);

            // Compute the new image size
            const int rows = static_cast<int>(data.img.rows * factor);
            const int cols = static_cast<int>(data.img.cols * factor);

            // Resize the data
            cv::Mat img, target;
            cv::resize(data.img, img, cv::Size(cols, rows), 0, 0, CV_INTER_LANCZOS4);
            cv::resize(data.target, target, cv::Size(cols, rows), 0, 0, CV_INTER_NN);

            cv::Mat finalImg = cv::Mat::zeros(data.img.rows, data.img.cols, CV_32FC3);
            cv::Mat finalTarget = 255 * cv::Mat::ones(data.img.rows, data.img.cols, CV_8UC1);

            // Create the final images
            // If the images are up-sampled, they have to be cropped
            // If the images are down-sampled, they have to be embedded into a bigger image
            if (factor > 1.0)
            {
                // Up-sample -> crop
                // Calculate the offset on both sides
                const int rowOffset = (rows - finalImg.rows) / 2;
                const int colOffset = (cols - finalImg.cols) / 2;

                for (int i = 0; i < finalImg.rows; i++)
                {
                    for (int j = 0; j < finalImg.cols; j++)
                    {
                        finalImg.at<cv::Vec3f>(i, j) = img.at<cv::Vec3f>(i + rowOffset, j + colOffset);
                        finalTarget.at<uchar>(i, j) = target.at<uchar>(i + rowOffset, j + colOffset);
                    }
                }
            }
            else
            {
                // Down-sample
                // Calculate the offset on both sides
                const int rowOffset = (finalImg.rows - rows) / 2;
                const int colOffset = (finalImg.cols - cols) / 2;

                for (int i = 0; i < img.rows; i++)
                {
                    for (int j = 0; j < img.cols; j++)
                    {
                        finalImg.at<cv::Vec3f>(i + rowOffset, j + colOffset) = img.at<cv::Vec3f>(i, j);
                        finalTarget.at<uchar>(i + rowOffset, j + colOffset) = target.at<uchar>(i, j);
                    }
                }
            }

            finalImg.copyTo(data.img);
            finalTarget.copyTo(data.target);
        }

    private:
        /**
         * The zooming range. The zooming factor will be sampled from (1 - range, 1 + range).
         */
        double range;
    };

    /**
     * Augments the image by randomly rotating the image.
     */
    class RotateAugmentor : public IAugmentor {
    public:
        /**
         * Initializes a new instance of the RotateAugmentor class.
         */
        RotateAugmentor(double range) : range(range) {}

        /**
         * Augments the given ImageLabelPair.
         */
        void augment(ImageLabelPair& data) const
        {
            std::random_device rd;
            std::mt19937 g(rd());
            std::uniform_real_distribution<double> d(-range, range);

            // Sample the rotation angle
            double factor = d(g);
            if (factor < 0)
            {
                factor += 360;
            }

            const int rows = static_cast<int>(data.img.rows);
            const int cols = static_cast<int>(data.img.cols);

            // Rotate the data
            cv::Mat img, target;
            cv::Mat M = cv::getRotationMatrix2D(cv::Point2f(cols/2,rows/2), factor, 1);

            cv::warpAffine(data.img, img, M, cv::Size(cols, rows));
            cv::warpAffine(data.target, target, M, cv::Size(cols, rows), CV_INTER_NN, cv::BORDER_CONSTANT, 255);

            img.copyTo(data.img);
            target.copyTo(data.target);
        }

    private:
        /**
         * The rotation angle range.
         */
        double range;
    };

    /**
     * Augments the image by randomly convolving it with a Gaussian kernel..
     */
    class BlurAugmentor : public IAugmentor {
    public:
        /**
         * Initializes a new instance of the BlurAugmentor class.
         */
        BlurAugmentor(double range) : range(range) {}

        /**
         * Augments the given ImageLabelPair.
         */
        void augment(ImageLabelPair& data) const
        {
            if (this->range <= 0)
            {
                return;
            }

            std::random_device rd;
            std::mt19937 g(rd());
            std::uniform_real_distribution<double> d(0, range);

            // Blur the data
            const double sigma = d(g);
            cv::Mat img;
            int kernelWidth = static_cast<int>(std::ceil(sigma) * 3);
            if ((kernelWidth % 2) == 0)
            {
                kernelWidth++;
            }
            cv::GaussianBlur(data.img, img, cv::Size(kernelWidth, kernelWidth), sigma);

            img.copyTo(data.img);
        }

    private:
        /**
         * The rotation angle range.
         */
        double range;
    };

    /**
     * Augments the image by randomly adjusting the saturation.
     */
    class SaturationAugmentor : public IAugmentor {
    public:
        /**
         * Initializes a new instance of the SaturationAugmentor class.
         */
        SaturationAugmentor(double rangeFrom, double rangeTo) : rangeFrom(rangeFrom), rangeTo(rangeTo) {}

        /**
         * Augments the given ImageLabelPair.
         */
        void augment(ImageLabelPair& data) const
        {
            std::random_device rd;
            std::mt19937 g(rd());
            std::uniform_real_distribution<double> d(rangeFrom, rangeTo);

            const int rows = static_cast<int>(data.img.rows);
            const int cols = static_cast<int>(data.img.cols);

            // Blur the data
            const float offset = static_cast<float>(d(g));
            cv::Mat img;
            cv::cvtColor(data.img, img, CV_BGR2HSV);

            // Adjust the saturation channel
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    img.at<cv::Vec3f>(i, j)[1] *= offset;
                    const auto value = img.at<cv::Vec3f>(i, j)[1];
                    img.at<cv::Vec3f>(i, j)[1] = std::max(0.0f, std::min(1.0f, value));
                }
            }
            
            cv::cvtColor(img, data.img, CV_HSV2BGR);
        }

    private:
        /**
         * The parameter range
         */
        double rangeFrom;
        double rangeTo;
    };

    /**
     * Augments the image by randomly adjusting the saturation.
     */
    class HueAugmentor : public IAugmentor {
    public:
        /**
         * Initializes a new instance of the HueAugmentor class.
         */
        HueAugmentor(double rangeFrom, double rangeTo) : rangeFrom(rangeFrom), rangeTo(rangeTo) {}

        /**
         * Augments the given ImageLabelPair.
         */
        void augment(ImageLabelPair& data) const
        {
            std::random_device rd;
            std::mt19937 g(rd());
            std::uniform_real_distribution<double> d(rangeFrom, rangeTo);

            const int rows = static_cast<int>(data.img.rows);
            const int cols = static_cast<int>(data.img.cols);

            // Blur the data
            const float offset = static_cast<float>(d(g));
            cv::Mat img;
            cv::cvtColor(data.img, img, CV_BGR2HSV);

            // Adjust the saturation channel
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    auto value = img.at<cv::Vec3f>(i, j)[0];
                    value += offset;

                    if (value > 360)
                    {
                        value -= 360;
                    }
                    else if (value < 0)
                    {
                        value += 360;
                    }
                    img.at<cv::Vec3f>(i, j)[0] = value;
                }
            }
            
            cv::cvtColor(img, data.img, CV_HSV2BGR);
        }

    private:
        /**
         * The parameter range
         */
        double rangeFrom;
        double rangeTo;
    };

    /**
     * Augments the image by randomly adjusting the saturation.
     */
    class BrightnessAugmentor : public IAugmentor {
    public:
        /**
         * Initializes a new instance of the BrightnessAugmentor class.
         */
        BrightnessAugmentor(double rangeFrom, double rangeTo) : rangeFrom(rangeFrom), rangeTo(rangeTo) {}

        /**
         * Augments the given ImageLabelPair.
         */
        void augment(ImageLabelPair& data) const
        {
            std::random_device rd;
            std::mt19937 g(rd());
            std::uniform_real_distribution<double> d(rangeFrom, rangeTo);

            const int rows = static_cast<int>(data.img.rows);
            const int cols = static_cast<int>(data.img.cols);

            // Blur the data
            const float offset = static_cast<float>(d(g));

            // Adjust the saturation channel
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    for (int c = 0; c < 3; c++)
                    {
                        auto value = data.img.at<cv::Vec3f>(i, j)[c];
                        value += offset;

                        if (value > 1)
                        {
                            value = 1;
                        }
                        else if (value < 0)
                        {
                            value = 0;
                        }
                        data.img.at<cv::Vec3f>(i, j)[c] = value;
                    }
                }
            }
        }

    private:
        /**
         * The parameter range
         */
        double rangeFrom;
        double rangeTo;
    };
}