#include <vector>
#include <thread>
#include <mutex>

namespace Chianti {

    /**
     * This is the interface of an iterator.
     */
    template <class T>
    class IIterator {
    public:
        /**
         * Returns the next element of 0 if the end of the resource has been obtained.
         * @return
         */
        virtual T* next() = 0;

        /**
         * Resets the iterator
         */
         virtual void reset() = 0;

        /**
         * Returns the number of elements that we iterate over
         */
         virtual size_t size() const = 0;
    };

    /**
     * This class iterates randomly over a list of elements.
     * This implementation is thread safe.
     */
    template <class T>
    class RandomIterator : public IIterator<T> {
    public:
        /**
         * Initializes a new instance of the RandomIterator class.
         *
         * @param resources The resource to iterator over randomly.
         */
        RandomIterator(const std::vector<T> & resources) : resources(resources), indices(resources.size()), index(0)
        {
            // Prepare the indices
            std::iota(indices.begin(), indices.end(), 0);
            shuffle();
        }

        /**
         * Returns the next element.
         *
         * @return The next element in the list.
         */
        T* next()
        {
            std::lock_guard<std::mutex> lock(mutex);

            // Do we need to shuffle the data?
            if (index >= resources.size())
            {
                shuffle();
            }

            T* result = &resources[indices[index]];
            index++;
            return result;
        }

        /**
         * Resets the iterator
         */
        void reset()
        {
            index = 0;
        }

        /**
         * Returns the number of elements that we iterate over
         */
        size_t size() const
        {
            return resources.size();
        }

    private:
        /**
         * Shuffles the order of elements.
         */
        void shuffle()
        {
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(indices.begin(), indices.end(), g);
            index = 0;
        }

        /**
         * The underlying resource
         */
        std::vector<T> resources;
        /**
         * The current index order.
         */
        std::vector<size_t> indices;
        /**
         * The current index.
         */
        size_t index;
        /**
         * The mutex that protect concurrent access to the iterator.
         */
        std::mutex mutex;
    };

    /**
     * This class iterates sequentially over the datapoints.
     */
    template <class T>
    class SequentialIterator : public IIterator<T> {
    public:
        /**
         * Initializes a new instance of the SequentialIterator class.
         *
         * @param resources The resource to iterator over randomly.
         */
        SequentialIterator(const std::vector<T> & resources) : resources(resources), index(0) { }

        /**
         * Returns the next element.
         *
         * @return The next element in the list.
         */
        T* next()
        {
            std::lock_guard<std::mutex> lock(mutex);

            // Do we need to shuffle the data?
            if (index >= resources.size())
            {
                index = 0;
            }

            T* result = &resources[index];
            index++;
            return result;
        }

        /**
         * Resets the iterator
         */
        void reset()
        {
            index = 0;
        }

        /**
         * Returns the number of elements that we iterate over
         */
        size_t size() const
        {
            return resources.size();
        }

    private:
        /**
         * The underlying resource
         */
        std::vector<T> resources;
        /**
         * The current index.
         */
        size_t index;
        /**
         * The mutex that protect concurrent access to the iterator.
         */
        std::mutex mutex;
    };

    /**
     * This class samples each element according to a predefined probability.
     */
    template <class T>
    class SampleIterator : public IIterator<T> {
    public:
        /**
         * Initializes a new instance of the SampleIterator class.
         *
         * @param resources The resource to iterator over randomly.
         */
        SampleIterator(const std::vector<T> & resources, const std::vector<double> & weights) : resources(resources), weights(weights)
        {
            assert(resources.size() == weights.size());
            normalizeWeights();
        }

        /**
         * Returns the next element.
         *
         * @return The next element in the list.
         */
        T* next()
        {
            // Sample a number between 0 and 1
            std::random_device rd;
            std::mt19937 g(rd());
            std::uniform_real_distribution<double> d(0, 1);
            const double u = d(g);
            double cumsum = 0;
            for (size_t i = 0; i < resources.size(); i++)
            {
                if (cumsum <= u && u < cumsum + weights[i])
                {
                    return &resources[i];
                }
                else
                {
                    cumsum += weights[i];
                }
            }

            return &resources[resources.size() - 1];
        }

        /**
         * Resets the iterator
         */
        void reset() {}

        /**
         * Returns the number of elements that we iterate over
         */
        size_t size() const
        {
            return resources.size();
        }

    private:
        void normalizeWeights()
        {
            // Normalize the weights
            double total = 0.0;
            for (size_t i = 0; i < weights.size(); i++) {
                weights[i] = std::max(weights[i], 0.0);
                total += weights[i];
            }

            for (size_t i = 0; i < weights.size(); i++) {
                weights[i] /= total;
            }
        }

        /**
         * The underlying resource
         */
        std::vector<T> resources;
        /**
         * The weights
         */
        std::vector<double> weights;
    };
}