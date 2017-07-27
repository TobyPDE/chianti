.. Chianti documentation master file, created by
   sphinx-quickstart on Thu Feb 16 23:06:23 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Chianti
===================================

This is a Python library for loading and augmenting training data asynchronously
in the background. It is primarily geared towards pipelines for semantic 
segmentation where you have a source image and a densely annotated target image.

The library consists of four major components:

* A set of image loaders that provide the raw image data. 
* A set of iterators that determine the order in which we iterate over a given 
  dataset
* A set of augmentors for synthetically increasing the diversity of the data set. 
* A data provider that connects all components and returns batches of augmented 
  images.

.. py:class:: DataProvider

    This class allows you to load batches of (source, target) pairs 
    asynchronously. Standard transformations:

    .. py:method:: __init__(augmentor, source_img_loader, target_img_loader, iterator, batch_size)

        Initializes a new instance of the DataProvider class.

        :param augmentor: An instance of :py:class:`Augmentor`.
        :param source_img_loader: An instance of :py:class:`Loader`.
        :param target_img_loader: An instance of :py:class:`Loader`.
        :param iterator: An instance of :py:class:`Iterator`.
        :param batchsize: The size of the image batches.
        :type augmentor: Augmentor
        :type source_img_loader: Loader
        :type arget_img_loader: Loader
        :type iterator: Iterator
        :type batchsize: int

    .. py:method:: next()

        Returns the next batch of images.

        :return: A tuple of two numpy arrays.
        
    .. py:method:: reset()

        Resets the underlying iterator to the beginning. This is useful if you 
        want to iterate over a dataset deterministically.

    .. py:method:: get_num_batches()

        Returns the total number of batches per epoch. 
        
        :return: The number of batches per epoch.
        :rtype: int
        

.. py:class:: Iterator

    A data iterator class.

    .. py:method:: next()

        Returns the next item. 

        :return: A tuple of two strings.
        
    .. py:method:: reset()

        Resets the iterator to its initial state. This also works for iterators
        that involve randomness.

    .. py:method:: get_num_elements()

        Returns the total number of elements in the structure the we iterate 
        over. 
        
        :return: The number of elements in the underlying structure.
        :rtype: int
        
    .. py:staticmethod:: Sequential(data_list)
        
        Factory method that creates a new sequential iterator. A sequential 
        iterator iterates deterministically over the provided list of elements.

        :param data_list: A list of string tuples.
        :type data_list: list
        
    .. py:staticmethod:: Random(data_list)
        
        Factory method that creates a new random iterator. A random
        iterator iterates over the dataset randomly in epochs. This means at the
        beginning of each epoch, the dataset is shuffled and the iterated over
        deterministically.

        :param data_list: A list of string tuples.
        :type data_list: list
        
    .. py:staticmethod:: WeightedRandom(data_list, weights)
        
        Factory method that creates a new weighted random iterator. This 
        iterator draws each training example independently from the dataset 
        with a probability proportional to the associated weights.

        :param data_list: A list of string tuples.
        :param weights: A list or numpy array of non-negative weights. There 
                        must be exactly one weight for each element of the data
                        list.
        :type data_list: list


.. py:class:: Loader
    
    A loader class for providing raw image data.

    .. py:staticmethod:: RGB()

        Factory method for creating a loader that simply loads RGB images from 
        the hard drive. 

    .. py:staticmethod:: Label()

        Factory method for creating a loader that loads 8bit class label images
        from the hard drive.

    .. py:staticmethod:: ValueMapper(value_map)

        Factory method for creating a loader that loads an 8bit class label
        image from the hard drive and then maps the pixel values according to
        the provided mapping.

        :param value_map: A list of numpy array of exactly 256 elements that 
                          represents a permutation. 

    .. py:staticmethod:: ColorMapper(color_map)

        Factory method for creating a loader that loads an RGB class label
        image from the hard drive and then maps the pixel values according to
        the provided mapping.

        :param color_map: A dict that maps colors (RGB tuples) to 8bit integer
                          values.


.. py:class:: Augmentor

    A data augmentation class. 
    
    .. py:staticmethod:: Subsample(factor)

        Factory method that creates an augmentor that subsamples the source and
        the target image by the given factor.

        :param factor: Subsampling factor.
        :type factor: int
    
    .. py:staticmethod:: Gamma(strength)

        Factory method that creates an augmentor that performs random gamma
        augmentation on the source image.

        :param strength: Strength of the augmentation. Must be a value between 0
                         and 0.5.
        :type strength: double
    
    .. py:staticmethod:: Translation(factor)

        Factory method that creates an augmentor that randomly translates both
        the source and the target image. On the source image, it uses reflection
        padding whereas on the target image it uses constant padding with void 
        labels (value -1).

        :param offset: Maximum translation offset in any direction.
        :type offset: int
    
    .. py:staticmethod:: Zooming(offset)

        Factory method that creates an augmentor that randomly zooms into the 
        image center. 

        :param factor: Maximum zooming factor.
        :type factor: double
    
    .. py:staticmethod:: Rotation(angel)

        Factory method that creates an augmentor that randomly rotates the 
        images.

        :param angel: Maximum rotation angel.
        :type angel: double
    
    .. py:staticmethod:: Crop(size, num_classes)

        Factory method that creates a crop augmentor. A crop augmentor randomly
        samples square crops from an image. The probability of a crop being 
        sampled is proportional to the entropy of the class distribution within
        the crop.

        :param size: The size of the crop. 
        :param num_classes: The total number of classes. 
        :type size: int
        :type num_classes: int

    .. py:staticmethod:: Saturation(delta_min, delta_max)

        Factory method that creates an augmentor that randomly augments the 
        image's saturation.

        :param delta_min: Minimum saturation multiplier. 
        :param delta_max: Maximum saturation multiplier. 
        :type delta_min: double
        :type delta_max: double

    .. py:staticmethod:: Hue(delta_min, delta_max)

        Factory method that creates an augmentor that randomly augments the 
        image's hue.

        :param delta_min: Minimum hue multiplier. 
        :param delta_max: Maximum hue multiplier. 
        :type delta_min: double
        :type delta_max: double
