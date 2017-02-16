.. Chianti documentation master file, created by
   sphinx-quickstart on Thu Feb 16 23:06:23 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Chianti
===================================

This Python extension is for people who train semantic segmentation systems with Python. 
It is primarily aimed at users of deep learning libraries that don't offer asynchronous data streams (e.g. Theano/Lasagne).
This extension allows you to load batches of (augmented) images and label images asynchronously. 
This allows you to maximize GPU load during training.

.. py:class:: DataProvider

    This class allows you to load batches of (source, target) pairs asynchronously. Standard transformations:

    - The source image (RGB) is loaded and then cast to float32. Its intensity values are scaled to [0, 1]
    - The target image (single channel 8-bit) is loaded and then cast to int32. Pixels of value 255 are mapped to the value -1, which is considered to be a void label.

    .. py:method:: __init__(iterator, batchsize, [augmentors])

        Initializes a new instance of the DataProvider class.

        :param iterator: An instance of :py:class:`Iterator`.
        :param batchsize: The size of the image batches.
        :param augmentors: A list of augmentation steps.
        :type iterator: Iterator
        :type batchsize: int
        :type augmentors: list(Augmentor)

    .. py:method:: next()

        Returns the next batch of images.

        :return: A tuple of two numpy arrays. The first array contains the source batch and the second entry contains the target batch.
        
    .. py:method:: reset()

        Resets the underlying iterator to the beginning. This is useful if you want to deterministically iterate over a dataset.

    .. py:method:: get_num_batches()

        Returns the total number of batches per epoch. 

        :return: The number of batches per epoch.
        :rtype: int
        

.. py:class:: Iterator

    A data iterator class. Use one of these factory methods in order to create new instances:

    - :py:func:`sequential_iterator`
    - :py:func:`random_iterator`


.. py:function:: sequential_iterator(files)

    Creates a new :py:class:`Iterator` instance that iterates sequentially over the dataset.

    :param files: A list of filename tuples. The first entry of each tuple is considered to be the filename of the source image. The second entry is considered to be the target filename.
    :type files: list(tuple[string, string])
    :return: A new sequantial iterator.
    :rtype: Iterator

.. py:function:: random_iterator(files)

    Creates a new :py:class:`Iterator` instance that iterates randomly over the dataset. The iterator uses epoch. This means that it traverses the entire dataset in a random order, which changes after each pass over the dataset.

    :param files: A list of filename tuples. The first entry of each tuple is considered to be the filename of the source image. The second entry is considered to be the target filename.
    :type files: list(tuple[string, string])
    :return: A new random iterator.
    :rtype: Iterator

.. py:class:: Augmentor

    A data augmentation class. Use one of these factory methods in order to create new instances:

    - :py:func:`subsample_augmentor`
    - :py:func:`gamma_augmentor`
    - :py:func:`translation_augmentor`
    - :py:func:`cityscapes_label_transformation_augmentor`
    

.. py:function:: subsample_augmentor(factor)

    Creates a new :py:class:`Augmentor` instance that resizes the images by subsampling them by a given subsampling factor.

    :param factor: The subsampling factor.
    :type factor: int
    :return: A new subsample augmentor.
    :rtype: Augmentor


.. py:function:: gamma_augmentor(gamma)

    Creates a new :py:class:`Augmentor` instance that applies brightness augmentation by performining random gamma corrections.

    :param gamma: Determines the strength of the gamma augmentation. Valid values are in (0, 0.5) where 0 corresponds to no augmentation and 0.5 corresponds to the strongest augmentation.
    :type gamma: double
    :return: A new gamma augmentor.
    :rtype: Augmentor

.. py:function:: translation_augmentor(offset)

    Creates a new :py:class:`Augmentor` instance that applies random translation augmentation.

    :param offset: The maximum offset by which an image is translated.
    :type offset: int
    :return: A new translation augmentor.
    :rtype: Augmentor

.. py:function:: cityscapes_label_transformation_augmentor()

    Creates a new :py:class:`Augmentor` instance that maps CityScapes label ids to CityScapes training ids. The resulting target images will have values in [-1, 18] where -1 corresponds to void labels.

    :return: A new cityscapes label transformation augmentor.
    :rtype: Augmentor
