# Chianti

This is a Python library for loading and augmenting training data asynchronously
in the background. It is primarily geared towards pipelines for semantic 
segmentation where you have a source image and a densely annotated target image.

The library consists of four major components:

* A set of image loaders that provide the raw image data. 
* A set of iterators that determine the order in which we iterate over a given 
  dataset
* A set of augmentors for synthetically increasing the diversity of the data 
  set. 
* A data provider that connects all components and returns batches of augmented 
  images.

# Installation

You need to have OpenCV and Boost installed. 
```
$ sudo apt-get install libopencv-dev libboost-all-dev

```

Check out the repository. 

```
$ git clone https://github.com/TobyPDE/chianti
```

Now, change into the directory and create a build folder.

```
$ cd chianti
$ mkdir build
$ cd build
```

Depending on your Python version (2.7 or 3.4), execute one of the two following 
commands

```
$ cmake .. -Dpython_version=2 -DCMAKE_BUILD_TYPE=Release
```

Or 

```
$ cmake .. -Dpython_version=3 -DCMAKE_BUILD_TYPE=Release
```

Build the library and the Python bindings.

```
$ make -j
```

Install the library system-wide.

```
$ sudo make install
```

# Documentation

Read here: [http://chianti.readthedocs.io/en/latest/](http://chianti.readthedocs.io/en/latest/)

# Usage

Assume that `files` is a list of filename tuples. The first entry of each entry
is the filename of the source image while the second entry is the target 
filename. Then the following creates a new data provider that iterates in 
epochs randomly over `files`. The pre-processing step consists of subsampling 
the images by a factor of 4. 

```
batch_size = 3
num_classes = 19
return pychianti.DataProvider(
    pychianti.Augmentor.Subsample(4),
    pychianti.Loader.RGB(),
    pychianti.Loader.Label(),
    pychianti.Iterator.Sequential(files),
    batch_size,
    num_classes)
```

# License

The MIT License (MIT) Copyright (c) 2017 Tobias Pohlen

The MIT License (MIT) Copyright (c) 2017 Google Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies 
of the Software, and to permit persons to whom the Software is furnished to do 
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE.
