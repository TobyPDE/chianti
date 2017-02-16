# Chianti

This Python extension is for people who train semantic segmentation system with Python. 
It is primarily geared towards users of deep learning libraries that don't offer data streams (i.e. Theano/Lasagne).
With this extension, you can load batches of (augmented) images and label images asynchronously. 
This allows you to maximize GPU load during training.

# Installation

```
$ pip install git+https://github.com/TobyPDE/chianti
``` 
# Documentation

Read here: [http://chianti.readthedocs.io/en/latest/]()

# Usage

Assume that `files` is a list of filename tuples. 
The first entry of each entry is the filename of the source image while the second entry is the target filename.
Then the following creates a new data provider that iterates in epochs randomly over `files`:
The pre-processing steps consist of subsampling the images by a factor of 4 and then applying random gamma and translation augmentation steps.

```
provider = chianti.DataProvider(
    iterator=chianti.random_iterator(files),
    batchsize=2,
    augmentors=[
        chianti.subsample_augmentor(4),
        chianti.gamma_augmentor(0.1),
        chianti.translation_augmentor(20)
    ]
)

```

# License

The MIT License (MIT) Copyright (c) 2017 Tobias Pohlen

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.