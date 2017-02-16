from distutils.core import setup, Extension
import numpy as np

module = Extension('chianti',
                   libraries=['opencv_core', 'opencv_highgui', 'opencv_imgproc', 'opencv_imgcodecs'],
                   include_dirs=[np.get_include()],
                   extra_link_args=['-lgomp'],
                   extra_compile_args=['-std=c++11', '-fopenmp'],
                   sources=['dataprovider.cpp'])

setup( name='Chianti',
       version='1.0',
       description='This package allows asynchronous loading and pre-processing of training data for semantic segmentation.',
       author='Tobias Pohlen',
       author_email='tobias.pohlen@gmail.com',
       ext_modules=[module])