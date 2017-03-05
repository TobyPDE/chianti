#include "Python.h"
#include "numpy/arrayobject.h"

#include "chianti/provider.h"
#include "utils.h"
#include <structmember.h>

// =====================================================================================================================
// PyIterator
// This class represents an iterator over pairs of image and label image paths. The class does not expose any methods
// to the python environment.
// =====================================================================================================================

typedef struct {
    PyObject_HEAD
    PyObject* imgs;
    PyObject* targets;
} PyBatch;

// Define the methods (heads up: there are no)
static PyMethodDef PyBatch_methods[] = {
        {NULL}
};

static PyMemberDef PyBatch_members[] = {
        {"imgs", T_OBJECT_EX, offsetof(PyBatch, imgs), 0, "Img batch"},
        {"targets", T_OBJECT_EX, offsetof(PyBatch, targets), 0, "Target batch"},
        {NULL}  /* Sentinel */
};

static void PyBatch_dealloc(PyBatch* self)
{
    Py_XDECREF(self->imgs);
    Py_XDECREF(self->targets);
}

static int PyBatch_init(PyBatch *self, PyObject *args, PyObject *kwds)
{
    self->imgs = Py_None;
    self->targets = Py_None;
}

static PyTypeObject PyBatchType = {
        PyVarObject_HEAD_INIT(NULL, 0)
        // tp_name
        "chianti.Iterator",
        // tp_basicsize
        sizeof(PyBatch),
        // tp_itemsize
        0,
        // tp_dealloc
        (destructor) PyBatch_dealloc,
        // tp_print
        0,
        // tp_getattr
        0,
        // tp_setattr
        0,
        // tp_reserved
        0,
        // tp_repr
        0,
        // tp_as_number
        0,
        // tp_as_sequence
        0,
        // tp_as_mapping
        0,
        // tp_hash
        0,
        // tp_call
        0,
        // tp_str
        0,
        // tp_getattro
        0,
        // tp_setattro
        0,
        // tp_as_buffer
        0,
        // tp_flags
        Py_TPFLAGS_DEFAULT,
        // tp_doc
        "A batch of images and targets.",
        // tp_traverse
        0,
        // tp_clear
        0,
        // tp_richcompare
        0,
        // tp_weaklistoffset
        0,
        // tp_iter
        0,
        // tp_iternext
        0,
        // tp_methods
        PyBatch_methods,
        // tp_members
        PyBatch_members,
        // tp_getset
        0,
        // tp_base
        0,
        // tp_dict
        0,
        // tp_descr_get
        0,
        // tp_descr_set
        0,
        // tp_dictoffset
        0,
        // tp_init
        (initproc)PyBatch_init};

// =====================================================================================================================
// PyIterator
// This class represents an iterator over pairs of image and label image paths. The class does not expose any methods
// to the python environment.
// =====================================================================================================================

typedef struct {
    PyObject_HEAD
    std::shared_ptr<Chianti::IIterator<std::pair<std::string, std::string>>> iterator;
} PyIterator;

// Define the methods (heads up: there are no)
static PyMethodDef PyIterator_methods[] = {
        {NULL}
};

static PyTypeObject PyIteratorType = {
        PyVarObject_HEAD_INIT(NULL, 0)
        // tp_name
        "chianti.Iterator",
        // tp_basicsize
        sizeof(PyIterator),
        // tp_itemsize
        0,
        // tp_dealloc
        0,
        // tp_print
        0,
        // tp_getattr
        0,
        // tp_setattr
        0,
        // tp_reserved
        0,
        // tp_repr
        0,
        // tp_as_number
        0,
        // tp_as_sequence
        0,
        // tp_as_mapping
        0,
        // tp_hash
        0,
        // tp_call
        0,
        // tp_str
        0,
        // tp_getattro
        0,
        // tp_setattro
        0,
        // tp_as_buffer
        0,
        // tp_flags
        Py_TPFLAGS_DEFAULT,
        // tp_doc
        "An iterator that iterates over pairs of images and label images.",
        // tp_traverse
        0,
        // tp_clear
        0,
        // tp_richcompare
        0,
        // tp_weaklistoffset
        0,
        // tp_iter
        0,
        // tp_iternext
        0,
        // tp_methods
        PyIterator_methods};

// =====================================================================================================================
// PyAugmentor
// This class represents a data augmentation step. Similar to the iterator class, it only carries a single pointer and
// does not offer any methods to the python runtime.
// =====================================================================================================================

typedef struct {
    PyObject_HEAD
    std::shared_ptr<Chianti::IAugmentor> augmentor;
} PyAugmentor;

// Define the methods (heads up: there are no)
static PyMethodDef PyAugmentor_methods[] = {
        {NULL}
};

static PyTypeObject PyAugmentorType = {
        PyVarObject_HEAD_INIT(NULL, 0)
        // tp_name
        "chianti.Augmentor",
        // tp_basicsize
        sizeof(PyAugmentor),
        // tp_itemsize
        0,
        // tp_dealloc
        0,
        // tp_print
        0,
        // tp_getattr
        0,
        // tp_setattr
        0,
        // tp_reserved
        0,
        // tp_repr
        0,
        // tp_as_number
        0,
        // tp_as_sequence
        0,
        // tp_as_mapping
        0,
        // tp_hash
        0,
        // tp_call
        0,
        // tp_str
        0,
        // tp_getattro
        0,
        // tp_setattro
        0,
        // tp_as_buffer
        0,
        // tp_flags
        Py_TPFLAGS_DEFAULT,
        // tp_doc
        "A data augmentation step.",
        // tp_traverse
        0,
        // tp_clear
        0,
        // tp_richcompare
        0,
        // tp_weaklistoffset
        0,
        // tp_iter
        0,
        // tp_iternext
        0,
        // tp_methods
        PyAugmentor_methods};


// =====================================================================================================================
// PyDataProvider
// This is the main data provider class. It returns a new batch of images on next(). Each batch consists of N
// (augmented) images and label images.
// =====================================================================================================================

typedef struct {
    PyObject_HEAD
    std::shared_ptr<Chianti::TensorDataProvider> dataProvider;
} PyDataProvider;

static int PyDataProvider_init(PyDataProvider* self, PyObject* args, PyObject *keywords)
{
    // The first two parameters are mandatory while the last parameter is optional
    static char *keywordList[] = {"iterator", "batchsize", "augmentors", 0};

    PyIterator* iterator = nullptr;
    int batchsize = 0;
    PyObject* augmentors = nullptr;

    if (!PyArg_ParseTupleAndKeywords(
            args,
            keywords,
            "O!i|O!",
            keywordList,
            &PyIteratorType, &iterator,
            &batchsize,
            &PyList_Type, &augmentors))
    {
        return 0;
    }

    auto cppIterator = iterator->iterator;
    auto cppAugmentor = std::make_shared<Chianti::CombinedAugmentor>();
    cppAugmentor->addAugmentor(std::make_shared<Chianti::CastToFloatAugmentor>());

    // Extract the iterators if there are any
    if (augmentors != nullptr)
    {
        try {
            PythonUtils::iterateList(augmentors, [cppAugmentor](int i, PyObject* entry)  {
                // Check if the given object is indeed an augmentor
                if (!PyObject_TypeCheck(entry, &PyAugmentorType))
                {
                    throw ConversionException("Augmentor list must only contain instances of Augmentor.");
                }
                cppAugmentor->addAugmentor(reinterpret_cast<PyAugmentor*>(entry)->augmentor);
            });
        } catch (ConversionException e)
        {
            PyErr_SetString(PyExc_RuntimeError, e.what());
            return 0;
        }
    }

    // Create the actual data provider
    auto dataProvider = std::make_shared<Chianti::DataProvider>(cppIterator, batchsize, cppAugmentor);

    // Create the data provider
    self->dataProvider = std::make_shared<Chianti::TensorDataProvider>(dataProvider);

    return 0;
}

template<size_t N>
int prod(const std::array<int, N> & a)
{
    int result = 1;
    for (size_t n = 0; n < N; n++)
    {
        result *= a[n];
    }
    return result;
}

static PyObject* PyDataProvider_next(PyDataProvider* self)
{
    if (self->dataProvider != nullptr && self->dataProvider->getBatchSize() > 0)
    {
        // Get the next batch
        auto batch = self->dataProvider->next();

        // Convert the batch to numpy arrays
        PyArrayObject* imgs = (PyArrayObject*) PyArray_FromDims(4, batch->imgsShape.begin(), NPY_FLOAT);
        std::memcpy(imgs->data, batch->imgs, prod(batch->imgsShape) * sizeof(float));
        PyArrayObject* target = (PyArrayObject*) PyArray_FromDims(3, batch->targetsShape.begin(), NPY_INT);
        std::memcpy(target->data, batch->targets, prod(batch->targetsShape) * sizeof(int));

        delete[] batch->imgs;
        delete[] batch->targets;

        // Create the return type (Batch)
        PyBatch* pyBatch = (PyBatch*)PyBatchType.tp_alloc(&PyBatchType, 0);

        pyBatch->imgs = (PyObject*) imgs;
        pyBatch->targets = (PyObject*) target;

        return (PyObject*) pyBatch;
    }
    else
    {
        Py_RETURN_NONE;
    }
}

static PyObject* PyDataProvider_reset(PyDataProvider* self)
{
    if (self->dataProvider != nullptr)
    {
        self->dataProvider->reset();
    }
    Py_RETURN_NONE;
}

static PyObject* PyDataProvider_getNumBatches(PyDataProvider* self)
{
    if (self->dataProvider != nullptr)
    {
        PyObject* result = PyLong_FromLong(self->dataProvider->getNumBatches());
        return result;
    }
    Py_RETURN_NONE;
}

static PyMethodDef PyDataProvider_methods[] = {
        {"next", (PyCFunction)PyDataProvider_next, METH_NOARGS, "Returns the next batch"
        },
        {"reset", (PyCFunction)PyDataProvider_reset, METH_NOARGS, "Resets the iterator"
        },
        {"get_num_batches", (PyCFunction)PyDataProvider_getNumBatches, METH_NOARGS, "Returns the number of batches"
        },
        {NULL}
};

static PyTypeObject PyDataProviderType = {
        PyVarObject_HEAD_INIT(NULL, 0)
        "chianti.DataProvider",             /* tp_name */
        sizeof(PyDataProvider), /* tp_basicsize */
        0,                         /* tp_itemsize */
        0,                         /* tp_dealloc */
        0,                         /* tp_print */
        0,                         /* tp_getattr */
        0,                         /* tp_setattr */
        0,                         /* tp_reserved */
        0,                         /* tp_repr */
        0,                         /* tp_as_number */
        0,                         /* tp_as_sequence */
        0,                         /* tp_as_mapping */
        0,                         /* tp_hash  */
        0,                         /* tp_call */
        0,                         /* tp_str */
        0,                         /* tp_getattro */
        0,                         /* tp_setattro */
        0,                         /* tp_as_buffer */
        Py_TPFLAGS_DEFAULT,        /* tp_flags */
        "Allows asynchronous batch pre-processing for semantic segmentation tasks.",           /* tp_doc */
        0,                         /* tp_traverse */
        0,                         /* tp_clear */
        0,                         /* tp_richcompare */
        0,                         /* tp_weaklistoffset */
        0,                         /* tp_iter */
        0,                         /* tp_iternext */
        PyDataProvider_methods,      /* tp_methods */
        0,                         /* tp_members */
        0,                         /* tp_getset */
        0,                         /* tp_base */
        0,                         /* tp_dict */
        0,                         /* tp_descr_get */
        0,                         /* tp_descr_set */
        0,                         /* tp_dictoffset */
        (initproc)PyDataProvider_init,                         /* tp_init */
        0,                         /* tp_alloc */
        0,                         /* tp_new */};


// =====================================================================================================================
// These are the ordinary functions contained in the package. They are all factory methods.
// =====================================================================================================================

static PyObject* sequentialIterator(PyObject* self, PyObject* args, PyObject *keywords)
{
    static char *keywordList[] = {"files", 0};

    PyObject* files = nullptr;

    if (!PyArg_ParseTupleAndKeywords(
            args,
            keywords,
            "O!",
            keywordList,
            &PyList_Type, &files))
    {
        return 0;
    }


    try {
        // Extract the list of tuples
        std::vector<std::pair<std::string, std::string>> tupleList;
        PythonUtils::iterateList(files, [&tupleList](int i, PyObject* tuple) {
            tupleList.push_back(std::make_pair<std::string, std::string>(
                    PythonUtils::objectToString(PythonUtils::getTupleEntry(0, tuple)),
                    PythonUtils::objectToString(PythonUtils::getTupleEntry(1, tuple))
            ));
        });

        // Make the python object
        PyIterator* iterator = (PyIterator *)PyIteratorType.tp_alloc(&PyIteratorType, 0);
        iterator->iterator = std::make_shared<Chianti::SequentialIterator<std::pair<std::string, std::string>> >(tupleList);

        return (PyObject*) iterator;
    } catch (ConversionException e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return 0;
    }
}

static PyObject* randomIterator(PyObject* self, PyObject* args, PyObject *keywords)
{
    static char *keywordList[] = {"files", 0};

    PyObject* files = nullptr;

    if (!PyArg_ParseTupleAndKeywords(
            args,
            keywords,
            "O!",
            keywordList,
            &PyList_Type, &files))
    {
        return 0;
    }


    try {
        // Extract the list of tuples
        std::vector<std::pair<std::string, std::string>> tupleList;
        PythonUtils::iterateList(files, [&tupleList](int i, PyObject* tuple) {
            tupleList.push_back(std::make_pair<std::string, std::string>(
                    PythonUtils::objectToString(PythonUtils::getTupleEntry(0, tuple)),
                    PythonUtils::objectToString(PythonUtils::getTupleEntry(1, tuple))
            ));
        });

        // Make the python object
        PyIterator* iterator = (PyIterator *)PyIteratorType.tp_alloc(&PyIteratorType, 0);
        iterator->iterator = std::make_shared<Chianti::RandomIterator<std::pair<std::string, std::string>> >(tupleList);

        return (PyObject*) iterator;
    } catch (ConversionException e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return 0;
    }
}

static PyObject* subsampleAugmentor(PyObject* self, PyObject* args, PyObject *keywords)
{
    static char *keywordList[] = {"factor", 0};

    int factor = 1;

    if (!PyArg_ParseTupleAndKeywords(
            args,
            keywords,
            "i",
            keywordList,
            &factor))
    {
        return 0;
    }

    PyAugmentor* augmentor = (PyAugmentor*)PyAugmentorType.tp_alloc(&PyAugmentorType, 0);
    augmentor->augmentor = std::make_shared<Chianti::SubsampleAugmentor>(factor);

    return (PyObject*) augmentor;
}

static PyObject* gammaAugmentor(PyObject* self, PyObject* args, PyObject *keywords)
{
    static char *keywordList[] = {"gamma", 0};

    double gamma = 1;

    if (!PyArg_ParseTupleAndKeywords(
            args,
            keywords,
            "d",
            keywordList,
            &gamma))
    {
        return 0;
    }

    PyAugmentor* augmentor = (PyAugmentor*)PyAugmentorType.tp_alloc(&PyAugmentorType, 0);
    augmentor->augmentor = std::make_shared<Chianti::GammaAugmentor>(gamma);

    return (PyObject*) augmentor;
}

static PyObject* translationAugmentor(PyObject* self, PyObject* args, PyObject *keywords)
{
    static char *keywordList[] = {"offset", 0};

    int offset = 1;

    if (!PyArg_ParseTupleAndKeywords(
            args,
            keywords,
            "i",
            keywordList,
            &offset))
    {
        return 0;
    }

    PyAugmentor* augmentor = (PyAugmentor*)PyAugmentorType.tp_alloc(&PyAugmentorType, 0);
    augmentor->augmentor = std::make_shared<Chianti::TranslationAugmentor>(offset);

    return (PyObject*) augmentor;
}

static PyObject* cityscapesLabelTransformationAugmentor(PyObject* self, PyObject* args, PyObject *keywords)
{
    PyAugmentor* augmentor = (PyAugmentor*)PyAugmentorType.tp_alloc(&PyAugmentorType, 0);
    augmentor->augmentor = std::make_shared<Chianti::CSLabelTransformationAugmentation>();

    return (PyObject*) augmentor;
}


// =====================================================================================================================
// Initialize the python extension
// =====================================================================================================================

static PyMethodDef chiantiMethods[] = {
        {"sequential_iterator", (PyCFunction)sequentialIterator, METH_VARARGS | METH_KEYWORDS, "Creates a new sequential (non random) iterator."},
        {"random_iterator", (PyCFunction)randomIterator, METH_VARARGS | METH_KEYWORDS, "Creates a new random (batch-based) iterator."},
        {"subsample_augmentor", (PyCFunction)subsampleAugmentor, METH_VARARGS | METH_KEYWORDS, "Creates a new augmentation step that subsamples the images."},
        {"gamma_augmentor", (PyCFunction)gammaAugmentor, METH_VARARGS | METH_KEYWORDS, "Creates a new random gamma augmentor."},
        {"translation_augmentor", (PyCFunction)translationAugmentor, METH_VARARGS | METH_KEYWORDS, "Creates a new random translation augmentor."},
        {"cityscapes_label_transformation_augmentor", (PyCFunction)cityscapesLabelTransformationAugmentor, METH_NOARGS, "Creates a new augmentation step to transform the cityscapes label ids to training ids."},
        {NULL, NULL, 0, NULL}
};

static struct PyModuleDef chiantiModule = {
        PyModuleDef_HEAD_INIT,
        // name of module
        "chianti",
        // module documentation, may be NULL
        0,
        -1,
        chiantiMethods
};

PyMODINIT_FUNC PyInit_chianti(void)
{
    PyObject* m;

    PyDataProviderType.tp_new = PyType_GenericNew;
    PyAugmentorType.tp_new = PyType_GenericNew;
    PyIteratorType.tp_new = PyType_GenericNew;
    PyBatchType.tp_new = PyType_GenericNew;

    if (PyType_Ready(&PyDataProviderType) < 0)
        return NULL;
    if (PyType_Ready(&PyAugmentorType) < 0)
        return NULL;
    if (PyType_Ready(&PyIteratorType) < 0)
        return NULL;
    if (PyType_Ready(&PyBatchType) < 0)
        return NULL;

    m = PyModule_Create(&chiantiModule);
    if (m == NULL)
        return NULL;

    Py_INCREF(&PyDataProviderType);
    Py_INCREF(&PyAugmentorType);
    Py_INCREF(&PyIteratorType);
    Py_INCREF(&PyBatchType);

    PyModule_AddObject(m, "DataProvider", (PyObject *)&PyDataProviderType);
    PyModule_AddObject(m, "Augmentor", (PyObject *)&PyAugmentorType);
    PyModule_AddObject(m, "Iterator", (PyObject *)&PyIteratorType);
    PyModule_AddObject(m, "Batch", (PyObject *)&PyBatchType);

    import_array();

    return m;
}