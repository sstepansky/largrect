#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_SSIZE_T_CLEAN
#include <Python.h>             // Python headers, mandatory
#include <numpy/arrayobject.h>  // NumPy array object headers
#include <stdbool.h>

typedef struct {
    unsigned long int area;
    unsigned int x0;
    unsigned int y0;
    unsigned int x1;
    unsigned int y1;
} Rect;


static void*
failure(PyObject *type, const char *message)
{
    PyErr_SetString(type, message);
    return NULL;
}


static PyObject*
largrect(PyObject *self, PyObject *args)
{
    PyArrayObject *a;

    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &a))
        return failure(PyExc_RuntimeError, "Failed to parse parameters.");

    if (PyArray_DESCR(a)->type_num != NPY_UINT8)
        return failure(PyExc_TypeError, "Type numpy.uint8 expected for image array.");

    if (PyArray_NDIM(a) != 2)
        return failure(PyExc_TypeError, "image must be a 2-dimensional array.");

    unsigned int r, c, dh;
    unsigned int nrows = PyArray_DIM(a, 0);
    unsigned int ncols = PyArray_DIM(a, 1);

    unsigned int *w = calloc((nrows * ncols), sizeof(int));
    unsigned int *h = calloc((nrows * ncols), sizeof(int));

    bool error;

    if (error = (w == NULL || h == NULL))
    {
        PyErr_SetString(PyExc_MemoryError,
                        "Could not allocate memory for arrays");
        goto cleanup;
    }

    unsigned int minw;
    unsigned long int area;
    Rect area_max = {0};

    npy_uint8 *p;
    p = PyArray_DATA(a);  // pointer to start of data

    for (r = 0; r < nrows; r++)
    {
        for (c = 0; c < ncols; c++)
        {
            if (*p++)  // non-zero
                continue;
            if (!r)
                h[c] = 1;
            else
                h[r * ncols + c] = h[(r - 1) * ncols + c] + 1;
            if (!c)
                w[r * ncols] = 1;
            else
                w[r * ncols + c] = w[r * ncols + c - 1] + 1;

            minw = w[r * ncols + c];
            for (dh = 0; dh < h[r * ncols + c]; dh++)
            {
                minw = ((minw) < (w[(r - dh) * ncols + c])) ? (minw) : (w[(r - dh) * ncols + c]);
                area = ((unsigned long int)dh + 1UL) * (unsigned long int)minw;
                if (error = (minw != area / (dh + 1)))
                {
                    PyErr_SetString(PyExc_OverflowError,
                                    "An integer wraparound occurred. Rectangle area too large to compute.");
                    goto cleanup;
                }
                if (area > area_max.area)
                {
                    area_max.area = area;
                    area_max.x0 = r - dh;
                    area_max.y0 = c - minw + 1;
                    area_max.x1 = r;
                    area_max.y1 = c;
                }
            }
        }
    }

cleanup:
    free(h);
    free(w);


    if (error)
        return NULL;

    PyObject *out = Py_BuildValue("iiii", area_max.x0, area_max.y0, area_max.x1, area_max.y1);

    return out;
}


PyDoc_STRVAR(
    largrect_doc,
    "largrect(image)\n"
    "--\n"
    "\n"
    "Find the largest rectangle containing 0 in an image\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "image : numpy.ndarray\n"
    "    The image as a 2D numpy.ndarray, with datatype numpy.uint8.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "out : tuple\n"
    "    The coordinates of the rectangle as (x0, y0, x1, y1), where (x0, y0)\n"
    "    is the upper left corner, and (x1, y1) is the bottom right corner.\n"
    "\n"
    "Examples\n"
    "--------\n"
    "    >>> import largrect\n"
    "    >>> import numpy as np\n"
    "    >>> x = np.array([[1, 0, 0, 2],\n"
    "                      [0, 0, 0, 2],\n"
    "                      [0, 0, 0, 3],\n"
    "                      [2, 0, 0, 0]], dtype=np.uint8)\n"
    "    >>> largrect.largrect(x)\n"
    "    (0, 1, 3, 2)\n"
);

/*  define functions in module */
static PyMethodDef largrect_Methods[] =
{
     {
        "largrect",
        largrect,
        METH_VARARGS,
        largrect_doc,
     },
     {NULL, NULL, 0, NULL} /* Sentinel */
};


/* module initialization */
static struct PyModuleDef largrect_module =
{
    PyModuleDef_HEAD_INIT,
    "largrect",
    "Small C module to find the largest rectangle containing 0 in an image",
    -1,
    largrect_Methods
};


PyMODINIT_FUNC
PyInit_largrect(void)
{
    PyObject *module;
    module = PyModule_Create(&largrect_module);
    if(module == NULL)
        return NULL;
    /* IMPORTANT: this must be called */
    import_array();
    if (PyErr_Occurred())
        return NULL;
    return module;
}
