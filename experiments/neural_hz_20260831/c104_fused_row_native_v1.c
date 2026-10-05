/* Complete C97 coefficient decision and reversible scaling. No native heap. */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

typedef struct { void *buf; npy_intp shape[1]; npy_intp strides[1]; } RowView;
#include <math.h>
#include <stdint.h>
#include <string.h>

static double read_double(const RowView *b, Py_ssize_t i) {
    double value;
    memcpy(&value, (char *)b->buf + i*b->strides[0], sizeof(value));
    return value;
}
static int64_t read_power(const RowView *b, Py_ssize_t i) {
    int64_t value;
    memcpy(&value, (char *)b->buf + i*b->strides[0], sizeof(value));
    return value;
}

static PyObject *prepare(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *objects[6], *result = NULL;
    RowView buffers[6] = {{0}};
    int minimum = 0, maximum = 0, first_exp = 0;
    int shift = 0, head = 0, has_value = 0, first_negative = 0;
    double top = 0., first_mantissa = 0.;
    if (!PyArg_ParseTuple(args, "OOOOOO", &objects[0], &objects[1],
            &objects[2], &objects[3], &objects[4], &objects[5])) return NULL;
    for (int k = 0; k < 6; ++k) {
        if (!PyArray_CheckExact(objects[k])) {
            PyErr_SetString(PyExc_ValueError,"canonical NumPy array required");
            goto done;
        }
        PyArrayObject *array = (PyArrayObject *)objects[k];
        int power = k == 1 || k == 3;
        if (PyArray_NDIM(array) != 1 || PyArray_ITEMSIZE(array) != 8
                || !PyArray_ISNOTSWAPPED(array)
                || PyArray_TYPE(array) != (power ? NPY_INT64 : NPY_FLOAT64)
                || (k >= 4 && !PyArray_ISWRITEABLE(array))) {
            PyErr_SetString(PyExc_ValueError,"canonical native double/int64 vector required");
            goto done;
        }
        buffers[k].buf = PyArray_DATA(array);
        buffers[k].shape[0] = PyArray_DIM(array,0);
        buffers[k].strides[0] = PyArray_STRIDE(array,0);
    }
    if (buffers[0].shape[0] != buffers[1].shape[0]
            || buffers[2].shape[0] != buffers[3].shape[0]
            || buffers[0].shape[0] != buffers[4].shape[0]
            || buffers[2].shape[0] != buffers[5].shape[0]
            || buffers[0].shape[0] > 64000000 - buffers[2].shape[0]) {
        PyErr_SetString(PyExc_ValueError,"complete bounded row shape differs");
        goto done;
    }
    for (int group = 0; group < 2; ++group) {
        RowView *v = &buffers[2*group], *p = &buffers[2*group+1];
        for (Py_ssize_t i = 0; i < v->shape[0]; ++i) {
            double value = read_double(v,i);
            int exponent;
            if (!isfinite(value) || value == 0.) {
                PyErr_SetString(PyExc_ValueError,"finite nonzero bounded coefficient vector required");
                goto done;
            }
            /* Bounds already established by original _bounded_powers. */
            int64_t power = read_power(p,i);
            double mantissa = frexp(fabs(value), &exponent);
            exponent += (int)power;
            if (!has_value || exponent < minimum) minimum = exponent;
            if (!has_value || exponent > maximum) { maximum = exponent; top = mantissa; }
            else if (exponent == maximum && mantissa > top) top = mantissa;
            if (group == 0 && i == 0) {
                first_exp = exponent; first_mantissa = mantissa; first_negative = value < 0.;
            }
            has_value = 1;
        }
    }
    if (has_value) {
        int lower = -19 - minimum, upper = (top == .5 ? 41 : 40) - maximum;
        if (lower > upper) { result = Py_NewRef(Py_None); goto done; }
        shift = lower > 0 ? lower : 0;
        if (shift > upper) shift = upper;
    }
    for (int group = 0; group < 2; ++group) {
        RowView *v = &buffers[2*group], *p = &buffers[2*group+1], *out = &buffers[4+group];
        for (Py_ssize_t i = 0; i < v->shape[0]; ++i) {
            double value = read_double(v,i);
            int exponent = (int)read_power(p,i) + shift;
            double scaled = ldexp(value,exponent), back = ldexp(scaled,-exponent);
            if (!isfinite(scaled) || !isfinite(back)) {
                PyErr_SetString(PyExc_FloatingPointError,"nonfinite reversible coefficient scaling");
                goto done;
            }
            if (back != value) {
                PyErr_SetString(PyExc_ValueError,"power-of-two scaling is not exactly reversible");
                goto done;
            }
            memcpy((char *)out->buf + i*out->strides[0], &scaled, sizeof(scaled));
        }
    }
    if (buffers[0].shape[0] && first_mantissa == .5) {
        int exponent = first_exp + shift - 1;
        if (exponent < -20 || exponent > 40) {
            PyErr_SetString(PyExc_ValueError,"derived head exponent outside proved window");
            goto done;
        }
        head = 1 + 2*(exponent+20) + first_negative;
    }
    result = Py_BuildValue("ii",shift,head);
done:
    /* args keeps all six arrays alive; no buffer exporter metadata is created. */
    return result;
}

static PyMethodDef methods[] = {
    {"prepare",prepare,METH_VARARGS,"Complete finite/window and inverse coefficient program."},
    {NULL,NULL,0,NULL}
};
static struct PyModuleDef module = {PyModuleDef_HEAD_INIT,"_c104_fused_row_v1",NULL,-1,methods,NULL,NULL,NULL,NULL};
PyMODINIT_FUNC PyInit__c104_fused_row_v1(void) {import_array(); return PyModule_Create(&module);}
