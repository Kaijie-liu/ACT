/* SPDX-License-Identifier: AGPL-3.0-or-later
 * Actual original coordinate checks + complete finite/window/inverse program.
 * The Python producer still validates original powers before signed copying. */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

typedef struct {
    char *buf; npy_intp size,stride; int width,is_unsigned,swapped;
} Vector;

static int fail(const char *message) { PyErr_SetString(PyExc_ValueError,message); return 0; }
static double real_at(const Vector *v,Py_ssize_t i) {
    double x; memcpy(&x,v->buf+i*v->stride,8); return x;
}
static int64_t power_at(const Vector *v,Py_ssize_t i) {
    int64_t x; memcpy(&x,v->buf+i*v->stride,8); return x;
}
static int coordinate(const Vector *v,Py_ssize_t i,int64_t limit,int64_t *out) {
    uint64_t bits=0;
    /* Native byte order is handled by memcpy; swapped input bytes are reversed
       before loading, for the original integer coordinate dtype only. */
    if (v->swapped) {
        unsigned char bytes[8];
        for (int j=0;j<v->width;++j) bytes[j]=(unsigned char)v->buf[i*v->stride+v->width-1-j];
        memcpy(&bits,bytes,(size_t)v->width);
    } else memcpy(&bits,v->buf+i*v->stride,(size_t)v->width);
    int64_t value;
    if (v->is_unsigned) {
        if (bits>=(uint64_t)limit) return fail("original unsigned coordinate outside frame");
        value=(int64_t)bits;
    } else {
        switch(v->width) {
            case 1:value=(int8_t)bits;break;
            case 2:value=(int16_t)bits;break;
            case 4:value=(int32_t)bits;break;
            case 8:memcpy(&value,&bits,8);break;
            default:return fail("unsupported original integer coordinate width");
        }
        if (value<0 || value>=limit) return fail("original signed coordinate outside frame");
    }
    *out=value; return 1;
}

static PyObject *prepare_checked(PyObject *self,PyObject *args) {
    (void)self;
    PyObject *objects[6],*result=NULL; PyArrayObject *outputs[2]={NULL,NULL};
    Vector inputs[6]={{0}}; long long nc,nb; double rhs;
    int minimum=0,maximum=0,first_exp=0,first_negative=0,has_value=0;
    double top=0.,first_mantissa=0.; int shift=0,head=0;
    if (!PyArg_ParseTuple(args,"OOOOOOLLd",&objects[0],&objects[1],&objects[2],
            &objects[3],&objects[4],&objects[5],&nc,&nb,&rhs)) return NULL;
    if (nc<0 || nb<0 || !isfinite(rhs)) { fail("invalid original frame or RHS"); goto done; }
    for (int k=0;k<6;++k) {
        if (!PyArray_CheckExact(objects[k])) { fail("canonical NumPy original row required"); goto done; }
        PyArrayObject *a=(PyArrayObject *)objects[k]; int type=PyArray_TYPE(a),col=k%3==0;
        if (PyArray_NDIM(a)!=1 || PyArray_SIZE(a)>64000000) { fail("bounded complete original vector required"); goto done; }
        if (col) {
            if (!PyTypeNum_ISINTEGER(type) || (PyArray_ITEMSIZE(a)!=1 && PyArray_ITEMSIZE(a)!=2 &&
                    PyArray_ITEMSIZE(a)!=4 && PyArray_ITEMSIZE(a)!=8)) { fail("original integer coordinates required"); goto done; }
        } else if (!PyArray_ISNOTSWAPPED(a) || PyArray_ITEMSIZE(a)!=8 ||
                    type!=(k%3==1?NPY_FLOAT64:NPY_INT64)) { fail("canonical float64/bounded int64 operands required"); goto done; }
        inputs[k]=(Vector){PyArray_DATA(a),PyArray_DIM(a,0),PyArray_STRIDE(a,0),
            (int)PyArray_ITEMSIZE(a),PyArray_DESCR(a)->kind=='u',!PyArray_ISNOTSWAPPED(a)};
    }
    if (inputs[0].size!=inputs[1].size || inputs[0].size!=inputs[2].size ||
        inputs[3].size!=inputs[4].size || inputs[3].size!=inputs[5].size ||
        inputs[1].size>64000000-inputs[4].size) { fail("complete original row shape differs"); goto done; }
    for (int group=0;group<2;++group) {
        Vector *c=&inputs[3*group],*v=c+1,*p=c+2; int64_t previous=-1,limit=group?nb:nc;
        for (Py_ssize_t i=0;i<v->size;++i) {
            int64_t column;
            if (!coordinate(c,i,limit,&column)) goto done;
            if (column<=previous) { fail("original row coordinates not strictly increasing"); goto done; }
            previous=column;
            double value=real_at(v,i); int exponent;
            if (!isfinite(value) || value==0.) { fail("finite nonzero defining coefficient required"); goto done; }
            double mantissa=frexp(fabs(value),&exponent);
            /* ORIGINAL _bounded_powers already proves the int64 power range. */
            exponent+=(int)power_at(p,i);
            if (!has_value || exponent<minimum) minimum=exponent;
            if (!has_value || exponent>maximum) { maximum=exponent;top=mantissa; }
            else if (exponent==maximum && mantissa>top) top=mantissa;
            if (group==0 && i==0) { first_exp=exponent;first_mantissa=mantissa;first_negative=value<0.; }
            has_value=1;
        }
    }
    if (has_value) {
        int lower=-19-minimum,upper=(top==.5?41:40)-maximum;
        if (lower>upper) { result=Py_NewRef(Py_None); goto done; }
        shift=lower>0?lower:0; if (shift>upper) shift=upper;
    }
    for (int group=0;group<2;++group) {
        Vector *v=&inputs[3*group+1],*p=v+1; npy_intp n=v->size;
        outputs[group]=(PyArrayObject *)PyArray_SimpleNew(1,&n,NPY_FLOAT64);
        if (!outputs[group]) goto done;
        double *out=PyArray_DATA(outputs[group]);
        for (Py_ssize_t i=0;i<n;++i) {
            double value=real_at(v,i); int exponent=(int)power_at(p,i)+shift;
            double scaled=ldexp(value,exponent),back=ldexp(scaled,-exponent);
            if (!isfinite(scaled) || !isfinite(back)) { PyErr_SetString(PyExc_FloatingPointError,"nonfinite reversible coefficient scaling"); goto done; }
            if (back!=value) { fail("power-of-two scaling is not exactly reversible"); goto done; }
            out[i]=scaled;
        }
    }
    double scaled_rhs=ldexp(rhs,shift),back_rhs=ldexp(scaled_rhs,-shift);
    if (!isfinite(scaled_rhs) || !isfinite(back_rhs)) { PyErr_SetString(PyExc_FloatingPointError,"nonfinite reversible RHS scaling"); goto done; }
    if (back_rhs!=rhs) { fail("RHS power-of-two scaling is not exactly reversible"); goto done; }
    if (inputs[0].size && first_mantissa==.5) {
        int exponent=first_exp+shift-1;
        if (exponent< -20 || exponent>40) { fail("derived head outside proved window"); goto done; }
        head=1+2*(exponent+20)+first_negative;
    }
    result=Py_BuildValue("OOdii",outputs[0],outputs[1],scaled_rhs,shift,head);
done:
    Py_XDECREF(outputs[0]);Py_XDECREF(outputs[1]);return result;
}
static PyMethodDef methods[]={
    {"prepare_checked",prepare_checked,METH_VARARGS,"Complete actual original coordinate and numerical preparation."},
    {NULL,NULL,0,NULL}
};
static struct PyModuleDef module={PyModuleDef_HEAD_INIT,"_c107_canonical_encoding_v1",NULL,-1,methods,NULL,NULL,NULL,NULL};
PyMODINIT_FUNC PyInit__c107_canonical_encoding_v1(void) { import_array(); return PyModule_Create(&module); }
