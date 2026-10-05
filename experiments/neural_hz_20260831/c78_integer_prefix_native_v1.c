/* Exact C5 integer tokens and exact sparse pointer-identity bits.
 * All owned heap objects are Python objects; GIL stays held throughout. */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <string.h>

typedef struct {
    PyObject *update;
    char *data;
    Py_ssize_t size, used, emitted;
} Writer;

static int flush(Writer *w) {
    if (!w->used) return 0;
    PyObject *view=PyMemoryView_FromMemory(w->data,w->used,PyBUF_READ);
    if (!view) return -1;
    PyObject *r=PyObject_CallOneArg(w->update,view);
    Py_DECREF(view);
    if (!r) return -1;
    Py_DECREF(r);w->used=0;return 0;
}

static int append(Writer *w,const char *data,Py_ssize_t n) {
    while (n) {
        if (w->used==w->size && flush(w)<0) return -1;
        Py_ssize_t left=w->size-w->used;
        Py_ssize_t take=n<left?n:left;
        memcpy(w->data+w->used,data,(size_t)take);
        w->used+=take;w->emitted+=take;data+=take;n-=take;
    }
    return 0;
}

static PyObject *scan(PyObject *self,PyObject *args) {
    (void)self;
    PyObject *items,*prefix,*pages,*update,*sizeof_fn,*buffer,*charge;
    Py_ssize_t start;int sequence;
    if (!PyArg_ParseTuple(args,"OnOOOOOOp",&items,&start,&prefix,&pages,
                         &update,&sizeof_fn,&buffer,&charge,&sequence)) return NULL;
    if ((!PyList_CheckExact(items) && !PyTuple_CheckExact(items)) ||
        !PyBytes_CheckExact(prefix) || !PyDict_CheckExact(pages) ||
        !PyByteArray_CheckExact(buffer) || PyByteArray_GET_SIZE(buffer)!=262144) {
        PyErr_SetString(PyExc_ValueError,"exact prefix inputs required");return NULL;
    }
    Py_ssize_t length=PyList_CheckExact(items)?PyList_GET_SIZE(items):PyTuple_GET_SIZE(items);
    if (start<0 || start>length || (!sequence && (start!=0 || length!=1))) {
        PyErr_SetString(PyExc_ValueError,"exact bounded integer prefix required");return NULL;
    }
    Writer w={update,PyByteArray_AS_STRING(buffer),PyByteArray_GET_SIZE(buffer),0,0};
    Py_ssize_t end=length-start>4096?start+4096:length;
    Py_ssize_t at=start,unique=0,shallow=0,created=0;
    const char *path=PyBytes_AS_STRING(prefix);Py_ssize_t path_n=PyBytes_GET_SIZE(prefix);
    for (;at<end;at++) {
        PyObject *value=PyList_CheckExact(items)?PyList_GET_ITEM(items,at):PyTuple_GET_ITEM(items,at);
        if (!PyLong_CheckExact(value)) break;
        if (append(&w,path,path_n)<0) return NULL;
        if (sequence) {
            char index[64];int n=PyOS_snprintf(index,sizeof(index),"%zd",at);
            if (n<0 || n>=(int)sizeof(index)) {
                PyErr_SetString(PyExc_ValueError,"index formatting failed");return NULL;
            }
            if (append(&w,index,n)<0 || append(&w,")\0",2)<0) return NULL;
        } else if (append(&w,"\0",1)<0) return NULL;
        PyObject *text=PyObject_Str(value);
        if (!text) return NULL;
        Py_ssize_t n;const char *digits=PyUnicode_AsUTF8AndSize(text,&n);
        if (!digits || append(&w,"('int', ",8)<0 || append(&w,digits,n)<0 || append(&w,")\0",2)<0) {
            Py_DECREF(text);return NULL;
        }
        Py_DECREF(text);
        uintptr_t address=(uintptr_t)value;
        PyObject *key=PyLong_FromUnsignedLongLong((unsigned long long)(address>>16));
        if (!key) return NULL;
        PyObject *page=PyDict_GetItemWithError(pages,key);
        if (!page) {
            if (PyErr_Occurred()) {Py_DECREF(key);return NULL;}
            PyObject *paid=PyObject_CallFunction(charge,"sn","c78_exact_pointer_page_zero",(Py_ssize_t)1024);
            if (!paid) {Py_DECREF(key);return NULL;}
            Py_DECREF(paid);
            page=PyByteArray_FromStringAndSize(NULL,8192);
            if (!page) {Py_DECREF(key);return NULL;}
            memset(PyByteArray_AS_STRING(page),0,8192);
            if (PyDict_SetItem(pages,key,page)<0) {Py_DECREF(page);Py_DECREF(key);return NULL;}
            Py_DECREF(page);created++;
        }
        Py_DECREF(key);
        if (!PyByteArray_CheckExact(page) || PyByteArray_GET_SIZE(page)!=8192) {
            PyErr_SetString(PyExc_ValueError,"identity page schema mismatch");return NULL;
        }
        unsigned char *bits=(unsigned char *)PyByteArray_AS_STRING(page);
        unsigned int bit=(unsigned int)(address&65535),mask=1U<<(bit&7);
        if (!(bits[bit>>3]&mask)) {
            PyObject *size=PyObject_CallOneArg(sizeof_fn,value);
            if (!size) return NULL;
            Py_ssize_t bytes=PyLong_AsSsize_t(size);Py_DECREF(size);
            if (bytes<0 || PyErr_Occurred()) return NULL;
            bits[bit>>3]|=(unsigned char)mask;unique++;shallow+=bytes;
        }
    }
    if (flush(&w)<0) return NULL;
    return Py_BuildValue("nnnnn",at,unique,shallow,w.emitted,created);
}

static PyMethodDef methods[]={{"scan",scan,METH_VARARGS,"Exact integer prefix, no input mutation."},{NULL,NULL,0,NULL}};
static struct PyModuleDef module={PyModuleDef_HEAD_INIT,"_c78_integer_prefix_v1",NULL,-1,methods,NULL,NULL,NULL,NULL};
PyMODINIT_FUNC PyInit__c78_integer_prefix_v1(void) {return PyModule_Create(&module);}
