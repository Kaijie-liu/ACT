/* SPDX-License-Identifier: AGPL-3.0-or-later
 * Complete original support -> dyadic bound -> positive defining row.
 * No coefficient multiplication/sum; original encoder is still the consumer. */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#define LIMIT 64000000
typedef struct { PyArrayObject *a; } Vec;
typedef struct { PyArrayObject *c,*v,*p; int64_t n,at,total; int top; } Row;

static int fail(const char *message) { PyErr_SetString(PyExc_ValueError,message); return 0; }
static int vector(PyObject *o, int typ, Vec *v) {
    if (!PyArray_CheckExact(o)) return fail("exact NumPy vector required");
    PyArrayObject *a=(PyArrayObject *)o;
    if (PyArray_NDIM(a)!=1 || !PyArray_ISNOTSWAPPED(a) || PyArray_SIZE(a)>LIMIT ||
        (typ==-1 ? (PyArray_TYPE(a)!=NPY_INT32 && PyArray_TYPE(a)!=NPY_INT64) :
         typ==-2 ? (PyArray_TYPE(a)!=NPY_FLOAT32 && PyArray_TYPE(a)!=NPY_FLOAT64) : PyArray_TYPE(a)!=typ))
        return fail("bounded native typed vector required");
    v->a=a; return 1;
}
static int64_t size(Vec v) { return PyArray_DIM(v.a,0); }
static void *at(Vec v,int64_t i) { return (char *)PyArray_DATA(v.a)+i*PyArray_STRIDE(v.a,0); }
static int64_t integer(Vec v,int64_t i) {
    if (PyArray_TYPE(v.a)==NPY_INT32) { int32_t n; memcpy(&n,at(v,i),4); return n; }
    int64_t n; memcpy(&n,at(v,i),8); return n;
}
static double number(Vec v,int64_t i) {
    if (PyArray_TYPE(v.a)==NPY_FLOAT32) { float n; memcpy(&n,at(v,i),4); return (double)n; }
    double n; memcpy(&n,at(v,i),8); return n;
}
static int truth(Vec v,int64_t i) { npy_bool n; memcpy(&n,at(v,i),1); return n!=0; }
static int parents(PyObject *keep,PyObject *slots,PyObject *powers,Vec *k,Vec *s,Vec *p) {
    return vector(keep,NPY_BOOL,k) && vector(slots,NPY_INT64,s) && vector(powers,NPY_INT32,p) &&
        ((size(*k)==size(*s) && size(*k)==size(*p)) || fail("parent shapes differ"));
}
static int exponent(double value,int64_t power,int *e) {
    if (!isfinite(value) || value==0. || power < -4096 || power > 4096)
        return fail("finite nonzero bounded defining summand required");
    frexp(fabs(value),e); *e+=(int)power; return 1;
}
static int visit(Row *r,int pass,int64_t col,double value,int64_t power) {
    int e;
    if (col<0 || !exponent(value,power,&e)) return PyErr_Occurred()?0:fail("missing original parent slot");
    if (!pass) {
        if (!r->n || e>r->top) r->top=e;
        ++r->n; return 1;
    }
    if (r->at>=r->n) return fail("source population changed during emission");
    int shift=e-(r->top-26); if (shift<0) shift=0;
    if (shift>26) return fail("source exponent changed during emission");
    r->total+=(INT64_C(1)<<shift);
    ((int64_t *)PyArray_DATA(r->c))[r->at]=col;
    ((double *)PyArray_DATA(r->v))[r->at]=-value;
    ((int64_t *)PyArray_DATA(r->p))[r->at]=power;
    ++r->at; return 1;
}
static int allocate(Row *r) {
    if (r->n<=0 || r->n>LIMIT/3-1) return fail("bounded nonzero defining row required");
    npy_intp n=r->n+1;
    r->c=(PyArrayObject *)PyArray_SimpleNew(1,&n,NPY_INT64);
    if (!r->c) return 0;
    r->v=(PyArrayObject *)PyArray_SimpleNew(1,&n,NPY_FLOAT64);
    if (!r->v) return 0;
    r->p=(PyArrayObject *)PyArray_SimpleNew(1,&n,NPY_INT64);
    return r->c && r->v && r->p;
}
static PyObject *finish(Row *r,int64_t pivot) {
    PyObject *result=NULL;
    if (!PyErr_Occurred()) {
        if (r->at!=r->n || pivot<0 || r->total<=0) fail("incomplete positive defining row");
        else {
            uint64_t t=(uint64_t)(r->total-1); int bits=0;
            while(t) { ++bits; t>>=1; }
            int unit=r->top-26+bits; if (unit<0) unit=0;
            if (unit>1023) fail("auxiliary power-of-two box is not finite representable");
            else {
                ((int64_t *)PyArray_DATA(r->c))[r->n]=pivot;
                ((double *)PyArray_DATA(r->v))[r->n]=1.;
                ((int64_t *)PyArray_DATA(r->p))[r->n]=unit;
                result=Py_BuildValue("OOOi",r->c,r->v,r->p,unit);
            }
        }
    }
    Py_XDECREF(r->c); Py_XDECREF(r->v); Py_XDECREF(r->p); return result;
}

static PyObject *csr_row(PyObject *self,PyObject *args) {
    (void)self; PyObject *o[6]; long long row,pivot; Vec ptr,idx,data,keep,slots,powers; Row r={0};
    if (!PyArg_ParseTuple(args,"OOOOOOLL",&o[0],&o[1],&o[2],&o[3],&o[4],&o[5],&row,&pivot)) return NULL;
    if (!vector(o[0],-1,&ptr) || !vector(o[1],-1,&idx) || !vector(o[2],-2,&data) ||
        !parents(o[3],o[4],o[5],&keep,&slots,&powers)) return NULL;
    if (row<0 || row>=size(ptr)-1 || size(idx)!=size(data)) { fail("CSR source shape differs"); return NULL; }
    int64_t a=integer(ptr,row),z=integer(ptr,row+1);
    if (a<0 || z<a || z>size(idx)) { fail("CSR row extent differs"); return NULL; }
    for (int pass=0;pass<2;++pass) {
        for (int64_t i=a;i<z;++i) {
            int64_t col=integer(idx,i);
            if (col<0 || col>=size(keep)) { fail("CSR input coordinate outside actual parent"); goto done; }
            double val=number(data,i);
            if (truth(keep,col) && val!=0. && !visit(&r,pass,integer(slots,col),val,integer(powers,col))) goto done;
        }
        if (!pass && !allocate(&r)) goto done;
    }
done: return finish(&r,pivot);
}

static int geometry(PyObject *o,int64_t *m) {
    if (!PyTuple_CheckExact(o) || PyTuple_GET_SIZE(o)!=14) return fail("complete convolution geometry required");
    for (int i=0;i<14;++i) {
        PyObject *v=PyTuple_GET_ITEM(o,i);
        if (!PyLong_CheckExact(v)) return fail("integer convolution geometry required");
        m[i]=PyLong_AsLongLong(v); if (PyErr_Occurred()) return 0;
        if (m[i]<0 || m[i]>LIMIT || ((i!=9 && i!=10) && m[i]==0)) return fail("bounded convolution geometry required");
    }
    return 1;
}
static PyObject *conv_row(PyObject *self,PyObject *args) {
    (void)self; PyObject *kernel,*k,*s,*p,*mask,*meta; long long row,pivot;
    Vec keep,slots,powers,active; Row r={0}; int64_t m[14];
    if (!PyArg_ParseTuple(args,"OOOOOOLL",&kernel,&k,&s,&p,&mask,&meta,&row,&pivot)) return NULL;
    if (!parents(k,s,p,&keep,&slots,&powers) || !geometry(meta,m)) return NULL;
    if (!PyArray_CheckExact(kernel)) { fail("exact original convolution kernel required"); return NULL; }
    PyArrayObject *w=(PyArrayObject *)kernel;
    if (PyArray_NDIM(w)!=4 || PyArray_TYPE(w)!=NPY_FLOAT64 || !PyArray_ISNOTSWAPPED(w) || PyArray_SIZE(w)>LIMIT) {
        fail("bounded original float64 kernel required"); return NULL;
    }
    int64_t batch=m[0],ci=m[1],hi=m[2],wi=m[3],co=m[4],ho=m[5],wo=m[6];
    int64_t sy=m[7],sx=m[8],py=m[9],px=m[10],dy=m[11],dx=m[12],groups=m[13];
    int64_t input=1,output=1;
    for (int j=0;j<4;++j) { if(input>LIMIT/m[j]) { fail("input geometry exceeds bound"); return NULL; } input*=m[j]; }
    int64_t dims[4]={batch,co,ho,wo};
    for (int j=0;j<4;++j) { if(output>LIMIT/dims[j]) { fail("output geometry exceeds bound"); return NULL; } output*=dims[j]; }
    if (size(keep)!=input || row<0 || row>=output || ci%groups || co%groups ||
        PyArray_DIM(w,0)!=co || PyArray_DIM(w,1)!=ci/groups || PyArray_DIM(w,2)<=0 || PyArray_DIM(w,3)<=0) {
        fail("actual convolution source geometry differs"); return NULL;
    }
    if (mask!=Py_None && (!vector(mask,NPY_BOOL,&active) || size(active)!=output || !truth(active,row))) {
        if (!PyErr_Occurred()) fail("inactive or mismatched convolution row");
        return NULL;
    }
    int64_t b=row/(co*ho*wo),oc=(row/(ho*wo))%co,oh=(row/wo)%ho,ow=row%wo;
    int64_t cig=ci/groups,group=oc/(co/groups);
    for (int pass=0;pass<2;++pass) {
        /* Positive dilation makes this the old argsort's exact global order. */
        for (int64_t c=0;c<cig;++c) for (int64_t kh=0;kh<PyArray_DIM(w,2);++kh) {
            int64_t ih=oh*sy-py+kh*dy; if (ih<0 || ih>=hi) continue;
            for (int64_t kw=0;kw<PyArray_DIM(w,3);++kw) {
                int64_t iw=ow*sx-px+kw*dx; if (iw<0 || iw>=wi) continue;
                int64_t col=((b*ci+group*cig+c)*hi+ih)*wi+iw;
                if (!truth(keep,col)) continue;
                double val; memcpy(&val,(char *)PyArray_DATA(w)+oc*PyArray_STRIDE(w,0)+c*PyArray_STRIDE(w,1)+
                    kh*PyArray_STRIDE(w,2)+kw*PyArray_STRIDE(w,3),8);
                if (val!=0. && !visit(&r,pass,integer(slots,col),val,integer(powers,col))) goto done;
            }
        }
        if (!pass && !allocate(&r)) goto done;
    }
done: return finish(&r,pivot);
}

static PyObject *sum_row(PyObject *self,PyObject *args) {
    (void)self; PyObject *items; long long row,pivot; Row r={0};
    if (!PyArg_ParseTuple(args,"OLL",&items,&row,&pivot)) return NULL;
    if (!PyTuple_CheckExact(items) || PyTuple_GET_SIZE(items)>LIMIT) { fail("bounded original sum parents required"); return NULL; }
    for (int pass=0;pass<2;++pass) {
        for (Py_ssize_t i=0;i<PyTuple_GET_SIZE(items);++i) {
            PyObject *t=PyTuple_GET_ITEM(items,i); Vec keep,slots,powers;
            if (!PyTuple_CheckExact(t) || PyTuple_GET_SIZE(t)!=4) { fail("complete sum parent descriptor required"); goto done; }
            if (!parents(PyTuple_GET_ITEM(t,0),PyTuple_GET_ITEM(t,1),PyTuple_GET_ITEM(t,2),&keep,&slots,&powers)) goto done;
            PyObject *n=PyTuple_GET_ITEM(t,3);
            if (!PyLong_CheckExact(n)) { fail("exact source multiplicity required"); goto done; }
            int64_t mult=PyLong_AsLongLong(n); if (PyErr_Occurred()) goto done;
            if (row<0 || row>=size(keep) || mult<1 || mult>(INT64_C(1)<<53)) { fail("bounded original sum coordinate/multiplicity required"); goto done; }
            if (truth(keep,row) && !visit(&r,pass,integer(slots,row),(double)mult,integer(powers,row))) goto done;
        }
        if (!pass && !allocate(&r)) goto done;
    }
done: return finish(&r,pivot);
}
static PyMethodDef methods[]={
    {"csr_row",csr_row,METH_VARARGS,"Original CSR support, bound and positive defining row."},
    {"conv_row",conv_row,METH_VARARGS,"Original Conv support, bound and positive defining row."},
    {"sum_row",sum_row,METH_VARARGS,"Original shared sum, bound and positive defining row."},
    {NULL,NULL,0,NULL}
};
static struct PyModuleDef module={PyModuleDef_HEAD_INIT,"_c106_defining_rows_v1",NULL,-1,methods,NULL,NULL,NULL,NULL};
PyMODINIT_FUNC PyInit__c106_defining_rows_v1(void) { import_array(); return PyModule_Create(&module); }
