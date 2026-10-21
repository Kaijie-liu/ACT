"""Default-off, owned snapshots of the actual native F0 problem and result.

No solve, propagation, normalization, sum_duplicates or mathematical acceptance
occurs here. Binary64 storage (including raw nonfinite candidate bits) is copied
losslessly. Observers receive no mutable alias to the live solve.
"""
import base64
import hashlib
import time

import numpy as np
import scipy.sparse as sp

from scoped_source.native_capture import (
    SCHEMA, GRADE, CaptureError, CaptureCapacity, CapturedSolve, CaptureLedger,
    canonical, checked_scope, bounded_metadata, require,
)


class NativeCapture:
    """One registered solve; the byte ledger and Work persist across captures.

    extra_hz/metadata are optional supplied upstream context, not independently
    verified lowering. Source labels and hashes remain distinct from a proof.
    """
    def __init__(self, *, scope, ledger, extra_hz=None, metadata=None):
        require(type(ledger) is CaptureLedger, "existing shared capture ledger")
        self.scope_bytes = checked_scope(scope)
        self.ledger = ledger
        require(extra_hz is None or (type(extra_hz) is dict and len(extra_hz) <= 8),
                "bounded upstream source mapping")
        self.extra_hz = dict(extra_hz or {})
        require(len(self.extra_hz) <= 8 and all(type(k) is str and k.isascii() and k.isidentifier()
                and len(k) <= 40 for k in self.extra_hz), "bounded upstream source names")
        self.metadata_bytes = bounded_metadata({} if metadata is None else metadata)
        self.record = None
        self.event = "REGISTERED"
        self.capture_seconds = 0.0

    def failure(self, reason, *, native_called):
        self._capture(event=reason, native_called=native_called)

    def validate_encodings(self, encodings):
        import json
        self.ledger.tick()
        scope = json.loads(self.scope_bytes)
        try:
            require([list(e.pair) for e in encodings] == scope["pairs"], "encoding pair/order binding")
            require(all(list(e.property_row) == scope["q"] and e.property_constant == scope["constant"]
                        for e in encodings), "encoding property binding")
        except (AttributeError, KeyError, TypeError) as exc:
            raise CaptureError("malformed encoding binding") from exc

    def hz(self, *, hz, input_hz, model, row, objective, result, native_called=True):
        self._capture(event="RETURNED", native_called=native_called,
                      hz=hz, input_hz=input_hz, model=model, row=row,
                      objective=objective, result=result)

    def monolithic(self, *, encodings, model, result):
        self._capture(event="RETURNED", native_called=True, encodings=encodings,
                      model=model, objective=model.objective, result=result)

    def _capture(self, **kwargs):
        require(self.event == "REGISTERED" and self.record is None, "single registered capture")
        self.event = "RETURNED" if kwargs.get("event") == "RETURNED" else kwargs["event"]
        started = time.monotonic()
        try:
            self.record = self._build(**kwargs)
            self.event = "CAPTURED"
        except CaptureError:
            self.event = "CAPTURE_FAILED"
            raise
        except (MemoryError, ValueError, TypeError, OverflowError, AttributeError, KeyError) as exc:
            self.event = "CAPTURE_FAILED"
            raise CaptureError("native capture failed: " + type(exc).__name__) from exc
        finally:
            self.capture_seconds += time.monotonic()-started

    def _build(self, *, event, native_called, model=None, row=None, objective=None,
               result=None, hz=None, input_hz=None, encodings=None):
        import json
        ledger = self.ledger
        ledger.tick()
        arrays, matrices, sources = {}, {}, {}
        scope = json.loads(self.scope_bytes)
        upper = 32768 + len(self.scope_bytes) + len(self.metadata_bytes)
        ledger.reserve(upper)

        def vector(name, value):
            nonlocal upper
            require(type(value) is np.ndarray and value.ndim == 1, "native one-dimensional buffer")
            require(value.dtype == np.float64 or (np.issubdtype(value.dtype, np.signedinteger)
                    and value.dtype.itemsize <= 8), "native binary64/integer buffer")
            require(name not in arrays, "unique array name")
            size = int(value.size)*8
            upper += 4*((size+2)//3) + 512
            ledger.reserve(upper)  # before astype, tobytes, base64 or JSON allocation
            arrays[name] = value
            return name

        def matrix(name, mat):
            require(sp.isspmatrix_csr(mat) and mat.dtype == np.float64, "native CSR binary64")
            require(name not in matrices and len(mat.shape) == 2, "unique matrix name")
            require(mat.data.ndim == mat.indices.ndim == mat.indptr.ndim == 1,
                    "CSR buffers")
            matrices[name] = {"shape": list(mat.shape),
                **{k: vector(name+"."+k, getattr(mat, k)) for k in ("data", "indices", "indptr")}}

        def source(name, value):
            require(hasattr(value, "Auc") and type(value.frame_id) is int,
                    "diagnostics require explicit sparse source frame")
            for key in ("c", "b", "ub"):
                vector(name+"."+key, getattr(value, key))
            for key in ("Gc", "Gb", "Ac", "Ab", "Auc", "Aub"):
                matrix(name+"."+key, getattr(value, key))
            sources[name] = {"frame_id": value.frame_id, "exact_flag": bool(value.exact),
                             "n_cont": value.n_cont, "n_bin": value.n_bin, "n_out": value.n_out}

        def lower(name, value):
            matrix(name+".A", value.A)
            matrix(name+".value_matrix", value.value_matrix)
            for key in ("row_lb", "row_ub", "var_lb", "var_ub", "integrality", "value_center"):
                vector(name+"."+key, getattr(value, key))

        native = None
        if model is not None:
            vector("native.objective", objective)
            if encodings is None:
                require(scope["path"] == "pair", "pair capture scope")
                lower("native", model)
                native = {"variables": model.n_var, "n_cont": model.n_cont, "n_bin": model.n_bin,
                          "objective_row": row, "objective_offset": float(model.value_center[row]).hex(),
                          "binary_map": "beta=2*z-1"}
                source("f0", hz)
                if input_hz is not None:
                    source("input", input_hz)
            else:
                require(scope["path"] == "monolithic" and [list(e.pair) for e in encodings] == scope["pairs"],
                        "ordered monolithic residual pair scope")
                matrix("native.A", model.A)
                for key in ("row_lb", "row_ub", "var_lb", "var_ub", "integrality"):
                    vector("native."+key, getattr(model, key))
                branches = []
                for i, (encoding, local) in enumerate(zip(encodings, model.lowered)):
                    lower("branch%d" % i, local)
                    source("f0_%d" % i, encoding.output_hz)
                    source("input_%d" % i, encoding.input_hz)
                    branches.append({"pair": list(encoding.pair), "n_cont": local.n_cont, "n_bin": local.n_bin})
                native = {"variables": int(model.objective.size), "objective_offset": 0.0.hex(),
                          "selector_offset": model.selector_offset, "local_offsets": list(model.local_offsets),
                          "branches": branches, "binary_map": "local variables are selector-homogenized"}
        if model is not None:
            for name, value in self.extra_hz.items():
                source("upstream_"+name, value)
        get = (lambda key: result.get(key)) if isinstance(result, dict) else (lambda key: getattr(result, key, None))
        raw_candidate = get("x")
        candidate = "MISSING"
        if raw_candidate is not None:
            require(isinstance(raw_candidate, np.ndarray) and raw_candidate.ndim == 1,
                    "native candidate array")
            vector("candidate", raw_candidate)
            candidate = "RAW_UNCHECKED"
        native_result = {key: (None if get(key) is None else int(get(key)))
                         for key in ("status", "mip_node_count")}
        for key in ("fun", "mip_dual_bound", "mip_gap"):
            native_result[key] = None if get(key) is None else float(get(key)).hex()
        # Freeze bytes only after a complete conservative size preflight.
        encoded = {}
        for name, value in arrays.items():
            ledger.tick()
            dtype = "<f8" if value.dtype == np.float64 else "<i8"
            ledger.visit(int(value.size))
            raw = value.astype(dtype, copy=False).tobytes(order="C")
            encoded[name] = {"dtype": dtype, "count": int(value.size),
                             "base64": base64.b64encode(raw).decode("ascii")}
            ledger.tick()
        value = {"schema": SCHEMA, "grade": GRADE, "scope": scope,
                 "event": event, "native_called": bool(native_called), "native": native,
                 "arrays": encoded, "matrices": matrices, "sources": sources,
                 "source_metadata": json.loads(self.metadata_bytes), "result": native_result,
                 "candidate": candidate, "deadline": ledger.work.deadline}
        ledger.tick()
        payload = canonical(value)
        require(len(payload) <= upper, "preflight encoding upper bound")
        ledger.visit((len(payload)+4095)//4096)
        anchor = hashlib.sha256(payload).hexdigest()
        ledger.tick()
        ledger.accept(len(payload))
        return CapturedSolve(payload, anchor)
