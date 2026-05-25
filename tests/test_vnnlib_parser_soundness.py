"""Regression tests for VNNLIB parser soundness invariants.

Added 2026-05-24 per advisor review. The HyZor SATSidecar exploratory
parser had a flat-union bug for multiple top-level ``(assert (or ...))``
forms; the same shape exists in ACAS-style specs. This file pins the
ACT parser's Cartesian-product semantic so a future refactor cannot
silently regress to flat union.

Tests are self-contained (no real benchmark dependency).
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
import torch

from act.front_end.vnnlib_loader.vnnlib_parser import (
    parse_vnnlib_queries,
    parse_vnnlib_to_tensors,
    list_vnnlib_variables,
    extract_label_from_vnnlib,
)


def _spec_holds(out_spec, y: np.ndarray) -> bool:
    """Evaluate an OutputSpec on concrete y under zero-tolerance UNSAFE
    semantics. Returns True iff y is in the unsafe set described by the
    spec (the convention used by ACT's _eval_unsafe_strict)."""
    kind = out_spec.kind
    if kind == "UNSAFE_LINEAR":
        C = np.asarray(out_spec.c, dtype=np.float64)
        d = np.asarray(out_spec.d, dtype=np.float64).reshape(-1)
        if C.ndim == 1:
            C = C.reshape(1, -1)
        # Convention: spec is the UNSAFE set; y in unsafe iff all rows hold.
        return bool(np.all(C @ y <= d))
    raise NotImplementedError(f"unhandled OutputSpec.kind = {kind!r}")


class TestVnnlibParserMultiOrCartesian(unittest.TestCase):
    """The two top-level OR-blocks must Cartesian-product to 4 queries,
    each conjoining ONE branch from EACH OR. Treating them as a flat
    union would falsely accept a witness satisfying ONE branch of OR_1
    but no branch of OR_2."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmpdir.name)
        spec = self.tmp / "multi_or.vnnlib"
        spec.write_text(
            "(declare-const X_0 Real)\n"
            "(declare-const Y_0 Real) (declare-const Y_1 Real)\n"
            "(assert (>= X_0 0.0)) (assert (<= X_0 1.0))\n"
            "(assert (or (<= Y_0 0.0) (>= Y_0 10.0)))\n"
            "(assert (or (<= Y_1 -1.0) (>= Y_1 5.0)))\n"
        )
        self.spec = spec

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_cartesian_count(self):
        queries = parse_vnnlib_queries(self.spec)
        self.assertEqual(
            len(queries), 4,
            "2 top-level OR blocks × 2 alternatives each must give 4 "
            "Cartesian queries, not 4 flat-union branches"
        )

    def test_phantom_witness_blocked_under_strict_eval(self):
        """y = (-1.5, 2): satisfies OR_1 branch 1 (Y_0 = -1.5 ≤ 0) but
        NO branch of OR_2 (Y_1 = 2 violates ≤ -1 and ≥ 5). The flat-union
        bug accepts this as SAT via OR_1. The correct Cartesian semantic
        rejects it (no query has all rows holding).
        """
        queries = parse_vnnlib_queries(self.spec)
        y = np.array([-1.5, 2.0])
        any_unsafe = any(_spec_holds(out_spec, y) for _, out_spec in queries)
        self.assertFalse(
            any_unsafe,
            "y=(-1.5, 2) is SAFE under correct Cartesian semantic; if any "
            "query reports unsafe, the parser has regressed to flat union"
        )

    def test_real_witness_accepted_under_strict_eval(self):
        """y = (-1.5, -2): satisfies OR_1 branch 1 (Y_0 ≤ 0) AND OR_2
        branch 1 (Y_1 ≤ -1). One Cartesian query must hold."""
        queries = parse_vnnlib_queries(self.spec)
        y = np.array([-1.5, -2.0])
        any_unsafe = any(_spec_holds(out_spec, y) for _, out_spec in queries)
        self.assertTrue(
            any_unsafe,
            "y=(-1.5, -2) is a real witness (passes both OR-blocks); if "
            "no query holds, the parser is dropping legitimate SATs"
        )

    def test_three_top_ors_cartesian(self):
        """Generalization: 2 × 3 × 2 = 12 Cartesian queries."""
        spec_path = self.tmp / "triple_or.vnnlib"
        spec_path.write_text(
            "(declare-const X_0 Real)\n"
            "(declare-const Y_0 Real) (declare-const Y_1 Real) (declare-const Y_2 Real)\n"
            "(assert (>= X_0 0.0)) (assert (<= X_0 1.0))\n"
            "(assert (or (<= Y_0 0.0) (>= Y_0 10.0)))\n"
            "(assert (or (<= Y_1 -1.0) (>= Y_1 5.0) (<= Y_1 -100.0)))\n"
            "(assert (or (<= Y_2 0.0) (>= Y_2 1.0)))\n"
        )
        queries = parse_vnnlib_queries(spec_path)
        self.assertEqual(len(queries), 12)


class TestVnnlibParserRealAcasxuMultiOr(unittest.TestCase):
    """Smoke test on a real ACAS Xu spec known to have ≥2 top-level ORs.
    Confirms the Cartesian count matches the expected product and per-
    query input box stays non-empty (no global-hoist collapse)."""

    def test_acasxu_multi_or_smoke(self):
        import re
        acasxu = Path("/data1/Kane/data/vnncomp2025_benchmarks/benchmarks/acasxu_2023/vnnlib")
        if not acasxu.is_dir():
            self.skipTest("acasxu vnnlib not present")
        target = None
        expected = None
        for f in sorted(acasxu.glob("*.vnnlib")):
            try:
                text = f.read_text()
            except Exception:
                continue
            n_or_asserts = len(re.findall(r"\(assert\s+\(or\b", text))
            if n_or_asserts >= 2:
                target = f
                expected = n_or_asserts
                break
        if target is None:
            self.skipTest("no multi-OR acasxu spec found")
        queries = parse_vnnlib_queries(target)
        # Cartesian product is product of |OR_i|; each |OR_i| ≥ 2 so total ≥ 2^expected
        self.assertGreaterEqual(
            len(queries), 1 << expected,
            f"{target.name} has {expected} top-OR blocks; Cartesian "
            f"product must be ≥ 2^{expected} = {1 << expected} queries"
        )
        for in_spec, out_spec in queries[:5]:
            lb = np.asarray(in_spec.lb).reshape(-1)
            ub = np.asarray(in_spec.ub).reshape(-1)
            self.assertTrue(
                np.all(lb <= ub),
                f"per-query input box must be non-empty; got lb={lb} ub={ub}"
            )


class TestVnnlibParserGzipSupport(unittest.TestCase):
    """R10: cgan_2023 + parts of nn4sys ship .vnnlib.gz. The parser must
    open compressed and uncompressed specs identically. Asserts that
    parse_vnnlib_queries / parse_vnnlib_to_tensors / list_vnnlib_variables /
    extract_label_from_vnnlib all yield the same result on a .gz spec as
    on its plain twin."""

    def setUp(self):
        import gzip
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmpdir.name)
        body = (
            "; label: 7\n"
            "(declare-const X_0 Real)\n"
            "(declare-const Y_0 Real) (declare-const Y_1 Real)\n"
            "(assert (>= X_0 0.0)) (assert (<= X_0 1.0))\n"
            "(assert (or (<= Y_0 0.0) (>= Y_0 10.0)))\n"
        )
        self.plain = self.tmp / "demo.vnnlib"
        self.plain.write_text(body)
        self.gz = self.tmp / "demo.vnnlib.gz"
        with gzip.open(self.gz, "wt", encoding="utf-8") as f:
            f.write(body)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_parse_queries_gz_matches_plain(self):
        q_plain = parse_vnnlib_queries(self.plain)
        q_gz = parse_vnnlib_queries(self.gz)
        self.assertEqual(len(q_plain), len(q_gz))
        self.assertEqual(len(q_gz), 2)

    def test_to_tensors_gz_matches_plain(self):
        t_plain, m_plain = parse_vnnlib_to_tensors(self.plain)
        t_gz, m_gz = parse_vnnlib_to_tensors(self.gz)
        self.assertTrue(torch.equal(t_plain, t_gz))
        self.assertEqual(m_plain["num_inputs"], m_gz["num_inputs"])
        self.assertEqual(m_plain["num_outputs"], m_gz["num_outputs"])

    def test_list_variables_gz_matches_plain(self):
        self.assertEqual(
            list_vnnlib_variables(self.plain),
            list_vnnlib_variables(self.gz),
        )

    def test_extract_label_gz_matches_plain(self):
        self.assertEqual(
            extract_label_from_vnnlib(self.plain),
            extract_label_from_vnnlib(self.gz),
        )

    def test_real_cgan_gz_parses(self):
        cgan = Path(
            "/data1/Kane/data/vnncomp2025_benchmarks/benchmarks/cgan_2023/vnnlib"
        )
        if not cgan.is_dir():
            self.skipTest("cgan_2023 vnnlib not present")
        gz_files = sorted(cgan.glob("*.vnnlib.gz"))
        if not gz_files:
            self.skipTest("no .vnnlib.gz under cgan_2023")
        info = list_vnnlib_variables(gz_files[0])
        self.assertGreater(info["num_inputs"], 0)
        self.assertGreater(info["num_outputs"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
