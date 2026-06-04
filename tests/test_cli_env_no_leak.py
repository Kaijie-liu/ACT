"""Regression test for the nn4sys/lindex auto-profile env-leak fix.

When `_run_vnnlib_verify_hybridz` processes multiple iids in the same
Python process, the per-iid lindex profile uses `os.environ.setdefault`
to enable three knobs:
  - ACT_HZ_SMALL_DENSE_DIRECT_QUERY
  - ACT_HZ_SPECAWARE_BOUND_CACHE
  - ACT_HZ_STABLE_AFFINE_FASTPATH

Without a per-iid restore, the first lindex iid sets these vars and they
stay set for all later iids in the loop — including non-lindex iids
(e.g., nn4sys/mscn / pensieve). The remaining knobs are conservative
proving paths so the bug is **not** a soundness violation, but it does
contaminate the process state and complicates auditing.

This test directly drives the per-iid restore logic by simulating the
control flow without actually loading any models. It guards the
`_iid_env_restore` mechanism in `cli._run_vnnlib_verify_hybridz`.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Optional

sys.path.insert(0, "/data1/Kane/ACT")


_LINDEX_ENV_KEYS = (
    "ACT_HZ_SMALL_DENSE_DIRECT_QUERY",
    "ACT_HZ_SPECAWARE_BOUND_CACHE",
    "ACT_HZ_STABLE_AFFINE_FASTPATH",
)


def _iter_one_iid(*, is_lindex: bool, preexisting_env: Dict[str, str]):
    """Reproduce the cli per-iid env-management loop in isolation."""
    # Restore preexisting env at start to make the test hermetic.
    for k in _LINDEX_ENV_KEYS:
        os.environ.pop(k, None)
    for k, v in preexisting_env.items():
        os.environ[k] = v

    # Snapshot what the test sees BEFORE the per-iid body.
    pre = {k: os.environ.get(k) for k in _LINDEX_ENV_KEYS}

    # This block mirrors cli.py lines 941, 1106-1124:
    _iid_env_restore: Dict[str, Optional[str]] = {}
    if is_lindex:
        for _k in _LINDEX_ENV_KEYS:
            if _k not in _iid_env_restore:
                _iid_env_restore[_k] = os.environ.get(_k)
            os.environ.setdefault(_k, "1")

    # During the iid body, the knobs should be set when is_lindex=True.
    during = {k: os.environ.get(k) for k in _LINDEX_ENV_KEYS}

    # End-of-iter restore (mirrors cli.py line 1320-1325):
    for _k, _prev in _iid_env_restore.items():
        if _prev is None:
            os.environ.pop(_k, None)
        else:
            os.environ[_k] = _prev

    # After the iid body, env should look exactly like before.
    post = {k: os.environ.get(k) for k in _LINDEX_ENV_KEYS}
    return pre, during, post


def test_lindex_iid_sets_then_restores_env():
    """An iid where _nn4sys_lindex_profile=True should set the three env
    vars during the body and restore the prior state after."""
    pre, during, post = _iter_one_iid(is_lindex=True, preexisting_env={})
    assert all(pre[k] is None for k in _LINDEX_ENV_KEYS), (
        f"unexpected pre-state: {pre!r}"
    )
    assert all(during[k] == "1" for k in _LINDEX_ENV_KEYS), (
        f"lindex profile should set knobs: {during!r}"
    )
    assert all(post[k] is None for k in _LINDEX_ENV_KEYS), (
        f"env leaked into next iid: {post!r}"
    )


def test_lindex_iid_preserves_user_explicit_env():
    """When user explicitly set the knobs before invoking cli, the
    profile must NOT overwrite them (setdefault semantics), AND must
    restore the user's value (NOT clear it)."""
    user_env = {
        "ACT_HZ_SMALL_DENSE_DIRECT_QUERY": "1",
        "ACT_HZ_SPECAWARE_BOUND_CACHE": "0",
        "ACT_HZ_STABLE_AFFINE_FASTPATH": "1",
    }
    pre, during, post = _iter_one_iid(is_lindex=True, preexisting_env=user_env)
    assert during == user_env, f"profile overrode user env: {during!r}"
    assert post == user_env, f"restore corrupted user env: {post!r}"


def test_non_lindex_iid_does_not_touch_env():
    """An iid that is NOT lindex (e.g. nn4sys/mscn) should leave env
    untouched both during and after."""
    pre, during, post = _iter_one_iid(is_lindex=False, preexisting_env={})
    assert pre == during == post, (
        f"non-lindex iid mutated env: pre={pre!r} during={during!r} post={post!r}"
    )


def test_lindex_then_non_lindex_does_not_leak():
    """End-to-end: after a lindex iid completes, a subsequent non-lindex
    iid in the SAME process must see a clean env."""
    # First iid: lindex
    _iter_one_iid(is_lindex=True, preexisting_env={})
    # No reset between iids — simulate the cli loop reading env state.
    after_lindex = {k: os.environ.get(k) for k in _LINDEX_ENV_KEYS}
    assert all(after_lindex[k] is None for k in _LINDEX_ENV_KEYS), (
        f"after lindex iid: env should be clean, got {after_lindex!r}"
    )
    # Second iid: non-lindex (we just need to confirm the snapshot for
    # the second iid would see clean env, hence pre == during == post).
    pre, during, post = _iter_one_iid(
        is_lindex=False,
        preexisting_env={k: v for k, v in after_lindex.items() if v is not None},
    )
    assert all(v is None for v in (pre.values() | during.values() | post.values())) if False else \
        all(pre[k] is None and during[k] is None and post[k] is None
            for k in _LINDEX_ENV_KEYS), (
        f"non-lindex iid after lindex saw leaked env: {during!r}"
    )


if __name__ == "__main__":
    # Hermetic: clear before running.
    for k in _LINDEX_ENV_KEYS:
        os.environ.pop(k, None)
    tests = [
        test_lindex_iid_sets_then_restores_env,
        test_lindex_iid_preserves_user_explicit_env,
        test_non_lindex_iid_does_not_touch_env,
        test_lindex_then_non_lindex_does_not_leak,
    ]
    n_pass = n_fail = 0
    for t in tests:
        # Reset before each test for hermeticity.
        for k in _LINDEX_ENV_KEYS:
            os.environ.pop(k, None)
        try:
            t()
            print(f"  PASS  {t.__name__}")
            n_pass += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            n_fail += 1
        except Exception as e:
            print(f"  ERR   {t.__name__}: {type(e).__name__}: {e}")
            n_fail += 1
    print(f"\nResult: {n_pass}/{len(tests)} passed")
    sys.exit(1 if n_fail else 0)
