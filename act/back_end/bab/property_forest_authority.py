"""Live-process authority boundary for property-separable BaB.

The serialized node-conservation receipt is an audit record, not a proof by
itself.  This module binds that record to the exact live ACT graph, encoded
ASSERT rows, input domain, BaB configuration, and tensor dtypes.  A
process-local MAC prevents a stale or edited metadata object from being
promoted after :func:`verify_bab_batched` returns.

The MAC is intentionally not a portable signature.  A saved JSON file cannot
re-create proof authority in another process; authority comes from the live
``VerifyStatus.CERTIFIED`` return and an immediate independent validation.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import hmac
import json
import math
import secrets
import struct
from pathlib import Path
from typing import Mapping, Optional, cast

import torch

from act.back_end.config import BaBConfig
from act.back_end.core import Net


__all__ = [
    "new_property_forest_run_token",
    "source_file_digests",
    "validate_bab_safe_capability",
]


_LIVE_SCHEMA = "act.property_forest_live_seal.v1"
_PROCESS_KEY = secrets.token_bytes(32)
_CAPABILITY_SENTINEL = object()
_LIVE_CAPABILITIES: dict[
    int, tuple["_LiveCapability", str, object]
] = {}


class _LiveCapability:
    """Opaque, process-local, single-use proof capability."""

    __slots__ = ("_identity",)

    def __init__(self, sentinel: object) -> None:
        if sentinel is not _CAPABILITY_SENTINEL:
            raise TypeError("live capabilities are verifier-issued only")
        self._identity = secrets.token_hex(32)


def new_property_forest_run_token() -> str:
    """Return a caller-held nonce that rejects same-process stale results."""

    return secrets.token_hex(32)


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def canonical_receipt_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _hash_len(digest: "hashlib._Hash", value: int) -> None:
    digest.update(struct.pack(">Q", int(value)))


def _hash_bytes(
    digest: "hashlib._Hash", tag: bytes, payload: bytes
) -> None:
    digest.update(tag)
    _hash_len(digest, len(payload))
    digest.update(payload)


def _hash_value(digest: "hashlib._Hash", value: object) -> None:
    """Hash supported live proof data with explicit type and length tags."""

    if value is None:
        digest.update(b"N")
        return
    if type(value) is bool:
        digest.update(b"B1" if value else b"B0")
        return
    if isinstance(value, enum.Enum):
        _hash_bytes(
            digest,
            b"E",
            (
                f"{type(value).__module__}.{type(value).__qualname__}:"
                f"{value.name}"
            ).encode("utf-8"),
        )
        return
    if type(value) is int:
        _hash_bytes(digest, b"I", str(value).encode("ascii"))
        return
    if type(value) is float:
        if math.isnan(value):
            raise ValueError("NaN cannot enter a proof binding")
        _hash_bytes(digest, b"F", struct.pack(">d", value))
        return
    if type(value) is str:
        _hash_bytes(digest, b"S", value.encode("utf-8"))
        return
    if isinstance(value, torch.dtype):
        _hash_bytes(digest, b"D", str(value).encode("ascii"))
        return
    if isinstance(value, torch.device):
        # Execution placement is not a mathematical model input.  Dtype is
        # bound separately; omitting device makes CPU/GPU replay comparable.
        _hash_bytes(digest, b"V", value.type.encode("ascii"))
        return
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
        if tensor.layout != torch.strided:
            raise TypeError("proof bindings require strided tensors")
        tensor = tensor.contiguous()
        digest.update(b"T")
        _hash_bytes(digest, b"d", str(tensor.dtype).encode("ascii"))
        _hash_value(digest, tuple(int(item) for item in tensor.shape))
        raw = (
            tensor.view(torch.uint8)
            .reshape(-1)
            .to(device="cpu")
            .numpy()
        )
        _hash_len(digest, int(raw.nbytes))
        digest.update(memoryview(raw))
        return
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        digest.update(b"C")
        _hash_bytes(
            digest,
            b"c",
            (
                f"{type(value).__module__}.{type(value).__qualname__}"
            ).encode("utf-8"),
        )
        fields = dataclasses.fields(value)
        _hash_len(digest, len(fields))
        for field in fields:
            _hash_bytes(digest, b"k", field.name.encode("utf-8"))
            _hash_value(digest, getattr(value, field.name))
        return
    if isinstance(value, Mapping):
        digest.update(b"M")
        encoded_keys: list[tuple[bytes, object, object]] = []
        for key, item in value.items():
            key_digest = hashlib.sha256()
            _hash_value(key_digest, key)
            encoded_keys.append((key_digest.digest(), key, item))
        encoded_keys.sort(key=lambda record: record[0])
        _hash_len(digest, len(encoded_keys))
        for _, key, item in encoded_keys:
            _hash_value(digest, key)
            _hash_value(digest, item)
        return
    if isinstance(value, tuple):
        digest.update(b"Q")
        _hash_len(digest, len(value))
        for item in value:
            _hash_value(digest, item)
        return
    if isinstance(value, list):
        digest.update(b"L")
        _hash_len(digest, len(value))
        for item in value:
            _hash_value(digest, item)
        return
    if isinstance(value, (set, frozenset)):
        digest.update(b"Z")
        item_digests = []
        for item in value:
            item_digest = hashlib.sha256()
            _hash_value(item_digest, item)
            item_digests.append(item_digest.digest())
        item_digests.sort()
        _hash_len(digest, len(item_digests))
        for item_digest in item_digests:
            digest.update(item_digest)
        return
    raise TypeError(
        "unsupported value in property-forest proof binding: "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _sha256_value(value: object) -> str:
    digest = hashlib.sha256()
    _hash_value(digest, value)
    return digest.hexdigest()


def property_forest_binding_digests(
    net: Net,
    config: BaBConfig,
) -> dict[str, str]:
    """Re-derive exact live model/spec/config/dtype bindings."""

    assertion_layers = [
        layer for layer in net.layers if layer.kind == "ASSERT"
    ]
    input_layers = [
        layer
        for layer in net.layers
        if layer.kind in {"INPUT", "INPUT_SPEC"}
    ]
    network_layers = [
        {
            "id": int(layer.id),
            "kind": str(layer.kind),
            "params": layer.params,
            "in_vars": tuple(int(value) for value in layer.in_vars),
            "out_vars": tuple(int(value) for value in layer.out_vars),
        }
        for layer in net.layers
    ]
    topology = {
        "preds": {
            int(layer_id): tuple(int(value) for value in values)
            for layer_id, values in net.preds.items()
        },
        "succs": {
            int(layer_id): tuple(int(value) for value in values)
            for layer_id, values in net.succs.items()
        },
    }
    encoded_rows = [
        {
            "id": int(layer.id),
            "params": layer.params,
            "in_vars": tuple(int(value) for value in layer.in_vars),
            "out_vars": tuple(int(value) for value in layer.out_vars),
        }
        for layer in assertion_layers
    ]
    input_domain = [
        {
            "id": int(layer.id),
            "kind": str(layer.kind),
            "params": layer.params,
            "in_vars": tuple(int(value) for value in layer.in_vars),
            "out_vars": tuple(int(value) for value in layer.out_vars),
        }
        for layer in input_layers
    ]
    dtype_records = [
        {
            "layer_id": int(layer.id),
            "parameter": str(name),
            "dtype": str(value.dtype),
            "shape": tuple(int(item) for item in value.shape),
            "floating": bool(value.is_floating_point()),
        }
        for layer in net.layers
        for name, value in sorted(layer.params.items())
        if isinstance(value, torch.Tensor)
    ]
    return {
        "network_sha256": _sha256_value(
            {"layers": network_layers, "topology": topology}
        ),
        "encoded_rows_sha256": _sha256_value(encoded_rows),
        "input_domain_sha256": _sha256_value(input_domain),
        "config_sha256": _sha256_value(config),
        "dtype_sha256": _sha256_value(
            {
                "parameter_dtypes": dtype_records,
                "torch_default_dtype": str(torch.get_default_dtype()),
            }
        ),
    }


def source_file_digests(
    source_paths: Mapping[str, Path | str],
) -> dict[str, str]:
    """Hash caller-named source files without trusting saved hash metadata."""

    output: dict[str, str] = {}
    for name, raw_path in sorted(source_paths.items()):
        if type(name) is not str or not name:
            raise ValueError("source binding names must be nonempty strings")
        path = Path(raw_path)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        output[name] = digest.hexdigest()
    return output


def _issue_property_forest_live_result(
    *,
    result: object,
    run_token: str,
    binding_digests: Mapping[str, str],
    source_digests: Mapping[str, str],
    node_receipt: Mapping[str, object],
    live_facts: Mapping[str, object],
) -> tuple[dict[str, object], object]:
    """Issue one opaque capability plus a non-authoritative audit seal.

    This is module-private and is called only at the terminal CERTIFIED return
    inside ``verify_bab_batched``.  The opaque object is held in the live
    ``VerifyResult`` and can be consumed exactly once.  The JSON-compatible
    seal alone can never mint or restore authority.
    """

    if not _is_sha256(run_token):
        raise ValueError("property-forest run token must be 32-byte hex")
    status = getattr(getattr(result, "status", None), "value", None)
    if (
        status != "certified"
        or getattr(result, "counterexample", None) is not None
    ):
        raise ValueError(
            "live capability issuance requires the terminal CERTIFIED "
            "VerifyResult with no counterexample"
        )
    if (
        set(source_digests) != {"onnx", "vnnlib"}
        or any(not _is_sha256(value) for value in source_digests.values())
    ):
        raise ValueError(
            "live capability requires exact pre-run onnx/vnnlib digests"
        )
    required_bindings = {
        "network_sha256",
        "encoded_rows_sha256",
        "input_domain_sha256",
        "config_sha256",
        "dtype_sha256",
    }
    if (
        set(binding_digests) != required_bindings
        or any(
            not _is_sha256(value)
            for value in binding_digests.values()
        )
    ):
        raise ValueError("invalid property-forest live binding digests")
    body: dict[str, object] = {
        "schema": _LIVE_SCHEMA,
        "proof_authority": False,
        "live_process_only": True,
        "portable_signature": False,
        "run_token_sha256": hashlib.sha256(
            run_token.encode("ascii")
        ).hexdigest(),
        "binding_digests": dict(binding_digests),
        "source_digests": dict(source_digests),
        "node_receipt_sha256": canonical_receipt_sha256(node_receipt),
        "live_facts": dict(live_facts),
    }
    mac = hmac.new(
        _PROCESS_KEY,
        _canonical_json_bytes(body),
        digestmod=hashlib.sha256,
    ).hexdigest()
    capability = _LiveCapability(_CAPABILITY_SENTINEL)
    _LIVE_CAPABILITIES[id(capability)] = (
        capability,
        mac,
        result,
    )
    return {**body, "process_mac_sha256": mac}, capability


def validate_property_forest_live_result(
    receipt: object,
    *,
    result: object,
    capability: object,
    run_token: str,
    net: Net,
    config: BaBConfig,
    expected_source_digests: Mapping[str, str],
    node_receipt: object,
    expected_live_facts: Mapping[str, object],
) -> tuple[bool, tuple[str, ...]]:
    """Consume one live capability and validate current trusted objects."""

    errors: list[str] = []
    issued = _LIVE_CAPABILITIES.pop(id(capability), None)
    if (
        issued is None
        or issued[0] is not capability
        or not isinstance(capability, _LiveCapability)
    ):
        errors.append("missing_stale_or_forged_live_capability")
    elif issued[2] is not result:
        errors.append("live_capability_result_identity_mismatch")
    status = getattr(getattr(result, "status", None), "value", None)
    if (
        status != "certified"
        or getattr(result, "counterexample", None) is not None
    ):
        errors.append("live_result_snapshot_not_certified")
    if type(receipt) is not dict:
        return False, tuple(sorted(set(errors + ["live_seal_not_dict"])))
    data = dict(receipt)
    expected_keys = {
        "schema",
        "proof_authority",
        "live_process_only",
        "portable_signature",
        "run_token_sha256",
        "binding_digests",
        "source_digests",
        "node_receipt_sha256",
        "live_facts",
        "process_mac_sha256",
    }
    if set(data) != expected_keys:
        errors.append("live_seal_schema_mismatch")
    if data.get("schema") != _LIVE_SCHEMA:
        errors.append("live_seal_version_mismatch")
    if data.get("proof_authority") is not False:
        errors.append("live_seal_must_remain_non_authoritative")
    if data.get("live_process_only") is not True:
        errors.append("live_process_scope_missing")
    if data.get("portable_signature") is not False:
        errors.append("portable_signature_must_be_false")
    if not _is_sha256(run_token):
        errors.append("invalid_expected_run_token")
    elif data.get("run_token_sha256") != hashlib.sha256(
        run_token.encode("ascii")
    ).hexdigest():
        errors.append("run_token_mismatch")
    try:
        current_bindings = property_forest_binding_digests(net, config)
    except Exception:
        current_bindings = {}
        errors.append("live_binding_rederivation_failed")
    if data.get("binding_digests") != current_bindings:
        errors.append("live_binding_mismatch")
    if (
        set(expected_source_digests) != {"onnx", "vnnlib"}
        or any(
            not _is_sha256(value)
            for value in expected_source_digests.values()
        )
        or data.get("source_digests")
        != dict(expected_source_digests)
    ):
        errors.append("sealed_source_digest_mismatch")
    try:
        node_sha256 = canonical_receipt_sha256(node_receipt)
    except Exception:
        node_sha256 = None
        errors.append("node_receipt_not_canonical")
    if data.get("node_receipt_sha256") != node_sha256:
        errors.append("node_receipt_mismatch")
    if data.get("live_facts") != dict(expected_live_facts):
        errors.append("live_facts_mismatch")
    facts = data.get("live_facts")
    expected_fact_keys = {
        "mode",
        "verifier_status",
        "spec_rows_total",
        "root_certified_rows",
        "forest_rows",
        "processed_nodes",
        "pool_remaining",
        "any_dropped_frontier_cap",
        "any_dropped_max_depth",
        "requested_max_batch_size",
        "initial_effective_max_batch_size",
        "time_budget_seconds",
        "solver_tier",
        "solver_backend",
        "solver_factory",
        "verification_dtype",
        "verification_device",
        "torch_default_dtype",
    }
    if type(facts) is not dict or set(facts) != expected_fact_keys:
        errors.append("live_facts_schema_mismatch")
    else:
        facts_map = cast(dict[str, object], facts)
        total = facts_map.get("spec_rows_total")
        root_rows = facts_map.get("root_certified_rows")
        forest_rows = facts_map.get("forest_rows")
        integer_fields = (
            "spec_rows_total",
            "processed_nodes",
            "pool_remaining",
            "initial_effective_max_batch_size",
        )
        if any(
            type(facts_map.get(name)) is not int
            or cast(int, facts_map[name]) < 0
            for name in integer_fields
        ):
            errors.append("live_fact_integer_invalid")
        if (
            type(root_rows) is not list
            or type(forest_rows) is not list
            or any(
                type(row_id) is not int or row_id < 0
                for row_id in (
                    list(root_rows)
                    if type(root_rows) is list
                    else []
                )
                + (
                    list(forest_rows)
                    if type(forest_rows) is list
                    else []
                )
            )
        ):
            errors.append("live_row_partition_malformed")
        elif type(total) is int:
            root_tuple = tuple(cast(list[int], root_rows))
            forest_tuple = tuple(cast(list[int], forest_rows))
            if (
                tuple(sorted(root_tuple)) != root_tuple
                or tuple(sorted(forest_tuple)) != forest_tuple
                or len(set(root_tuple)) != len(root_tuple)
                or len(set(forest_tuple)) != len(forest_tuple)
                or set(root_tuple) & set(forest_tuple)
                or sorted(root_tuple + forest_tuple)
                != list(range(total))
            ):
                errors.append("live_row_partition_incomplete")
        if (
            facts_map.get("verifier_status") != "certified"
            or facts_map.get("pool_remaining") != 0
            or facts_map.get("any_dropped_frontier_cap") is not False
            or facts_map.get("any_dropped_max_depth") is not False
            or facts_map.get("solver_tier")
            not in {"dual", "dual_alpha", "dual_alpha_eta"}
            or facts_map.get("solver_backend")
            != "act.back_end.solver.solver_dual.DualSolver"
            or type(facts_map.get("solver_factory")) is not str
            or not cast(str, facts_map.get("solver_factory"))
            or facts_map.get("verification_dtype")
            not in {"torch.float32", "torch.float64"}
            or facts_map.get("verification_device")
            not in {"cpu", "cuda"}
            or facts_map.get("torch_default_dtype")
            not in {"torch.float32", "torch.float64"}
            or type(facts_map.get("time_budget_seconds")) is not float
            or not math.isfinite(
                cast(float, facts_map.get("time_budget_seconds"))
            )
            or cast(float, facts_map.get("time_budget_seconds")) <= 0.0
            or type(
                facts_map.get("initial_effective_max_batch_size")
            )
            is not int
            or cast(
                int,
                facts_map.get("initial_effective_max_batch_size"),
            )
            < 1
            or (
                facts_map.get("requested_max_batch_size") is not None
                and type(facts_map.get("requested_max_batch_size"))
                not in {int, str}
            )
        ):
            errors.append("live_safe_terminal_facts_invalid")
        mode = facts_map.get("mode")
        if mode == "root_presolve":
            if (
                type(total) is not int
                or forest_rows != []
                or root_rows != list(range(total))
            ):
                errors.append("root_presolve_partition_invalid")
            if (
                type(node_receipt) is not dict
                or set(node_receipt)
                != {
                    "schema",
                    "proof_authority",
                    "spec_rows_total",
                    "strictly_certified_rows",
                    "forest_rows",
                    "complete",
                }
                or node_receipt.get("schema")
                != "act.property_forest_root_presolve_receipt.v1"
                or node_receipt.get("complete") is not True
                or node_receipt.get("proof_authority") is not False
                or node_receipt.get("spec_rows_total") != total
                or node_receipt.get("strictly_certified_rows")
                != root_rows
                or node_receipt.get("forest_rows") != []
            ):
                errors.append("root_presolve_receipt_invalid")
        elif mode == "complete_forest":
            if not forest_rows:
                errors.append("complete_forest_has_no_forest_rows")
            node_totals = (
                node_receipt.get("totals")
                if type(node_receipt) is dict
                else None
            )
            exact_node_valid = False
            exact_node_errors: tuple[str, ...] = ()
            if (
                type(forest_rows) is list
                and all(type(value) is int for value in forest_rows)
                and type(facts_map.get("processed_nodes")) is int
                and type(facts_map.get("pool_remaining")) is int
            ):
                from act.back_end.bab.bab import (
                    _validate_property_forest_receipt,
                )

                (
                    exact_node_valid,
                    exact_node_errors,
                ) = _validate_property_forest_receipt(
                    node_receipt,
                    expected_row_ids=tuple(
                        cast(list[int], forest_rows)
                    ),
                    expected_processed=cast(
                        int, facts_map["processed_nodes"]
                    ),
                    expected_pool_remaining=cast(
                        int, facts_map["pool_remaining"]
                    ),
                )
            if (
                type(node_receipt) is not dict
                or node_receipt.get("complete") is not True
                or node_receipt.get("proof_authority") is not False
                or node_receipt.get("root_rows") != forest_rows
                or type(node_totals) is not dict
                or node_totals.get("processed")
                != facts_map.get("processed_nodes")
                or not exact_node_valid
            ):
                errors.append("complete_forest_receipt_invalid")
                errors.extend(
                    f"complete_forest_node:{error}"
                    for error in exact_node_errors
                )
        else:
            errors.append("live_mode_invalid")
    claimed_mac = data.pop("process_mac_sha256", None)
    if not _is_sha256(claimed_mac):
        errors.append("process_mac_malformed")
    else:
        expected_mac = hmac.new(
            _PROCESS_KEY,
            _canonical_json_bytes(data),
            digestmod=hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(claimed_mac, expected_mac):
            errors.append("process_mac_mismatch")
        if issued is not None and not hmac.compare_digest(
            issued[1], claimed_mac
        ):
            errors.append("capability_seal_mismatch")
    return not errors, tuple(sorted(set(errors)))


def build_property_forest_safe_proof_receipt(
    *,
    result: object,
    live_seal: object,
    capability: object,
    run_token: str,
    net: Net,
    config: BaBConfig,
    node_receipt: object,
    live_facts: Mapping[str, object],
    source_paths: Mapping[str, Path | str],
    source_digests_before_run: Mapping[str, str],
) -> tuple[Optional[dict[str, object]], tuple[str, ...]]:
    """Promote only an immediately validated live CERTIFIED forest result."""

    valid, errors = validate_property_forest_live_result(
        live_seal,
        result=result,
        capability=capability,
        run_token=run_token,
        net=net,
        config=config,
        expected_source_digests=source_digests_before_run,
        node_receipt=node_receipt,
        expected_live_facts=live_facts,
    )
    error_list = list(errors)
    if set(source_paths) != {"onnx", "vnnlib"}:
        error_list.append("source_path_set_incomplete")
    if (
        set(source_digests_before_run) != {"onnx", "vnnlib"}
        or any(
            not _is_sha256(value)
            for value in source_digests_before_run.values()
        )
    ):
        error_list.append("pre_run_source_digest_set_invalid")
    try:
        source_after = source_file_digests(source_paths)
    except Exception:
        source_after = {}
        error_list.append("source_rehash_failed")
    if source_after != dict(source_digests_before_run):
        error_list.append("source_changed_during_run")
    if not valid or error_list:
        return None, tuple(sorted(set(error_list)))
    body: dict[str, object] = {
        "schema": "act.property_forest_safe_proof.v1",
        "proof_authority": True,
        "authority_source": "live_dual_unsat_and_complete_partition",
        "authority_scope": "this_live_trusted_run_only",
        "authority_trust_boundary": (
            "trusted ACT verifier code in this process; arbitrary "
            "same-process code execution is outside the boundary"
        ),
        "portable_signature": False,
        "cross_process_revalidation_required": True,
        "serialized_receipt_reauthorizes": False,
        "source_digests": source_after,
        "binding_digests": property_forest_binding_digests(net, config),
        "node_receipt_sha256": canonical_receipt_sha256(node_receipt),
        "live_seal_mac_sha256": cast(
            Mapping[str, object], live_seal
        )["process_mac_sha256"],
        "live_facts": dict(live_facts),
    }
    body["receipt_sha256"] = canonical_receipt_sha256(body)
    return body, ()


def validate_bab_safe_capability(
    result: object,
    *,
    net: Net,
    solver_factory: object,
    config: BaBConfig,
    max_batch_size: object,
    time_budget_s: float,
    expected_dtype: str,
    expected_device: str,
    run_token: str,
    source_paths: Mapping[str, Path | str],
    source_digests_before_run: Mapping[str, str],
) -> tuple[Optional[dict[str, object]], tuple[str, ...]]:
    """Validate and consume the only formal property-forest SAFE boundary.

    The validator reads ``result.status`` itself, re-derives all live digests,
    independently checks node conservation, verifies the exact invocation
    context, and consumes the verifier-issued opaque capability.  A copied
    JSON seal, a caller-supplied expected status, or a self-hash is
    insufficient.
    """

    errors: list[str] = []
    status = getattr(result, "status", None)
    status_value = getattr(status, "value", status)
    if status_value != "certified":
        errors.append("result_not_certified")
    if getattr(result, "counterexample", None) is not None:
        errors.append("certified_result_has_counterexample")
    metadata = getattr(result, "metadata", None)
    if type(metadata) is not dict:
        return None, tuple(sorted(set(errors + ["result_metadata_invalid"])))
    meta = cast(dict[str, object], metadata)
    if meta.get("property_separable_bab") is not True:
        errors.append("property_forest_not_enabled")
    live_facts = meta.get("property_forest_live_facts")
    live_seal = meta.get("property_forest_live_seal")
    capability = meta.pop("_property_forest_live_capability", None)
    if type(live_facts) is not dict:
        errors.append("missing_live_facts")
        facts: dict[str, object] = {}
    else:
        facts = cast(dict[str, object], live_facts)

    factory_binding = (
        f"{getattr(solver_factory, '__module__', type(solver_factory).__module__)}."
        f"{getattr(solver_factory, '__qualname__', type(solver_factory).__qualname__)}"
    )
    normalized_max_batch = (
        max_batch_size
        if isinstance(max_batch_size, (int, str))
        else None
    )
    if facts.get("solver_factory") != factory_binding:
        errors.append("solver_factory_mismatch")
    if facts.get("requested_max_batch_size") != normalized_max_batch:
        errors.append("max_batch_size_mismatch")
    if (
        type(time_budget_s) not in {int, float}
        or not math.isfinite(float(time_budget_s))
        or float(time_budget_s) <= 0.0
        or facts.get("time_budget_seconds") != float(time_budget_s)
    ):
        errors.append("time_budget_mismatch")
    normalized_dtype = (
        expected_dtype
        if expected_dtype.startswith("torch.")
        else f"torch.{expected_dtype}"
    )
    if facts.get("verification_dtype") != normalized_dtype:
        errors.append("verification_dtype_mismatch")
    if facts.get("verification_device") != expected_device:
        errors.append("verification_device_mismatch")

    mode = facts.get("mode")
    if mode == "root_presolve":
        node_receipt = meta.get(
            "property_forest_root_presolve_receipt"
        )
        if (
            meta.get("resolved_by") != "root_presolve"
            or meta.get("pool_remaining") != 0
            or meta.get("spec_rows_kept") != 0
            or meta.get("property_forest_root_rows") != []
            or meta.get("property_forest_root_certified_rows")
            != facts.get("root_certified_rows")
        ):
            errors.append("root_presolve_metadata_mismatch")
    elif mode == "complete_forest":
        node_receipt = meta.get(
            "property_forest_node_conservation_receipt"
        )
        forest_rows = facts.get("forest_rows")
        processed = facts.get("processed_nodes")
        pool_remaining = facts.get("pool_remaining")
        if (
            type(forest_rows) is not list
            or any(type(value) is not int for value in forest_rows)
            or type(processed) is not int
            or type(pool_remaining) is not int
        ):
            errors.append("forest_expected_facts_malformed")
        else:
            from act.back_end.bab.bab import (
                _validate_property_forest_receipt,
            )

            node_valid, node_errors = (
                _validate_property_forest_receipt(
                    node_receipt,
                    expected_row_ids=tuple(cast(list[int], forest_rows)),
                    expected_processed=processed,
                    expected_pool_remaining=pool_remaining,
                )
            )
            if not node_valid:
                errors.extend(
                    f"node_receipt:{error}" for error in node_errors
                )
        if (
            meta.get("property_forest_node_conservation_valid")
            is not True
            or meta.get("property_forest_node_conservation_errors")
            != []
            or meta.get("property_forest_coverage_complete") is not True
            or meta.get("property_forest_root_rows") != forest_rows
            or meta.get("property_forest_root_certified_rows")
            != facts.get("root_certified_rows")
            or meta.get("nodes") != processed
            or meta.get("pool_remaining") != 0
            or meta.get("any_dropped_frontier_cap") is not False
            or meta.get("any_dropped_max_depth") is not False
        ):
            errors.append("complete_forest_metadata_mismatch")
    else:
        node_receipt = {}
        errors.append("unknown_live_proof_mode")

    safe_receipt, live_errors = (
        build_property_forest_safe_proof_receipt(
            result=result,
            live_seal=live_seal,
            capability=capability,
            run_token=run_token,
            net=net,
            config=config,
            node_receipt=node_receipt,
            live_facts=facts,
            source_paths=source_paths,
            source_digests_before_run=source_digests_before_run,
        )
    )
    errors.extend(live_errors)
    if errors or safe_receipt is None:
        return None, tuple(sorted(set(errors)))
    safe_receipt["result_status"] = "certified"
    safe_receipt["invocation"] = {
        "solver_factory": factory_binding,
        "max_batch_size": normalized_max_batch,
        "time_budget_seconds": float(time_budget_s),
        "dtype": normalized_dtype,
        "device": expected_device,
    }
    safe_receipt["receipt_sha256"] = canonical_receipt_sha256(
        {
            key: value
            for key, value in safe_receipt.items()
            if key != "receipt_sha256"
        }
    )
    return safe_receipt, ()
