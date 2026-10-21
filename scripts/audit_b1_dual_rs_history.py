"""Reconstruct saved Dual RS test identities without numpy, torch or a model.

This is a fixed saved-source audit, not an executable selection protocol or a
proof of training correctness. stdout is a deterministic, compact receipt.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path
import struct


ROOT = Path(__file__).resolve().parents[1]
TABLE = Path(
    "/data1/Kane/MOE/baselines/recent_moe_20260921/dual_rs/data/sigma_label/"
    "map10e4/0.250_0.500_1.000_test.npy"
)
SOURCES = {
    str(TABLE): "4559674ff67ea69526e32d804d46e67738a1978e4147506e6248d149f1619a88",
    "configs/recent_moe/dual_rs_epoch_control_r1.json":
        "16c00252fdf5e892a8c45cf9851995349cd2f49d41a9224a66349f713bf6518f",
    "configs/recent_moe/dual_rs_training_r1.json":
        "de9c3a9b7ff1346715e999e00821786e9c3dceb65aadea96ac0c900952bdf066",
    "docs/dual_rs_epoch_control_archive_20260921_r1.json":
        "36f93bde89c07b4623f5c2c50be0612df946679db3ae0f724d186b0556fed982",
    "docs/dual_rs_training_landing_archive_20260922.json":
        "e39d412b09576219c154335e6f7c4bfb370ca951b0138573fb080f475594a364",
    "scripts/dual_rs_epoch_pipeline.py":
        "04323428759d59c1285ff70fd74c12cae8fe25c3d540b3de3600af74e6ddccd8",
    "scripts/dual_rs_training_state.py":
        "36ddce918c956c1277c26e980a6e9165c17142f9f6cbb8f3b0cfb16bca718c78",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def legacy_integer_list_digest(indices: list[int]) -> str:
    """Only the list/int branch of the hash-bound legacy digest, no import."""
    digest = hashlib.sha256()
    digest.update(b"list")
    digest.update(str(len(indices)).encode())
    for index in indices:
        require(type(index) is int, "non-integer identity")
        digest.update(("int:" + repr(index) + ";").encode())
    return digest.hexdigest()


def eligible_indices(raw: bytes) -> list[int]:
    # Only the one frozen NPY layout is accepted; this is not a general loader.
    require(raw[:8] == b"\x93NUMPY\x01\x00", "NPY magic/version")
    header_length = struct.unpack_from("<H", raw, 8)[0]
    offset = 10 + header_length
    require(offset <= 4096, "NPY header capacity")
    header = ast.literal_eval(raw[10:offset].decode("ascii"))
    require(header == {"descr": "<f8", "fortran_order": True,
                       "shape": (10000, 5)}, "NPY frozen shape/order/dtype")
    require(len(raw) == offset + 10000 * 5 * 8, "NPY body size")
    # Column -2 is column 3. Fortran storage makes this a contiguous column.
    values = struct.unpack_from("<10000d", raw, offset + 3 * 10000 * 8)
    require(all(math.isfinite(value) for value in values), "nonfinite radius")
    return [index for index, value in enumerate(values) if value != 0.0]


def audit() -> dict:
    raw_sources = {}
    bindings = {}
    total = 0
    for name, expected in SOURCES.items():
        path = Path(name) if Path(name).is_absolute() else ROOT / name
        size = path.stat().st_size
        require(size <= 1024 * 1024, "saved-source size cap")
        total += size
        require(total <= 2 * 1024 * 1024, "saved-source cumulative cap")
        with path.open("rb") as stream:
            raw = stream.read(1024 * 1024 + 1)
        require(len(raw) == size, "saved source changed while reading")
        actual = hashlib.sha256(raw).hexdigest()
        require(actual == expected, "saved-source identity: " + name)
        raw_sources[name] = raw
        bindings[name] = {"sha256": actual, "bytes": len(raw)}

    epoch_config = json.loads(raw_sources["configs/recent_moe/dual_rs_epoch_control_r1.json"])
    train_config = json.loads(raw_sources["configs/recent_moe/dual_rs_training_r1.json"])
    epoch_archive = json.loads(raw_sources["docs/dual_rs_epoch_control_archive_20260921_r1.json"])
    train_archive = json.loads(raw_sources["docs/dual_rs_training_landing_archive_20260922.json"])
    all_indices = eligible_indices(raw_sources[str(TABLE)])
    prefix = all_indices[:256]
    require(len(all_indices) == 8719 and len(prefix) == 256, "cohort counts")
    require(epoch_config["prefix_count"] == 256 and epoch_config["test_count"] == 256,
            "control prefix contract")
    require(train_config["prefix_count"] is None and train_config["test_count"] == 8719,
            "training test cohort contract")
    for cfg, indices in ((epoch_config, prefix), (train_config, all_indices)):
        require(cfg["test_label_table"] == str(TABLE), "label table binding")
        require(legacy_integer_list_digest(indices) == cfg["test_indices_sha256"],
                "legacy test identity digest")
    require(epoch_archive["config_sha256"] == SOURCES["configs/recent_moe/dual_rs_epoch_control_r1.json"],
            "control config/archive identity")
    require(train_archive["config_sha256"] == SOURCES["configs/recent_moe/dual_rs_training_r1.json"],
            "training config/archive identity")
    for arm, epochs in (("reference", [1, 2]), ("resume", [2])):
        rows = epoch_archive["epoch_records"][arm]
        require([row["completed_epoch"] for row in rows] == epochs, "control epochs")
        require(all(row["binding"]["test_indices_sha256"] == epoch_config["test_indices_sha256"]
                    for row in rows), "control saved test binding")
    worker = train_archive["worker_terminal"]
    final = train_archive["final_epoch"]
    require(worker["completed_epochs"] == final["completed_epoch"] == train_config["epochs"] == 90,
            "registered final epoch")
    require(worker["selected_checkpoint"] == "epoch090.pt", "final checkpoint selection")
    require(worker["binding"]["test_indices_sha256"] == final["binding"]["test_indices_sha256"]
            == train_config["test_indices_sha256"], "training saved test binding")
    require(train_config["checkpoint_selection"] ==
            "epoch90 only, regardless of accuracy or future certification", "fixed selection rule")
    return {
        "schema": "B1_DUAL_RS_HISTORY_RECONSTRUCTION_V1",
        "status": "SAVED_IDENTITY_AUDIT_PASS_NOT_SELECTION_AUTHORITY",
        "dataset": "CIFAR10_TEST", "real_model_calls": 0,
        "native_solver_calls": 0, "bytes_read": total,
        "sources": bindings,
        "control_exclusion": {
            "count": len(prefix), "indices": prefix,
            "legacy_digest": legacy_integer_list_digest(prefix),
            "meaning": "explicit completed-epoch/save-restore test control cohort",
        },
        "fixed_recipe_aggregate_test_use": {
            "count": len(all_indices), "legacy_digest": legacy_integer_list_digest(all_indices),
            "checkpoint_epoch": 90,
            "meaning": "prior test evaluation disclosed, not 8719 ACT verification endpoints",
        },
        "limits": [
            "Does not independently reprove training or author certification.",
            "Hash binding does not replace the documented dataset-mapping code review.",
            "Not clean-correct predictions for any of the three ACT models.",
            "Does not close other historical exclusions or authorize a new model call.",
        ],
    }


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, sort_keys=True))
