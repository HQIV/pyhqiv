import json
from pathlib import Path

import numpy as np
import pytest

from pyhqiv.so8_generators import (
    SO8_JSON_KEY_CHECKSUM,
    SO8_JSON_KEY_TENSOR,
    dump_so8_generators_json,
    load_so8_generators,
    load_so8_generators_auto,
    load_so8_generators_from_json,
    so8_tensor_sha256_hex,
    upper_triangle_coord,
)


def test_so8_generators_are_antisymmetric() -> None:
    gens = load_so8_generators().tensor
    for k in range(gens.shape[0]):
        A = gens[k]
        assert np.allclose(A + A.T, 0.0, atol=1e-12)


def test_so8_generators_are_linearly_independent_in_upper_triangle_coords() -> None:
    gens = load_so8_generators().tensor
    coord_mat = np.stack([upper_triangle_coord(gens[k]) for k in range(28)], axis=1)  # (28,28)
    rank = np.linalg.matrix_rank(coord_mat, tol=1e-10)
    assert rank == 28


def test_so8_generators_json_matches_lean(tmp_path: Path) -> None:
    lean = load_so8_generators()
    out = tmp_path / "g.json"
    dump_so8_generators_json(out, lean)
    from_json = load_so8_generators_from_json(out)
    assert np.allclose(from_json.tensor, lean.tensor, atol=0.0, rtol=0.0)


def test_packaged_so8_json_matches_lean() -> None:
    pkg_json = Path(__file__).resolve().parents[1] / "src" / "pyhqiv" / "so8_generators.json"
    if not pkg_json.is_file():
        return
    lean = load_so8_generators()
    js = load_so8_generators_from_json(pkg_json)
    assert np.allclose(js.tensor, lean.tensor, atol=0.0, rtol=0.0)


def test_load_so8_generators_auto_uses_packaged_json_when_present() -> None:
    pkg_json = Path(__file__).resolve().parents[1] / "src" / "pyhqiv" / "so8_generators.json"
    if not pkg_json.is_file():
        return
    auto = load_so8_generators_auto()
    lean = load_so8_generators()
    assert np.allclose(auto.tensor, lean.tensor, atol=0.0, rtol=0.0)


def test_so8_json_payload_shape() -> None:
    pkg_json = Path(__file__).resolve().parents[1] / "src" / "pyhqiv" / "so8_generators.json"
    if not pkg_json.is_file():
        return
    data = json.loads(pkg_json.read_text(encoding="utf-8"))
    arr = np.asarray(data[SO8_JSON_KEY_TENSOR], dtype=np.float64)
    assert arr.shape == (28, 8, 8)


def test_so8_json_checksum_mismatch_raises(tmp_path: Path) -> None:
    lean = load_so8_generators()
    out = tmp_path / "bad.json"
    dump_so8_generators_json(out, lean)
    data = json.loads(out.read_text(encoding="utf-8"))
    data[SO8_JSON_KEY_CHECKSUM] = "0" * 64
    bad = tmp_path / "bad2.json"
    bad.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        load_so8_generators_from_json(bad, verify_checksum=True)


def test_so8_json_checksum_can_be_skipped(tmp_path: Path) -> None:
    lean = load_so8_generators()
    out = tmp_path / "bad.json"
    dump_so8_generators_json(out, lean)
    data = json.loads(out.read_text(encoding="utf-8"))
    data[SO8_JSON_KEY_CHECKSUM] = "0" * 64
    bad = tmp_path / "bad2.json"
    bad.write_text(json.dumps(data), encoding="utf-8")
    g = load_so8_generators_from_json(bad, verify_checksum=False)
    assert np.allclose(g.tensor, lean.tensor)


def test_so8_json_missing_tensor_key(tmp_path: Path) -> None:
    p = tmp_path / "empty.json"
    p.write_text(json.dumps({"foo": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        load_so8_generators_from_json(p)


def test_so8_json_wrong_shape(tmp_path: Path) -> None:
    p = tmp_path / "shape.json"
    p.write_text(
        json.dumps({SO8_JSON_KEY_TENSOR: np.zeros((2, 2)).tolist(), SO8_JSON_KEY_CHECKSUM: ""}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="shape"):
        load_so8_generators_from_json(p, verify_checksum=False)


def test_so8_tensor_sha256_stable_on_lean(tmp_path: Path) -> None:
    g = load_so8_generators()
    h1 = so8_tensor_sha256_hex(g.tensor)
    out = tmp_path / "g.json"
    dump_so8_generators_json(out, g)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data[SO8_JSON_KEY_CHECKSUM] == h1


def test_load_so8_generators_auto_explicit_json_path(tmp_path: Path) -> None:
    lean = load_so8_generators()
    out = tmp_path / "g.json"
    dump_so8_generators_json(out, lean)
    auto = load_so8_generators_auto(json_path=out)
    assert np.allclose(auto.tensor, lean.tensor)

