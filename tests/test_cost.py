"""Tests for harness/cost.py and codex validation."""

import json
from pathlib import Path

import pytest

from harness import cli, cost


def test_compute_cost_and_unpriced_errors():
    w = {"r_grad": 2.5}
    assert cost.compute_cost(10, 4, w) == pytest.approx(20.0)
    with pytest.raises(ValueError, match="missing r_m"):
        cost.compute_cost(10, 4, None)
    with pytest.raises(ValueError, match="r_hvp"):
        cost.compute_cost(10, 4, w, hvp=1)
    with pytest.raises(ValueError, match="r_hess"):
        cost.compute_cost(10, 4, w, hess=1)
    w2 = {"r_grad": 2.5, "r_hvp": 5.0, "r_hess": 15.0}
    assert cost.compute_cost(10, 4, w2, hvp=2, hess=1) == pytest.approx(45.0)


def test_weights_for_rejects_missing_or_invalid(tmp_path):
    p = tmp_path / "costs.json"
    p.write_text(json.dumps({"good": {"r_grad": 1.8}, "bad": {"r_grad": 0.5}}))
    assert cost.weights_for("good", path=p) == {"r_grad": 1.8}
    with pytest.raises(KeyError, match="missing cost weight"):
        cost.weights_for("absent", path=p)
    with pytest.raises(ValueError, match="invalid r_grad"):
        cost.weights_for("bad", path=p)


def test_codex_validate_cli():
    assert cli.main(["codex", "validate"]) == 0
