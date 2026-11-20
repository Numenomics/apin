"""Tests for the Slice 2 adapt/transition kernel ABI, canonical tuning, aux_spec, per-step
provenance, frozen-kernel replay, capability gating, and subprocess isolation."""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from harness import cost, grid, logdensity, models, sampler, score
from harness.logdensity import (
    Budget,
    BudgetExceeded,
    CapabilityError,
    Chain,
    ContractError,
    Counter,
    canonical_digest,
    check_imports,
    deserialize_tree,
    execute_run,
    serialize_tree,
)

EXAMPLES = Path(__file__).parent.parent / "examples" / "samplers"


class FakeModel:
    """Standard normal in 3D that rejects x[0] > 5, supporting grad/hvp/hess."""

    def param_unc_num(self):
        return 3

    def log_density(self, x, propto, jacobian):
        if x[0] > 5:
            raise RuntimeError("normal_lpdf: domain error")
        return -0.5 * float(x @ x)

    def log_density_gradient(self, x, propto, jacobian):
        return self.log_density(x, propto, jacobian), -np.asarray(x, dtype=float)

    def log_density_hessian_vector_product(self, x, v, propto, jacobian):
        return self.log_density(x, propto, jacobian), -np.asarray(v, dtype=float)

    def log_density_hessian(self, x, propto, jacobian):
        lp, g = self.log_density_gradient(x, propto, jacobian)
        return lp, g, -np.eye(3, dtype=float)


def _stay_adapt(oracle, budget, rng):
    x0 = rng.uniform(-1.0, 1.0, size=oracle.dim)
    oracle.log_density(x0)
    return {"step": 0.1, "scale": np.ones(oracle.dim)}, (x0, None), [1.0], "none"


def _stay_transition(tuning, chain, oracle, rng):
    return chain.position, None


def test_counter_counts_and_enforces_budget():
    c = Counter(FakeModel(), max_evaluations=5, max_seconds=60)
    for _ in range(3):
        c.log_density(np.zeros(3))
    c.log_density_gradient(np.ones(3))
    lp, g = c.log_density_gradient(np.ones(3))
    assert (c.density, c.gradient) == (3, 2)
    assert lp == -1.5 and np.array_equal(g, -np.ones(3))
    with pytest.raises(BudgetExceeded):
        c.log_density(np.zeros(3))
    assert c.exceeded and c.total_calls == 5


def test_oracle_capabilities_block_unpriced_hvp_and_hess():
    c = Counter(FakeModel(), max_evaluations=20, max_seconds=60)
    o = c.oracle()
    assert o.block_shape is None
    with pytest.raises(CapabilityError):
        o.hvp(np.zeros(3), np.ones(3))
    with pytest.raises(CapabilityError):
        o.hess(np.zeros(3))

    c2 = Counter(FakeModel(), max_evaluations=20, max_seconds=60,
                 capabilities={"hvp", "hess", "block_shape"}, block_shape=((3,),))
    o2 = c2.oracle()
    assert o2.block_shape == ((3,),)
    lp, hv = o2.hvp(np.zeros(3), np.ones(3))
    assert lp == 0.0 and np.array_equal(hv, -np.ones(3))
    lp, g, h = o2.hess(np.zeros(3))
    assert lp == 0.0 and np.array_equal(h, -np.eye(3))
    assert (c2.hvp_calls, c2.hess_calls) == (1, 1)


def test_oracle_exposes_no_counter_or_model():
    c = Counter(FakeModel(), max_evaluations=20, max_seconds=60)
    o = c.oracle()
    for name in ("log_density", "log_density_gradient", "hvp", "hess"):
        f = getattr(o, name)
        assert not any(hasattr(f, a) for a in ("__self__", "model", "counter", "budget", "step_seen"))
    with pytest.raises(AttributeError):
        o.dim = 99


def test_rejected_point_is_charged_and_not_recorded():
    c = Counter(FakeModel(), max_evaluations=10, max_seconds=60)
    bad = np.array([6.0, 0.0, 0.0])
    assert c.log_density(bad) == -math.inf
    lp, g = c.log_density_gradient(bad)
    assert lp == -math.inf and np.all(np.isnan(g))
    assert c.density + c.gradient == 2 and len(c.step_seen) == 0


def test_canonical_serialization_and_digest_sensitivity(tmp_path):
    tree = {"a": 1, "b": True, "c": [1.5, None, ("x", np.array([1.0, 2.0]))]}
    sha1, nbytes = serialize_tree(tree, tmp_path / "t")
    loaded = deserialize_tree(tmp_path / "t")
    assert canonical_digest(loaded) == sha1 and nbytes > 0

    # 1 ULP float mutation changes digest
    mutated = {"a": 1, "b": True, "c": [1.5, None, ("x", np.array([np.nextafter(1.0, 2.0), 2.0]))]}
    assert canonical_digest(mutated) != sha1

    # Non-canonical object rejected
    with pytest.raises(ContractError):
        canonical_digest({"fn": lambda x: x})


def test_import_lint_and_examples():
    for src in ("import os", "from subprocess import run", "from . import helper", "eval('1')", "open('x')"):
        with pytest.raises(ImportError):
            check_imports(src)
    for f in EXAMPLES.glob("*.py"):
        check_imports(f.read_text())


def test_load_submission_requires_adapt_and_transition(tmp_path):
    f = tmp_path / "bad.py"
    f.write_text("def adapt(oracle, budget, rng): pass\n")
    with pytest.raises(AttributeError, match="adapt.*transition"):
        logdensity.load_submission(f)


def test_execute_run_accepts_valid_kernel_and_records_split_counts(tmp_path):
    draws, summary = execute_run(
        FakeModel(), (_stay_adapt, _stay_transition), Budget(1, 30, 500, 60.0), run_seed=42, work_dir=tmp_path
    )
    assert draws.shape == (1, 30, 3)
    assert summary["adapt"]["density_evals"] == 1
    assert summary["sample"]["density_evals"] == 0
    assert summary["replay"]["replayed_singles"] == logdensity.REPLAY_STEPS
    assert summary["replay"]["replayed_window"] == logdensity.REPLAY_WINDOW


def test_step_provenance_rejects_unevaluated_new_position(tmp_path):
    def fake_move(tuning, chain, oracle, rng):
        return chain.position + 0.1, None  # never evaluated!

    with pytest.raises(ContractError, match="evaluated with a finite log density"):
        execute_run(FakeModel(), (_stay_adapt, fake_move), Budget(1, 5, 100, 60.0), run_seed=1, work_dir=tmp_path)


def test_frozen_tuning_mutation_is_caught(tmp_path):
    def mutate(tuning, chain, oracle, rng):
        tuning["scale"][0] += 1e-12
        return chain.position, None

    with pytest.raises(ContractError, match="mutated"):
        execute_run(FakeModel(), (_stay_adapt, mutate), Budget(1, 10, 100, 60.0), run_seed=1, work_dir=tmp_path)


def test_hidden_global_state_in_transition_fails_replay(tmp_path):
    sub = tmp_path / "stateful.py"
    sub.write_text(
        "import numpy as np\n"
        "_COUNT = 0\n"
        "def adapt(oracle, budget, rng):\n"
        "    x = np.zeros(oracle.dim)\n"
        "    oracle.log_density(x)\n"
        "    return {'s': 0.1}, (x, None), [1.0], 'none'\n"
        "def transition(tuning, chain, oracle, rng):\n"
        "    global _COUNT\n"
        "    _COUNT += 1\n"
        "    x = chain.position + 1e-3 * _COUNT\n"
        "    oracle.log_density(x)\n"
        "    return x, None\n"
    )
    with pytest.raises(ContractError, match="replay"):
        execute_run(FakeModel(), sub, Budget(1, 10, 500, 60.0), run_seed=7, work_dir=tmp_path)


def test_clean_post_adapt_subprocess_blocks_adapt_global_smuggling(tmp_path):
    """A submission that stashes state in a module global during adapt() sees a fresh empty
    global when transition() runs in its own clean child process."""
    sub = tmp_path / "smuggle.py"
    sub.write_text(
        "import numpy as np\n"
        "_SECRET = None\n"
        "def adapt(oracle, budget, rng):\n"
        "    global _SECRET\n"
        "    _SECRET = 12345\n"
        "    x = np.zeros(oracle.dim)\n"
        "    oracle.log_density(x)\n"
        "    return {'ok': True}, (x, None), [1.0], 'none'\n"
        "def transition(tuning, chain, oracle, rng):\n"
        "    if _SECRET is None:\n"
        "        raise RuntimeError('global did not survive post-adapt process isolation')\n"
        "    return chain.position, None\n"
    )
    bdir = tmp_path / "bundle"
    bdir.mkdir()
    # Run adapt in process 1, sample in process 2 via python -c using FakeModel
    code_adapt = (
        f"from pathlib import Path; from harness.logdensity import Budget, run_adapt_phase; "
        f"from tests.test_logdensity import FakeModel; "
        f"run_adapt_phase(FakeModel(), Path({str(sub)!r}), Budget(1, 5, 100, 30.0), 1, Path({str(bdir)!r}))"
    )
    code_sample = (
        f"import json; from pathlib import Path; from harness.logdensity import Budget, run_sample_phase; "
        f"from tests.test_logdensity import FakeModel; "
        f"bdir = Path({str(bdir)!r}); meta = json.loads((bdir / 'adapt_meta.json').read_text()); "
        f"run_sample_phase(FakeModel(), Path({str(sub)!r}), Budget(1, 5, 100, 30.0), 1, bdir, meta)"
    )
    root = Path(__file__).parent.parent
    r1 = subprocess.run([sys.executable, "-c", code_adapt], cwd=root, capture_output=True, text=True)
    assert r1.returncode == 0, r1.stderr
    r2 = subprocess.run([sys.executable, "-c", code_sample], cwd=root, capture_output=True, text=True)
    assert r2.returncode != 0 and "did not survive post-adapt process isolation" in r2.stderr


def test_lifted_aux_and_multiwalker_tempered_kernel(tmp_path):
    """Typed aux (NRPT parity bit uniform_discrete(2)) + K=2 tempered walkers record cleanly."""

    def nrpt_adapt(oracle, budget, rng):
        pos = rng.uniform(-1.0, 1.0, size=(2, oracle.dim))
        for w in range(2):
            oracle.log_density(pos[w])
        return {"step": 0.2}, Chain(pos, 0), [0.5, 1.0], {"kind": "uniform_discrete", "k": 2}

    def nrpt_transition(tuning, chain, oracle, rng):
        pos = chain.position.copy()
        parity = int(chain.aux)
        for w in range(2):
            cand = pos[w] + tuning["step"] * rng.standard_normal(oracle.dim)
            if math.isfinite(oracle.log_density(cand)):
                pos[w] = cand
        return Chain(pos, 1 - parity)

    draws, summary = execute_run(
        FakeModel(), (nrpt_adapt, nrpt_transition), Budget(1, 25, 1000, 60.0), run_seed=9, work_dir=tmp_path
    )
    assert draws.shape == (2, 25, 3)
    assert summary["walkers"] == 2 and summary["betas"] == [0.5, 1.0]


def test_invalid_betas_and_aux_rejected(tmp_path):
    def no_cold_beta(oracle, budget, rng):
        x = np.zeros(oracle.dim)
        oracle.log_density(x)
        return {}, (x, None), [0.5], "none"

    with pytest.raises(ContractError, match="beta must equal 1.0"):
        execute_run(FakeModel(), (no_cold_beta, _stay_transition), Budget(1, 5, 50, 60.0), 1, tmp_path)

    def bad_unit_sphere(oracle, budget, rng):
        x = np.zeros(oracle.dim)
        oracle.log_density(x)
        return {}, (x, np.array([2.0, 0.0, 0.0])), [1.0], {"kind": "unit_sphere", "dim": 3}

    with pytest.raises(ContractError, match="unit norm"):
        execute_run(FakeModel(), (bad_unit_sphere, _stay_transition), Budget(1, 5, 50, 60.0), 1, tmp_path)


def test_example_hmc_and_ula_on_fake_model(tmp_path):
    """Ported example HMC recovers N(0, I) cleanly and passes bit-exact replay; ULA also runs cleanly."""
    hmc_draws, hmc_sum = execute_run(
        FakeModel(), EXAMPLES / "hmc.py", Budget(1, 400, 10**6, 60.0), run_seed=1, work_dir=tmp_path / "hmc"
    )
    assert hmc_draws.shape == (1, 400, 3)
    flat = hmc_draws[0]
    assert np.all(np.abs(flat.mean(axis=0)) < 0.25) and np.all(np.abs(flat.var(axis=0) - 1.0) < 0.35)
    assert hmc_sum["adapt"]["gradient_evals"] > 0 and hmc_sum["sample"]["gradient_evals"] > 0

    ula_draws, ula_sum = execute_run(
        FakeModel(), EXAMPLES / "ula_wrong.py", Budget(1, 200, 10**6, 60.0), run_seed=2, work_dir=tmp_path / "ula"
    )
    assert ula_draws.shape == (1, 200, 3)
    assert ula_sum["sample"]["density_evals"] > 0 and ula_sum["sample"]["gradient_evals"] > 0


def test_budget_overrun_caught_even_if_submission_swallows_it(tmp_path):
    def swallow(oracle, budget, rng):
        x = np.zeros(oracle.dim)
        try:
            for _ in range(budget.max_evaluations + 2):
                oracle.log_density(x)
        except BudgetExceeded:
            pass
        return {}, (x, None), [1.0], "none"

    with pytest.raises(BudgetExceeded, match="caught"):
        execute_run(FakeModel(), (swallow, _stay_transition), Budget(1, 2, 3, 60.0), 1, tmp_path)


@pytest.mark.slow
def test_examples_end_to_end(tmp_path):
    """End-to-end BridgeStan test with clean multi-process isolation, replay, and r_m scoring."""
    names = "eight_schools_noncentered,blr"
    ms = models.select(names)
    for label, file in (("hmc", "hmc.py"), ("ula", "ula_wrong.py")):
        s = sampler.get("logdensity", sampler.Settings(
            build_dir=Path(__file__).parent.parent / "build",
            options={"sampler": str(EXAMPLES / file)},
        ))
        done = grid.run_grid(s, ms, reps=1, runs_dir=tmp_path / "runs", label=label, log=lambda *_: None)
        assert len(done) == 2
    g = score.declared(names, reps=1, build_dir=Path(__file__).parent.parent / "build")
    out = score.score(tmp_path / "runs", tmp_path / "scores", g, log=lambda *_: None)
    rows = {(r["label"], r["model"]): r for r in out["per_run"]}
    for m in ("eight_schools_noncentered", "blr"):
        hmc, ula = rows["hmc", m], rows["ula", m]
        assert hmc["status"] != "wrong", hmc["reason"]
        assert hmc["evaluations"] == hmc["density_evals"] + hmc["gradient_evals"] > 0
        assert math.isfinite(hmc["cost"]) and hmc["cost"] >= hmc["evaluations"]
        assert ula["status"] != "right", ula["reason"]
        assert math.isnan(ula["ess_per_cost"])
