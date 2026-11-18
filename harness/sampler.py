"""The adapter interface. A sampler is one file in harness/adapters/.

An adapter gets a RunRequest (model, data, seed, chains, budget, output
directory) and must write one CSV per chain to `out_dir/chain_<c>.csv`, with a
header row of variable names and one row per post-warmup draw. Columns whose
names end in `__` are sampler diagnostics and are not scored. It returns a
Timing record. Nothing else in the harness knows which sampler ran.

To add a sampler, copy adapters/nutpie.py, change `name`, and implement `run`.
"""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .models import Model


@dataclass
class Settings:
    """Everything an adapter may need from the command line.

    cmdstan: CmdStan tree (falls back to $CMDSTAN).
    build_dir: where compiled models are cached.
    options: free-form `--opt key=value` pairs; each adapter documents its own.
    """

    cmdstan: str | None = None
    build_dir: Path = Path("build")
    options: dict[str, str] = field(default_factory=dict)


@dataclass
class RunRequest:
    model: Model
    data_file: Path
    seed: int
    chains: int
    warmup: int
    draws: int
    out_dir: Path


@dataclass
class Timing:
    """What one run cost.

    wall_s: wall clock for the whole run, chains in parallel, from start of
        sampling to the last chain finishing (compile time excluded).
    chain_s: each chain's own time as the sampler reports it, warmup plus
        sampling. Their sum is the second wall convention (see PROTOCOL.md).
    grads_sampling: gradient evaluations in the sampling phase, all chains.
    grads_warmup: the same for warmup, or None when the sampler does not report it.
    evaluations: every density and gradient evaluation, warmup included, when the
        harness counts them itself (the logdensity adapter); else None, and the
        scorer uses grads_sampling + grads_warmup.
    """

    wall_s: float
    chain_s: list[float]
    warmup_s: list[float | None]
    sampling_s: list[float | None]
    grads_sampling: int | None
    grads_warmup: int | None
    divergences: int | None
    extra: dict = field(default_factory=dict)
    evaluations: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class Sampler:
    name: str = ""

    def __init__(self, settings: Settings):
        self.settings = settings

    def check(self) -> None:
        """Raise with a plain message if the sampler cannot run here."""

    def run(self, req: RunRequest) -> Timing:
        raise NotImplementedError

    def describe(self) -> dict:
        """Build identity recorded next to each run (paths, versions, flags)."""
        return {}

    def opt(self, key: str, default: str | None = None) -> str | None:
        return self.settings.options.get(key, default)


def registry() -> dict[str, type[Sampler]]:
    from . import adapters

    found = {}
    for info in pkgutil.iter_modules(adapters.__path__):
        mod = importlib.import_module(f"{adapters.__name__}.{info.name}")
        cls = getattr(mod, "Adapter", None)
        if cls is not None and issubclass(cls, Sampler):
            found[cls.name] = cls
    return found


def get(name: str, settings: Settings) -> Sampler:
    reg = registry()
    if name not in reg:
        raise KeyError(f"unknown sampler {name!r}; known: {', '.join(sorted(reg))}")
    return reg[name](settings)
