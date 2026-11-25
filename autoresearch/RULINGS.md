# apin Competition & Autoresearch Rulings (`RULINGS.md`)

Versioned rulebook modeled on `openai/parameter-golf` (issue #402) and `KellerJordan/modded-nanogpt`. Every discovered loophole or borderline optimization gets a numbered ruling stating whether it is **Legal**, **Track-Restricted**, or **Banned**, and how the harness enforces it.

---

### R-001 · Markov Chain Kernel Scope (Legal vs. Banned)
- **Status:** **Binding (all tracks)**
- **Ruling:** Entries must be Markov chain samplers implementing `adapt(oracle, budget, rng)` and `transition(tuning, chain, oracle, rng)`. Non-Markov methods (VI, Pathfinder, SMC, normalizing flows, polynomial transport, nested sampling, importance sampling) are **Legal** inside `adapt()` or as a proposal inside an exact Metropolis-Hastings Markov kernel, and **Banned** as standalone draw generators.

### R-002 · Undeclared State, Globals, and Cross-Chain Smuggling
- **Status:** **Banned**
- **Ruling:** All evolving Markov state MUST live inside `chain = (position, aux)` with stationary law declared in `aux_spec`. Caching state across `transition()` calls via module globals, closures, class attributes, filesystem writes, C-extension static buffers, or ambient RNG (`random`, `np.random` global state) is **Banned**.
- **Enforcement:** `transition()` executes in a fresh post-adapt child process (`OMP_NUM_THREADS=1`); SHA-256 hash of `tuning` checked pre/post; 20 random steps and short multi-step windows replayed bit-for-bit in fresh processes.

### R-003 · Target Identification & Oracle Probing
- **Status:** **Banned**
- **Ruling:** Inspecting stack frames, filesystem paths, BridgeStan library names, dimension fingerprints, or target constants to branch to model-specific hyperparameters is **Banned**.
- **Enforcement:** Input-blind `oracle` hides names, paths, seeds, and phase; analytic shapes vary dimension, shift, sign, scale, and rotation per trial; external PRs run in a network-disabled sandbox with BridgeStan outside the container.

### R-004 · Anonymous Block-Shape Metadata (Kronecker / Structured Preconditioning)
- **Status:** **Track-Restricted (opt-in capability `block_shape`)**
- **Ruling:** Reading anonymous tensor/matrix shapes of parameter blocks (e.g., `(N, P)` without parameter names) is **Legal** only when declared in `capabilities_requested: ["block_shape"]`. Results are tagged with that capability so general black-box entries and shape-aware entries remain transparent.

### R-005 · Higher-Order Oracles (HVPs and Hessians)
- **Status:** **Track-Restricted (explicitly priced only)**
- **Ruling:** Calling `oracle.hvp` or `oracle.hess` is **Banned** unless the release publishes frozen cost weights `r_hvp` / `r_hess` and the submission declares the capability. Unpriced higher-order calls raise a hard error.

### R-006 · Unadjusted Samplers & Non-Reversible Step Controllers in L1
- **Status:** **Banned in L1/L2/L3/L5; Track-Restricted to L4**
- **Ruling:** Kernels without an exact target-preserving stationary distribution (ULA, unadjusted MCLMC, Foster ULD `ALIGN`/`ShOULD`/`QUICSORT`, or unadjusted adaptive step-size controllers like raw `PIDController` inside `transition()`) are **Banned** from L1/L2/L3/L5 and routed exclusively to **Track L4**.

### R-007 · Stuck / Identity Kernels & MDB Targeting
- **Status:** **Banned**
- **Ruling:** Deliberately slowing mixing to pass exact-draw stationarity, or tuning bias to sit just below the Tier 0/1 Minimum-Detectable-Bias (MDB) threshold, is **Banned**.
- **Enforcement:** Invariance is always paired with dispersed-start mixing / empirical efficiency; Tier 2 runs at $\ge 10\times$ finer MDB on fresh seeds.

### R-008 · Structured Invariance Argument (`invariance.md`)
- **Status:** **Required for L1/L2/L3/L5 promotion**
- **Ruling:** Every adjusted record must provide `invariance.md` in **involutive-MCMC form** (declare `(position, aux)` and `aux_spec`, express each sub-move as MH, involution + Jacobian, or Gibbs, and state composition). Owner audits all non-library custom blocks against submitted code; LLM-only invariance certification is **Banned**.

### R-009 · Record-Claim Statistical Bar & Systems Waiver
- **Status:** **Binding**
- **Ruling:** Claiming a new SOTA record requires beating the prior record by $\ge 5\%$ with **paired multi-seed significance $p < 0.01$** (hierarchical bootstrap against the frozen $\ge 20$-rep baseline) and clearing the Blum–Hardt Ladder threshold on the sealed holdout. **Waiver:** Lane S2 pure systems speedups waive statistical testing *only* when draws are **byte-identical** at fixed seed across all verification models.

### R-010 · PR Queue Cap & License Purity
- **Status:** **Binding**
- **Ruling:** Maximum **1 open record PR per contributor** (and 1 open promotion candidate per autoresearch lane). All submitted code must be permissively licensed (`BSD-3-Clause`, `MIT`, or `Apache-2.0`) and free of copyleft (`GPL`/`AGPL`) or unlicensed code.

### R-011 · Carrying a Gradient Between `transition()` Calls
- **Status:** **Banned**
- **Ruling:** A sampler may not carry the gradient or the log density of its current position from one `transition()` call to the next. `chain` holds only `(position, aux)`, and `aux` must follow the law declared in `aux_spec`, independent of position. A stored gradient is a function of the position, so it cannot live in `aux`, and R-002 bans every other place to keep it. Each `transition()` call therefore pays for the first gradient it uses. Reusing a gradient within one call is **Legal**.
- **Enforcement:** Automatic. The bill counts every density and gradient call, the statistical independence test on `aux` in `invariance.py` catches a gradient stored in `aux`, and the replay check compares both `(position, aux)` digests and per-step oracle call counts `[density, gradient, hvp, hess]` in reverse step order in a fresh child process, catching module-global or closure caches across calls. Each lane's starting code pays the same cost, so the speed-up over it is not affected.

