# Pair-Hamming Structured Max-Margin Implementation Plan

> **Execution policy:** Implement by functional batch under the global lightweight development and verification strategy. Do not apply task-by-task review or full-validation gates unless the batch risk warrants them.

**Goal:** Add exact device-resident pair-Hamming cost-augmented structured max-margin to frozen GEC four-arm training and verify only a two-step-per-arm CUDA smoke.

**Architecture:** A standalone autograd Function evaluates max-product with torch tensors on CPU/CUDA, stores O(L^2) decisions and traceback applies the Danskin gradient. A derived config selects margin instead of NLL while retaining GEC data, context, optimizer, resume and unfreeze machinery; metadata separates checkpoint metrics.

**Tech Stack:** Python, PyTorch, NumPy brute-force test oracle, pytest, existing CUDA/Triton inside-outside.

---

### Batch 1: Contract and exact objective

**Files:**
- Create: `docs/plans/2026-11-26-structured-margin-candidate-design.md`
- Create: `src/rnajepa/structured_margin.py`
- Test: `tests/test_structured_margin.py`

**Risk:** High — combinatorial or subgradient errors invalidate the candidate.

**Implementation:** Freeze formula/legal-set/tie/normalization; implement O(L^3) time, O(L^2) table max-product, cost augmentation, target validation, float64 internals and device traceback.

**Minimum verification:** Independent exhaustive L=4..6 and L<=12 fixtures for values/unique-maximizer gradients; mask, ties, extremes, AMP and CPU/CUDA.

**Independent review:** Yes — numerical-core and final three-round audits.

### Batch 2: GEC integration

**Files:**
- Create: `configs/gec_structured_margin_convergence_v1.json`
- Modify: `tools/train_plan_a_gec_convergence_v1.py`
- Test: `tests/test_structured_margin_training.py`

**Risk:** High — NLL drift or checkpoint reuse would break comparisons.

**Implementation:** Select objective only from the new config; preserve old config and harness. Bind run/checkpoint/smoke identity to both contracts; use post-terminal validation margin and explicit objective/kernel metadata.

**Minimum verification:** Config/seed/decoder guards, margin convergence, identity tests and existing GEC/NLL suites.

**Independent review:** Yes — inspect every trainer hunk.

### Batch 3: Bounded smoke and delivery

**Files:**
- Create: `tools/run_structured_margin_smoke_v1.sh`
- Create: `tools/verify_structured_margin_smoke_v1.py`
- Create: `tests/test_structured_margin_smoke.py`

**Risk:** Medium — stale artifacts or unbounded execution could be misread as scientific evidence.

**Implementation:** Reuse shared residual init and interruption/resume in a new root. Verify two steps per arm, finite gradients, matched batches, changed residuals, fixed decoder and no CPU fallback. Do not create a full-training launcher.

**Minimum verification:** Verifier tests, actual two-step CUDA smoke, three rounds of tests/diff/artifact audit, commit and push.

**Independent review:** Yes — final code/test/artifact/resource audit.

**Execution grouping:** One implementation agent for coupled batches, with numerical-core and final delivery audits. No seeds 1/2, full training or decoder search.
