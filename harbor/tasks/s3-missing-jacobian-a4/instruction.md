# Fix missing log-det-Jacobian & pullback gradient in Lane A4 sinh-arcsinh transport

The Lane A4 nonlinear sinh-arcsinh transport in `/app/autoresearch/lanes/A4/submission.py` omits the exact `log_abs_det_jac` in `forward()` and the `grad_z(log_abs_det_jac)` term in `pullback_grad()`, causing `harness/judge.py`'s finite-difference Jacobian verifier to reject it.

Fix `/app/autoresearch/lanes/A4/submission.py` so `forward()`, `pullback_grad()`, and `inverse()` match the exact diffeomorphism and pass invariance on Student-t targets.

Verify locally with:
```bash
uv run apin autoresearch eval --lane A4 --run-id local --dim 4 --repeats 1 --no-queue --hypothesis 'fix jacobian'
```
Only edit `/app/autoresearch/lanes/A4/submission.py`.
