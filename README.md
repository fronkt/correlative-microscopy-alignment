# correlative-microscopy-alignment

Multi-scale alignment of correlative materials microscopy image pairs using
foundational dense matchers (RoMa, ELoFTR, MatchAnything) wrapped in a
scale-aware pyramidal patching layer.

See `docs/context.md`, `docs/research_plan.md`, `docs/task_plan.md` for the
research framing and engineering decomposition. Live TODO is `tasks/todo.md`.

## Layout
```
src/cma/
  data/         # image-pair types, synthetic generator
  pyramid/      # multi-scale tile extraction + back-projection
  matchers/     # Matcher ABC + (later) RoMa / ELoFTR / MatchAnything
  estimators/   # MAGSAC++ consensus + transform fitting
  metrics/      # P_match@k, mu_err, med_err, success_rate
  pipeline/     # register(I_s, I_t, backbone) -> H, diagnostics
tests/          # pytest suite, synthetic-pair acceptance test
configs/        # hydra configs
```

## Quick test
```
pip install -e .[dev]
pytest -q
```

The default `pytest` run uses synthetic image pairs + an oracle matcher to
verify pyramid round-tripping, metrics, RANSAC fitting, and end-to-end
homography recovery to <0.5 px. Real backbones land behind feature flags
once their weights are vendored.

## Status
Phase 0 (setup) + the testable core of Phases 1–3 scaffolded. Real
backbones, AmalgaMatch loader, and full eval harness are next.
