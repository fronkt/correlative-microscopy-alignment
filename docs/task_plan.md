# Task Plan (engineering-level)

This plan decomposes the research plan into concrete code/data tasks.
Owner column left for assignment.

## Phase 0 — Setup (W1)
| ID  | Task | Owner | Done When |
|-----|------|-------|-----------|
| 0.1 | Create repo `correlative-microscopy-alignment`, set up `pyproject.toml`, ruff, pre-commit |  | CI green on empty repo |
| 0.2 | Pin env: torch 2.x + cu12, kornia, opencv-python, scikit-image, hydra-core |  | `uv sync` reproducible on CUDA box |
| 0.3 | Obtain AmalgaMatch (license, download script, integrity hash) |  | Local mirror + checksum logged |
| 0.4 | Pull pretrained weights: RoMa, ELoFTR, MatchAnything |  | All three load + run on one sample |

## Phase 1 — Baselines (W2)
| ID  | Task |
|-----|------|
| 1.1 | Dataset loader: yields `(I_s, I_t, K_gt, scale_meta, group, subclass)` |
| 1.2 | Metric harness: `P_match@k`, `mu_err`, `med_err`, `success_rate`, runtime, mem |
| 1.3 | Control A: zero-shot ELoFTR / RoMa / MatchAnything end-to-end on test split |
| 1.4 | Control B: SIFT keypoints + MMI + homography fit |
| 1.5 | Persist baseline results as `results/baselines.parquet` + plots |

## Phase 2 — Pyramidal Wrapper (W3-W4)
| ID  | Task |
|-----|------|
| 2.1 | `pyramid.build(I_s, scale_ratio, tile_size, overlap=0.5)` -> list of tiles + back-projection metadata |
| 2.2 | Scale-aware fallback when `scale_meta` missing (estimator + flag) |
| 2.3 | `Matcher` ABC + concrete subclasses for RoMa / ELoFTR / MatchAnything |
| 2.4 | Tile-batched inference, fp16, peak-mem budget guard |
| 2.5 | Correspondence aggregator with tile-id + confidence tags |

## Phase 3 — Consensus + Transform (W5)
| ID  | Task |
|-----|------|
| 3.1 | MAGSAC++ wrapper (opencv USAC) with configurable threshold |
| 3.2 | Affine + Homography fitters; per-pair model selection |
| 3.3 | End-to-end pipeline glue: `register(I_s, I_t, backbone) -> H, diagnostics` |
| 3.4 | Unit tests: synthetic pair w/ known H recovers H to <0.5 px |

## Phase 4 — Full Evaluation (W6)
| ID  | Task |
|-----|------|
| 4.1 | Run Experimental group on test split, all 3 backbones |
| 4.2 | Headline table: Control A vs Control B vs Experimental, per group |
| 4.3 | Paired bootstrap CIs, significance markers |
| 4.4 | First-pass writeup of results section |

## Phase 5 — Ablations + Sensitivity (W7-W8)
| ID  | Task |
|-----|------|
| 5.1 | FOV sweep {50, 25, 10, 5, 2}% — synthetic crops from I_s, locate breakdown |
| 5.2 | Pyramid depth ablation (1, 2, 3, 4 levels) |
| 5.3 | Overlap ablation {25, 50, 75}% |
| 5.4 | RANSAC threshold sweep |
| 5.5 | Transform-family ablation (affine vs homography) |
| 5.6 | Scale-metadata error sensitivity (+/-20%) |

## Phase 6 — Writeup + Release (W9-W10)
| ID  | Task |
|-----|------|
| 6.1 | Tech report (figures, tables, methods, discussion of failure modes) |
| 6.2 | README with single-command reproduction |
| 6.3 | Release tagged v0.1, archived results bundle |

## Cross-Cutting
- Determinism: seed everything; record `git rev-parse HEAD` per result.
- Every result row carries `{backbone, config_hash, dataset_hash}`.
- All sweeps driven by hydra configs in `configs/`.
