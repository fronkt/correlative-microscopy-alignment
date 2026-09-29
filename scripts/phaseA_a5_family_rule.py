"""Phase A / A5: characterize the exploratory 'RoMa-family-only' S1 pick rule (E4 of exploratory_triage.py).

Exploratory on data already seen; every variant tried is logged in results/phaseA/a5_log.md.
Run from the repo root:  python scripts/phaseA_a5_family_rule.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from cma.triage import ci95, cluster_bootstrap, mcnemar_exact

OUT = Path("results/phaseA")
OUT.mkdir(parents=True, exist_ok=True)
B = 10_000
BEST = "ma_roma|pyramid_v2|none|s0"
E4 = {"n_ok": 56, "best_single_n": 47, "won": 14, "lost": 5, "oracle_n": 80}


def load() -> pd.DataFrame:
    df = pd.read_csv("results/triage/candidates.csv")
    df["cand"] = (df.backbone + "|" + df["mode"] + "|" + df["transform"].astype(str) + "|s"
                  + df.seed.astype(int).astype(str))
    ok = df.status.eq("ok")
    df["err"] = np.where(ok, pd.to_numeric(df.mu_ed, errors="coerce"), np.inf)
    df["err"] = df["err"].fillna(np.inf)
    nm = pd.to_numeric(df.n_matches, errors="coerce")
    ni = pd.to_numeric(df.n_inliers, errors="coerce")
    df["nm"] = np.where(ok, nm, 0).astype(float)
    df["ni"] = np.where(ok, ni, 0).astype(float)
    df["S1"] = np.where(ok & (nm > 0), ni / nm, -np.inf)
    return df


df = load()
voters = sorted(df.loc[df.pool.isin(["core", "transform"]), "cand"].unique())
assert len(voters) == 15
pairs = sorted(df.pair_id.unique())
meta = df.drop_duplicates("pair_id").set_index("pair_id").loc[pairs, ["scene", "group", "subclass"]]
scenes = meta.scene.values
groups = meta.group.values
piv = lambda col: df.pivot(index="pair_id", columns="cand", values=col).reindex(pairs)[voters]  # noqa: E731
E, S1, NM, NI = piv("err"), piv("S1").fillna(-np.inf), piv("nm"), piv("ni")

dense = [c for c in voters if c.split("|")[0] in ("roma", "ma_roma")]
assert len(dense) == 12
base_ok = (E[BEST] <= 20).values
assert int(base_ok.sum()) == E4["best_single_n"]


def pick(score: pd.DataFrame, cols: list[str], mask: pd.DataFrame | None = None):
    """Argmax over cols (ties -> first in sorted voter order, as idxmax in E4). Pairs with no eligible
    candidate get pick None (counted as failure)."""
    s = score[cols].astype(float)
    if mask is not None:
        s = s.where(mask[cols])
    valid = s.notna().any(axis=1)
    pk = s.fillna(-np.inf).idxmax(axis=1)
    pk = pk.where(valid, None)
    return pk


def evaluate(pk: pd.Series) -> dict:
    ok = np.array([(p is not None and E.loc[i, p] <= 20) for i, p in zip(pairs, pk)])
    b, c = int((ok & ~base_ok).sum()), int((~ok & base_ok).sum())
    return {"ok": ok, "n_ok": int(ok.sum()), "sr20": float(ok.mean()), "won": b, "lost": c,
            "mcnemar_p": mcnemar_exact(b, c), "gain_n": int(ok.sum() - base_ok.sum())}


def picks_summary(pk: pd.Series) -> dict:
    vc = pk.fillna("NONE").value_counts()
    return {k: int(v) for k, v in vc.items()}


def brief(r: dict, pk: pd.Series) -> dict:
    return {k: r[k] for k in ("n_ok", "sr20", "won", "lost", "gain_n", "mcnemar_p")} | {"picks": picks_summary(pk)}


# ---------------------------------------------------------------- reproduce E4
pk_e4 = pick(S1, dense)
r_e4 = evaluate(pk_e4)
oracle_dense = int((E[dense].min(axis=1) <= 20).sum())
exp = json.loads(Path("results/triage/exploratory.json").read_text())["E4"]
repro = {"n_candidates": len(dense), "n_ok": r_e4["n_ok"], "best_single_n": int(base_ok.sum()),
         "won": r_e4["won"], "lost": r_e4["lost"], "mcnemar_p": r_e4["mcnemar_p"], "oracle_n": oracle_dense,
         "dense_candidates": dense,
         "matches_exploratory_json": bool(
             r_e4["n_ok"] == exp["n_ok"] and r_e4["won"] == exp["won"] and r_e4["lost"] == exp["lost"]
             and oracle_dense == exp["oracle_n"] and abs(r_e4["mcnemar_p"] - exp["mcnemar_p"]) < 1e-12)}
assert repro["matches_exploratory_json"], repro
assert (r_e4["n_ok"], r_e4["won"], r_e4["lost"], oracle_dense) == (56, 14, 5, 80)

# ties at the max among dense candidates
mx = S1[dense].max(axis=1)
ties = int(((S1[dense].eq(mx, axis=0)).sum(axis=1) > 1).sum())
ties_diff_outcome = 0
for i in pairs:
    tied = [c for c in dense if S1.loc[i, c] == mx[i]]
    if len(tied) > 1 and len({bool(E.loc[i, c] <= 20) for c in tied}) > 1:
        ties_diff_outcome += 1
repro["pairs_with_S1_tie_at_max"] = ties
repro["tied_pairs_where_outcome_depends_on_tiebreak"] = ties_diff_outcome

# ---------------------------------------------------------------- 1. wins / losses
rows = []
for k, i in enumerate(pairs):
    if r_e4["ok"][k] != base_ok[k]:
        p = pk_e4[i]
        rows.append({"outcome": "win" if r_e4["ok"][k] else "loss", "pair_id": i, "group": meta.loc[i, "group"],
                     "subclass": meta.loc[i, "subclass"], "scene": meta.loc[i, "scene"], "chosen": p,
                     "chosen_err": float(E.loc[i, p]), "chosen_S1": float(S1.loc[i, p]),
                     "chosen_n_matches": float(NM.loc[i, p]),
                     "best_single_err": float(E.loc[i, BEST]), "best_single_S1": float(S1.loc[i, BEST]),
                     "best_single_n_matches": float(NM.loc[i, BEST])})
wl = pd.DataFrame(rows).sort_values(["outcome", "group", "pair_id"], ascending=[False, True, True])
wl.to_csv(OUT / "a5_wins_losses.csv", index=False)
assert (wl.outcome == "win").sum() == 14 and (wl.outcome == "loss").sum() == 5
wl_sum = {"wins_by_group": wl[wl.outcome == "win"].group.value_counts().to_dict(),
          "losses_by_group": wl[wl.outcome == "loss"].group.value_counts().to_dict(),
          "wins_by_scene_n": int(wl[wl.outcome == "win"].scene.nunique()),
          "wins_chosen": wl[wl.outcome == "win"].chosen.value_counts().to_dict(),
          "losses_chosen": wl[wl.outcome == "loss"].chosen.value_counts().to_dict(),
          "wins_best_single_err_finite": int(np.isfinite(wl[wl.outcome == "win"].best_single_err).sum())}

# ---------------------------------------------------------------- 2. leave-one-group / scene out
def sub_eval(r: dict, keep: np.ndarray) -> dict:
    ok, bo = r["ok"][keep], base_ok[keep]
    b, c = int((ok & ~bo).sum()), int((~ok & bo).sum())
    return {"n_pairs": int(keep.sum()), "rule_n": int(ok.sum()), "best_n": int(bo.sum()), "gain_n": int(ok.sum() - bo.sum()),
            "won": b, "lost": c, "mcnemar_p": mcnemar_exact(b, c)}


logo = {g: sub_eval(r_e4, groups != g) for g in sorted(set(groups))}
per_group = {g: sub_eval(r_e4, groups == g) for g in sorted(set(groups))}
loso = {s: sub_eval(r_e4, scenes != s) for s in sorted(set(scenes))}
loso_gain = [v["gain_n"] for v in loso.values()]
loso_p = [v["mcnemar_p"] for v in loso.values()]
loso_sum = {"n_scenes": len(loso), "gain_min": int(min(loso_gain)), "gain_max": int(max(loso_gain)),
            "p_min": float(min(loso_p)), "p_max": float(max(loso_p))}
# scenes contributing the wins (drop the scene -> gain falls the most)
contrib = sorted(((s, (r_e4["ok"] & ~base_ok)[scenes == s].sum() - (~r_e4["ok"] & base_ok)[scenes == s].sum())
                  for s in loso), key=lambda t: -t[1])[:5]
loso_sum["top_net_contributing_scenes"] = [(s, int(v)) for s, v in contrib]

# ---------------------------------------------------------------- 3. mechanism rules
log_rows: list[dict] = []
rules: dict[str, dict] = {}
pk_store: dict[str, pd.Series] = {}


def run(name: str, pk: pd.Series, note: str = "") -> dict:
    r = evaluate(pk)
    rules[name] = brief(r, pk) | {"note": note}
    pk_store[name] = pk
    log_rows.append({"variant": name, **{k: rules[name][k] for k in ("n_ok", "won", "lost", "mcnemar_p")}, "note": note})
    r["_pk"] = pk
    return r


R = {}
R["R5_S1_all15"] = run("R5_S1_all15 (baseline: max S1 over all 15; = H3 select_S1)", pick(S1, voters))
R["R1_all_N0"] = R["R5_S1_all15"]
for N in (50, 100, 200, 500, 1000, 5000):
    mask = NM >= N
    key = f"R1_S1_all15_nmatches>={N}"
    R[key] = run(key, pick(S1, voters, mask))
    rules[key]["pairs_with_no_eligible_candidate"] = int((~mask.any(axis=1)).sum())
    rules[key]["eligible_frac_by_family"] = {
        "sparse(sift,loftr,matchanything)": float(mask[[c for c in voters if c.split("|")[0] not in ("roma", "ma_roma")]].values.mean()),
        "dense(roma,ma_roma)": float(mask[dense].values.mean())}
R["R2"] = run("R2_S1_dense12 (= E4)", pk_e4)
direct = [c for c in dense if "|direct|" in c]
pyr = [c for c in dense if "|pyramid_v2|" in c]
R["R3a"] = run("R3a_S1_dense_direct10", pick(S1, direct))
R["R3b"] = run("R3b_S1_dense_pyramid2", pick(S1, pyr))
R["R3c"] = run("R3c_S1_roma_only6 (extra)", pick(S1, [c for c in dense if c.startswith("roma|")]))
R["R3d"] = run("R3d_S1_ma_roma_only6 (extra)", pick(S1, [c for c in dense if c.startswith("ma_roma|")]))
R["R4"] = run("R4_max_n_inliers_all15", pick(NI, voters))
R["R4b"] = run("R4b_max_n_inliers_dense12 (extra)", pick(NI, dense))

# ---------------------------------------------------------------- 4. bootstrap (gain on SR20, scene-clustered)
def boot(r: dict) -> dict:
    ok = r["ok"]
    b = cluster_bootstrap(lambda idx: ok[idx].mean() - base_ok[idx].mean(), scenes, B=B)
    lo, hi = ci95(b)
    return {"gain": float(ok.mean() - base_ok.mean()), "lo": float(lo), "hi": float(hi), "n_boot": int(len(b)),
            "frac_boot_gain_le_0": float((np.asarray(b) <= 0).mean())}


r1_keys = [k for k in R if k.startswith("R1_S1_all15_nmatches")]
best_r1 = max(r1_keys, key=lambda k: (rules[k]["n_ok"], -rules[k]["mcnemar_p"]))
boots = {"R2 (E4)": boot(R["R2"]), f"best R1 = {best_r1}": boot(R[best_r1])}
r1_ev = evaluate(R[best_r1]["_pk"])
r1_logo = {g: sub_eval(r1_ev, groups != g) for g in sorted(set(groups))}

# ---------------------------------------------------------------- 5. multiplicity
# Pick rules compared against the best single candidate on these 187 pairs:
prior = ["H3 select_S1 (all 15)", "H3 select_S2", "H3 select_S3", "E4 dense-only S1"]
mine = [k for k in rules if not k.startswith("R2") and not k.startswith("R5")]  # R2=E4, R5=select_S1 already counted
n_variants = len(prior) + len(mine)
p_list = {k: rules[k]["mcnemar_p"] for k in rules}
best_p_name = min(p_list, key=p_list.get)
mult = {"prior_rules_on_these_pairs": prior, "new_rules_this_task": mine, "n_variants_total": n_variants,
        "best_p_variant": best_p_name, "best_p": p_list[best_p_name],
        "bonferroni_adj_best_p": min(1.0, p_list[best_p_name] * n_variants),
        "bonferroni_adj_E4_p": min(1.0, rules["R2_S1_dense12 (= E4)"]["mcnemar_p"] * n_variants),
        "bonferroni_alpha_per_test": 0.05 / n_variants,
        "note": "counts pick rules only; excludes LOGO/LOSO (sensitivity of one rule) and upstream E1-E9 diagnostics. "
                "E4's own choice of 'dense-only' was made after seeing SIFT dominate the S1 picks (E3), so the "
                "true forking-path count is larger than stated."}

for k in R:
    R[k].pop("ok", None)
res = {"reproduction_E4": repro, "wins_losses_summary": wl_sum, "logo_E4": logo, "per_group_E4": per_group,
       "loso_E4_summary": loso_sum, "rules": rules, "bootstrap_gain": boots, "best_R1": best_r1,
       "logo_best_R1": r1_logo, "multiplicity": mult, "B": B, "best_single": BEST}
(OUT / "a5_family_rule.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
print(json.dumps({k: res[k] for k in ("reproduction_E4", "wins_losses_summary", "logo_E4", "loso_E4_summary",
                                      "bootstrap_gain", "best_R1", "logo_best_R1", "multiplicity")}, indent=1, default=str))
for k, v in rules.items():
    print(k, {a: v[a] for a in ("n_ok", "won", "lost", "mcnemar_p")}, v["picks"])
