"""Analytics (SPEC §11). Reads the DuckDB replica of telemetry.db ONLY — never the corpus.

Every endpoint takes the same filter set (model, cipher, strategy, run, defense, date range).
"""
from __future__ import annotations

import functools
import html
import json
import math
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
from fastapi import APIRouter, Depends, Query
from fastapi.responses import HTMLResponse
from scipy import stats
from sqlalchemy import text

from app.analytics_store import Filters, flat_rows, model_size_b, query, snapshot
from app.config import settings
from app.db.telemetry.session import get_engine

router = APIRouter(prefix="/api/analytics", tags=["analytics"])
METRICS = ("avalanche_score", "shannon_entropy", "diffusion_score", "confusion_chi2")
CORR_SCOPES = {"all": "", "encoding_family": "family_raw = 'encoding'"}


def filters(model: str | None = None, cipher: str | None = None, strategy: str | None = None,
            run_id: int | None = None, defense: str | None = None, date_from: str | None = None,
            date_to: str | None = None, family: str | None = None, category: str | None = None) -> Filters:
    return Filters(model, cipher, strategy, run_id, defense, date_from, date_to, family, category)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - h) / d), min(1.0, (c + h) / d))


def logistic_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float] | None:
    """IRLS logistic regression y ~ sigmoid(b0 + b1*x). Returns coefficients on the raw x scale."""
    if len(x) < 5 or y.min() == y.max() or np.std(x) < 1e-12:
        return None
    mu, sd = x.mean(), x.std()
    X = np.c_[np.ones(len(x)), (x - mu) / sd]
    b = np.zeros(2)
    for _ in range(50):
        p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))
        W = np.clip(p * (1 - p), 1e-6, None)
        step = np.linalg.solve(X.T @ (X * W[:, None]) + 1e-4 * np.eye(2), X.T @ (y - p))
        b = np.clip(b + step, -50, 50)  # perfectly separable data would diverge
        if np.abs(step).max() < 1e-8:
            break
    return float(b[0] - b[1] * mu / sd), float(b[1] / sd)


def _rate(rows: list[dict], key=lambda r: bool(r["bypass"])) -> dict:
    n = len(rows)
    k = sum(1 for r in rows if key(r))
    lo, hi = wilson(k, n)
    return {"n": n, "k": k, "rate": k / n if n else 0.0, "ci_low": lo, "ci_high": hi}


def _corr(rows: list[dict], metric: str) -> dict:
    pts = [(r[metric], int(bool(r["bypass"]))) for r in rows if r.get(metric) is not None]
    out: dict = {"metric": metric, "n": len(pts), "spearman": None, "logistic": None}
    if len(pts) >= 3:
        x, y = np.array([p[0] for p in pts], float), np.array([p[1] for p in pts], float)
        if len(set(x)) > 1 and len(set(y)) > 1:
            rho, pv = stats.spearmanr(x, y)
            out["spearman"] = {"rho": float(rho), "p": float(pv)}
        fit = logistic_fit(x, y)
        if fit:
            out["logistic"] = {"intercept": fit[0], "coef": fit[1]}
    return out


@router.get("/filters")
def filter_options():
    def distinct(col: str) -> list:
        return [r[col] for r in query(f"SELECT DISTINCT {col} FROM flat WHERE {col} IS NOT NULL ORDER BY 1")]
    rng = query("SELECT MIN(created_at) lo, MAX(created_at) hi FROM flat")[0]
    # n counts only attempts that reach the rates (attacker-refused excluded, as in Filters.where)
    runs = query("SELECT run_id, ANY_VALUE(run_name) AS run_label, "
                 "COUNT(*) FILTER (WHERE COALESCE(generated_by, '') <> 'attacker_refused') n, "
                 "COUNT(*) FILTER (WHERE generated_by = 'attacker_refused') refused FROM flat GROUP BY 1 ORDER BY 1 DESC")
    runs = [{"run_id": r["run_id"], "name": r["run_label"], "n": r["n"], "attacker_refused": r["refused"]} for r in runs]
    return {"models": distinct("target_model"), "ciphers": distinct("cipher_type"),
            "strategies": distinct("strategy_name"), "defenses": distinct("defense"), "runs": runs,
            "date_min": str(rng["lo"])[:10] if rng["lo"] else None, "date_max": str(rng["hi"])[:10] if rng["hi"] else None}


@router.get("/overview")
def overview(f: Filters = Depends(filters)):
    rows = flat_rows(f, "bypass, filter_evaded, cost_usd, target_model, run_id")
    n = len(rows)
    per_model: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        per_model[r["target_model"]].append(r)
    with get_engine().connect() as c:
        active = c.execute(text("SELECT COUNT(*) FROM runs WHERE status='running'")).scalar_one()
    refused = len(flat_rows(Filters(**{**f.__dict__, "include_attacker_refused": True}), "generated_by")) - n
    # attacker_refused: attempts whose attacker model declined; NOT in total_attempts or any rate
    return {"total_attempts": n, "attacker_refused": refused, "confirmed": sum(bool(r["bypass"]) for r in rows),
            "bypass_rate": _rate(rows)["rate"] if n else 0.0,
            "filter_evasion_rate": sum(bool(r["filter_evaded"]) for r in rows) / n if n else 0.0,
            "spend_usd": sum(r["cost_usd"] or 0 for r in rows), "active_runs": active,  # global (not filter-scoped)
            "total_runs": len({r["run_id"] for r in rows}),
            "per_model": [{"model": k, "attempts": len(v), "bypass_rate": _rate(v)["rate"]} for k, v in per_model.items()]}


@router.get("/correlation")
def correlation(metric: str = Query("avalanche_score"), scope: str = Query("all"), f: Filters = Depends(filters)):
    """Headline plot: crypto metric vs bypass. Points + per-cipher aggregates + logistic fit + Spearman.

    scope=all (default, unchanged) uses every attempt; scope=encoding_family restricts to attempts of the
    `encoding` attack family only (plan §A.5), so framing families don't contaminate the correlation."""
    if metric not in METRICS:
        return {"error": f"metric must be one of {METRICS}"}
    if scope not in CORR_SCOPES:
        return {"error": f"scope must be one of {tuple(CORR_SCOPES)}"}
    rows = [r for r in flat_rows(f, cond=CORR_SCOPES[scope]) if r[metric] is not None]
    out = _corr(rows, metric) | {"points": [], "by_cipher": [], "curve": []}
    if scope != "all":
        out["scope"] = scope
    if not rows:
        return out
    agg: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        agg[r["cipher_type"]].append(r)
    for c, v in agg.items():
        out["by_cipher"].append({"cipher": c, "x": float(np.mean([r[metric] for r in v])), "bypass_rate": _rate(v)["rate"],
                                 "n": len(v), "ci_low": _rate(v)["ci_low"], "ci_high": _rate(v)["ci_high"]})
    out["points"] = [{"x": r[metric], "y": int(bool(r["bypass"])), "cipher": r["cipher_type"]} for r in rows[:2000]]
    if out["logistic"]:
        xs = np.linspace(min(r[metric] for r in rows), max(r[metric] for r in rows), 40)
        b0, b1 = out["logistic"]["intercept"], out["logistic"]["coef"]
        out["curve"] = [{"x": float(a), "p": float(1 / (1 + math.exp(-(b0 + b1 * a))))} for a in xs]
    return out


@router.get("/metric-table")
def metric_table(f: Filters = Depends(filters)):
    rows = flat_rows(f)
    return [_corr([r for r in rows if r[m] is not None], m) for m in METRICS]


def _ols(x: list[float], y: list[float]) -> dict | None:
    if len(x) < 3 or len(set(x)) < 2:
        return None
    s = stats.linregress(x, y)
    fin = lambda v: float(v) if v is not None and math.isfinite(v) else None  # noqa: E731  (JSON rejects NaN)
    return {"slope": fin(s.slope), "intercept": fin(s.intercept), "r2": fin(s.rvalue ** 2)}


@router.get("/scaling")
def scaling(f: Filters = Depends(filters)):
    rows = flat_rows(f, "cipher_type, plaintext_size_bytes, ciphertext_size_bytes, encryption_time_ms, decryption_time_ms")
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by[r["cipher_type"]].append(r)
    fits = {c: {"enc_vs_plaintext": _ols([r["plaintext_size_bytes"] for r in v], [r["encryption_time_ms"] for r in v]),
                "dec_vs_ciphertext": _ols([r["ciphertext_size_bytes"] for r in v], [r["decryption_time_ms"] for r in v]),
                "expansion": float(np.mean([r["ciphertext_size_bytes"] / max(1, r["plaintext_size_bytes"]) for r in v]))}
            for c, v in by.items()}
    return {"points": [{"cipher": r["cipher_type"], "plaintext_bytes": r["plaintext_size_bytes"],
                        "ciphertext_bytes": r["ciphertext_size_bytes"], "enc_ms": r["encryption_time_ms"],
                        "dec_ms": r["decryption_time_ms"]} for r in rows[:3000]], "fits": fits}


@router.get("/pareto")
def pareto(f: Filters = Depends(filters)):
    by: dict[str, list[dict]] = defaultdict(list)
    for r in flat_rows(f):
        by[r["cipher_type"]].append(r)
    res = []
    for c, v in by.items():
        rt = _rate(v)
        res.append({"cipher": c, "bypass_rate": rt["rate"], "ci_low": rt["ci_low"], "ci_high": rt["ci_high"], "n": len(v),
                    "mean_cost_ms": float(np.mean([r["encryption_time_ms"] + r["decryption_time_ms"] + r["latency_ms"] for r in v])),
                    "mean_cost_usd": float(np.mean([r["cost_usd"] for r in v]))})
    # mark the non-dominated frontier: higher bypass with lower cost
    for a in res:
        a["frontier"] = not any(b is not a and b["bypass_rate"] >= a["bypass_rate"] and b["mean_cost_ms"] <= a["mean_cost_ms"]
                                and (b["bypass_rate"] > a["bypass_rate"] or b["mean_cost_ms"] < a["mean_cost_ms"]) for b in res)
    return res


def _kappa(a: list[bool], b: list[bool]) -> float | None:
    n = len(a)
    if n < 2:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = (sum(a) / n) * (sum(b) / n) + (1 - sum(a) / n) * (1 - sum(b) / n)
    return None if pe == 1 else (po - pe) / (1 - pe)


@router.get("/judge-agreement")
def judge_agreement(f: Filters = Depends(filters)):
    """Three independent signals (§9): regex canary, judge LLM, Llama Guard. Pairwise agreement + Cohen's kappa."""
    by: dict[str, list[dict]] = defaultdict(list)
    for r in flat_rows(f):
        by[r["strategy_name"]].append(r)
    out = []
    for strat, v in by.items():
        row: dict = {"strategy": strat, "n": len(v)}
        for name, (ka, kb) in {"regex_judge": ("regex_verdict", "judge_lm_verdict"),
                               "regex_guard": ("regex_verdict", "llamaguard_verdict"),
                               "judge_guard": ("judge_lm_verdict", "llamaguard_verdict")}.items():
            pairs = [(bool(r[ka]), bool(r[kb])) for r in v if r[ka] is not None and r[kb] is not None]
            row[name] = {"n": len(pairs), "agreement": (sum(x == y for x, y in pairs) / len(pairs)) if pairs else None,
                         "kappa": _kappa([p[0] for p in pairs], [p[1] for p in pairs]) if pairs else None}
        out.append(row)
    return out


@router.get("/standards")
def standards(f: Filters = Depends(filters)):
    by: dict[tuple, list[dict]] = defaultdict(list)
    for r in flat_rows(f):
        by[(r["owasp_tag"] or "untagged", r["atlas_tag"] or "untagged")].append(r)
    return [{"owasp": k[0], "atlas": k[1], "attempts": len(v), "bypass_rate": _rate(v)["rate"],
             "ci_low": _rate(v)["ci_low"], "ci_high": _rate(v)["ci_high"]} for k, v in by.items()]


@router.get("/leaderboard")
def leaderboard(f: Filters = Depends(filters)):
    by: dict[str, list[dict]] = defaultdict(list)
    for r in flat_rows(f):
        by[r["target_model"]].append(r)
    res = []
    for model, v in by.items():
        rt = _rate(v)
        res.append({"model": model, "attempts": rt["n"], "asr": rt["rate"], "asr_ci_low": rt["ci_low"],
                    "asr_ci_high": rt["ci_high"], "filter_evasion_rate": _rate(v, lambda r: bool(r["filter_evaded"]))["rate"]})
    return sorted(res, key=lambda r: r["asr"])  # most robust first


@router.get("/defense-ablation")
def defense_ablation(f: Filters = Depends(filters)):
    """Each defense vs the undefended baseline, compared ONLY on cells (model × dataset × cipher × strategy)
    that exist in both arms, so the delta isn't confounded by different mixes of models or ciphers."""
    f2 = Filters(**{**f.__dict__, "defense": None})
    rows = flat_rows(f2)
    cells: dict[tuple, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        cells[(r["target_model"], r["dataset_name"], r["cipher_type"], r["strategy_name"])][r["defense"]].append(r)
    names = sorted({r["defense"] for r in rows} - {"none"})
    scope = {"defense_filter_ignored": bool(f.defense), "comparison": "matched cells (model, dataset, cipher, strategy)",
             "filters_applied": {k: v for k, v in f2.__dict__.items() if v not in (None, "")}}
    out = []
    base_all = [r for r in rows if r["defense"] == "none"]
    out.append({"defense": "none", **_rate(base_all), "matched_cells": None, "baseline_rate": None, "delta_vs_none": None,
                "delta_ci_low": None, "delta_ci_high": None, "scope": scope})
    for d in names:
        arm, base, n_cells = [], [], 0
        for arms in cells.values():
            if arms.get(d) and arms.get("none"):
                arm += arms[d]; base += arms["none"]; n_cells += 1
        rt, rb = _rate(arm), _rate(base)
        # 95% CI for the difference of two proportions (Wald)
        se = math.sqrt(rt["rate"] * (1 - rt["rate"]) / rt["n"] + rb["rate"] * (1 - rb["rate"]) / rb["n"]) if rt["n"] and rb["n"] else None
        delta = (rt["rate"] - rb["rate"]) if (rt["n"] and rb["n"]) else None
        out.append({"defense": d, **rt, "matched_cells": n_cells, "baseline_rate": rb["rate"] if rb["n"] else None,
                    "baseline_n": rb["n"], "delta_vs_none": delta,
                    "delta_ci_low": None if se is None else delta - 1.96 * se, "delta_ci_high": None if se is None else delta + 1.96 * se,
                    "scope": scope, "note": None if n_cells else "no cells with both arms — run the same config with and without this defense"})
    return out


@router.get("/model-size")
def model_size(f: Filters = Depends(filters)):
    by: dict[str, list[dict]] = defaultdict(list)
    for r in flat_rows(f):
        by[r["target_model"]].append(r)
    res = [{"model": m, "size_b": model_size_b(m), **_rate(v)} for m, v in by.items()]
    return sorted(res, key=lambda r: (r["size_b"] is None, r["size_b"] or 0, r["model"]))


@router.get("/heatmap")
def heatmap(f: Filters = Depends(filters)):
    by: dict[tuple, list[dict]] = defaultdict(list)
    for r in flat_rows(f):
        by[(r["cipher_type"], r["strategy_name"])].append(r)
    return [{"cipher": c, "strategy": s, **_rate(v)} for (c, s), v in by.items()]


@router.get("/timeline")
def timeline(f: Filters = Depends(filters)):
    by: dict[int, list[dict]] = defaultdict(list)
    for r in flat_rows(f):
        by[r["run_id"]].append(r)
    return [{"run_id": k, "name": v[0]["run_name"], "created_at": str(min(r["created_at"] for r in v))[:19], **_rate(v)}
            for k, v in sorted(by.items())]


# ---------------------------------------------------------------------------------------------------------
# Robustness Profile (plan §A.5 + §D.5). Built ONLY from attempts with a real family (`a.family IS NOT NULL`);
# legacy strategy rows (DirectCipherWrap, PAIR, ...) never appear. Attacker-refused attempts stay out of every
# rate (Filters.where) and are counted separately in `attacker_refused`.
# ---------------------------------------------------------------------------------------------------------
FAMILY_ONLY = "family_raw IS NOT NULL"
_PROFILE_COLS = ("family_raw, category, channel, delivery, cipher_type, target_model, dataset_name, objective_id, "
                 "defense, bypass")


def atlas_parent(tag: str | None) -> str | None:
    """Parent ATLAS technique id: `AML.T0051.001` -> `AML.T0051` (sub-technique suffix dropped)."""
    if not tag:
        return None
    parts = tag.split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else tag


@functools.lru_cache(maxsize=1)
def _family_meta() -> dict[str, dict]:
    from app.engine.families import list_families
    return {f["name"]: f for f in list_families()}


def _meta(name: str) -> dict:
    """Registry metadata; tolerant of family names written by stories not loaded here."""
    m = _family_meta().get(name) or {}
    return {"citation": m.get("citation"), "owasp_tag": m.get("owasp_tag"), "atlas_tag": m.get("atlas_tag"),
            "atlas_parent": atlas_parent(m.get("atlas_tag")), "registry_category": m.get("category"),
            "registry_channel": m.get("channel")}


def _profile_rows(f: Filters, cond: str = "") -> list[dict]:
    return flat_rows(f, _PROFILE_COLS, cond=FAMILY_ONLY + (f" AND ({cond})" if cond else ""))


def _refused(f: Filters, cond: str = "") -> int:
    """Attacker-refused family attempts matching the same filters (excluded from every rate)."""
    c = FAMILY_ONLY + " AND generated_by = 'attacker_refused'" + (f" AND ({cond})" if cond else "")
    return len(flat_rows(Filters(**{**f.__dict__, "include_attacker_refused": True}), "attempt_id", cond=c))


def _mode(vals: list) -> str | None:
    vals = [v for v in vals if v]
    return max(sorted(set(vals)), key=vals.count) if vals else None


def _diff(arm: dict, base: dict) -> dict:
    """Difference of two proportions with a 95% Wald CI, clamped to [-1, 1]."""
    if not (arm["n"] and base["n"]):
        return {"delta": None, "delta_ci_low": None, "delta_ci_high": None}
    d = arm["rate"] - base["rate"]
    se = math.sqrt(arm["rate"] * (1 - arm["rate"]) / arm["n"] + base["rate"] * (1 - base["rate"]) / base["n"])
    return {"delta": d, "delta_ci_low": max(-1.0, d - 1.96 * se), "delta_ci_high": min(1.0, d + 1.96 * se)}


@router.get("/family-heatmap")
def family_heatmap(f: Filters = Depends(filters)):
    """Rows = attack families, columns = target models, cell = bypass rate (filter_evaded AND semantic_complied)
    with a 95% Wilson CI and n. Per-family category/channel/tags/citation; ATLAS rolled up by parent technique."""
    rows = _profile_rows(f)
    by_fam: dict[str, list[dict]] = defaultdict(list)
    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        by_fam[r["family_raw"]].append(r)
        cells[(r["family_raw"], r["target_model"])].append(r)
    fams = []
    for name in sorted(by_fam, key=lambda n: (_mode([r["category"] for r in by_fam[n]]) or "", n)):
        v, m = by_fam[name], _meta(name)
        fams.append({"family": name, "category": _mode([r["category"] for r in v]) or m["registry_category"],
                     "channel": _mode([r["channel"] for r in v]) or m["registry_channel"],
                     "channels": sorted({r["channel"] for r in v if r["channel"]}),
                     **{k: m[k] for k in ("citation", "owasp_tag", "atlas_tag", "atlas_parent")}, **_rate(v)})
    by_atlas: dict[str, list[dict]] = defaultdict(list)
    tags: dict[str, set] = defaultdict(set)
    for fam in fams:
        key = fam["atlas_parent"] or "untagged"
        by_atlas[key] += by_fam[fam["family"]]
        if fam["atlas_tag"]:
            tags[key].add(fam["atlas_tag"])
    atlas = [{"atlas": k, "sub_ids": sorted(tags[k]),
              "families": [x["family"] for x in fams if (x["atlas_parent"] or "untagged") == k], **_rate(v)}
             for k, v in sorted(by_atlas.items())]
    return {"models": sorted({r["target_model"] for r in rows}), "families": fams, "by_atlas": atlas,
            "attacker_refused": _refused(f),
            "cells": [{"family": fa, "model": mo, **_rate(v)} for (fa, mo), v in sorted(cells.items())]}


@router.get("/family-radar")
def family_radar(f: Filters = Depends(filters)):
    """One polygon per model over the family axes. A model with no attempts on a family has no entry for it
    (never a fabricated 0)."""
    rows = _profile_rows(f)
    per: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        per[r["target_model"]][r["family_raw"]].append(r)
    return {"families": sorted({r["family_raw"] for r in rows}), "attacker_refused": _refused(f),
            "models": [{"model": mo, "families": {fa: _rate(v) for fa, v in sorted(fv.items())}}
                       for mo, fv in sorted(per.items())]}


@router.get("/compounding")
def compounding(f: Filters = Depends(filters)):
    """Per family: bare (cipher NONE) vs family + cipher, compared ONLY on matched cells
    (model x dataset x objective x defense) present in both arms — the defense_ablation pattern."""
    f2 = Filters(**{**f.__dict__, "cipher": None})  # both arms are needed, so a cipher filter can't apply
    rows = _profile_rows(f2)
    cells: dict[str, dict[tuple, dict[str, list[dict]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        key = (r["target_model"], r["dataset_name"], r["objective_id"], r["defense"])
        cells[r["family_raw"]][key][r["cipher_type"]].append(r)
    scope = {"comparison": "matched cells (model, dataset, objective, defense)", "cipher_filter_ignored": bool(f.cipher),
             "filters_applied": {k: v for k, v in f2.__dict__.items() if v not in (None, "", False)}}
    out = []
    for fam in sorted(cells):
        fc = cells[fam]
        ciphers = sorted({c for arms in fc.values() for c in arms} - {"NONE"})
        per_cipher = []
        for c in ciphers:
            arm, base, nc = [], [], 0
            for arms in fc.values():
                if arms.get(c) and arms.get("NONE"):
                    arm += arms[c]; base += arms["NONE"]; nc += 1
            ra, rb = _rate(arm), _rate(base)
            per_cipher.append({"cipher": c, "matched_cells": nc, "ciphered": ra, "bare": rb, **_diff(ra, rb)})
        ciph_all: list[dict] = []
        bare_all: list[dict] = []
        n_cells = 0
        for arms in fc.values():  # pooled: each matched cell's ciphered attempts vs its bare ones
            ca = [r for c, v in arms.items() if c != "NONE" for r in v]
            if ca and arms.get("NONE"):
                ciph_all += ca; bare_all += arms["NONE"]; n_cells += 1
        ra, rb = _rate(ciph_all), _rate(bare_all)
        note = (None if n_cells else "never run with a cipher; add it to wrap_with_cipher" if not ciphers
                else "never run bare (cipher NONE) on the same cells; include NONE in the run ciphers")
        cats = [r["category"] for arms in fc.values() for v in arms.values() for r in v]
        out.append({"family": fam, "category": _mode(cats), "matched_cells": n_cells, "bare": rb, "ciphered": ra,
                    **_diff(ra, rb), "by_cipher": per_cipher, "note": note})
    return {"families": out, "scope": scope, "attacker_refused": _refused(f2)}


@router.get("/injection-breakdown")
def injection_breakdown(f: Filters = Depends(filters)):
    """Injection families only (§D.5): direct vs indirect split, and bypass rate by delivery carrier over
    indirect attempts only (direct injection has no carrier). Both are also split per model."""
    cond = "category = 'injection'"
    rows = _profile_rows(f, cond)
    ind = [r for r in rows if r["channel"] == "indirect"]

    def group(rs: list[dict], key: str) -> list[dict]:
        by: dict[str, list[dict]] = defaultdict(list)
        bym: dict[tuple, list[dict]] = defaultdict(list)
        for r in rs:
            by[r[key] or "unspecified"].append(r)
            bym[(r[key] or "unspecified", r["target_model"])].append(r)
        return [{key: k, "families": sorted({r["family_raw"] for r in v}), **_rate(v),
                 "per_model": [{"model": m, **_rate(x)} for (kk, m), x in sorted(bym.items()) if kk == k]}
                for k, v in sorted(by.items())]

    return {"by_channel": group(rows, "channel"), "by_delivery": group(ind, "delivery"),
            "has_direct": any(r["channel"] == "direct" for r in rows), "has_indirect": bool(ind),
            "models": sorted({r["target_model"] for r in rows}), "attacker_refused": _refused(f, cond),
            "note": None if ind else "No indirect-injection attempts yet. Run an indirect family against an agentic target."}


def build_report(f: Filters) -> dict:
    """Guardrail Robustness Report: aggregates only — no prompts, payloads or raw responses.

    Every section is read inside ONE pinned snapshot, so counts agree even while a run is in progress.
    """
    with snapshot():
        rows = flat_rows(f, "target_model, dataset_name, seed, run_id")
        models = sorted({r["target_model"] for r in rows})
        sc = scaling(f)
        return {
            "provenance": {"title": "Guardrail Robustness Report", "operator": settings.operator_name,
                           "generated_at": datetime.now(timezone.utc).isoformat(), "license": "CC-BY-4.0",
                           "filters": {k: v for k, v in f.__dict__.items() if v not in (None, "")},
                           "models": models, "datasets": sorted({r["dataset_name"] for r in rows}),
                           "seeds": sorted({r["seed"] for r in rows}), "runs": sorted({r["run_id"] for r in rows}),
                           "attempts": len(rows), "simulated_data": any(m.startswith("mock-") for m in models),
                           "redaction": "aggregate rates and metrics only; no prompts, payloads or responses"},
            "overview": overview(f), "leaderboard": leaderboard(f), "heatmap": heatmap(f),
            "metric_correlations": metric_table(f), "pareto": pareto(f), "scaling_fits": sc["fits"],
            "defense_ablation": defense_ablation(f), "model_size": model_size(f), "standards": standards(f),
            "judge_agreement": judge_agreement(f), "robustness_profile": family_heatmap(f),
        }


def _profile_html(prof: dict) -> str:
    """Family x model table (rate, 95% CI, n) with a numbered citation footnote per family."""
    e = html.escape
    pc = lambda x: f"{x * 100:.1f}%"  # noqa: E731
    if not prof["families"]:
        return "<p>No attack-family attempts match these filters (legacy strategy runs are not part of the profile).</p>"
    cells = {(c["family"], c["model"]): c for c in prof["cells"]}
    notes = {f["family"]: i + 1 for i, f in enumerate(x for x in prof["families"] if x["citation"])}
    th = "".join(f"<th>{e(h)}</th>" for h in ["Family", "Category", "Channel", "ATLAS", *prof["models"]])
    trs = []
    for fam in prof["families"]:
        ref = f"<sup>[{notes[fam['family']]}]</sup>" if fam["family"] in notes else ""
        atlas = e(fam["atlas_parent"] or "—") + (f" <small>({e(fam['atlas_tag'])})</small>"
                                                  if fam["atlas_tag"] and fam["atlas_tag"] != fam["atlas_parent"] else "")
        tds = []
        for mo in prof["models"]:
            c = cells.get((fam["family"], mo))
            tds.append("<td>—</td>" if c is None else
                       f"<td>{pc(c['rate'])} [{pc(c['ci_low'])} – {pc(c['ci_high'])}] n={c['n']}</td>")
        trs.append(f"<tr><td>{e(fam['family'])}{ref}</td><td>{e(fam['category'] or '—')}</td>"
                   f"<td>{e(', '.join(fam['channels']) or '—')}</td><td>{atlas}</td>{''.join(tds)}</tr>")
    foot = "".join(f"<li value={i}>{e(name)}: {e(next(x['citation'] for x in prof['families'] if x['family'] == name))}</li>"
                   for name, i in notes.items())
    return (f"<p>Bypass rate (filter evaded AND complied) per attack family and target model, with 95% Wilson interval "
            f"and attempts. Attacker-refused attempts excluded: {prof['attacker_refused']}.</p>"
            f"<table><tr>{th}</tr>{''.join(trs)}</table><ol class=refs>{foot}</ol>")


def _table(headers: list[str], rows: list[list]) -> str:
    e = html.escape
    th = "".join(f"<th>{e(h)}</th>" for h in headers)
    tr = "".join("<tr>" + "".join(f"<td>{e(str(c))}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><tr>{th}</tr>{tr}</table>"


@router.get("/report")
def report(format: str = "json", f: Filters = Depends(filters)):
    """format=json | html. The HTML page has a print stylesheet: use your browser's Print → Save as PDF."""
    rep = build_report(f)
    if format != "html":
        return rep
    e, p = html.escape, rep["provenance"]
    pc = lambda x: "—" if x is None else f"{x * 100:.1f}%"  # noqa: E731
    num = lambda x, d=3: "—" if x is None else f"{x:.{d}f}"  # noqa: E731
    sections = [
        ("Overview", _table(["Attempts", "Confirmed bypasses", "Bypass rate", "Passed the filter", "Attacker refused (excluded)"],
                            [[rep["overview"]["total_attempts"], rep["overview"]["confirmed"], pc(rep["overview"]["bypass_rate"]), pc(rep["overview"]["filter_evasion_rate"]),
                              rep["overview"].get("attacker_refused", 0)]])),
        ("Robustness Profile: attack family × model (bypass rate, 95% CI)", _profile_html(rep["robustness_profile"])),
        ("Models, most robust first (attack success rate, 95% CI)", _table(["Model", "Attempts", "Success rate", "95% CI"], [[r["model"], r["attempts"], pc(r["asr"]), f"{pc(r['asr_ci_low'])} – {pc(r['asr_ci_high'])}"] for r in rep["leaderboard"]])),
        ("Cipher property vs bypass (Spearman)", _table(["Property", "n", "ρ", "p"], [[m["metric"], m["n"], num(m["spearman"]["rho"]) if m["spearman"] else "—", f"{m['spearman']['p']:.2g}" if m["spearman"] else "—"] for m in rep["metric_correlations"]])),
        ("Cipher × strategy", _table(["Cipher", "Strategy", "n", "Bypass rate"], [[h["cipher"], h["strategy"], h["n"], pc(h["rate"])] for h in rep["heatmap"]])),
        ("Cost frontier", _table(["Cipher", "Bypass rate", "Mean cost (ms)", "On frontier"], [[x["cipher"], pc(x["bypass_rate"]), num(x["mean_cost_ms"], 1), "yes" if x["frontier"] else ""] for x in rep["pareto"]])),
        ("Defense ablation (matched cells only)", _table(["Defense", "Bypass rate", "Baseline", "Change", "Matched cells"], [[d["defense"], pc(d["rate"]), pc(d["baseline_rate"]), "—" if d["delta_vs_none"] is None else f"{d['delta_vs_none'] * 100:+.1f} pp", d["matched_cells"] if d["matched_cells"] is not None else "—"] for d in rep["defense_ablation"]])),
        ("Model size", _table(["Model", "Size (B params)", "Attempts", "Bypass rate"], [[m["model"], m["size_b"] if m["size_b"] is not None else "—", m["n"], pc(m["rate"])] for m in rep["model_size"]])),
        ("OWASP LLM Top 10 / MITRE ATLAS", _table(["OWASP", "ATLAS", "Attempts", "Bypass rate"], [[s_["owasp"], s_["atlas"], s_["attempts"], pc(s_["bypass_rate"])] for s_ in rep["standards"]])),
        ("Judge agreement", _table(["Strategy", "n", "Canary ↔ judge", "Canary ↔ Guard", "Judge ↔ Guard"], [[j["strategy"], j["n"]] + [pc(j[k]["agreement"]) for k in ("regex_judge", "regex_guard", "judge_guard")] for j in rep["judge_agreement"]])),
    ]
    warn = "<p class=warn><b>This report contains simulated (mock) model data. Do not cite it as a result.</b></p>" if p["simulated_data"] else ""
    body = "".join(f"<h2>{e(t)}</h2>{tb}" for t, tb in sections)
    return HTMLResponse(f"""<!doctype html><meta charset=utf-8><title>{e(p['title'])}</title>
<style>body{{font:15px/1.5 system-ui;max-width:920px;margin:2rem auto;padding:0 1rem;color:#111}}table{{border-collapse:collapse;width:100%;margin:.5rem 0 1.5rem}}td,th{{border:1px solid #ccc;padding:4px 8px;text-align:left}}
.warn{{background:#fde68a;padding:8px}}button{{padding:6px 14px}}@media print{{button{{display:none}}h2{{break-after:avoid}}table{{break-inside:avoid}}}}</style>
<button onclick="window.print()">Print / save as PDF</button>
<h1>{e(p['title'])}</h1><p>Operator {e(p['operator'])} · {e(p['generated_at'])} · {e(p['license'])}<br>
Models: {e(', '.join(p['models']) or 'none')} · datasets: {e(', '.join(p['datasets']))} · seeds: {p['seeds']} · {p['attempts']} attempts in {len(p['runs'])} runs<br>
Filters: {e(json.dumps(p['filters']) if p['filters'] else 'none')}</p>{warn}<p><i>{e(p['redaction'])}</i></p>{body}""")
