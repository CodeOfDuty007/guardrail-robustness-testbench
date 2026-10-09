"""Analytics (SPEC §11). Reads the DuckDB replica of telemetry.db ONLY — never the corpus.

Every endpoint takes the same filter set (model, cipher, strategy, run, defense, date range).
"""
from __future__ import annotations

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


def filters(model: str | None = None, cipher: str | None = None, strategy: str | None = None,
            run_id: int | None = None, defense: str | None = None, date_from: str | None = None,
            date_to: str | None = None) -> Filters:
    return Filters(model, cipher, strategy, run_id, defense, date_from, date_to)


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
    runs = query("SELECT run_id, ANY_VALUE(run_name) AS run_label, COUNT(*) n FROM flat GROUP BY 1 ORDER BY 1 DESC")
    runs = [{"run_id": r["run_id"], "name": r["run_label"], "n": r["n"]} for r in runs]
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
    return {"total_attempts": n, "confirmed": sum(bool(r["bypass"]) for r in rows),
            "bypass_rate": _rate(rows)["rate"] if n else 0.0,
            "filter_evasion_rate": sum(bool(r["filter_evaded"]) for r in rows) / n if n else 0.0,
            "spend_usd": sum(r["cost_usd"] or 0 for r in rows), "active_runs": active,  # global (not filter-scoped)
            "total_runs": len({r["run_id"] for r in rows}),
            "per_model": [{"model": k, "attempts": len(v), "bypass_rate": _rate(v)["rate"]} for k, v in per_model.items()]}


@router.get("/correlation")
def correlation(metric: str = Query("avalanche_score"), f: Filters = Depends(filters)):
    """Headline plot: crypto metric vs bypass. Points + per-cipher aggregates + logistic fit + Spearman."""
    if metric not in METRICS:
        return {"error": f"metric must be one of {METRICS}"}
    rows = [r for r in flat_rows(f) if r[metric] is not None]
    out = _corr(rows, metric) | {"points": [], "by_cipher": [], "curve": []}
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
            "judge_agreement": judge_agreement(f),
        }


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
        ("Overview", _table(["Attempts", "Confirmed bypasses", "Bypass rate", "Passed the filter"],
                            [[rep["overview"]["total_attempts"], rep["overview"]["confirmed"], pc(rep["overview"]["bypass_rate"]), pc(rep["overview"]["filter_evasion_rate"])]])),
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
