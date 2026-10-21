"""End-to-end REX pipeline on the fast config (B -> F0 -> F1 -> A1 -> A2, plus the RR control).

Thresholds are set against the *measured* untuned paper default (fast config, seeds 0-2, see README),
not tuned toward a target:
  paper default : B, F1 forget/retain = 1.0. A1 forget = 1.0 / 1.0 / 0.79 (excision removes 0-21%).
                  A2 forget = RR forget = 0.0, retain >= 0.995.  -> A2 ~= RR at toy scale.
  pushout (NOT from paper): A1 forget = 0.0; push_out_weight=0: A1 forget = 0.58 / 0.38 / 0.04.
We assert what holds and deliberately do NOT assert "A1 forgets" or "A2 < RR" for the paper default.
"""
import json
import os
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from app.defenses.rex.data import generate_corpora  # noqa: E402
from app.defenses.rex.evaluate import answer, generate  # noqa: E402
from app.defenses.rex.relocate import (  # noqa: E402
    STAGES, RexConfig, default_config, fast_config, load_model, rex_pipeline, run_rex_pipeline,
    save_artifacts,
)


def _strip(m):
    return {k: v for k, v in m.items() if k not in ("timings_s", "save_dir")}


@pytest.fixture(scope="module")
def art():
    return rex_pipeline(0, fast_config())


@pytest.fixture(scope="module")
def pushout(art):
    return rex_pipeline(0, replace(fast_config(), variant="prelearned_pushout"), base=art.base).metrics


@pytest.fixture(scope="module")
def pushout0(art):
    cfg = replace(fast_config(), variant="prelearned_pushout", push_out_weight=0.0)
    return rex_pipeline(0, cfg, base=art.base).metrics


def _check_common(m):
    st = m["stages"]
    assert set(st) == set(STAGES)
    for k in ("B", "F1"):
        assert st[k]["forget_acc"] >= 0.9 and st[k]["retain_acc"] >= 0.9, k
    for k in ("A1", "A2", "RR"):                                       # general capability preserved
        assert st["B"]["retain_acc"] - st[k]["retain_acc"] <= 0.1, k
    assert st["RR"]["forget_acc"] <= 0.2                               # 100-step RMU erases the toy facts
    assert st["A2"]["forget_acc"] <= 0.2
    p, e = m["params"], m["experts_per_layer"]
    assert p["A1"] < p["F1"] and e["A1"] < e["F1"]                     # excision is physical
    assert p["A1"] == p["A2"] == p["RR"] == p["B"]


def test_paper_default_stages(art):
    m = art.metrics
    assert m["variant"] == "paper" and "paper recipe" in m["provenance"]
    assert m["config"]["source_scale"] == 0.5 and m["config"]["clone_noise"] == 1e-5
    assert m["expansion"]["method"] == "differential_activation"
    _check_common(m)
    s = m["summary"]
    assert set(s["forget_acc"]) == {"B", "F1", "A1", "A2", "RR"}
    # measured, reported, not asserted as a success: excision alone removes little in the toy
    assert 0.0 <= s["excision_forget_drop"] <= 1.0
    assert abs(s["a2_minus_rr_forget"]) <= 0.2                          # A2 ~= RR
    # training-time routing constraints were met on general tokens (rank hinge -> 0)
    assert m["relocation_log"][-1]["rank"] == 0.0
    assert m["routing"]["retain_target_selection_rate"] <= 0.2


def test_pushout_ablation_is_labelled_and_relocates(pushout, pushout0):
    for m in (pushout, pushout0):
        assert m["variant"] == "prelearned_pushout" and "NOT from the paper" in m["provenance"]
        _check_common(m)
    assert pushout["stages"]["A1"]["forget_acc"] <= 0.2
    assert pushout0["stages"]["A1"]["forget_acc"] > pushout["stages"]["A1"]["forget_acc"]


def test_unknown_variant_rejected():
    with pytest.raises(ValueError):
        RexConfig(variant="nope")


def test_metrics_json_able(art):
    m = art.metrics
    assert json.loads(json.dumps(m)) == m
    assert set(m["timings_s"]) >= {"base_train_s", "relocate_s", "cleanup_s", "total_s"}


def test_same_seed_identical_metrics(art):
    again = run_rex_pipeline(0, fast_config())
    assert _strip(again) == _strip(art.metrics)
    assert again["delivered_sha256"] == art.metrics["delivered_sha256"]


def test_reusing_base_gives_identical_results(art):
    again = rex_pipeline(0, fast_config(), base=art.base).metrics
    assert _strip(again) == _strip(art.metrics)


def test_pipeline_restores_torch_global_state():
    before = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled())
    rng = torch.get_rng_state()
    tiny = replace(fast_config(), base_steps=1, relocate_steps=1, rmu_steps=1)
    run_rex_pipeline(1, tiny)
    run_rex_pipeline(1, replace(tiny, variant="prelearned_pushout"))
    assert (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled()) == before
    assert torch.equal(torch.get_rng_state(), rng)


def test_generate_answer_plain_types_and_save_load_roundtrip(art, tmp_path):
    c = generate_corpora(0, fast_config().data)
    fact = c.forget[0]
    prompt = " ".join(fact.prompt[1:])
    got_full = answer(art.full, art.tokenizer, prompt)
    assert isinstance(got_full, str) and got_full == fact.answer[0]
    assert answer(art.delivered, art.tokenizer, prompt) != fact.answer[0]
    copy = c.retain_test[0]
    out = generate(art.delivered, art.tokenizer, " ".join(copy.prompt[1:]))
    assert isinstance(out, str) and out.split() == list(copy.answer)

    d = save_artifacts(art, tmp_path / "rex")
    assert {p.name for p in d.iterdir()} >= {"full.pt", "excised.pt", "delivered.pt", "vocab.json",
                                             "metrics.json"}
    for name, model in (("excised", art.excised), ("delivered", art.delivered), ("full", art.full)):
        loaded, tok = load_model(d, name)
        assert loaded.n_experts == model.n_experts
        assert answer(loaded, tok, prompt) == answer(model, art.tokenizer, prompt)
    assert json.loads((d / "metrics.json").read_text())["delivered_sha256"] == art.metrics["delivered_sha256"]
    with pytest.raises(ValueError):
        load_model(d, "../full")


@pytest.mark.skipif(not os.environ.get("REX_FULL"), reason="minutes-scale; set REX_FULL=1")
def test_default_config_full_pipeline():
    m = run_rex_pipeline(0, default_config())
    _check_common(m)
    assert m["params"]["F1"] > 1_000_000
    assert m["timings_s"]["total_s"] < 240
