"""End-to-end REX pipeline on the fast config (~10 s per run on a laptop CPU).

Thresholds (fast config, documented in README): base/full forget+retain >= 0.9; excised forget_acc
<= 0.2 and retain drop <= 0.1; excised strictly smaller. Measured on seeds 0-2: base/full 1.0/1.0,
excised forget 0.0, excised retain 1.0.
"""
import json
import os
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from app.defenses.rex.data import generate_corpora  # noqa: E402
from app.defenses.rex.evaluate import answer, generate  # noqa: E402
from app.defenses.rex.relocate import (  # noqa: E402
    default_config, fast_config, load_model, rex_pipeline, run_rex_pipeline, save_artifacts,
)


def _strip(m):
    return {k: v for k, v in m.items() if k not in ("timings_s", "save_dir")}


@pytest.fixture(scope="module")
def art():
    return rex_pipeline(0, fast_config())


def _check_acs(m):
    assert m["base"]["forget_acc"] >= 0.9 and m["base"]["retain_acc"] >= 0.9
    assert m["full"]["forget_acc"] >= 0.9 and m["full"]["retain_acc"] >= 0.9
    assert m["excised"]["forget_acc"] <= 0.2
    assert m["base"]["retain_acc"] - m["excised"]["retain_acc"] <= 0.1
    assert m["full"]["retain_acc"] - m["excised"]["retain_acc"] <= 0.1
    assert m["params"]["excised"] < m["params"]["full"]
    assert m["experts_per_layer"]["excised"] < m["experts_per_layer"]["full"]


def test_base_learns_and_excision_removes_forget_keeps_retain(art):
    m = art.metrics
    _check_acs(m)
    assert m["excised"]["forget_answer_prob"] < 0.2 < m["full"]["forget_answer_prob"]
    assert m["routing"]["forget_mass_on_targets"] > 0.5 > m["routing"]["retain_mass_on_targets"]


def test_metrics_json_able(art):
    m = art.metrics
    assert json.loads(json.dumps(m)) == m
    assert set(m["timings_s"]) >= {"base_train_s", "relocate_s", "total_s"}


def test_same_seed_identical_metrics(art):
    again = run_rex_pipeline(0, fast_config())
    assert _strip(again) == _strip(art.metrics)
    assert again["excised_sha256"] == art.metrics["excised_sha256"]


def test_pipeline_restores_torch_global_state(art):
    before = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled())
    rng = torch.get_rng_state()
    run_rex_pipeline(1, replace(fast_config(), base_steps=1, relocate_steps=1))
    assert (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled()) == before
    assert torch.equal(torch.get_rng_state(), rng)


def test_generate_answer_plain_types_and_save_load_roundtrip(art, tmp_path):
    c = generate_corpora(0, fast_config().data)
    fact = c.forget[0]
    prompt = " ".join(fact.prompt[1:])
    got_full = answer(art.full, art.tokenizer, prompt)
    assert isinstance(got_full, str) and got_full == fact.answer[0]
    assert answer(art.excised, art.tokenizer, prompt) != fact.answer[0]
    copy = c.retain_test[0]
    out = generate(art.excised, art.tokenizer, " ".join(copy.prompt[1:]))
    assert isinstance(out, str) and out.split() == list(copy.answer)

    d = save_artifacts(art, tmp_path / "rex")
    assert {p.name for p in d.iterdir()} >= {"full.pt", "excised.pt", "vocab.json", "metrics.json"}
    model, tok = load_model(d, "excised")
    assert model.n_experts == art.excised.n_experts
    assert answer(model, tok, prompt) == answer(art.excised, art.tokenizer, prompt)
    full, _ = load_model(d, "full")
    assert answer(full, tok, prompt) == fact.answer[0]
    assert json.loads((d / "metrics.json").read_text())["excised_sha256"] == art.metrics["excised_sha256"]
    with pytest.raises(ValueError):
        load_model(d, "../full")


@pytest.mark.skipif(not os.environ.get("REX_FULL"), reason="minutes-scale; set REX_FULL=1")
def test_default_config_full_pipeline():
    m = run_rex_pipeline(0, default_config())
    _check_acs(m)
    assert m["params"]["full"] > 1_000_000
    assert m["timings_s"]["total_s"] < 180
