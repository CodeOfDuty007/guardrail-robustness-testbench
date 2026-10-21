"""Toy MoE + REX surgery: expansion, gradient masking, excision is physical and exact."""
import pytest

torch = pytest.importorskip("torch")

from app.defenses.rex.data import DataConfig, Tokenizer, generate_corpora  # noqa: E402
from app.defenses.rex.evaluate import encode_examples, seq_len  # noqa: E402
from app.defenses.rex.relocate import (  # noqa: E402
    _accumulate, _lm_loss, active_mask, excise_experts, expand_experts, param_groups,
)
from app.defenses.rex.toy_moe import MoEConfig, build_model, param_count  # noqa: E402


@pytest.fixture(scope="module")
def setup():
    c = generate_corpora(0, DataConfig(n_forget=8, n_retain_train=16, n_retain_test=8, copy_len=3))
    tok = Tokenizer(c.vocab)
    T = seq_len(c.all_examples())
    cfg = MoEConfig(vocab_size=tok.size, max_len=T, d_model=32, n_heads=2, d_ff=64)
    model = build_model(cfg, seed=0).eval()
    return model, encode_examples(tok, c.forget, T), encode_examples(tok, c.retain_test, T)


def test_shapes_and_router_logits_per_expert(setup):
    model, forget, _ = setup
    logits, probs = model(forget.ids)
    assert logits.shape == (*forget.ids.shape, model.cfg.vocab_size)
    assert len(probs) == model.cfg.n_layers
    assert all(p.shape == (*forget.ids.shape, 8) for p in probs)
    assert torch.allclose(probs[0].sum(-1), torch.ones(forget.ids.shape))


def test_seeded_init_is_deterministic(setup):
    model, _, _ = setup
    other = build_model(model.cfg, seed=0)
    assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), other.state_dict().values()))
    assert not torch.equal(build_model(model.cfg, seed=1).tok.weight, model.tok.weight)


def test_expansion_adds_k_experts_and_masked_view_equals_base(setup):
    model, forget, _ = setup
    full = expand_experts(model, 2, torch.Generator().manual_seed(0)).eval()
    assert full.n_experts == 10 and full.cfg.n_experts == 10
    assert all(layer.router.shape == (10, model.cfg.d_model) for layer in full.moe_layers())
    assert param_count(full) > param_count(model)
    # with targets hidden from routing, the expanded model computes exactly the base model
    base_logits, _ = model(forget.ids)
    masked_logits, _ = full(forget.ids, active=active_mask(10, [8, 9]))
    assert torch.allclose(base_logits, masked_logits, atol=1e-6)


def test_excision_is_physical_and_matches_masked_full(setup):
    model, forget, retain = setup
    full = expand_experts(model, 2, torch.Generator().manual_seed(0))
    with torch.no_grad():  # make targets differ from their source copies
        for layer in full.moe_layers():
            for e in layer.experts[8:]:
                for p in e.parameters():
                    p.add_(torch.randn_like(p) * 0.5)
            layer.router[8:] += 1.0
    full.eval()
    excised = excise_experts(full, [8, 9]).eval()
    assert excised.n_experts == 8 < full.n_experts
    assert param_count(excised) < param_count(full)
    assert all(len(layer.experts) == 8 and layer.router.shape[0] == 8 for layer in excised.moe_layers())
    for b in (forget, retain):
        a, _ = excised(b.ids)
        m, _ = full(b.ids, active=active_mask(10, [8, 9]))
        assert torch.allclose(a, m, atol=1e-6)
    with pytest.raises(ValueError):
        excise_experts(full, list(range(9)))
    for bad in ([10], [-1], [8, 8], [8, 99]):
        with pytest.raises(ValueError):
            excise_experts(full, bad)


def test_selective_gradient_mask_only_touches_allowed_params(setup):
    model, forget, _ = setup
    full = expand_experts(model, 2, torch.Generator().manual_seed(0)).train()
    groups = param_groups(full, 8)
    every = list(full.parameters())
    assert sum(len(g) for g in groups.values()) == len(every)          # partition: complete ...
    assert len({id(p) for g in groups.values() for p in g}) == len(every)  # ... and disjoint
    logits, _ = full(forget.ids)
    _accumulate(_lm_loss(logits, forget), groups["target"])
    allowed = {id(p) for p in groups["target"]}
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in groups["target"])
    assert all(p.grad is None for p in every if id(p) not in allowed)
