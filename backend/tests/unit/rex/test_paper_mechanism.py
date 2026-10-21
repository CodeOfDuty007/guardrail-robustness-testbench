"""Outcome-independent checks of the paper recipe's mechanics (tiny model, fast)."""
import copy
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from app.defenses.rex import relocate as R  # noqa: E402
from app.defenses.rex.cleanup import rmu_cleanup, rmu_trainable  # noqa: E402
from app.defenses.rex.data import DataConfig, Tokenizer, generate_corpora  # noqa: E402
from app.defenses.rex.evaluate import encode_examples, seq_len  # noqa: E402
from app.defenses.rex.toy_moe import MoEConfig, build_model  # noqa: E402

N, K = 8, 2


@pytest.fixture(scope="module")
def setup():
    c = generate_corpora(0, DataConfig(n_forget=8, n_retain_train=16, n_retain_test=8, copy_len=3))
    tok = Tokenizer(c.vocab)
    T = seq_len(c.all_examples())
    base = build_model(MoEConfig(vocab_size=tok.size, max_len=T, d_model=32, n_heads=2, d_ff=64), seed=0)
    cfg = replace(R.fast_config(), forget_batch=8, retain_batch=16, relocate_steps=3, rmu_steps=3)
    return base.eval(), encode_examples(tok, c.forget, T), encode_examples(tok, c.retain_train, T), cfg


def _expanded(base):
    return R.expand_experts(base, K, torch.Generator().manual_seed(0), sources=[[3, 5], [1, 6]],
                            scale=0.5, noise=1e-5, expert_noise=True)


def _snap(params):
    return [p.detach().clone() for p in params]


def test_expansion_clones_sources_and_scales_them(setup):
    base = setup[0]
    f0 = _expanded(base)
    assert f0.n_experts == N + K
    for layer_b, layer_f, src in zip(base.moe_layers(), f0.moe_layers(), [[3, 5], [1, 6]]):
        for j, s in enumerate(src):
            clone, orig = layer_f.experts[N + j], layer_b.experts[s]
            assert torch.allclose(clone.fc1.weight, orig.fc1.weight, atol=1e-4)      # unscaled copy + noise
            assert not torch.equal(clone.fc1.weight, orig.fc1.weight)                 # noise applied
            assert torch.allclose(layer_f.router[N + j], layer_b.router[s], atol=1e-4)
            assert torch.allclose(layer_f.experts[s].fc2.weight, 0.5 * orig.fc2.weight)  # source scaled
        untouched = [i for i in range(N) if i not in src]
        assert all(torch.equal(layer_f.experts[i].fc2.weight, layer_b.experts[i].fc2.weight) for i in untouched)


def test_differential_activation_picks_largest_delta(setup, monkeypatch):
    base, forget, retain, _ = setup
    real = R.selection_frequency(base, forget)                    # real stats: K selections per token
    assert all(abs(float(f.sum()) - K) < 1e-5 for f in real)
    fh = torch.tensor([0.9, 0.8, 0.1, 0.3, 0.0, 0.0, 0.0, 0.0])
    fs = torch.tensor([0.9, 0.1, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5])   # expert 0 is general-purpose: delta 0
    monkeypatch.setattr(R, "selection_frequency", lambda m, b: [fh, fh] if b is forget else [fs, fs])
    sources, deltas = R.differential_activation(base, forget, retain, 2)
    assert sources == [[1, 3], [1, 3]]
    assert deltas[0][0] == 0.0


def test_mixed_batch_mask_equals_excised_view_on_general_tokens(setup):
    base, forget, retain, _ = setup
    f0 = _expanded(base).eval()
    ids, _, _, harm, active = R.mixed_batch(forget, retain, f0.n_experts, N)
    with torch.no_grad():
        mixed, _ = f0(ids, active=active)
        gen_view, _ = f0(retain.ids, active=R.active_mask(f0.n_experts, [N, N + 1]))
        harm_view, _ = f0(forget.ids)
    assert harm[: len(forget)].all() and not harm[len(forget):].any()
    assert torch.allclose(mixed[len(forget):], gen_view, atol=1e-6)   # targets never dispatched on general
    assert torch.allclose(mixed[: len(forget)], harm_view, atol=1e-6)


def test_gradient_routing_of_the_two_losses(setup):
    base, forget, retain, cfg = setup
    f0 = _expanded(base).train()
    g = R.param_groups(f0, N)
    loss_h, loss_s, _ = R.paper_losses(f0, N, forget, retain, cfg)
    gs = torch.autograd.grad(loss_s, g["target"], retain_graph=True, allow_unused=True)
    assert all(x is None or float(x.abs().sum()) == 0.0 for x in gs)   # general data cannot reach targets
    gh = torch.autograd.grad(loss_h, g["orig"], retain_graph=True, allow_unused=True)
    assert any(x is not None and float(x.abs().sum()) > 0 for x in gh)  # ... harm data could: mask needed
    R._accumulate(loss_h, g["target"] + g["router"])
    assert all(p.grad is None for p in g["orig"] + g["shared"])


def test_relocate_paper_freezes_shared_params(setup):
    base, forget, retain, cfg = setup
    f1 = _expanded(base)
    g = R.param_groups(f1, N)
    before = {k: _snap(v) for k, v in g.items()}
    R.relocate_paper(f1, N, forget, retain, cfg, torch.Generator().manual_seed(0))
    assert all(torch.equal(a, b) for a, b in zip(before["shared"], g["shared"]))
    for k in ("orig", "target", "router"):
        assert any(not torch.equal(a, b) for a, b in zip(before[k], g[k])), k
    assert all(p.requires_grad for p in f1.parameters())                 # restored afterwards


def test_mass_and_rank_hinges():
    # one token, 4 originals + 2 targets, K=2
    satisfied = torch.tensor([[[3.0, 2.0, 0.0, 0.0, -1.0, -2.0]]])     # targets well below the boundary
    violated = torch.tensor([[[3.0, 0.0, 0.0, 0.0, 2.0, -2.0]]])       # a target inside the top-K
    assert float(R._rank_violation(satisfied, 4, 2, 0.1).sum()) == 0.0
    assert float(R._rank_violation(violated, 4, 2, 0.1).sum()) == pytest.approx(2.1)
    assert float(R._target_gate_mass(satisfied, 4, 2).sum()) == 0.0
    z = float(R._target_gate_mass(violated, 4, 2).sum())
    assert z == pytest.approx(torch.tensor([3.0, 2.0]).softmax(0)[1].item())


def test_rmu_updates_only_expert_mlps_up_to_monitor_and_is_deterministic(setup):
    base, forget, retain, cfg = setup
    cfg = replace(cfg, rmu_monitor_layer=0)
    snapshot = copy.deepcopy(base.state_dict())
    a = rmu_cleanup(base, forget, retain, cfg, seed=0, steer=10.0)
    b = rmu_cleanup(base, forget, retain, cfg, seed=0, steer=10.0)
    assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))
    assert all(torch.equal(v, snapshot[k]) for k, v in base.state_dict().items())   # input not mutated
    train = {n for n, p in a.named_parameters() if any(p is q for q in rmu_trainable(a, 0))}
    assert train and all(n.startswith("blocks.0.moe.experts.") for n in train)
    ref = dict(base.named_parameters())
    frozen_same = [torch.equal(p, ref[n]) for n, p in a.named_parameters() if n not in train]
    assert frozen_same and all(frozen_same)                     # routers, attention, layer-1 experts, ...
    assert any(not torch.equal(p, ref[n]) for n, p in a.named_parameters() if n in train)


def test_paper_lm_loss_is_one_token_mean_split_by_domain(setup):
    base, forget, retain, cfg = setup
    f0 = _expanded(base).eval()
    zero = replace(cfg, mass_floor_weight=0.0, mass_ceiling_weight=0.0, rank_weight=0.0)
    loss_h, loss_s, _ = R.paper_losses(f0, N, forget, retain, zero)
    ids, targets, _, _, active = R.mixed_batch(forget, retain, f0.n_experts, N)
    logits, _ = f0(ids, active=active)
    combined = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                                                 ignore_index=-100)
    assert float((loss_h + loss_s).detach()) == pytest.approx(float(combined.detach()), rel=1e-5)
