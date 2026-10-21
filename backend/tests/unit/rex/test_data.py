"""Synthetic-data ACs: deterministic from seed, verifiably synthetic, no real-hazard keywords."""
import re
from dataclasses import replace

import pytest

pytest.importorskip("torch")  # rex is the optional [rex] extra; skip the whole suite without it

from app.defenses.rex.data import (  # noqa: E402
    ANSWER_RE, ENTITY_RE, DataConfig, Example, Tokenizer, corpus_text, generate_corpora, verify_synthetic,
)

# Independent of data.HAZARD_DENYLIST on purpose: the test must not trust the module's own list.
REAL_HAZARD_KEYWORDS = [
    "anthrax", "sarin", "ricin", "novichok", "tabun", "soman", "mustard", "toxin", "pathogen", "virus",
    "bacteria", "botulinum", "smallpox", "ebola", "plague", "bioweapon", "nerve agent", "chemical",
    "synthesis", "precursor", "explosive", "detonator", "bomb", "tnt", "rdx", "uranium", "plutonium",
    "enrich", "centrifuge", "warhead", "weapon", "malware", "ransomware", "exploit", "shellcode",
    "keylogger", "botnet", "rootkit", "zero-day", "gene", "culture", "aerosol",
]
CFG = DataConfig(n_forget=40, n_retain_train=300, n_retain_test=100)


def test_same_seed_identical_corpora():
    a, b = generate_corpora(7, CFG), generate_corpora(7, CFG)
    assert a.forget == b.forget and a.retain_train == b.retain_train and a.retain_test == b.retain_test
    assert a.vocab == b.vocab and corpus_text(a) == corpus_text(b)


def test_different_seed_different_corpora():
    a, b = generate_corpora(7, CFG), generate_corpora(8, CFG)
    assert a.forget != b.forget
    assert a.retain_train != b.retain_train
    assert {x.answer for x in a.forget} != {x.answer for x in b.forget}


def test_forget_entities_invented_and_answers_random_generator_tokens():
    c = generate_corpora(3, CFG)
    assert len(c.forget) == CFG.n_forget
    entities = [ex.prompt[5] for ex in c.forget]
    assert all(ENTITY_RE.match(e) for e in entities)
    assert len(set(entities)) == len(entities)
    answers = [ex.answer[0] for ex in c.forget]
    assert all(ANSWER_RE.match(a) for a in answers)
    assert len(set(answers)) == len(answers)
    # no vowels -> no English word can appear; and each has a digit
    assert all(not re.search(r"[aeiou]", a) and re.search(r"\d", a) for a in answers)
    # answers are exactly what the seeded generator produces (regenerate -> identical)
    assert answers == [ex.answer[0] for ex in generate_corpora(3, CFG).forget]
    # every forget prompt is the fixed invented template
    assert all(ex.prompt[:5] == ("<bos>", "the", "access", "rune", "for") and ex.prompt[6] == "is"
               for ex in c.forget)


def test_denylist_never_appears_in_corpus_or_vocab():
    for seed in range(5):
        c = generate_corpora(seed, CFG)
        text = (corpus_text(c) + " " + " ".join(c.vocab)).lower()
        tokens = set(text.split())
        # substring match for keywords with a vowel (random answers have none, so only real words could
        # match); whole-token match for vowel-less acronyms that random strings may contain by chance
        hits = [k for k in REAL_HAZARD_KEYWORDS
                if (k in text if re.search(r"[aeiou]", k) else k in tokens)]
        assert not hits, (seed, hits)


def test_retain_heldout_disjoint_from_train():
    c = generate_corpora(1, CFG)
    train = {ex.answer for ex in c.retain_train}
    test = {ex.answer for ex in c.retain_test}
    assert len(train) == CFG.n_retain_train and len(test) == CFG.n_retain_test
    assert not train & test
    assert all(ex.prompt[-1] == "=" and ex.prompt[2:-1] == ex.answer for ex in c.retain_test)


def test_verify_synthetic_rejects_tampered_corpus():
    c = generate_corpora(0, CFG)
    bad = replace(c, forget=[Example(c.forget[0].prompt, ("weapon",)), *c.forget[1:]])
    with pytest.raises(ValueError):
        verify_synthetic(bad)
    bad2 = replace(c, forget=[Example(("<bos>", "the", "access", "rune", "for", "Acme-Corp", "is"),
                                      c.forget[0].answer)])
    with pytest.raises(ValueError):
        verify_synthetic(bad2)


def test_tokenizer_roundtrip_and_deterministic_vocab():
    c = generate_corpora(0, CFG)
    tok = Tokenizer(c.vocab)
    ex = c.forget[0]
    ids = tok.encode([*ex.prompt, *ex.answer])
    assert tok.decode(ids) == [*ex.prompt, *ex.answer]
    assert tok.encode(["never-seen"]) == [tok.unk_id]
    assert c.vocab[:4] == ["<pad>", "<bos>", "<eos>", "<unk>"]
    with pytest.raises(ValueError):
        Tokenizer(["x", *c.vocab])


def test_impossible_copy_space_rejected():
    with pytest.raises(ValueError):
        generate_corpora(0, DataConfig(n_retain_train=60, n_retain_test=10, copy_len=2, n_symbols=8))


def test_heldout_prompts_are_new_phrasings_of_the_same_facts():
    from app.defenses.rex.data import HELDOUT_TEMPLATES, FORGET_TEMPLATE
    c = generate_corpora(0)
    assert len(c.forget_heldout) == len(HELDOUT_TEMPLATES) * len(c.forget)
    assert FORGET_TEMPLATE not in HELDOUT_TEMPLATES
    train = {ex.prompt for ex in c.forget}
    facts = {ex.prompt[5]: ex.answer for ex in c.forget}
    for ex in c.forget_heldout:
        assert ex.prompt not in train
        ent = next(w for w in ex.prompt if w.startswith("VAULT-"))
        assert facts[ent] == ex.answer
        assert all(w in c.vocab for w in ex.prompt)
