"""S1-A1: all single-turn jailbreak families run end-to-end on the default (offline) mock target."""
from collections import defaultdict

from tests.integration.test_run import wait_done

S1_A1 = ["roleplay_persona", "authority_context", "persuasion", "refusal_suppression",
         "hypothetical_framing", "many_shot", "low_resource_lang", "payload_splitting"]
# Seed whose simulated guard draws fall inside the SIM_FAMILY_BOOST band, so families separate.
# (At the default seed every canary draw sits below 0.9 - 0.35 and the profile is flat.)
SEED = 42


def test_all_families_run_one_attempt_each_and_profile_is_non_flat(client):
    fams = ["encoding", *S1_A1]
    rid = client.post("/api/runs", json={"families": fams, "ciphers": ["NONE"], "seed": SEED}).json()["id"]
    assert wait_done(client, rid)["status"] == "done"
    atts = client.get(f"/api/runs/{rid}/attempts").json()
    assert len(atts) == 3 * len(fams)  # 3 canary objectives x 9 families, cipher NONE
    per = defaultdict(list)
    for a in atts:
        assert a["family"] in fams and a["category"] == "jailbreak" and a["cipher_type"] == "NONE"
        per[a["family"]].append(a)
    assert {f: len(v) for f, v in per.items()} == {f: 3 for f in fams}
    assert all(len({a["objective_id"] for a in v}) == 3 for v in per.values())
    outcomes = {f: tuple(sorted((a["objective_id"], a["status"]) for a in v)) for f, v in per.items()}
    assert len(set(outcomes.values())) >= 2, outcomes


def test_persuasion_wrapped_with_cipher_runs_bare_and_ciphered(client):
    rid = client.post("/api/runs", json={"families": ["persuasion"], "wrap_with_cipher": ["persuasion"],
                                         "ciphers": ["NONE", "STANDARD_BASE64"]}).json()["id"]
    assert wait_done(client, rid)["status"] == "done"
    atts = client.get(f"/api/runs/{rid}/attempts").json()
    assert {a["family"] for a in atts} == {"persuasion"}
    assert sorted(a["cipher_type"] for a in atts) == ["NONE"] * 3 + ["STANDARD_BASE64"] * 3
