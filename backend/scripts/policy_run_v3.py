"""Extended battery: fills in strategy x objective combos not yet tested against the reduced
policy_violation set (confirm_harmful_capability, nullify_safety_policy), plus a prompt-size
sweep via ManyShot_N (Anthropic many-shot jailbreaking: shot count / prompt length as the
manipulated variable, same objective, same wording per shot).
Sized for free tier (5 RPM/model): small batches, 65s pause between."""
import time
import httpx

c = httpx.Client(base_url="http://localhost:8000", timeout=60)
T, A, J = 5, 6, 4

BATCHES = [
    ("DirectCipherWrap", ["NONE"], 1),
    ("PersonaRoleplay", ["NONE"], 1),
    ("InstructionOverride", ["NONE"], 1),
    ("ManyShot", ["NONE"], 1),       # baseline: 2 shots
    ("ManyShot_4", ["NONE"], 1),
    ("ManyShot_16", ["NONE"], 1),
    ("ManyShot_64", ["NONE"], 1),
    ("ManyShot_128", ["NONE"], 1),
    ("PAIR", ["NONE"], 2),
    ("TAP", ["NONE"], 2),
    ("Crescendo", ["NONE"], 2),
]

results = []
for n, (strat, ciphers, iters) in enumerate(BATCHES, 1):
    cfg = {"name": f"gemini-v3-{n}-{strat}", "target_provider_id": T, "judge_provider_id": J,
           "dataset": "policy_violation", "limit": 2, "ciphers": ciphers, "strategies": [strat],
           "max_iterations": iters, "concurrency": 1}
    if strat in ("PAIR", "TAP", "Crescendo"):
        cfg["attacker_provider_id"] = A
    rid = c.post("/api/runs", json=cfg).json()["id"]
    for _ in range(140):
        time.sleep(3)
        s = c.get(f"/api/runs/{rid}").json()
        if s["status"] in ("done", "error"):
            break
    rows = [r for r in c.get("/api/explorer/attempts?limit=300").json()["rows"] if r["run_id"] == rid]
    tally = {}
    for r in rows:
        tally[r["status"]] = tally.get(r["status"], 0) + 1
    line = f"batch {n} {strat} {ciphers} run={rid} status={s['status']} err={s.get('error')} attempts={tally}"
    print(line, flush=True)
    results.append(line)
    wins = [r["id"] for r in rows if r["status"] == "success"]
    if wins:
        print("POLICY VIOLATION CONFIRMED: attempt ids", wins, flush=True)
    time.sleep(65)

print("---done---", flush=True)
