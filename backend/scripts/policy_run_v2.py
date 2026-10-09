"""Policy-violation objectives (confirm_harmful_capability, nullify_safety_policy) tested across
a wider battery of prompt-framing methods from the jailbreak literature. Cipher=NONE isolates the
wording/framing signal from encoding; one cipher-wrapped run included for comparison.
Sized for free tier (5 RPM/model): small batches, 65s pause between."""
import time
import httpx

c = httpx.Client(base_url="http://localhost:8000", timeout=60)
T, A, J = 5, 6, 4

BATCHES = [
    ("RefusalSuppression", ["NONE"], 1),
    ("PrefixInjection", ["NONE"], 1),
    ("HypotheticalFraming", ["NONE"], 1),
    ("PayloadSplitting", ["NONE"], 1),
    ("PAIR", ["NONE"], 2),
    ("PayloadSplitting", ["STANDARD_BASE64"], 1),
]

results = []
for n, (strat, ciphers, iters) in enumerate(BATCHES, 1):
    cfg = {"name": f"gemini-methodv2-{n}-{strat}", "target_provider_id": T, "judge_provider_id": J,
           "dataset": "policy_violation", "limit": 2, "ciphers": ciphers, "strategies": [strat],
           "max_iterations": iters, "concurrency": 1}
    if strat in ("PAIR", "TAP", "Crescendo"):
        cfg["attacker_provider_id"] = A
    rid = c.post("/api/runs", json=cfg).json()["id"]
    for _ in range(120):
        time.sleep(3)
        s = c.get(f"/api/runs/{rid}").json()
        if s["status"] in ("done", "error"):
            break
    rows = [r for r in c.get("/api/explorer/attempts?limit=200").json()["rows"] if r["run_id"] == rid]
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
