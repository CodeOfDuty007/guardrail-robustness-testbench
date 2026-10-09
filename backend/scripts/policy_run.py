"""Run policy_violation objectives against the Gemini target using varied prompt-framing
strategies (not just cipher encoding). Sized for free tier (5 RPM/model): small batches, pauses."""
import time
import httpx

c = httpx.Client(base_url="http://localhost:8000", timeout=60)
T, A, J = 5, 6, 4

BATCHES = [
    ("InstructionOverride", ["NONE"], 1),
    ("PersonaRoleplay", ["NONE"], 1),
    ("ManyShot", ["NONE"], 1),
    ("PAIR", ["NONE"], 2),
    ("DirectCipherWrap", ["STANDARD_BASE64"], 1),
]

results = []
for n, (strat, ciphers, iters) in enumerate(BATCHES, 1):
    cfg = {"name": f"gemini-policy-{n}-{strat}", "target_provider_id": T, "judge_provider_id": J,
           "dataset": "policy_violation", "limit": 5, "ciphers": ciphers, "strategies": [strat],
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
    time.sleep(65)  # respect 5 RPM quota window

print("---done---", flush=True)
