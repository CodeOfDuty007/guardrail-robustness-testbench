import json

from app.compute import nodes


def test_node_register_probe_adopt(client, monkeypatch):
    async def fake_probe(node, token=None):
        return {"status": "online", "models": ["llama3:8b", "qwen2.5:7b"], "gpu_info": "llama3:8b (5.1 GB VRAM)",
                "error": None, "latency_ms": 12.5}
    monkeypatch.setattr(nodes, "probe", fake_probe)
    n = client.post("/api/compute/nodes", json={"label": "rig", "base_url": "http://gpu:11434", "token": "sk-should-not-persist-0123456789"}).json()
    assert n["status"] == "online" and n["models"] == ["llama3:8b", "qwen2.5:7b"] and "token" not in json.dumps(n)
    ps = client.post(f"/api/compute/nodes/{n['id']}/adopt", params={"role": "target"}, json=["llama3:8b"]).json()
    assert ps[0]["base_url"] == "http://gpu:11434" and ps[0]["compute_node_id"] == n["id"]
    assert client.post("/api/compute/nodes", json={"label": "x", "base_url": "ftp://bad"}).status_code == 422


def test_offline_node_reports_error(client):
    n = client.post("/api/compute/nodes", json={"label": "dead", "base_url": "http://127.0.0.1:9"}).json()
    assert n["status"] == "offline"
