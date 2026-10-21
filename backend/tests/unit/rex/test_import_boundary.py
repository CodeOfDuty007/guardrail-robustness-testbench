"""torch is an optional extra: the app (and the rex package itself) must import without it."""
import subprocess
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[3]

# A meta-path finder that makes torch look uninstalled (sys.modules['torch']=None confuses scipy).
CODE = """
import sys
class _NoTorch:
    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ModuleNotFoundError("No module named 'torch' (blocked by test)")
        return None
sys.meta_path.insert(0, _NoTorch())
import app.main
import app.defenses.rex as rex
assert "run_rex_pipeline" in rex.__all__
assert "torch" not in sys.modules
try:
    rex.run_rex_pipeline
except ModuleNotFoundError:
    print("ok")
"""


def test_app_imports_without_torch():
    r = subprocess.run([sys.executable, "-c", CODE], cwd=BACKEND, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().endswith("ok")
