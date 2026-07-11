import itertools, os, sys
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from security.fusion_engine import compute_trust_score as engine_compute_trust_score, detect_conflicts as engine_detect_conflicts
from backend.app import compute_trust as backend_compute_trust, detect_conflicts as backend_detect_conflicts

SIGNAL_KEYS = ["identity", "liveness", "visual", "audio", "injection"]
BACKEND_KEYS = {"identity": "identity_score", "liveness": "liveness_score", "visual": "visual_deepfake_score", "audio": "audio_deepfake_score", "injection": "injection_risk_score"}
GRID = [0.0, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 1.0]

def band_to_common(band):
    b = band.lower()
    if b == "trusted": return "trusted"
    if b == "suspicious": return "suspicious"
    if b in ("potential fraud", "fraud"): return "fraud"
    return b

def run_engine(c):
    conflicts = engine_detect_conflicts(c)
    result = engine_compute_trust_score(dict(c))
    trust = result["trust_score"]
    penalty = 0.0
    if conflicts["conflict_detected"]:
        penalty = max({"identity_vs_visual_deepfake": 0.10, "liveness_vs_injection": 0.30}.get(x, 0.0) for x in conflicts["conflict_types"])
        trust = int(trust * (1 - penalty))
    capped = False
    if c["identity"] < 0.60 and trust >= 80:
        trust, capped = 79, True
    band = "Trusted" if trust >= 80 else "Suspicious" if trust >= 50 else "Potential Fraud"
    return trust, band, capped

def run_backend(c):
    flat = {BACKEND_KEYS[k]: c[k] * 100 for k in SIGNAL_KEYS}
    sub = {BACKEND_KEYS[k]: c[k] for k in SIGNAL_KEYS}
    conflicts = backend_detect_conflicts(flat)
    trust, band, hard_deny, capped, _ = backend_compute_trust(flat, sub, conflicts)
    assert hard_deny == []
    return trust, band, capped

@pytest.mark.parametrize("i,l,v,a,j", list(itertools.product(GRID, repeat=5)))
def test_same_score_and_band(i, l, v, a, j):
    c = {"identity": i, "liveness": l, "visual": v, "audio": a, "injection": j}
    et, eb, ec = run_engine(dict(c))
    bt, bb, bc = run_backend(dict(c))
    assert abs(et - bt) <= 1, f"{c}: engine={et} backend={bt}"
    assert band_to_common(eb) == band_to_common(bb), f"{c}: engine={eb} backend={bb}"
    assert ec == bc, f"{c}: engine_capped={ec} backend_capped={bc}"

def test_wrong_person_attack_blocked():
    c = {"identity": 0.50, "liveness": 1.0, "visual": 1.0, "audio": 1.0, "injection": 1.0}
    et, eb, ec = run_engine(dict(c))
    bt, bb, bc = run_backend(dict(c))
    assert et == 79 and bt == 79
    assert band_to_common(eb) == "suspicious" and band_to_common(bb) == "suspicious"
    assert ec and bc
