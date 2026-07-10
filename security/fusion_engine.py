"""
DeepfakeGuard - Member 3 (M3) Trust Score Fusion Engine
Week 3 - "All Five Signals, One Score"

This module takes 5 different signals, normalizes them, computes a weighted sum,
smooths with EMA, detects conflicts, and produces an explainable trust score.

Usage:
    from security.fusion_engine import TrustScoreFusionEngine
    
    engine = TrustScoreFusionEngine(alpha=0.3)
    result = engine.process_signals({
        "identity": {"similarity_score": 0.81},
        "liveness": {"challenge_passed": True, "response_time_ms": 3000},
        "visual": {"deepfake_probability": 0.74},
        "audio": {"spoof_probability": 0.03},
        "injection": {"frame_timing_anomaly_score": 0.1, "virtual_camera_detected": False}
    })
"""

from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================

WEIGHTS = {
    "identity": 0.25,
    "liveness": 0.25,
    "visual": 0.20,
    "audio": 0.20,
    "injection": 0.10,
}

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"

SIGNAL_TIMEOUT_SECONDS = 3.0
LIVENESS_TIMEOUT_MS = 7000

# Conflict penalties (deducted from trust score)
CONFLICT_PENALTIES = {
    "identity_vs_visual_deepfake": 0.10,  # 10% penalty
    "liveness_vs_injection": 0.30,       # 30% penalty (more dangerous)
}

# Security hardening: minimum identity trust to reach "Trusted"
IDENTITY_TRUST_THRESHOLD = 0.60


# ============================================================
# SIGNAL NORMALIZATION
# ============================================================

def normalize_signal(signal_name: str, raw_payload: dict) -> float:
    """Converts a raw signal payload to a 0.0-1.0 trust contribution."""
    
    if signal_name == "identity":
        similarity = raw_payload.get("similarity_score", 0.0)
        return max(0.0, min(1.0, float(similarity)))
    
    elif signal_name == "liveness":
        challenge_passed = raw_payload.get("challenge_passed", False)
        response_time_ms = raw_payload.get("response_time_ms", float('inf'))
        if not challenge_passed:
            return 0.0
        elif response_time_ms <= LIVENESS_TIMEOUT_MS:
            return 1.0
        else:
            return 0.7
    
    elif signal_name == "visual":
        deepfake_prob = raw_payload.get("deepfake_probability", 0.5)
        return 1.0 - max(0.0, min(1.0, float(deepfake_prob)))
    
    elif signal_name == "audio":
        spoof_prob = raw_payload.get("spoof_probability", 0.5)
        return 1.0 - max(0.0, min(1.0, float(spoof_prob)))
    
    elif signal_name == "injection":
        anomaly_score = raw_payload.get("frame_timing_anomaly_score", 0.0)
        virtual_camera = raw_payload.get("virtual_camera_detected", False)
        trust = 1.0 - max(0.0, min(1.0, float(anomaly_score)))
        if virtual_camera:
            trust = min(trust, 0.2)
        return trust
    
    else:
        raise ValueError(f"Unknown signal: {signal_name}")


# ============================================================
# TRUST SCORE COMPUTATION
# ============================================================

def compute_trust_score(trust_contributions: dict) -> dict:
    """Computes trust score via weighted sum."""
    missing_signals = []
    for signal_name in WEIGHTS:
        if signal_name not in trust_contributions:
            trust_contributions[signal_name] = 0.5
            missing_signals.append(signal_name)
    
    raw = sum(WEIGHTS[k] * trust_contributions[k] for k in WEIGHTS)
    score = int(raw * 100)
    
    if score >= 80:
        band = "Trusted"
    elif score >= 50:
        band = "Suspicious"
    else:
        band = "Potential Fraud"
    
    return {
        "trust_score": score,
        "band": band,
        "raw_score": raw,
        "missing_signals": missing_signals,
        "trust_contributions": trust_contributions.copy()
    }


# ============================================================
# EXPLAINABILITY
# ============================================================

def generate_reason(trust_contributions: dict) -> str:
    """Finds the lowest signals and generates a human-readable reason."""
    sorted_signals = sorted(trust_contributions.items(), key=lambda x: x[1])
    worst = sorted_signals[0]
    second_worst = sorted_signals[1]
    
    if worst[1] > 0.6:
        return "All signals within normal range."
    
    display_names = {
        "identity": "identity match",
        "liveness": "liveness challenge",
        "visual": "visual deepfake risk",
        "audio": "audio spoof risk",
        "injection": "injection risk"
    }
    
    return (
        f"Score reduced primarily due to {display_names.get(worst[0], worst[0])} "
        f"({worst[1]:.2f}) and {display_names.get(second_worst[0], second_worst[0])} "
        f"({second_worst[1]:.2f})."
    )


# ============================================================
# CONFLICT DETECTION
# ============================================================

def detect_conflicts(trust_contributions: dict) -> dict:
    """Detects suspicious signal combinations."""
    conflicts = []
    
    if (trust_contributions.get("identity", 0) > 0.7 and 
        trust_contributions.get("visual", 1) < 0.3):
        conflicts.append("identity_vs_visual_deepfake")
    
    if (trust_contributions.get("liveness", 0) > 0.8 and 
        trust_contributions.get("injection", 1) < 0.3):
        conflicts.append("liveness_vs_injection")
    
    return {
        "conflict_detected": len(conflicts) > 0,
        "conflict_types": conflicts
    }


# ============================================================
# EMA SMOOTHING
# ============================================================

class TrustScoreEMA:
    """Exponential Moving Average smoothing for trust score."""
    
    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self.ema_score = None
    
    def update(self, new_score: int) -> int:
        if self.ema_score is None:
            self.ema_score = float(new_score)
        else:
            self.ema_score = self.alpha * new_score + (1 - self.alpha) * self.ema_score
        return int(self.ema_score)
    
    def reset(self):
        self.ema_score = None


# ============================================================
# MAIN FUSION ENGINE
# ============================================================

class TrustScoreFusionEngine:
    """
    DeepfakeGuard M3 - Trust Score Fusion Engine
    
    Combines 5 signals: identity, liveness, visual, audio, injection
    Output: trust_score, band, reason, conflict_detected, sub_scores
    """
    
    def __init__(self, alpha: float = 0.3):
        self.ema = TrustScoreEMA(alpha=alpha)
        self.last_signal_time = {}
        self.current_contributions = {}
    
    def process_signals(self, signals: dict, timestamp: float = None) -> dict:
        if timestamp is None:
            timestamp = datetime.now().timestamp()
        
        trust_contributions = {}
        signal_missing = []
        
        for signal_name in WEIGHTS:
            if signal_name in signals:
                try:
                    trust_contributions[signal_name] = normalize_signal(signal_name, signals[signal_name])
                    self.last_signal_time[signal_name] = timestamp
                except Exception:
                    signal_missing.append(signal_name)
                    trust_contributions[signal_name] = 0.5
            else:
                last_time = self.last_signal_time.get(signal_name, 0)
                if timestamp - last_time > SIGNAL_TIMEOUT_SECONDS:
                    signal_missing.append(signal_name)
                    trust_contributions[signal_name] = 0.5
                else:
                    trust_contributions[signal_name] = self.current_contributions.get(signal_name, 0.5)
        
        self.current_contributions = trust_contributions.copy()
        
        result = compute_trust_score(trust_contributions)
        conflicts = detect_conflicts(trust_contributions)
        
        # ============================================================
        # CONFLICT PENALTY
        # ============================================================
        max_penalty = 0.0
        if conflicts["conflict_detected"] and conflicts["conflict_types"]:
            max_penalty = max(CONFLICT_PENALTIES.get(c, 0.0) for c in conflicts["conflict_types"])
            penalized_score = int(result["trust_score"] * (1 - max_penalty))
            result["trust_score"] = penalized_score
            if penalized_score >= 80:
                result["band"] = "Trusted"
            elif penalized_score >= 50:
                result["band"] = "Suspicious"
            else:
                result["band"] = "Potential Fraud"
        
        # ============================================================
        # SECURITY HARDENING: Identity threshold
        # ============================================================
        identity_capped = False
        if trust_contributions["identity"] < IDENTITY_TRUST_THRESHOLD and result["band"] == "Trusted":
            result["trust_score"] = 79
            result["band"] = "Suspicious"
            identity_capped = True
        
        ema_score = self.ema.update(result["trust_score"])
        reason = generate_reason(trust_contributions)
        
        if identity_capped:
            reason += " [Identity match below threshold — capped at Suspicious]"
        
        if conflicts["conflict_detected"]:
            reason += f" [Conflict penalty: -{int(max_penalty * 100)}% applied]"
        
        if signal_missing:
            reason += f" [Missing: {', '.join(signal_missing)}]"
        
        return {
            "trust_score": result["trust_score"],
            "ema_trust_score": ema_score,
            "band": result["band"],
            "reason": reason,
            "conflict_detected": conflicts["conflict_detected"],
            "conflict_types": conflicts["conflict_types"],
            "sub_scores": {
                "identity_trust": trust_contributions["identity"],
                "liveness_trust": trust_contributions["liveness"],
                "visual_trust": trust_contributions["visual"],
                "audio_trust": trust_contributions["audio"],
                "injection_trust": trust_contributions["injection"]
            },
            "signal_missing": signal_missing,
            "weights_used": WEIGHTS,
            "timestamp": timestamp
        }
    
    def reset(self):
        """Resets the engine for a new session."""
        self.ema.reset()
        self.last_signal_time = {}
        self.current_contributions = {}


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    # Test 1: All signals perfect
    engine = TrustScoreFusionEngine()
    result = engine.process_signals({
        "identity": {"similarity_score": 1.0},
        "liveness": {"challenge_passed": True, "response_time_ms": 2000},
        "visual": {"deepfake_probability": 0.0},
        "audio": {"spoof_probability": 0.0},
        "injection": {"frame_timing_anomaly_score": 0.0, "virtual_camera_detected": False}
    })
    assert result["trust_score"] == 100
    assert result["band"] == "Trusted"
    print("✅ Test 1 passed: All signals perfect -> Trust=100")
    
    # Test 2: Visual and Audio bad
    engine.reset()
    result = engine.process_signals({
        "identity": {"similarity_score": 1.0},
        "liveness": {"challenge_passed": True, "response_time_ms": 2000},
        "visual": {"deepfake_probability": 1.0},
        "audio": {"spoof_probability": 1.0},
        "injection": {"frame_timing_anomaly_score": 0.0, "virtual_camera_detected": False}
    })
    assert result["trust_score"] == 54  # 60 - 10% conflict penalty
    assert result["band"] == "Suspicious"
    print("✅ Test 2 passed: Visual+Audio bad -> Trust=54 (Conflict penalty applied)")
    
    # Test 3: Conflict detection
    engine.reset()
    result = engine.process_signals({
        "identity": {"similarity_score": 0.85},
        "liveness": {"challenge_passed": True, "response_time_ms": 2000},
        "visual": {"deepfake_probability": 0.9},
        "audio": {"spoof_probability": 0.1},
        "injection": {"frame_timing_anomaly_score": 0.0, "virtual_camera_detected": False}
    })
    assert result["conflict_detected"] == True
    assert "identity_vs_visual_deepfake" in result["conflict_types"]
    print("✅ Test 3 passed: Conflict detection working")
    
    # Test 4: Identity threshold hardening
    engine.reset()
    result = engine.process_signals({
        "identity": {"similarity_score": 0.50},  # Below threshold
        "liveness": {"challenge_passed": True, "response_time_ms": 2000},
        "visual": {"deepfake_probability": 0.0},
        "audio": {"spoof_probability": 0.0},
        "injection": {"frame_timing_anomaly_score": 0.0, "virtual_camera_detected": False}
    })
    assert result["trust_score"] == 79
    assert result["band"] == "Suspicious"
    assert "capped at Suspicious" in result["reason"]
    print("✅ Test 4 passed: Identity threshold hardening working")
    
    print("\n🎉 All tests passed successfully!")