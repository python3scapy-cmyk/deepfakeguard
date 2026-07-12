"""
M3 Week 3 Test Script
Tests 4 attack scenarios.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from security.fusion_engine import TrustScoreFusionEngine

def print_result(result, expected_band):
    print(f"\n   📊 Result:")
    print(f"      Trust Score: {result['trust_score']}")
    print(f"      EMA Score: {result['ema_trust_score']}")
    print(f"      Band: {result['band']}")
    print(f"      Expected: {expected_band}")
    print(f"      Status: {'✅ PASS' if result['band'] == expected_band else '❌ FAIL'}")
    print(f"      Conflict: {result['conflict_detected']}")
    if result['conflict_types']:
        print(f"      Conflict Types: {result['conflict_types']}")
    print(f"      Reason: {result['reason'][:100]}...")
    return result['band'] == expected_band

def scenario_1_happy_path():
    print("\n" + "="*60)
    print("🟢 SCENARIO 1: HAPPY PATH (Normal User)")
    print("="*60)
    print("   Description: Real person, all signals clean")
    print("   Expected: Trust ≥ 80, Band = Trusted")
    
    engine = TrustScoreFusionEngine()
    result = engine.process_signals({
        "identity": {"similarity_score": 0.92},
        "liveness": {"challenge_passed": True, "response_time_ms": 2500},
        "visual": {"deepfake_probability": 0.05},
        "audio": {"spoof_probability": 0.02},
        "injection": {"frame_timing_anomaly_score": 0.0, "virtual_camera_detected": False}
    })
    return print_result(result, "Trusted")

def scenario_2_video_replay():
    print("\n" + "="*60)
    print("🔴 SCENARIO 2: VIDEO REPLAY ATTACK")
    print("="*60)
    print("   Description: Video played on phone, liveness fails")
    print("   Expected: Trust < 50, Band = Potential Fraud")
    
    engine = TrustScoreFusionEngine()
    result = engine.process_signals({
        "identity": {"similarity_score": 0.88},
        "liveness": {"challenge_passed": False, "response_time_ms": 0},
        "visual": {"deepfake_probability": 0.85},
        "audio": {"spoof_probability": 0.10},
        "injection": {"frame_timing_anomaly_score": 0.1, "virtual_camera_detected": False}
    })
    return print_result(result, "Potential Fraud")

def scenario_3_cloned_voice():
    print("\n" + "="*60)
    print("🔴 SCENARIO 3: CLONED VOICE ATTACK")
    print("="*60)
    print("   Description: Cloned voice, visual normal")
    print("   Expected: Audio spoof rises, score drops")
    
    engine = TrustScoreFusionEngine()
    result = engine.process_signals({
        "identity": {"similarity_score": 0.90},
        "liveness": {"challenge_passed": True, "response_time_ms": 3000},
        "visual": {"deepfake_probability": 0.08},
        "audio": {"spoof_probability": 0.92},
        "injection": {"frame_timing_anomaly_score": 0.0, "virtual_camera_detected": False}
    })
    return print_result(result, "Suspicious")

def scenario_4_virtual_camera():
    print("\n" + "="*60)
    print("🔴 SCENARIO 4: VIRTUAL CAMERA ATTACK")
    print("="*60)
    print("   Description: Fake stream via OBS Virtual Camera")
    print("   Expected: Injection risk rises, conflict alert")
    
    engine = TrustScoreFusionEngine()
    result = engine.process_signals({
        "identity": {"similarity_score": 0.85},
        "liveness": {"challenge_passed": True, "response_time_ms": 4000},
        "visual": {"deepfake_probability": 0.75},
        "audio": {"spoof_probability": 0.15},
        "injection": {"frame_timing_anomaly_score": 0.8, "virtual_camera_detected": True}
    })
    return print_result(result, "Potential Fraud")

def test_ema_smoothing():
    print("\n" + "="*60)
    print("📊 EMA SMOOTHING TEST")
    print("="*60)
    print("   Scenario: Sudden drop from ~97 to ~31")
    print("   Roadmap spec: alpha=0.3, should take ~3 cycles to cross below 50")
    
    engine = TrustScoreFusionEngine(alpha=0.3)
    
    # First 3 updates: normal user
    for _ in range(3):
        result = engine.process_signals({
            "identity": {"similarity_score": 0.95},
            "liveness": {"challenge_passed": True, "response_time_ms": 2000},
            "visual": {"deepfake_probability": 0.05},
            "audio": {"spoof_probability": 0.02},
            "injection": {"frame_timing_anomaly_score": 0.0, "virtual_camera_detected": False}
        })
    
    print(f"   Initial EMA: {result['ema_trust_score']}")
    
    # Then attack starts
    for i in range(5):
        result = engine.process_signals({
            "identity": {"similarity_score": 0.85},
            "liveness": {"challenge_passed": False, "response_time_ms": 0},
            "visual": {"deepfake_probability": 0.9},
            "audio": {"spoof_probability": 0.85},
            "injection": {"frame_timing_anomaly_score": 0.5, "virtual_camera_detected": False}
        })
        print(f"   Loop {i+1}: Trust={result['trust_score']}, EMA={result['ema_trust_score']}, Band={result['band']}")
        if result['band'] == "Potential Fraud":
            print(f"   ✅ Dropped below 50! (Loop {i+1})")
            return True
    
    return False

def main():
    print("="*60)
    print("🛡️  M3 WEEK 3 - TRUST SCORE FUSION TEST")
    print("="*60)
    
    results = []
    results.append(("Happy Path", scenario_1_happy_path()))
    results.append(("Video Replay", scenario_2_video_replay()))
    results.append(("Cloned Voice", scenario_3_cloned_voice()))
    results.append(("Virtual Camera", scenario_4_virtual_camera()))
    results.append(("EMA Smoothing", test_ema_smoothing()))
    
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} - {name}")
    
    print(f"\n   Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! M3 Week 3 is ready!")
    else:
        print("\n⚠️ Some tests failed.")

if __name__ == "__main__":
    main()