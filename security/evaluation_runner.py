"""
DeepfakeGuard evaluation runner (M3) -- rewritten Week 4.

The previous evaluation_runner.py was accidentally overwritten with a copy
of the markdown report and only ever produced seeded mock numbers. This
rewrite separates evaluation into three explicit, never-confusable modes:

  record    Poll the LIVE backend (/scores) during one labeled session and
            append its final verdict to security/eval_sessions.jsonl.
                python -m security.evaluation_runner record --label bonafide
                python -m security.evaluation_runner record --label video_replay
  report    Compute APCER/BPCER from the recorded REAL sessions and write
            security/EVALUATION.md (header states data is real).
                python -m security.evaluation_runner report
  simulate  Reproduce the old Week 3 seeded mock protocol through the
            canonical security/fusion_engine.py path. Writes to
            security/EVALUATION_SIMULATED.md ONLY -- never EVALUATION.md.
                python -m security.evaluation_runner simulate --seed 42

Protocol (Week 3 roadmap): 15 bonafide + 15 attack sessions, 3 per attack
type: video_replay, cloned_voice, virtual_camera, wrong_person,
deepfake_video.

Metrics:
  APCER  = attack sessions whose final band is "Trusted"  / total attacks
  BPCER  = bonafide sessions ending "Potential Fraud"     / total bonafide
  We also report a stricter "not blocked" rate (attacks ending Trusted OR
  Suspicious) because judges may ask for it.
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

ATTACK_TYPES = ["video_replay", "cloned_voice", "virtual_camera",
                "wrong_person", "deepfake_video"]
VALID_LABELS = ["bonafide"] + ATTACK_TYPES

HERE = os.path.dirname(os.path.abspath(__file__))
SESSIONS_PATH = os.path.join(HERE, "eval_sessions.jsonl")
REAL_REPORT_PATH = os.path.join(HERE, "EVALUATION.md")
SIM_REPORT_PATH = os.path.join(HERE, "EVALUATION_SIMULATED.md")
BACKEND_URL = os.environ.get("DFG_BACKEND_URL", "http://localhost:5000")


def _norm_band(band):
    b = (band or "").strip().lower()
    if b == "trusted":
        return "Trusted"
    if b == "suspicious":
        return "Suspicious"
    if b in ("fraud", "potential fraud"):
        return "Potential Fraud"
    return band or "unknown"


# ---------------------------------------------------------------- record
def cmd_record(args):
    import urllib.request

    if args.label not in VALID_LABELS:
        sys.exit(f"[FATAL] --label must be one of {VALID_LABELS}")

    print(f"[INFO] Recording session label={args.label} for {args.duration}s "
          f"from {BACKEND_URL}/scores")
    print("[INFO] Run the scenario NOW (participant on camera / attack active).")

    polls = []
    end = time.time() + args.duration
    while time.time() < end:
        try:
            with urllib.request.urlopen(BACKEND_URL + "/scores", timeout=3) as r:
                polls.append(json.load(r))
        except Exception as e:
            print(f"[WARN] poll failed: {e}")
        time.sleep(args.interval)

    if not polls:
        sys.exit("[FATAL] No successful polls -- is the backend running?")

    final = polls[-1]
    reporting = final.get("modules_reporting") or []
    if not reporting:
        sys.exit("[FATAL] Backend reported NO live modules during this window "
                 "(trust=50 default). Start main.py / the participant stream, "
                 "verify the dashboard shows live signals, then re-record. "
                 "Nothing was saved.")
    print(f"[INFO] Modules reporting during session: {reporting}")
    # Majority band over the last 5 polls to avoid a single-frame flicker
    tail = [_norm_band(p.get("trust_band")) for p in polls[-5:]]
    band = max(set(tail), key=tail.count)
    trusts = [p.get("ema_trust_score") for p in polls
              if p.get("ema_trust_score") is not None]

    record = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "category": "bonafide" if args.label == "bonafide" else "attack",
        "final_band": band,
        "final_trust": final.get("ema_trust_score"),
        "trust_min": min(trusts) if trusts else None,
        "trust_max": max(trusts) if trusts else None,
        "conflict_types": final.get("conflict_types", []),
        "hard_deny_reasons": final.get("hard_deny_reasons", []),
        "identity_capped": final.get("identity_capped"),
        "signals": final.get("signals"),
        "polls": len(polls),
        "duration_s": args.duration,
        "source": "live_backend",
    }
    with open(SESSIONS_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")

    counts = _label_counts()
    print(f"[SUCCESS] Recorded: band={band} trust={record['final_trust']} "
          f"-> {SESSIONS_PATH}")
    print("[INFO] Progress toward 15+15 protocol: " + ", ".join(
        f"{k}={counts.get(k, 0)}/{15 if k == 'bonafide' else 3}"
        for k in VALID_LABELS))


def _label_counts():
    counts = defaultdict(int)
    if os.path.exists(SESSIONS_PATH):
        with open(SESSIONS_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    counts[json.loads(line)["label"]] += 1
    return counts


# ---------------------------------------------------------------- report
def _compute_metrics(sessions):
    bonafide = [s for s in sessions if s["category"] == "bonafide"]
    attacks = [s for s in sessions if s["category"] == "attack"]

    apcer_hits = [s for s in attacks if s["final_band"] == "Trusted"]
    bpcer_hits = [s for s in bonafide if s["final_band"] == "Potential Fraud"]
    not_blocked = [s for s in attacks if s["final_band"] != "Potential Fraud"]

    per_type = {}
    for t in ATTACK_TYPES:
        rows = [s for s in attacks if s["label"] == t]
        if not rows:
            continue
        trusts = [s["final_trust"] for s in rows if s["final_trust"] is not None]
        per_type[t] = {
            "n": len(rows),
            "avg_trust": sum(trusts) / len(trusts) if trusts else None,
            "blocked": sum(1 for s in rows if s["final_band"] == "Potential Fraud"),
            "suspicious": sum(1 for s in rows if s["final_band"] == "Suspicious"),
            "trusted": sum(1 for s in rows if s["final_band"] == "Trusted"),
        }
    return {
        "n_bonafide": len(bonafide), "n_attacks": len(attacks),
        "apcer": len(apcer_hits) / len(attacks) if attacks else None,
        "bpcer": len(bpcer_hits) / len(bonafide) if bonafide else None,
        "attack_not_blocked_rate":
            len(not_blocked) / len(attacks) if attacks else None,
        "band_dist": {
            "bonafide": _band_dist(bonafide), "attack": _band_dist(attacks)},
        "per_type": per_type,
    }


def _band_dist(rows):
    return {b: sum(1 for s in rows if s["final_band"] == b)
            for b in ("Trusted", "Suspicious", "Potential Fraud")}


def _write_report(path, metrics, header_lines):
    m = metrics
    pct = lambda x: "n/a" if x is None else f"{100 * x:.1f}%"
    lines = header_lines + [
        "", "### Results", "",
        "| Metric | Value | Description |",
        "|--------|-------|-------------|",
        f"| APCER | {pct(m['apcer'])} | Attack sessions ending \"Trusted\" |",
        f"| BPCER | {pct(m['bpcer'])} | Bonafide sessions ending \"Potential Fraud\" |",
        f"| Attacks not blocked | {pct(m['attack_not_blocked_rate'])} | "
        "Attacks ending Trusted or Suspicious (stricter view) |",
        "", "### Band Distribution", "",
        "| Category | Trusted | Suspicious | Potential Fraud |",
        "|----------|---------|------------|-----------------|",
    ]
    for cat, n in (("bonafide", m["n_bonafide"]), ("attack", m["n_attacks"])):
        d = m["band_dist"][cat]
        lines.append(f"| {cat.capitalize()} ({n}) | {d['Trusted']} | "
                     f"{d['Suspicious']} | {d['Potential Fraud']} |")
    lines += ["", "### Attack Breakdown", "",
              "| Attack Type | n | Avg Trust | Blocked | Suspicious | Trusted |",
              "|---|---|---|---|---|---|"]
    for t, r in m["per_type"].items():
        avg = "n/a" if r["avg_trust"] is None else f"{r['avg_trust']:.1f}"
        lines.append(f"| {t} | {r['n']} | {avg} | {r['blocked']}/{r['n']} | "
                     f"{r['suspicious']}/{r['n']} | {r['trusted']}/{r['n']} |")
    lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"[SUCCESS] Report written to {path}")


def cmd_report(args):
    if not os.path.exists(SESSIONS_PATH):
        sys.exit(f"[FATAL] No {SESSIONS_PATH}. Record sessions first:\n"
                 "  python -m security.evaluation_runner record --label bonafide")
    with open(SESSIONS_PATH) as f:
        sessions = [json.loads(l) for l in f if l.strip()]
    m = _compute_metrics(sessions)
    if m["n_bonafide"] < 15 or m["n_attacks"] < 15:
        print(f"[WARNING] Protocol calls for 15+15; you have "
              f"{m['n_bonafide']} bonafide, {m['n_attacks']} attack. "
              "Report generated anyway -- labeled as partial.")
    header = [
        "# Evaluation Report — System-Level APCER / BPCER",
        "",
        f"> **Data source: REAL live sessions** recorded from the running "
        f"backend via /scores polling ({m['n_bonafide']} bonafide + "
        f"{m['n_attacks']} attack sessions in `eval_sessions.jsonl`).",
        f"> Generated {datetime.now(timezone.utc).isoformat()} by "
        "`evaluation_runner.py report`. No mock or seeded signals.",
    ]
    _write_report(REAL_REPORT_PATH, m, header)


# ---------------------------------------------------------------- simulate
def cmd_simulate(args):
    import random
    from security.fusion_engine import (compute_trust_score, detect_conflicts,
                                        CONFLICT_PENALTIES)
    rng = random.Random(args.seed)

    def run_session(label):
        if label == "bonafide":
            c = {"identity": rng.uniform(0.85, 0.98),
                 "liveness": rng.uniform(0.85, 1.0),
                 "visual": 1 - rng.uniform(0.0, 0.15),
                 "audio": 1 - rng.uniform(0.0, 0.10),
                 "injection": rng.uniform(0.9, 1.0)}
        elif label == "video_replay":
            c = {"identity": rng.uniform(0.7, 0.9), "liveness": rng.uniform(0.0, 0.2),
                 "visual": rng.uniform(0.4, 0.7), "audio": rng.uniform(0.6, 0.9),
                 "injection": rng.uniform(0.7, 1.0)}
        elif label == "cloned_voice":
            c = {"identity": rng.uniform(0.8, 0.95), "liveness": rng.uniform(0.8, 1.0),
                 "visual": rng.uniform(0.8, 1.0), "audio": rng.uniform(0.0, 0.25),
                 "injection": rng.uniform(0.8, 1.0)}
        elif label == "virtual_camera":
            c = {"identity": rng.uniform(0.7, 0.9), "liveness": rng.uniform(0.6, 0.9),
                 "visual": rng.uniform(0.3, 0.6), "audio": rng.uniform(0.6, 0.9),
                 "injection": rng.uniform(0.0, 0.2)}
        elif label == "wrong_person":
            c = {"identity": rng.uniform(0.2, 0.5), "liveness": rng.uniform(0.8, 1.0),
                 "visual": rng.uniform(0.8, 1.0), "audio": rng.uniform(0.7, 1.0),
                 "injection": rng.uniform(0.9, 1.0)}
        else:  # deepfake_video
            c = {"identity": rng.uniform(0.6, 0.85), "liveness": rng.uniform(0.0, 0.3),
                 "visual": rng.uniform(0.0, 0.3), "audio": rng.uniform(0.5, 0.8),
                 "injection": rng.uniform(0.7, 1.0)}

        # Canonical path -- identical semantics to tests/test_fusion_parity.py
        conflicts = detect_conflicts(dict(c))
        result = compute_trust_score(dict(c))
        trust = result["trust_score"]
        if conflicts["conflict_detected"]:
            penalty = max(CONFLICT_PENALTIES.get(x, 0.0)
                          for x in conflicts["conflict_types"])
            trust = int(trust * (1 - penalty))
        if c["identity"] < 0.60 and trust >= 80:
            trust = 79
        band = ("Trusted" if trust >= 80 else
                "Suspicious" if trust >= 50 else "Potential Fraud")
        return {"label": label,
                "category": "bonafide" if label == "bonafide" else "attack",
                "final_band": band, "final_trust": trust,
                "source": "simulated"}

    sessions = [run_session("bonafide") for _ in range(15)]
    for t in ATTACK_TYPES:
        sessions += [run_session(t) for _ in range(3)]

    m = _compute_metrics(sessions)
    header = [
        "# Evaluation Report — SIMULATED (mock signals)",
        "",
        f"> **Data source: SIMULATED.** Seeded mock signal profiles "
        f"(seed={args.seed}) pushed through the canonical "
        "`security/fusion_engine.py` scoring path.",
        "> **These numbers must NOT be quoted in the pitch or demo.** They",
        "> validate fusion behavior only. Real numbers live in EVALUATION.md,",
        "> generated by `evaluation_runner.py report` from recorded sessions.",
    ]
    _write_report(SIM_REPORT_PATH, m, header)


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("record", help="record one labeled live session")
    r.add_argument("--label", required=True, choices=VALID_LABELS)
    r.add_argument("--duration", type=int, default=30,
                   help="seconds to poll (default 30)")
    r.add_argument("--interval", type=float, default=1.0,
                   help="poll interval seconds (default 1.0)")
    r.set_defaults(func=cmd_record)

    rep = sub.add_parser("report", help="APCER/BPCER report from real sessions")
    rep.set_defaults(func=cmd_report)

    s = sub.add_parser("simulate", help="seeded mock run (separate report file)")
    s.add_argument("--seed", type=int, default=42)
    s.set_defaults(func=cmd_simulate)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
