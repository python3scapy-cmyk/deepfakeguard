#!/usr/bin/env python3
"""
scripts/record_session.py

Polls the DeepfakeGuard backend's GET /scores once per second for a fixed
duration and appends every reading, plus one summary row, to
evaluation/sessions.csv. This exists so nobody hand-transcribes trust
scores off the dashboard into the evaluation report -- that's how a repo
ends up with two contradictory APCER numbers in two different files.

╔══════════════════════════════════════════════════════════════════════╗
║ YOU MUST PASS --code FOR BROWSER SESSIONS.                            ║
║                                                                        ║
║ /scores without ?code= serves the LOCAL room (main.py's loopback      ║
║ pipeline). Browser participants live in CODED rooms. Recording        ║
║ without --code while the session runs in the browser produces a CSV   ║
║ full of signal_missing_count=5 rows -- an hour of recording, zero     ║
║ usable data. (This is exactly what happened to the first bonafide_01  ║
║ run.) This script now HARD-STOPS if every early poll shows all five   ║
║ signals missing, instead of quietly writing garbage.                  ║
╚══════════════════════════════════════════════════════════════════════╝

IMPORTANT SUBTLETY: /scores returns TWO different notions of "current
state":
  - trust_band        -- computed from the RAW (unsmoothed) trust score
                          every single poll. Can flicker frame to frame.
  - ema_trust_score    -- the EMA-smoothed score (alpha=0.3). This is what
                          the dashboard displays as the big number, but the
                          dashboard's *band* badge is driven by trust_band,
                          not by a band derived from the EMA score.
These two can disagree, especially right at a threshold crossing. This
script records BOTH:
  - final_band_raw_majority : majority vote of trust_band over the last
                               `--tail` polls (damps single-frame flicker)
  - final_band_ema          : band re-derived from the final ema_trust_score
                               using the same 80/50 thresholds
If they disagree, the script prints a warning. Decide up front which one
you're reporting as "the" final band in EVALUATION.md, and be consistent
across all 30 sessions -- don't switch definitions mid-protocol.

Usage:
    # Browser session in room K7M2Q9, auto-focused most recent client:
    python scripts/record_session.py --label bonafide_01 --code K7M2Q9 --duration 60

    # Pin a specific client in a multi-client room:
    python scripts/record_session.py --label attack_video_replay_02 \
        --code K7M2Q9 --session web_ab12 --duration 45

    # Local main.py pipeline (the ONLY case where omitting --code is right):
    python scripts/record_session.py --label bonafide_03 --local --duration 60

Label convention (required by scripts/compute_metrics.py):
    bonafide_<n>          genuine-user sessions
    attack_<type>_<n>     attack sessions, where <type> is one of:
                          video_replay, cloned_voice, virtual_camera,
                          wrong_person, deepfake_video

Workflow per session:
    1. POST /session-reset (or click "Reset Session" on the dashboard)
    2. Start the actual scenario (sit down, hold up the phone, start OBS, etc.)
    3. THEN start this script -- so the recording window covers real
       behavior, not idle "Awaiting signals" frames from before the
       scenario began.
"""
import argparse
import csv
import os
import sys
import time
import urllib.parse
from collections import Counter
from datetime import datetime, timezone

import requests

DEFAULT_BASE = "http://localhost:5000/scores"
DEFAULT_OUT = "evaluation/sessions.csv"
FIELDNAMES = [
    "session_label", "row_type", "timestamp",
    "trust_score", "ema_trust_score", "trust_band",
    "signal_missing_count",
    "final_band_raw_majority", "final_band_ema", "band_disagreement",
]

# If this many consecutive polls FROM THE START all report every signal
# missing, the recording is aimed at an empty room. Abort loudly.
DEAD_ROOM_POLL_LIMIT = 8
TOTAL_SIGNALS = 5


def build_url(base, code, session):
    """Append ?code= / &session= to the /scores URL."""
    params = {}
    if code:
        params["code"] = code.upper()
    if session:
        params["session"] = session
    if not params:
        return base
    sep = "&" if "?" in base else "?"
    return base + sep + urllib.parse.urlencode(params)


def poll_once(url):
    try:
        resp = requests.get(url, timeout=2.0)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[WARN] poll failed: {e}", file=sys.stderr)
        return None


def band_from_score(score, high=80, mid=50):
    """Re-derive a band from a numeric score using the SAME thresholds as
    backend/app.py's compute_trust() -- keep these in sync by hand if that
    function's thresholds ever change."""
    if score is None:
        return None
    if score >= high:
        return "trusted"
    elif score >= mid:
        return "suspicious"
    return "fraud"


def majority_band(bands, tail=5):
    bands = [b for b in bands if b is not None]
    if not bands:
        return None
    window = bands[-tail:]
    return Counter(window).most_common(1)[0][0]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--label", required=True,
                    help="e.g. bonafide_03 or attack_wrong_person_02")
    ap.add_argument("--code", default=None,
                    help="room code of the browser session, e.g. K7M2Q9. "
                         "REQUIRED for browser sessions (see banner above).")
    ap.add_argument("--session", default=None,
                    help="optional client session_id (e.g. web_ab12) to pin "
                         "when several clients share the room; otherwise the "
                         "room's most recently active client is recorded.")
    ap.add_argument("--local", action="store_true",
                    help="explicitly record the LOCAL room (main.py pipeline). "
                         "Without this flag, omitting --code is an error.")
    ap.add_argument("--duration", type=float, default=60.0, help="seconds to record")
    ap.add_argument("--interval", type=float, default=1.0, help="seconds between polls")
    ap.add_argument("--tail", type=int, default=5,
                    help="number of trailing polls used for the raw-band majority vote")
    ap.add_argument("--url", default=DEFAULT_BASE,
                    help="base /scores URL (code/session are appended for you)")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    if not args.code and not args.local:
        print("[ERROR] no --code given. Browser sessions live in CODED rooms; "
              "recording without --code polls the empty LOCAL room and writes "
              "junk (signal_missing_count=5 on every row). Pass --code XXXXXX, "
              "or pass --local if you really are recording main.py's pipeline.",
              file=sys.stderr)
        sys.exit(2)

    url = build_url(args.url, args.code, args.session)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    write_header = not os.path.exists(args.out)

    raw_bands = []
    rows = []
    missing_counts = []
    print(f"Recording '{args.label}' for {args.duration:.0f}s from {url} ...")
    start = time.time()
    last_data = None
    while time.time() - start < args.duration:
        data = poll_once(url)
        ts = datetime.now(timezone.utc).isoformat()
        if data is not None:
            last_data = data
            band = data.get("trust_band")
            trust = data.get("trust_score")
            ema = data.get("ema_trust_score")
            missing = len(data.get("signal_missing") or [])
            raw_bands.append(band)
            missing_counts.append(missing)
            rows.append({
                "session_label": args.label, "row_type": "sample",
                "timestamp": ts, "trust_score": trust,
                "ema_trust_score": ema, "trust_band": band,
                "signal_missing_count": missing,
                "final_band_raw_majority": "", "final_band_ema": "",
                "band_disagreement": "",
            })
            print(f"  {ts}  trust={trust}  ema={ema}  band={band}  missing={missing}")

            # Dead-room detection: N polls in, still zero signals -> the
            # recorder is pointed at the wrong room. Stop before an entire
            # session's worth of junk lands in the CSV.
            if (len(missing_counts) == DEAD_ROOM_POLL_LIMIT
                    and all(m >= TOTAL_SIGNALS for m in missing_counts)):
                print(f"\n[ERROR] first {DEAD_ROOM_POLL_LIMIT} polls all report "
                      f"every signal missing -- this room has no live data. "
                      f"Nothing was written to {args.out}.\n"
                      f"        Recording URL was: {url}\n"
                      f"        Check: is the participant connected and verifying? "
                      f"Is --code the code shown on the admin dashboard? "
                      f"(For main.py, use --local.)", file=sys.stderr)
                sys.exit(3)
        time.sleep(args.interval)

    fb_raw = majority_band(raw_bands, tail=args.tail)
    fb_ema = band_from_score(last_data.get("ema_trust_score")) if last_data else None
    disagree = bool(fb_raw and fb_ema and fb_raw != fb_ema)

    summary = {
        "session_label": args.label, "row_type": "summary",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trust_score": last_data.get("trust_score") if last_data else None,
        "ema_trust_score": last_data.get("ema_trust_score") if last_data else None,
        "trust_band": raw_bands[-1] if raw_bands else None,
        "signal_missing_count": len(last_data.get("signal_missing") or []) if last_data else None,
        "final_band_raw_majority": fb_raw,
        "final_band_ema": fb_ema,
        "band_disagreement": disagree,
    }
    rows.append(summary)

    with open(args.out, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    print(f"\nfinal_band_raw_majority (last {args.tail} polls): {fb_raw}")
    print(f"final_band_ema (from final EMA score):          {fb_ema}")
    if disagree:
        print("[WARN] raw-band majority and EMA-derived band DISAGREE for this "
              "session. Note it -- don't silently pick whichever one looks better.")

    # Data-quality verdict for this session, so a half-dead recording is
    # caught NOW instead of at compute_metrics time.
    if missing_counts:
        frac_all_missing = sum(1 for m in missing_counts if m >= TOTAL_SIGNALS) / len(missing_counts)
        if frac_all_missing > 0.5:
            print(f"[WARN] {frac_all_missing:.0%} of samples had ALL signals "
                  f"missing. compute_metrics.py will EXCLUDE this session. "
                  f"Re-record it.", file=sys.stderr)

    print(f"Appended {len(rows)} rows to {args.out}")

    if not raw_bands:
        print("[ERROR] No successful polls -- is backend/app.py running on "
              "localhost:5000? Nothing usable was recorded for this session.",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
