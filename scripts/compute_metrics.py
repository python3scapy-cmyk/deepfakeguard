#!/usr/bin/env python3
"""
scripts/compute_metrics.py

Reads evaluation/sessions.csv (written by record_session.py), pulls the
one "summary" row per session_label, and computes APCER/BPCER plus a
per-attack-type breakdown. Generates the exact table that belongs in
security/EVALUATION.md's real-evaluation section -- so that table is
produced from recorded data instead of retyped by hand (which is how you
get a report that doesn't match what actually happened).

Uses final_band_raw_majority by default. Pass --band-source ema to use
final_band_ema instead. Whichever you pick, use the SAME one for every
session in the run -- see the note in record_session.py about why these
two can disagree.

Usage:
    python scripts/compute_metrics.py evaluation/sessions.csv
    python scripts/compute_metrics.py evaluation/sessions.csv --band-source ema
"""
import argparse
import csv
import sys
from collections import defaultdict

ATTACK_TYPES = ["video_replay", "cloned_voice", "virtual_camera",
                "wrong_person", "deepfake_video"]


def classify(label):
    if label.startswith("bonafide"):
        return "bonafide", None
    for t in ATTACK_TYPES:
        if label.startswith(f"attack_{t}"):
            return "attack", t
    return "unknown", None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv_path", nargs="?", default="evaluation/sessions.csv")
    ap.add_argument("--band-source", choices=["raw_majority", "ema"], default="raw_majority")
    args = ap.parse_args()

    band_field = ("final_band_raw_majority" if args.band_source == "raw_majority"
                  else "final_band_ema")

    summaries = {}
    disagreements = []
    with open(args.csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["row_type"] != "summary":
                continue
            summaries[row["session_label"]] = row  # last write wins if a session was rerun
            if row.get("band_disagreement") == "True":
                disagreements.append(row["session_label"])

    bonafide, attacks = [], defaultdict(list)
    unrecognized = []
    for label, row in summaries.items():
        kind, atype = classify(label)
        if kind == "bonafide":
            bonafide.append(row)
        elif kind == "attack":
            attacks[atype].append(row)
        else:
            unrecognized.append(label)

    if unrecognized:
        print(f"[WARN] unrecognized session_label(s), skipped: {unrecognized}\n"
              f"       expected 'bonafide_N' or 'attack_<type>_N'", file=sys.stderr)

    n_bona = len(bonafide)
    n_bona_fraud = sum(1 for r in bonafide if r[band_field] == "fraud")
    bpcer = (n_bona_fraud / n_bona * 100) if n_bona else float("nan")

    all_attacks = [r for rows in attacks.values() for r in rows]
    n_attack = len(all_attacks)
    n_attack_trusted = sum(1 for r in all_attacks if r[band_field] == "trusted")
    apcer = (n_attack_trusted / n_attack * 100) if n_attack else float("nan")

    print(f"Band source: {band_field}")
    print(f"Sessions found: {n_bona} bonafide, {n_attack} attack "
          f"({', '.join(f'{t}:{len(r)}' for t, r in attacks.items()) or 'none'})\n")

    print(f"BPCER = {n_bona_fraud}/{n_bona} bonafide sessions ended 'fraud'   = {bpcer:.1f}%")
    print(f"APCER = {n_attack_trusted}/{n_attack} attack sessions ended 'trusted' = {apcer:.1f}%\n")

    print(f"{'Attack type':16s} {'trusted':>8s} {'suspicious':>11s} {'fraud':>7s} {'n':>4s}")
    for t in ATTACK_TYPES:
        rows = attacks.get(t, [])
        c = {"trusted": 0, "suspicious": 0, "fraud": 0}
        for r in rows:
            c[r[band_field]] = c.get(r[band_field], 0) + 1
        n = len(rows)
        print(f"{t:16s} {c.get('trusted',0):>8d} {c.get('suspicious',0):>11d} "
              f"{c.get('fraud',0):>7d} {n:>4d}")

    if disagreements:
        print(f"\n[WARN] {len(disagreements)} session(s) where the raw-majority "
              f"band and EMA-derived band disagreed: {disagreements}")

    if n_bona != 15 or n_attack != 15:
        print(f"\n[WARN] Roadmap protocol requires 15 bonafide + 15 attack "
              f"(3 per type, 5 types). Currently {n_bona} bonafide / {n_attack} attack. "
              f"Do not publish these as the final evaluation numbers until the "
              f"full protocol is recorded.")


if __name__ == "__main__":
    main()