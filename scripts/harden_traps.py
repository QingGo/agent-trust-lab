#!/usr/bin/env python3
"""Harden low-discrimination traps from a comparison evaluation.

Reads a comparison.json from a multi-model evaluation, identifies traps
where all models score similarly well (high ceiling, near-zero spread),
and uses LLM rewriting to increase difficulty.

Usage:
    python scripts/harden_traps.py results/cmp_3models/comparison.json
    python scripts/harden_traps.py results/cmp_3models/comparison.json --dry-run
    python scripts/harden_traps.py results/cmp_3models/comparison.json --intensity aggressive
"""

import os
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from agent_trust_lab.redteam.hardener import HardenerConfig, TrapHardener  # noqa: E402


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Harden low-discrimination traps from a comparison evaluation"
    )
    parser.add_argument(
        "comparison_path",
        help="Path to comparison.json from multi-model evaluation",
    )
    parser.add_argument(
        "--trap-library",
        default=str(_PROJECT_ROOT / "src" / "agent_trust_lab" / "traps" / "data"),
        help="Path to trap YAML library (default: src/agent_trust_lab/traps/data)",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory for hardened YAMLs (default: overwrite originals)",
    )
    parser.add_argument(
        "--intensity",
        default="moderate",
        choices=["light", "moderate", "aggressive"],
        help="Hardening intensity (default: moderate)",
    )
    parser.add_argument(
        "--max-spread",
        type=float,
        default=0.05,
        help="Max inter-model trust spread to consider hardenable (default: 0.05)",
    )
    parser.add_argument(
        "--min-max-trust",
        type=float,
        default=0.90,
        help="Min max trust (ceiling) to consider hardenable (default: 0.90)",
    )
    parser.add_argument(
        "--model",
        default="deepseek-v4-flash",
        help="LLM model for hardening (default: deepseek-v4-flash)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without writing files",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip creating .bak backups of original files",
    )

    args = parser.parse_args()

    config = HardenerConfig(
        trap_library_path=args.trap_library,
        output_dir=args.output_dir,
        model=args.model,
        intensity=args.intensity,
        backup_originals=not args.no_backup,
        dry_run=args.dry_run,
    )

    hardener = TrapHardener(config)
    hardened = hardener.harden_from_comparison(
        comparison_path=args.comparison_path,
        max_spread=args.max_spread,
        min_max_trust=args.min_max_trust,
    )

    written = 0
    skipped = 0
    for h in hardened:
        tid = h.get("trap_id", "?")
        tt = h.get("trap_type", "?")
        dif = h.get("difficulty", "?")
        trap_file = os.path.join(
            args.trap_library, "general", f"{tid}.yaml"
        )
        if os.path.exists(trap_file + ".bak"):
            print(f"  SKIP {tid} ({tt}) -- already hardened (.bak exists)")
            skipped += 1
            continue
        print(f"  {tid} ({tt}) → difficulty={dif}")
        hardener.write_hardened(h)
        written += 1

    print(f"\nHardened: {written} new, Skipped: {skipped} already done")


if __name__ == "__main__":
    main()
