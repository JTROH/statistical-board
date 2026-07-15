"""CLI: `python -m stat_board "your question" --data file.csv`.

Runs the standalone statistical board (API-driven). The Claude Code skill
`/stat-board` does the same thing on your Claude Code session with no API key.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from . import config, orchestrator
from . import transcript as transcript_mod


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="stat_board",
        description="Adversarial statistical board: turn a dataset + question into a "
                    "vetted PDF report grounded in real computation.",
    )
    parser.add_argument("question", help="The statistical question, e.g. 'Do the groups differ?'")
    parser.add_argument("--data", required=True, help="Dataset (CSV or JSON).")
    parser.add_argument("--group-col", help="One-factor long-format: column of group labels.")
    parser.add_argument("--value-col", help="Outcome column (long-format group value, or the multi-factor outcome).")
    parser.add_argument("--factor", action="append", dest="factors",
                        help="Multi-factor: a categorical factor column (repeat for each).")
    parser.add_argument("--covariate", action="append", dest="covariates",
                        help="Multi-factor: a numeric covariate column (repeat for each).")
    parser.add_argument("--type", type=int, choices=[1, 2, 3], default=2, dest="typ",
                        help="Multi-factor: ANOVA SS type, fallback if the judge doesn't "
                             "restate one (default 2).")
    parser.add_argument("--paired", action="store_true", help="Paired / repeated-measures design.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level (default 0.05).")
    parser.add_argument("--rounds", type=int, default=config.MAX_ROUNDS,
                        help=f"Max judge-led rounds (default {config.MAX_ROUNDS}).")
    parser.add_argument("--out", default="reports", help="Output directory (default: reports/).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Stub every agent call — exercise the flow with no API calls.")
    args = parser.parse_args()

    if not args.dry_run and not config.credentials_present():
        print("No Anthropic credentials found. Set ANTHROPIC_API_KEY (or use --dry-run).",
              file=sys.stderr)
        return 1

    try:
        result = asyncio.run(orchestrator.run(
            args.question, args.data, group_col=args.group_col, value_col=args.value_col,
            factors=args.factors, covariates=args.covariates, typ=args.typ,
            paired=args.paired, alpha=args.alpha, max_rounds=args.rounds, dry_run=args.dry_run))
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130

    paths = transcript_mod.save(result, args.out)
    print(f"\n✓ Report (PDF): {paths['pdf']}")
    print(f"✓ Report (MD):  {paths['md']}")
    print(f"✓ Transcript:   {paths['json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
