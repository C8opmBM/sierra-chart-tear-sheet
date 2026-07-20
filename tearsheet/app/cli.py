"""Command-line interface for tearsheet."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tearsheet",
        description="Generate a trade tear sheet from a Sierra Chart TradeActivityLog file.",
    )
    p.add_argument("--input", required=True, metavar="FILE",
                   help="Path to TradeActivityLog_*.txt")
    p.add_argument("--output", default="report.html", metavar="FILE",
                   help="Output HTML path (default: report.html)")
    p.add_argument("--starting-balance", type=float, default=None, metavar="AMOUNT",
                   help="Account balance before the first trade. Only used when the "
                        "log has no Account Balance rows at all, in which case the "
                        "equity curve is reconstructed from realized trade P&L "
                        "instead (some prop-firm eval/funded accounts never post "
                        "Account Balance activity). Ignored otherwise. If omitted "
                        "in that fallback case, the curve starts at 0 (relative "
                        "cumulative P&L, not an absolute balance).")
    return p


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    from tearsheet.app.main import run
    run(input_path, args.output, starting_balance=args.starting_balance)


if __name__ == "__main__":
    main()
