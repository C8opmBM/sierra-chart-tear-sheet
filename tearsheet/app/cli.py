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
    p.add_argument("--risk-capital", type=float, default=None, metavar="AMOUNT",
                   help="Amount of capital actually at risk, when it's smaller than "
                        "--starting-balance / the account's displayed balance — e.g. a "
                        "prop-firm evaluation or funded account where the shown balance "
                        "is mostly a trailing-drawdown buffer, and the amount you can "
                        "actually lose before failing is much smaller. When set, Monte "
                        "Carlo's drawdown %% and ruin-probability figures are computed "
                        "against this amount instead of the account balance, while the "
                        "dollar-denominated equity curve is left untouched. Example: a "
                        "$50,000 Lucid account with a $2,000 trailing max drawdown -> "
                        "--starting-balance 50000 --risk-capital 2000.")
    return p


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    from tearsheet.app.main import run
    run(input_path, args.output, starting_balance=args.starting_balance, risk_capital=args.risk_capital)


if __name__ == "__main__":
    main()
