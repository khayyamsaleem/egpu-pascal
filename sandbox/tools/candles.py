#!/usr/bin/env python3
"""Draw a candlestick chart from OHLCV CSV -- stdlib + matplotlib, no other dependencies.

The sandbox mounts this repo read-only at /repo, so a model that needs a real candlestick chart can
just run this rather than hand-rolling one (which small models get wrong: they tend to plot open and
close as two lines and call it a candlestick).

    python3 /repo/sandbox/tools/candles.py --out /work/DOCN.png --title "DOCN, last 7 days" < data.csv
    python3 /repo/sandbox/tools/candles.py --csv /work/data.csv --out /work/DOCN.png

Input: CSV with a header, columns date,open,high,low,close[,volume] (what `stock_history` returns).
Output: a PNG (also printed to stdout as the path, so it shows up in a tool result).
"""
import argparse
import csv
import re
import sys
from datetime import datetime


def _rows_from(lines, positional):
    out = []
    for r in (lines if positional else csv.DictReader(lines)):
        try:
            if positional:
                p = [x.strip() for x in r.split(",")]
                if len(p) < 5:
                    continue
                d, o, h, l, c, v = p[0], p[1], p[2], p[3], p[4], (p[5] if len(p) > 5 else 0)
            else:
                d = r.get("date") or r.get("Date")
                o, h, l, c, v = r["open"], r["high"], r["low"], r["close"], (r.get("volume") or 0)
            out.append(dict(d=str(d).strip(), o=float(o), h=float(h), l=float(l), c=float(c),
                            v=float(v or 0)))
        except (KeyError, TypeError, ValueError, AttributeError):
            continue          # a header row, a provenance line, or a gap: skip, don't die
    return out


def load(fileobj, symbol=None, n=None):
    """Accept the output of the `stock_history` tool as-is: it tolerates the banner line, a header
    row, or no header at all (columns are read by position then)."""
    raw = [l.strip() for l in fileobj if l.strip()]
    # the tool prefixes a provenance line, "[yahoo] DOCN, 7 daily rows ..." -- take the symbol from
    # it so the chart labels itself even when the caller passes no --title/--symbol
    found = None
    for l in raw:
        m = re.match(r"\[[a-z]+\]\s+([A-Za-z0-9._-]{1,12})[,:]", l)
        if m:
            found = m.group(1)
            break
    lines = [l for l in raw if not l.startswith("[")]
    if not lines:
        raise SystemExit("nothing on input")
    rows = _rows_from(lines, positional=False) or _rows_from(lines, positional=True)
    if n:
        rows = rows[-int(n):]
    if not rows:
        raise SystemExit("no OHLC rows found -- expected CSV columns date,open,high,low,close[,volume]")
    return rows, (symbol or found)


def draw(rows, out_path, title=None, symbol=None, style="candle"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    n = len(rows)
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * n), 4.6), dpi=110)
    width = 0.6
    for i, r in enumerate(rows):
        up = r["c"] >= r["o"]
        col = "#1a9850" if up else "#d73027"
        # wick: the full low-high range, then the body from open to close
        ax.plot([i, i], [r["l"], r["h"]], color=col, linewidth=1.1, zorder=2)
        lo, hi = min(r["o"], r["c"]), max(r["o"], r["c"])
        if style == "candle":
            ax.add_patch(Rectangle((i - width / 2, lo), width, max(hi - lo, 1e-9),
                                   facecolor=col, edgecolor=col, zorder=3))
        if n <= 20:           # with few days, annotate the close so the numbers are readable
            ax.annotate("%.2f" % r["c"], (i, r["h"]), textcoords="offset points", xytext=(0, 6),
                        ha="center", fontsize=7, color="#333333")
    ax.set_xticks(range(n))
    labels = []
    for r in rows:
        try:
            labels.append(datetime.strptime(r["d"], "%Y-%m-%d").strftime("%b %d"))
        except ValueError:
            labels.append(r["d"])
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("price")
    ax.set_title(title or ((("%s: " % symbol) if symbol else "") + "daily open/high/low/close" +
                           (", %d days" % len(rows)) + "  (green = close above open)"))
    ax.grid(True, axis="y", alpha=0.25, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser(description="OHLCV CSV -> candlestick PNG")
    ap.add_argument("csv_path", nargs="?", default=None,
                    help="input CSV (or use --csv, or stdin)")
    ap.add_argument("--csv", default=None, help="input CSV (default: the positional path, else stdin)")
    ap.add_argument("--out", default="/work/candles.png", help="output PNG path")
    ap.add_argument("--title", default=None)
    ap.add_argument("--symbol", default=None)
    ap.add_argument("--days", type=int, default=None, help="use only the last N rows")
    ap.add_argument("--style", default="candle", choices=["candle", "line"])
    a = ap.parse_args(argv)
    src = a.csv or a.csv_path
    rows, sym = load(open(src) if src else sys.stdin, a.symbol, a.days)
    path = draw(rows, a.out, a.title, sym, a.style)
    print("%d rows -> %s" % (len(rows), path))
    for r in rows:
        print("  %s open %.2f high %.2f low %.2f close %.2f" % (r["d"], r["o"], r["h"], r["l"], r["c"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
