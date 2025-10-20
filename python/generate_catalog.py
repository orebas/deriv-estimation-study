#!/usr/bin/env python3
"""
Generate a compact catalog summary from minimal_pilot_results.csv

Usage:
  python generate_catalog.py <input_csv> <output_csv> [max_order]

Outputs columns:
  method,language,works,first_deriv_pass,max_order_pass,rmse_o0,rmse_o1,...
"""

import sys
import csv
import math
from collections import defaultdict


def parse_float(x: str) -> float:
    try:
        v = float(x)
        return v
    except Exception:
        return float("nan")


def main():
    if len(sys.argv) < 3:
        print("Usage: python generate_catalog.py <input_csv> <output_csv> [max_order]")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    max_order = 4
    if len(sys.argv) >= 4:
        try:
            max_order = int(sys.argv[3])
        except Exception:
            pass

    # Read rows
    rows = []
    with open(input_csv, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    # Aggregate RMSE by method and order
    method_lang = {}
    rmse_by_method = defaultdict(dict)  # method -> order -> rmse
    for r in rows:
        method = r.get("method", "").strip()
        lang = r.get("language", "").strip()
        order = r.get("deriv_order", "")
        if method == "" or order == "":
            continue
        try:
            k = int(order)
        except Exception:
            continue
        rmse = parse_float(r.get("rmse", "nan"))
        if not math.isfinite(rmse):
            continue
        method_lang[method] = lang
        rmse_by_method[method][k] = rmse

    FIRST_THRESH = 1.0
    ORDER_THRESH = 100.0

    # Write summary
    orders = list(range(0, max_order + 1))
    fieldnames = [
        "method",
        "language",
        "works",
        "first_deriv_pass",
        "max_order_pass",
    ] + [f"rmse_o{k}" for k in orders]

    with open(output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for method in sorted(rmse_by_method.keys()):
            data = rmse_by_method[method]
            lang = method_lang.get(method, "")
            rmse0 = data.get(0, float("nan"))
            works = math.isfinite(rmse0)
            rmse1 = data.get(1, float("nan"))
            first_pass = math.isfinite(rmse1) and (rmse1 < FIRST_THRESH)
            max_pass = -1
            for k in orders:
                rk = data.get(k, float("nan"))
                if math.isfinite(rk) and rk < ORDER_THRESH:
                    max_pass = k
            out = {
                "method": method,
                "language": lang,
                "works": str(bool(works)).lower(),
                "first_deriv_pass": str(bool(first_pass)).lower(),
                "max_order_pass": max_pass,
            }
            for k in orders:
                out[f"rmse_o{k}"] = data.get(k, "")
            w.writerow(out)

    print(f"Wrote summary to {output_csv}")


if __name__ == "__main__":
    main()


