from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from factor_gen.gm3.adapter import GM3DataAdapter
from factor_gen.qsk_v5 import make_panel, split_by_time, run


def main():
    p = argparse.ArgumentParser(description="factor_gen QuantSkills-style V5")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--index", default="SHSE.000300")
    p.add_argument("--frequency", default="1d")
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    adapter = GM3DataAdapter(token=cfg.get("gm_token"))
    symbols = cfg.get("symbols") or adapter.get_constituents(args.index, args.end)
    df = adapter.fetch(symbols, args.start, args.end, args.frequency)
    panel = make_panel(df)
    train, valid, test = split_by_time(panel, float(cfg.get("train_ratio", .60)), float(cfg.get("valid_ratio", .20)))
    result = run(train, valid, test, cfg)
    print("\n===== FACTOR GEN V5 =====")
    print("STATUS       :", result["status"])
    print("TEST IC      :", result["test"]["ic"])
    print("MEAN IC      :", result["test"]["mean_ic"])
    print("MEDIAN IC    :", result["test"]["median_ic"])
    print("IC-IR        :", result["test"]["ic_ir"])
    print("POSITIVE     :", result["test"]["positive_ratio"])
    print("MONOTONICITY :", result["test"]["monotonicity"])
    print("TURNOVER     :", result["test"]["turnover"])
    print("DECAY        :", result["test"]["decay"])
    print("FORMULA      :", result["expression"])
    print("FACTOR FILE  :", result["factor"])
    print("MODEL FILE   :", result["model"])


if __name__ == "__main__":
    main()
