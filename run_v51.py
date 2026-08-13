from __future__ import annotations

import argparse
import logging
from pathlib import Path
import yaml

from factor_gen.gm3.adapter import GM3DataAdapter
from factor_gen.qsk_v5 import make_panel, split_by_time, run


def main():
    p = argparse.ArgumentParser(description="factor_gen V5.1 weekly trend signal miner")
    p.add_argument("--config", default="config_v51.yaml")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--index", default="SHSE.000300")
    p.add_argument("--frequency", default="1d")
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    adapter = GM3DataAdapter(token=cfg.get("gm_token"))
    symbols = cfg.get("symbols") or adapter.get_constituents(args.index, args.end)
    df = adapter.fetch(symbols, args.start, args.end, args.frequency)
    panel = make_panel(df, horizon=int(cfg.get("horizon", 5)))
    train, valid, test = split_by_time(panel, float(cfg.get("train_ratio", .60)), float(cfg.get("valid_ratio", .20)))
    result = run(train, valid, test, cfg)
    print("\n===== FACTOR GEN V5.1 WEEKLY SIGNAL =====")
    for k in ("status", "expression", "factor", "model"):
        print(f"{k.upper():16}:", result.get(k))
    print("HORIZON          :", result.get("horizon"))
    print("TEST MEAN IC     :", result.get("test", {}).get("mean_ic"))
    print("TEST IC-IR       :", result.get("test", {}).get("ic_ir"))
    print("SIGNAL ACTIVE    :", result.get("signal", {}).get("active_ratio"))
    print("SIGNAL HIT RATE  :", result.get("signal", {}).get("hit_rate"))
    print("SEARCH            :", result.get("search"))


if __name__ == "__main__":
    main()
