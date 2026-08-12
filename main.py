"""factor_gen V4.2 formal release - sole entry point for 掘金3.
Run: python main.py
"""
import logging
from factor_gen.v42 import run

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

if __name__ == "__main__":
    result=run()
    print("\n===== FACTOR GEN V4.2 =====")
    print(f"STATUS      : {result.status}")
    print(f"TEST IC     : {result.test_mean_ic:.5f}")
    print(f"MEDIAN IC   : {result.median_ic:.5f}")
    print(f"IC-IR       : {result.ic_ir:.3f}")
    print(f"POSITIVE    : {result.positive_ratio:.2%}")
    print(f"FORMULA     : {result.formula}")
    print("===========================")
