"""End-to-end V4 pipeline: GM3 data -> Transformer -> IC -> search -> export."""
from __future__ import annotations
import logging
from .gm3_adapter import GM3DataAdapter
from .features import make_features
from .transformer import TransformerFactor
from .evaluation import evaluate_factor
from .search import search_combinations
from .exporter import export_gm3_factor

LOG = logging.getLogger("factor_gen")

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    LOG.info("GM3 AI Factor Generator V4 starting")
    data = GM3DataAdapter().load()
    features, target = make_features(data)
    model = TransformerFactor()
    score = model.fit_predict(features, target)
    base = evaluate_factor(score, target)
    LOG.info("Transformer factor: mean IC=%.5f median IC=%.5f status=%s", base.mean_ic, base.median_ic, base.status)
    best = search_combinations(features, target, max_candidates=1_000_000)
    result = export_gm3_factor(best)
    LOG.info("Best factor exported: %s", result)
