"""Command-line entry points (also exposed as asof-* console scripts).

    asof-prepare     --config configs/default.yaml
    asof-build-index --config ... --domain beijing_housing
    asof-run         --config ... --domain beijing_housing --method asof
    asof-ablation    --config ... --domain beijing_housing
    asof-tables      --runs data/runs --out data/tables
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .jsonl import write_json


def _base_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--override", nargs="*", default=[],
                    help="extra YAMLs merged over the base (providers/ablations)")


# --------------------------------------------------------------------------- #
def prepare_data_main(argv: list[str] | None = None) -> None:
    from .data import LOADERS, write_canonical

    ap = argparse.ArgumentParser(description="Build the canonical benchmark from datasets/")
    _base_args(ap)
    ap.add_argument("--domains", nargs="*", help="subset of domains (default: config)")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.override)
    root, out = cfg.path("datasets"), cfg.path("canonical")
    domains = a.domains or cfg["domains"]

    all_stats = {}
    for d in domains:
        if d not in LOADERS:
            print(f"[skip] no loader for domain '{d}'")
            continue
        notices, questions = LOADERS[d](root)
        st = write_canonical(d, notices, questions, out)
        all_stats[d] = st
        print(f"[{d:16s}] notices={st['notices']:4d}  questions={st['questions']:4d}  "
              f"types={st['questions_by_type']}")
    write_json(out / "stats.json", all_stats)
    print(f"\nwrote canonical benchmark -> {out}")


# --------------------------------------------------------------------------- #
def build_index_main(argv: list[str] | None = None) -> None:
    from .index import build_domain_index

    ap = argparse.ArgumentParser(description="Build dense+sparse index (GPU step)")
    _base_args(ap)
    ap.add_argument("--domain", required=True)
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.override)
    build_domain_index(cfg, a.domain)


def run_experiment_main(argv: list[str] | None = None) -> None:
    from .eval.runner import run_experiment

    ap = argparse.ArgumentParser(description="Run one method on one domain")
    _base_args(ap)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--method", default="asof",
                    choices=["asof", "latest_wins", "whole_doc", "hybrid_rag",
                             "timelyrag", "full_rebuild", "longctx", "longctx_visible"])
    ap.add_argument("--split", default="test")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.override)
    run_experiment(cfg, a.domain, a.method, split=a.split)


def run_ablation_main(argv: list[str] | None = None) -> None:
    from .eval.runner import run_ablation

    ap = argparse.ArgumentParser(description="Run all ablation variants on a domain")
    _base_args(ap)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--variants-dir", default="configs/ablations")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.override)
    run_ablation(cfg, a.domain, Path(a.variants_dir))


def make_tables_main(argv: list[str] | None = None) -> None:
    from .eval.tables import make_tables

    ap = argparse.ArgumentParser(description="Results json -> paper LaTeX tables")
    ap.add_argument("--runs", default="data/runs")
    ap.add_argument("--out", default="data/tables")
    a = ap.parse_args(argv)
    make_tables(Path(a.runs), Path(a.out))
