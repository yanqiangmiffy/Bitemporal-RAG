"""Turn run JSONs into the paper's LaTeX tables, with REAL measured numbers.

Writes <out>/main_results_real.tex and <out>/ablation_real.tex (the ``_real``
suffix avoids clobbering the \\mock placeholder tables until you've reviewed them).
Aggregates joint accuracy by question type across all runs of each method.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from ..jsonl import read_json

TYPES = ["Pre", "Post", "Unch", "Win", "Inap", "Unseen"]
METHOD_ORDER = ["hybrid_rag", "latest_wins", "whole_doc", "timelyrag",
                "longctx", "longctx_visible", "full_rebuild", "asof"]
METHOD_NAMES = {
    "hybrid_rag": "Hybrid RAG", "latest_wins": "Latest-wins",
    "whole_doc": "Whole-doc inval.", "timelyrag": "TimelyRAG~\\cite{timelyrag}",
    "longctx": "Long-context (all docs)",
    "longctx_visible": "Long-context (date-filtered)",
    "full_rebuild": "Full rebuild (ref.)", "asof": "\\textbf{\\asof{} (ours)}",
}
ABLATION_ORDER = ["full", "no_visible_time", "no_condition_split", "no_exact_id", "small_extractor"]
ABLATION_NAMES = {
    "full": "\\asof{} (full)", "no_visible_time": "\\;\\; $-$ visible-time axis",
    "no_condition_split": "\\;\\; $-$ condition split", "no_exact_id": "\\;\\; $-$ exact id match",
    "small_extractor": "\\;\\; smaller extractor",
}


def _agg(runs: list[dict]) -> tuple[dict, float]:
    ok = defaultdict(int)
    tot = defaultdict(int)
    n_ok = n = 0
    for r in runs:
        for it in r["metrics"].get("per_item", []):
            t = it["qtype"]
            j = int(it["joint_ok"])
            ok[t] += j
            tot[t] += 1
            n_ok += j
            n += 1
    by = {t: round(100 * ok[t] / tot[t], 1) for t in TYPES if tot[t]}
    overall = round(100 * n_ok / n, 1) if n else 0.0
    return by, overall


def _row(label: str, by: dict, overall: float, bold: bool) -> str:
    cells = []
    for t in TYPES:
        v = by.get(t)
        if v is None:
            cells.append("--")
        else:
            cells.append(f"\\textbf{{{v}}}" if bold else f"{v}")
    ov = f"\\textbf{{{overall}}}" if bold else f"{overall}"
    return f"{label:28s} & " + " & ".join(cells) + f" & {ov} \\\\"


def _table(rows: list[str], caption: str, label: str, header_first: str) -> str:
    head = f"{header_first} & " + " & ".join(TYPES) + " & Overall \\\\"
    return "\n".join([
        "% Auto-generated from data/runs by asof.eval.tables — REAL measured numbers.",
        "\\begin{table}[t]", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
        "\\centering", "\\footnotesize", "\\begin{tabular}{@{}lccccccc@{}}", "\\toprule",
        head, "\\midrule", *rows, "\\bottomrule", "\\end{tabular}", "\\end{table}", "",
    ])


def make_tables(runs_dir: Path, out_dir: Path) -> None:
    runs = [read_json(p) for p in sorted(Path(runs_dir).glob("*.json"))]
    if not runs:
        print(f"[tables] no run JSONs in {runs_dir}; run experiments first.")
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    main = [r for r in runs if not r["run"].startswith("ablation__")]
    by_method: dict[str, list[dict]] = defaultdict(list)
    for r in main:
        by_method[r["method"]].append(r)
    rows = []
    for m in METHOD_ORDER:
        if m in by_method:
            by, ov = _agg(by_method[m])
            rows.append(_row(METHOD_NAMES[m], by, ov, bold=(m == "asof")))
    if rows:
        tex = _table(rows, "Joint accuracy (\\%) by question type (measured).",
                     "tab:main", "Method")
        (out_dir / "main_results_real.tex").write_text(tex, encoding="utf-8")
        print(f"[tables] wrote {out_dir/'main_results_real.tex'} ({len(rows)} methods)")

    abl = [r for r in runs if r["run"].startswith("ablation__")]
    by_var = {r["run"].replace("ablation__", ""): r for r in abl}
    arows = []
    for v in ABLATION_ORDER:
        if v in by_var:
            by, ov = _agg([by_var[v]])
            arows.append(_row(ABLATION_NAMES.get(v, v), by, ov, bold=(v == "full")))
    if arows:
        tex = _table(arows, "Ablation: Joint accuracy (\\%) by question type (measured).",
                     "tab:ablation", "Variant")
        (out_dir / "ablation_real.tex").write_text(tex, encoding="utf-8")
        print(f"[tables] wrote {out_dir/'ablation_real.tex'} ({len(arows)} variants)")
