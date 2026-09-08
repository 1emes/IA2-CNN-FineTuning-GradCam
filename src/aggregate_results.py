"""Agrega métricas de múltiplas sementes produzidas por experiment.py."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np


def mean_and_std(values: list[float]) -> tuple[float, float]:
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return mean, std


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default="results/aggregate")
    parser.add_argument("--expected-seeds", nargs="+", type=int, default=[42, 123, 2026])
    parser.add_argument(
        "--expected-seeds",
        default="42,123,2026",
        help="Sementes obrigatórias, separadas por vírgula; use vazio para não validar.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for path in sorted(results_dir.glob("seed_*/summary.json")):
        with path.open(encoding="utf-8") as stream:
            summaries.append((path.parent.name, json.load(stream)))
    if not summaries:
        raise SystemExit("Nenhum results/seed_*/summary.json foi encontrado.")
    observed_seeds = {
        int(seed_name.removeprefix("seed_"))
        for seed_name, _summary in summaries
    }
    expected_seeds = set(args.expected_seeds)
    if observed_seeds != expected_seeds:
        missing = sorted(expected_seeds - observed_seeds)
        unexpected = sorted(observed_seeds - expected_seeds)
        raise SystemExit(
            f"Conjunto de sementes incompleto. Ausentes: {missing}; inesperadas: {unexpected}."
        )
    expected = {
        f"seed_{seed.strip()}" for seed in args.expected_seeds.split(",") if seed.strip()
    }
    found = {seed for seed, _ in summaries}
    if expected and found != expected:
        missing = sorted(expected - found)
        unexpected = sorted(found - expected)
        raise SystemExit(
            "Conjunto de sementes incompleto ou inesperado. "
            f"Ausentes: {missing or 'nenhuma'}; inesperadas: {unexpected or 'nenhuma'}."
        )

    classification_values: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    gradcam_values: dict[str, list[float]] = defaultdict(list)
    for _seed, summary in summaries:
        for row in summary["classification"]:
            strategy = row["strategy"]
            for metric, value in row.items():
                if metric != "strategy":
                    classification_values[strategy][metric].append(float(value))
        for metric, value in summary["gradcam"].items():
            gradcam_values[metric].append(float(value))

    classification_rows = []
    for strategy, metrics in sorted(classification_values.items()):
        row = {"strategy": strategy, "runs": len(summaries)}
        for metric, values in sorted(metrics.items()):
            row[f"{metric}_mean"], row[f"{metric}_std"] = mean_and_std(values)
        classification_rows.append(row)

    gradcam_rows = []
    for metric, values in sorted(gradcam_values.items()):
        mean, std = mean_and_std(values)
        gradcam_rows.append({"metric": metric, "mean": mean, "std": std, "runs": len(values)})

    write_csv(output_dir / "classification_aggregate.csv", classification_rows)
    write_csv(output_dir / "gradcam_aggregate.csv", gradcam_rows)
    representative_figure = results_dir / "seed_42" / "figures" / "gradcam_examples.png"
    if representative_figure.exists():
        figure_dir = output_dir / "figures"
        figure_dir.mkdir(exist_ok=True)
        shutil.copy2(representative_figure, figure_dir / "gradcam_examples.png")
    with (output_dir / "aggregate_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "seeds": sorted(observed_seeds),
                "classification": classification_rows,
                "gradcam": gradcam_rows,
            },
            stream,
            indent=2,
        )
    print(f"Agregação gravada em: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
