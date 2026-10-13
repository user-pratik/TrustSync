"""Generate publication report tables, CSV exports, numbers dict, and figures."""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any


def wilson(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Compute 95% Wilson score confidence interval for a binomial proportion.

    Args:
        k: Number of successes / positive outcomes.
        n: Total number of trials.
        confidence: Confidence level (default 0.95).

    Returns:
        tuple[float, float]: (lower_bound, upper_bound) rounded to 4 decimals.
    """
    if n <= 0:
        return (0.0, 0.0)
    z = 1.959963984540054  # 95% two-sided normal critical value
    p = float(k) / float(n)
    denom = 1.0 + (z * z) / float(n)
    center = (p + (z * z) / (2.0 * float(n))) / denom
    spread = (z / denom) * math.sqrt((p * (1.0 - p)) / float(n) + (z * z) / (4.0 * float(n) * float(n)))
    lower = max(0.0, center - spread)
    upper = min(1.0, center + spread)
    return (round(lower, 4), round(upper, 4))


def fmt_pct(val: float | None) -> str:
    """Format float into percentage string with 1 decimal."""
    if val is None:
        return "--"
    return f"{val * 100.0:.1f}%"


def generate_assets(results_path: str | Path = "reports/results.json", report_dir: str | Path = "report") -> None:
    """Generate all markdown tables, CSV exports, numbers JSON, and figure copies.

    Args:
        results_path: Path to input reports/results.json.
        report_dir: Output report directory.
    """
    res_path = Path(results_path)
    if not res_path.is_file():
        raise FileNotFoundError(f"Results file {res_path} not found.")

    rep_dir = Path(report_dir)
    rep_dir.mkdir(parents=True, exist_ok=True)
    fig_dest = rep_dir / "figures"
    fig_dest.mkdir(parents=True, exist_ok=True)

    # Copy any generated figure images
    fig_src = res_path.parent / "figures"
    if fig_src.is_dir():
        for png in fig_src.glob("*.png"):
            shutil.copyfile(png, fig_dest / png.name)

    data = json.loads(res_path.read_text(encoding="utf-8"))

    # Select headline metrics source
    if data.get("loso_learned") and data["loso_learned"].get("pooled"):
        headline = data["loso_learned"]["pooled"]
        headline_source = "loso_learned"
    elif data.get("loso_rule") and data["loso_rule"].get("pooled"):
        headline = data["loso_rule"]["pooled"]
        headline_source = "loso_rule"
    else:
        headline = data.get("current", {})
        headline_source = "current"

    dataset = data.get("dataset", {})
    n_clips = dataset.get("n_clips", headline.get("n_clips", 0))

    numbers: dict[str, Any] = {}
    numbers["n_clips"] = n_clips
    numbers["headline_source"] = headline_source

    # 1. Headline Metrics Table & CSV
    headline_rows = []
    headline_metrics = [
        ("Decision Accuracy", "decision_accuracy"),
        ("Spoof FAR", "far_spoof"),
        ("All-Reject FAR", "far_all_reject"),
        ("False Rejection Rate (FRR)", "frr"),
        ("Attribution Accuracy", "attribution_accuracy"),
        ("Addressee Accuracy", "addressee_accuracy"),
    ]

    for label, key in headline_metrics:
        val = headline.get(key, 0.0)
        numbers[key] = val
        k = int(round(val * n_clips))
        ci_low, ci_high = wilson(k, n_clips)
        numbers[f"{key}_k"] = k
        numbers[f"{key}_n"] = n_clips
        numbers[f"{key}_ci_low"] = ci_low
        numbers[f"{key}_ci_high"] = ci_high
        numbers[f"{key}_pct"] = round(val * 100.0, 1)

        pct_str = fmt_pct(val)
        count_str = f"{k}/{n_clips}"
        ci_str = f"[{ci_low*100.0:.1f}%, {ci_high*100.0:.1f}%]"
        headline_rows.append([label, count_str, pct_str, ci_str])

    # Liveness F1
    liv = headline.get("liveness", {})
    f1_val = liv.get("f1", 0.0)
    numbers["liveness_f1"] = f1_val
    numbers["liveness_f1_pct"] = round(f1_val * 100.0, 1)
    headline_rows.append(["Liveness F1", "--", fmt_pct(f1_val), "--"])

    with open(rep_dir / "table_headline.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Metric", "Count", "Percentage", "95% CI"])
        w.writerows(headline_rows)

    # 2. Ablation Table & CSV
    ablation = data.get("ablation", [])
    ablation_rows = []
    for row in ablation:
        name = row.get("name", "")
        acc = row.get("accuracy", 0.0)
        far = row.get("far_all_reject", 0.0)
        frr = row.get("frr", 0.0)
        f1 = row.get("liveness_f1", 0.0)
        ablation_rows.append([name, fmt_pct(acc), fmt_pct(far), fmt_pct(frr), fmt_pct(f1)])
        numbers[f"ablation_{name.lower()}_acc"] = acc
        numbers[f"ablation_{name.lower()}_far"] = far
        numbers[f"ablation_{name.lower()}_frr"] = frr
        numbers[f"ablation_{name.lower()}_f1"] = f1

    with open(rep_dir / "table_ablation.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Variant", "Accuracy", "FAR (All Reject)", "FRR", "Liveness F1"])
        w.writerows(ablation_rows)

    # 3. Per-Scenario Table & CSV
    per_scen = headline.get("per_scenario", {})
    scen_rows = []
    for sc_name, sc_data in sorted(per_scen.items()):
        sn = sc_data.get("n", 0)
        sacc = sc_data.get("accuracy", 0.0)
        srate = sc_data.get("accept_rate", 0.0)
        scen_rows.append([sc_name, str(sn), fmt_pct(sacc), fmt_pct(srate)])
        numbers[f"scenario_{sc_name}_n"] = sn
        numbers[f"scenario_{sc_name}_acc"] = sacc

    with open(rep_dir / "table_per_scenario.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Scenario", "N", "Accuracy", "Accept Rate"])
        w.writerows(scen_rows)

    # 4. Confusion Matrix Table & CSV
    conf = headline.get("confusion", {})
    all_decisions = ["ACCEPT", "REJECT_SPOOF", "REJECT_AMBIGUOUS", "REJECT_NOT_ADDRESSED", "REJECT_LOW_QUALITY", "REJECT_NO_FACE", "NONE"]
    conf_rows = []
    for exp_label, dec_dict in sorted(conf.items()):
        row = [exp_label] + [str(dec_dict.get(d, 0)) for d in all_decisions]
        conf_rows.append(row)

    with open(rep_dir / "table_confusion.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Expected"] + all_decisions)
        w.writerows(conf_rows)

    # 5. Leave-One-Subject-Out Folds Table & CSV
    loso_folds = (data.get("loso_learned") or data.get("loso_rule") or {}).get("folds", {})
    loso_rows = []
    for subj, s_metrics in sorted(loso_folds.items()):
        loso_rows.append([
            subj,
            str(s_metrics.get("n_clips", 0)),
            fmt_pct(s_metrics.get("decision_accuracy", 0.0)),
            fmt_pct(s_metrics.get("far_all_reject", 0.0)),
        ])

    with open(rep_dir / "table_loso_folds.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Subject", "Clips", "Accuracy", "FAR"])
        w.writerows(loso_rows)

    # 6. Dataset Counts Table & CSV
    scen_counts = dataset.get("per_scenario_counts", {})
    ds_rows = [["Total Clips", str(n_clips)]]
    for s_name, count in sorted(scen_counts.items()):
        ds_rows.append([f"Scenario: {s_name}", str(count)])

    with open(rep_dir / "table_dataset.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Category", "Count"])
        w.writerows(ds_rows)

    # Combine into report/tables.md
    tables_md = []
    tables_md.append("## Headline Metrics\n")
    tables_md.append("| Metric | Count | Percentage | 95% Wilson CI |")
    tables_md.append("| :--- | :---: | :---: | :---: |")
    for r in headline_rows:
        tables_md.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} |")
    tables_md.append("\n## Gating Ablation Study (B0 to A5)\n")
    tables_md.append("| Variant | Accuracy | FAR (All Reject) | FRR | Liveness F1 |")
    tables_md.append("| :--- | :---: | :---: | :---: | :---: |")
    for r in ablation_rows:
        tables_md.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} |")
    tables_md.append("\n## Per-Scenario Evaluation\n")
    tables_md.append("| Scenario | N | Accuracy | Accept Rate |")
    tables_md.append("| :--- | :---: | :---: | :---: |")
    for r in scen_rows:
        tables_md.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} |")
    tables_md.append("\n## Confusion Matrix\n")
    tables_md.append("| Expected | " + " | ".join(all_decisions) + " |")
    tables_md.append("| :--- | " + " | ".join([":---:"] * len(all_decisions)) + " |")
    for r in conf_rows:
        tables_md.append(f"| {r[0]} | " + " | ".join(r[1:]) + " |")
    tables_md.append("\n## Leave-One-Subject-Out Cross-Validation\n")
    tables_md.append("| Subject | Clips | Accuracy | FAR |")
    tables_md.append("| :--- | :---: | :---: | :---: |")
    for r in loso_rows:
        tables_md.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} |")
    tables_md.append("\n## Dataset Distribution\n")
    tables_md.append("| Category | Count |")
    tables_md.append("| :--- | :---: |")
    for r in ds_rows:
        tables_md.append(f"| {r[0]} | {r[1]} |")

    (rep_dir / "tables.md").write_text("\n".join(tables_md) + "\n", encoding="utf-8")

    # Write numbers.json
    (rep_dir / "numbers.json").write_text(json.dumps(numbers, indent=2), encoding="utf-8")
    print(f"Report assets written to {rep_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate TrustSync report assets from results.json.")
    parser.add_argument("--results", type=str, default="reports/results.json", help="Path to reports/results.json")
    parser.add_argument("--out", type=str, default="report", help="Output report directory")
    args = parser.parse_args()

    generate_assets(results_path=args.results, report_dir=args.out)
    return 0


if __name__ == "__main__":
    main()
