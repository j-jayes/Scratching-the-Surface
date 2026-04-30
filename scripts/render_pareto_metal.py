"""Render the (escalation_rate, F1) Pareto curve from threshold_sweep.json.

Output: ``reports/pareto_metal.png`` and ``reports/pareto_metal.json``
(per-domain points with the chosen operating point starred).

This is the picture the website's evaluation page should carry to make the
\u03c4-calibration argument visually rather than in prose.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

REPORTS = Path("reports")
SWEEP = REPORTS / "threshold_sweep.json"
SUMMARY = Path("models/patchcore_metal/summary.json")
OUT_PNG = REPORTS / "pareto_metal.png"
OUT_JSON = REPORTS / "pareto_metal.json"

# Hand-picked operating points (Phase K calibration).
OPERATING = {"severstal": -0.5, "ksdd2": 1.0}


def main() -> None:
    if not SWEEP.exists():
        raise SystemExit(f"Missing {SWEEP} \u2014 run threshold_sweep.py first.")

    sweep = json.loads(SWEEP.read_text(encoding="utf-8"))
    by_domain = sweep.get("per_domain", sweep)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"severstal": "tab:orange", "ksdd2": "tab:blue"}
    serialised = {}
    for domain, dom_data in by_domain.items():
        points = dom_data.get("curve", dom_data) if isinstance(dom_data, dict) else dom_data
        # Each point: {tau, f1, escalation_rate, ...}
        xs = [p["escalation_rate"] for p in points]
        ys = [p["f1"] for p in points]
        ax.plot(xs, ys, marker="o", linewidth=1.5, markersize=4,
                color=colors.get(domain, "gray"), label=f"{domain} (sweep)")
        # Star the operating point.
        op_tau = OPERATING.get(domain)
        op = next((p for p in points if abs(p["tau"] - op_tau) < 1e-6), None)
        if op:
            ax.scatter(op["escalation_rate"], op["f1"], marker="*",
                       s=320, edgecolor="black", linewidth=1.2,
                       color=colors.get(domain, "gray"), zorder=5,
                       label=f"{domain} \u03c4={op_tau} (chosen)")
        serialised[domain] = {"points": points, "operating_tau_z": op_tau}

    ax.set_xlabel("Escalation rate to L2/L3 (per-frame)")
    ax.set_ylabel("F1 (binary defect / no_defect)")
    ax.set_title("PatchCore L1 \u2014 cost vs accuracy Pareto by domain")
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)

    OUT_JSON.write_text(json.dumps(serialised, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_PNG} and {OUT_JSON}")


if __name__ == "__main__":
    main()
