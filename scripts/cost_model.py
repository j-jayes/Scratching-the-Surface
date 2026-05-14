"""Phase 4 — Cloud-vs-on-prem cost model + break-even chart.

Three deployment modes are compared as a function of monthly image volume:

1. **Azure-hosted (managed):**
   - Azure OpenAI gpt-4.1-mini per call
   - One always-on ACA replica (preprocessing + dispatcher)

2. **Hybrid (Azure VLM + on-prem ResNet50):**
   - ResNet50 on a commodity CPU box handles every frame at marginal cost
     (electricity only).
   - A configurable *escalation rate* of frames is sent to Azure OpenAI for a
     second opinion (the "low-confidence" share).

3. **Fully on-prem:**
   - ResNet50 + Qwen3-VL on a single L40S workstation.
   - Capex amortised over 36 months, plus power + 10 %/yr maintenance.

Outputs:
- ``website/assets/cost/breakeven.png``  — 3-line break-even chart
- ``reports/cost_model.json``            — assumptions + crossover points

Pricing snapshot is May 2026 (Azure West Europe, OpenRouter, Newegg L40S).
Tweak the constants below to re-run with fresh numbers.

Run::

    uv run python scripts/cost_model.py
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


# ── Assumptions ─────────────────────────────────────────────────────────────
@dataclass
class Assumptions:
    # Azure OpenAI gpt-4.1-mini, USD/1M tokens (May 2026, West Europe).
    aoai_price_in_per_M: float = 0.15
    aoai_price_out_per_M: float = 0.60
    # Per-call token budget measured from the Phase 0a probe + bench traces.
    tokens_in_per_call: int = 500
    tokens_out_per_call: int = 100

    # ACA always-on replica for routing/preprocess (1 vCPU, 2 GiB).
    aca_vcpu: float = 1.0
    aca_ram_gib: float = 2.0
    aca_price_vcpu_sec: float = 0.000024  # USD per vCPU-second
    aca_price_ram_sec_gib: float = 0.0000025  # USD per GiB-second

    # OpenRouter Qwen3-VL-30B-A3B-Instruct (open-weights), USD/1M tokens.
    qwen_or_price_in_per_M: float = 0.13
    qwen_or_price_out_per_M: float = 0.52

    # On-prem CPU box — only really cost is power + amortised hardware.
    cpu_box_capex_usd: float = 1500.0          # 1× workstation
    cpu_box_amortise_months: int = 36
    cpu_box_power_w: float = 50.0              # avg
    electricity_usd_per_kwh: float = 0.18      # EU average

    # On-prem GPU box for self-hosted Qwen3-VL.
    l40s_box_capex_usd: float = 12000.0        # workstation + 1× L40S
    l40s_box_amortise_months: int = 36
    l40s_box_power_w: float = 350.0            # idle+inference avg
    l40s_maintenance_pct_per_year: float = 0.10
    # Achievable Qwen3-VL throughput on one L40S via vLLM (image+text).
    # Conservative — published benches range 4-8 img/s for 30B-A3B activations.
    qwen_l40s_throughput_imgs_per_sec: float = 5.0

    # ResNet50 on CPU: ~50 ms/image at batch=1 (M2 measurement, will be
    # similar on a Xeon/Ryzen). 20 imgs/s.
    resnet50_cpu_throughput_imgs_per_sec: float = 20.0

    # Hybrid escalation rate — share of frames that get a VLM second opinion.
    hybrid_escalation_rate: float = 0.10

    # Volume sweep.
    volume_min: int = 10_000
    volume_max: int = 10_000_000


def aoai_per_image_cost(a: Assumptions) -> float:
    return (a.tokens_in_per_call * a.aoai_price_in_per_M / 1e6
            + a.tokens_out_per_call * a.aoai_price_out_per_M / 1e6)


def qwen_or_per_image_cost(a: Assumptions) -> float:
    return (a.tokens_in_per_call * a.qwen_or_price_in_per_M / 1e6
            + a.tokens_out_per_call * a.qwen_or_price_out_per_M / 1e6)


def aca_monthly_cost(a: Assumptions) -> float:
    secs_per_month = 30 * 24 * 3600
    return (a.aca_vcpu * a.aca_price_vcpu_sec
            + a.aca_ram_gib * a.aca_price_ram_sec_gib) * secs_per_month


def cpu_box_monthly_cost(a: Assumptions) -> float:
    capex = a.cpu_box_capex_usd / a.cpu_box_amortise_months
    power_kwh = a.cpu_box_power_w * 24 * 30 / 1000
    power = power_kwh * a.electricity_usd_per_kwh
    return capex + power


def l40s_box_monthly_cost(a: Assumptions) -> float:
    capex = a.l40s_box_capex_usd / a.l40s_box_amortise_months
    maint = a.l40s_box_capex_usd * a.l40s_maintenance_pct_per_year / 12
    power_kwh = a.l40s_box_power_w * 24 * 30 / 1000
    power = power_kwh * a.electricity_usd_per_kwh
    return capex + maint + power


def cost_curves(a: Assumptions, n_points: int = 200) -> dict:
    volumes = np.geomspace(a.volume_min, a.volume_max, n_points)

    aoai_per = aoai_per_image_cost(a)
    aca_fixed = aca_monthly_cost(a)

    azure_only = aoai_per * volumes + aca_fixed
    # Hybrid: ResNet50 free per call (already paid in box capex), only the
    # escalation share hits Azure.
    hybrid = (aoai_per * volumes * a.hybrid_escalation_rate
              + aca_fixed
              + cpu_box_monthly_cost(a))
    # Fully on-prem: flat box cost (covers all volumes up to the box throughput)
    onprem = np.full_like(volumes, l40s_box_monthly_cost(a) + cpu_box_monthly_cost(a))

    # Highlight throughput cap on the on-prem line — flag if volume exceeds
    # a single-box capacity over a 30-day month at full duty cycle.
    secs_per_month = 30 * 24 * 3600
    onprem_capacity = a.qwen_l40s_throughput_imgs_per_sec * secs_per_month
    return {
        "volumes": volumes,
        "azure_only": azure_only,
        "hybrid": hybrid,
        "onprem": onprem,
        "onprem_capacity": onprem_capacity,
    }


def find_crossover(volumes: np.ndarray, a: np.ndarray, b: np.ndarray) -> float | None:
    """Volume at which ``a`` first becomes ≥ ``b`` (or None if never)."""
    diff = a - b
    sign_change = np.where(np.diff(np.sign(diff)))[0]
    if len(sign_change) == 0:
        return None
    i = sign_change[0]
    # linear-interp in log-space
    x0, x1 = np.log10(volumes[i]), np.log10(volumes[i + 1])
    y0, y1 = diff[i], diff[i + 1]
    if y1 - y0 == 0:
        return float(volumes[i])
    x = x0 - y0 * (x1 - x0) / (y1 - y0)
    return float(10 ** x)


def render_chart(curves: dict, a: Assumptions, out_path: Path) -> None:
    import matplotlib.pyplot as plt  # local import
    fig, ax = plt.subplots(figsize=(8, 5))
    v = curves["volumes"]
    ax.loglog(v, curves["azure_only"], label="Azure-only (AOAI gpt-4.1-mini)",
              color="#d95f02", lw=2)
    ax.loglog(v, curves["hybrid"],
              label=f"Hybrid (CPU ResNet50 + {int(a.hybrid_escalation_rate*100)} % AOAI escalation)",
              color="#7570b3", lw=2)
    ax.loglog(v, curves["onprem"],
              label="Fully on-prem (L40S + CPU box)",
              color="#1b9e77", lw=2)

    # Crossovers.
    for (a_arr, b_arr, name, colour) in [
        (curves["azure_only"], curves["hybrid"], "Azure↔Hybrid", "#d95f02"),
        (curves["hybrid"], curves["onprem"], "Hybrid↔On-prem", "#1b9e77"),
    ]:
        x = find_crossover(v, a_arr, b_arr)
        if x is not None and curves["volumes"][0] < x < curves["volumes"][-1]:
            y = np.interp(np.log10(x), np.log10(v), np.log10(a_arr))
            y = 10 ** y
            ax.scatter([x], [y], color=colour, zorder=5, s=60, marker="o",
                       edgecolor="black")
            ax.annotate(f"{name}\n≈ {x:,.0f} img/mo",
                        (x, y), textcoords="offset points",
                        xytext=(8, -16), fontsize=9)

    # Capacity bound on on-prem.
    cap = curves["onprem_capacity"]
    if cap < curves["volumes"][-1]:
        ax.axvline(cap, color="#1b9e77", linestyle=":", alpha=0.5)
        ax.text(cap, ax.get_ylim()[0] * 2,
                f"  1× L40S capacity ≈ {cap:,.0f}/mo",
                color="#1b9e77", fontsize=8, rotation=90, va="bottom")

    ax.set_xlabel("Images per month")
    ax.set_ylabel("Total monthly cost (USD)")
    ax.set_title("Defect-detection cost: Azure vs Hybrid vs On-prem")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-png",
                   default="website/assets/cost/breakeven.png")
    p.add_argument("--out-json", default="reports/cost_model.json")
    args = p.parse_args()

    a = Assumptions()
    curves = cost_curves(a)
    crossover_az_hy = find_crossover(curves["volumes"],
                                     curves["azure_only"], curves["hybrid"])
    crossover_hy_op = find_crossover(curves["volumes"],
                                     curves["hybrid"], curves["onprem"])
    out = {
        "assumptions": asdict(a),
        "per_image_cost_usd": {
            "aoai_gpt41mini": round(aoai_per_image_cost(a), 6),
            "qwen3_vl_openrouter": round(qwen_or_per_image_cost(a), 6),
        },
        "monthly_fixed_cost_usd": {
            "aca_replica": round(aca_monthly_cost(a), 2),
            "cpu_box_amortised": round(cpu_box_monthly_cost(a), 2),
            "l40s_box_amortised": round(l40s_box_monthly_cost(a), 2),
        },
        "crossover_volumes_per_month": {
            "azure_to_hybrid": (round(crossover_az_hy) if crossover_az_hy else None),
            "hybrid_to_onprem": (round(crossover_hy_op) if crossover_hy_op else None),
        },
        "onprem_l40s_monthly_capacity_imgs": int(curves["onprem_capacity"]),
    }
    out_json = ROOT / args.out_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_json}")
    print(json.dumps(out, indent=2))

    out_png = ROOT / args.out_png
    render_chart(curves, a, out_png)
    print(f"\nWrote {out_png}")


if __name__ == "__main__":
    main()
