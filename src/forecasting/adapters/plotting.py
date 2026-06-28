"""Comparison plots for the step-3.5 report (matplotlib, Agg — headless/Docker safe).

`render_report(report, out_dir)` writes the figure set and returns the paths:
  - pred_vs_real.png        aggregate actual vs each model over the final holdout + sample stores
  - error_by_horizon.png    RMSLE vs days-ahead (1..horizon) per model — should rise with distance
  - seasonal.png            grouped RMSLE bars per season per model (the xgboost↔hybrid ablation)
  - residuals.png           residual (actual − predicted) distribution per model
  - feature_importance_<model>.png   top-gain features per tree model

Pure rendering off a `ComparisonReport` (domain data) — no metric maths, no MLflow here.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..domain.entities import DATE, STORE  # noqa: E402

_PRIMARY = "rmsle"


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def _pred_vs_real(report, out_dir: Path) -> Path:
    """Aggregate (summed over stores) actual vs each model, plus three sample stores."""
    models = list(report.holdout_preds)
    base = report.holdout_preds[models[0]]
    sample = list(base[STORE].drop_duplicates().head(3))
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    panels = [("All stores (total/day)", None)] + [(f"store {s}", s) for s in sample]
    for ax, (title, store) in zip(axes.ravel(), panels, strict=False):
        for name in models:
            df = report.holdout_preds[name]
            if store is not None:
                df = df[df[STORE] == store]
            g = df.groupby(DATE, as_index=False).sum(numeric_only=True).sort_values(DATE)
            ax.plot(g[DATE], g["y_pred"], label=name, linewidth=1.3)
        # actual once (shared y_true), thick + dashed so models read against it
        df = base if store is None else base[base[STORE] == store]
        g = df.groupby(DATE, as_index=False).sum(numeric_only=True).sort_values(DATE)
        ax.plot(g[DATE], g["y_true"], label="actual", color="black", linewidth=2.2, linestyle="--")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=45)
    axes.ravel()[0].legend(fontsize=8)
    fig.suptitle(f"Final {report.final_horizon}-day forecast vs actual")
    return _save(fig, out_dir / "pred_vs_real.png")


def _error_by_horizon(report, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, res in report.results.items():
        bh = res.by_horizon
        if bh is None or bh.empty:
            continue
        ax.plot(bh["horizon_offset"], bh[_PRIMARY], marker="o", label=name)
    ax.set_xlabel("days ahead")
    ax.set_ylabel("RMSLE")
    ax.set_title(f"Error by horizon (1..{report.horizon})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return _save(fig, out_dir / "error_by_horizon.png")


def _seasonal(report, out_dir: Path) -> Path:
    seasons = ["spring", "summer", "autumn", "winter"]
    models = [n for n, r in report.results.items() if r.by_segment]
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.8 / max(len(models), 1)
    for i, name in enumerate(models):
        seg = report.results[name].by_segment
        vals = [seg.get(s, {}).get(_PRIMARY, 0.0) for s in seasons]
        x = [j + i * width for j in range(len(seasons))]
        ax.bar(x, vals, width=width, label=name)
    ax.set_xticks([j + width * (len(models) - 1) / 2 for j in range(len(seasons))])
    ax.set_xticklabels(seasons)
    ax.set_ylabel("RMSLE")
    ax.set_title("RMSLE by season (xgboost ↔ timesfm_hybrid = the ablation)")
    ax.legend(fontsize=8)
    return _save(fig, out_dir / "seasonal.png")


def _residuals(report, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, df in report.holdout_preds.items():
        if df.empty:
            continue
        resid = (df["y_true"] - df["y_pred"]).to_numpy()
        ax.hist(resid, bins=40, histtype="step", label=name, linewidth=1.3)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("residual (actual − predicted)")
    ax.set_ylabel("count")
    ax.set_title("Residual distribution (final holdout)")
    ax.legend(fontsize=8)
    return _save(fig, out_dir / "residuals.png")


def _feature_importance(name: str, imp: dict[str, float], out_dir: Path) -> Path:
    top = sorted(imp.items(), key=lambda kv: kv[1], reverse=True)[:20][::-1]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh([f for f, _ in top], [v for _, v in top])
    ax.set_xlabel("gain")
    ax.set_title(f"Feature importance — {name}")
    return _save(fig, out_dir / f"feature_importance_{name}.png")


def render_report(report, out_dir: Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        _pred_vs_real(report, out_dir),
        _error_by_horizon(report, out_dir),
        _seasonal(report, out_dir),
        _residuals(report, out_dir),
    ]
    for name, imp in report.importances.items():
        if imp:
            paths.append(_feature_importance(name, imp, out_dir))
    return paths
