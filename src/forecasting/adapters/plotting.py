"""Comparison plots for the step-4 report (matplotlib, Agg — headless/Docker safe).

`render_report(report, out_dir)` writes the figure set and returns the paths:
  - holdout_scores.png      final-holdout score per model, 3 metrics, with/without the naive (2x3)
  - pred_vs_real.png        aggregate actual-vs-models over the holdout + 2 sample stores
  - seasonal.png            by-season error per model, 3 metrics, with/without the naive (2x3)
  - error_by_horizon.png    error vs days-ahead per model, 3 metrics, with/without the naive (2x3)
  - error_by_prefecture.png each metric by prefecture, per model (stacked 3x1)
  - feature_importance_<model>.png   top relative-gain features per tree model
  - residuals.png           residual distribution per model (outliers clipped)
  - index.html              all of the above bundled into one scannable report (the CV headline)

Pure rendering off a `ComparisonReport` (domain data) — every number is precomputed; no metric
maths and no MLflow here.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..domain.entities import DATE, STORE  # noqa: E402

_METRICS = ("rmsle", "mae", "weighted_mae")
_METRIC_LABEL = {"rmsle": "RMSLE", "mae": "MAE", "weighted_mae": "weighted MAE"}
_PRIMARY = "rmsle"


def _save(fig, path: Path) -> Path:
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def _model_colors(models) -> dict[str, tuple]:
    """Stable color per model name, shared across every chart."""
    cmap = plt.get_cmap("tab10")
    return {name: cmap(i % 10) for i, name in enumerate(models)}


def _month_axis(ax) -> None:
    """Weekly date ticks + month-name boundaries (readable on a ~39-day window)."""
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter("%d"))
    ax.tick_params(axis="x", which="major", length=0, pad=18, labelsize=10)
    ax.tick_params(axis="x", which="minor", labelsize=7, colors="0.5")
    for d in ax.xaxis.get_minorticklabels():
        d.set_rotation(0)
    ax.grid(True, which="major", axis="x", alpha=0.25)
    ax.grid(True, which="both", axis="y", alpha=0.15)


def _series(df, store=None):
    """(dates, summed value) for a model frame, optionally one store."""
    if store is not None:
        df = df[df[STORE] == store]
    g = df.groupby(DATE, as_index=False).sum(numeric_only=True).sort_values(DATE)
    return g[DATE], g


def _families(models) -> list[tuple[str, list[str]]]:
    """Group models so each forecast panel holds only a couple of lines vs actual."""
    naive = [m for m in models if "naive" in m]
    timesfm = [m for m in models if "timesfm" in m]
    trees = [m for m in models if m not in naive and m not in timesfm]
    out = [("Baseline", naive), ("Trees", trees), ("TimesFM", timesfm)]
    return [(label, group) for label, group in out if group]


def _pred_vs_real(report, out_dir: Path) -> Path:
    """One row per model family (Baseline / Trees / TimesFM), each vs actual — so a panel never
    holds more than ~2 model lines. Columns: aggregate total/day, then 2 sample stores."""
    models = list(report.holdout_preds)
    colors = _model_colors(models)
    base = report.holdout_preds[models[0]]
    sample = list(base[STORE].drop_duplicates().head(2))
    views = [("All stores — total/day", None)] + [(f"store {s}", s) for s in sample]
    fams = _families(models)

    fig, axes = plt.subplots(len(fams), len(views), figsize=(5 * len(views), 3 * len(fams)),
                             squeeze=False)
    for r, (fam_label, group) in enumerate(fams):
        for c, (view_title, store) in enumerate(views):
            ax = axes[r][c]
            for name in group:
                x, g = _series(report.holdout_preds[name], store)
                ax.plot(x, g["y_pred"], label=name, color=colors[name], linewidth=1.5)
            x, g = _series(base, store)
            ax.plot(x, g["y_true"], label="actual", color="black", linewidth=2.0, linestyle="--")
            if r == 0:
                ax.set_title(view_title, fontsize=10)
            if c == 0:
                ax.set_ylabel(fam_label, fontsize=11, fontweight="bold")
                ax.legend(fontsize=7, loc="upper right")
            _month_axis(ax)
    fig.suptitle(f"Final {report.final_horizon}-day forecast vs actual", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(fig, out_dir / "pred_vs_real.png")


def _grouped_bars(ax, categories, models, value_of, colors) -> None:
    """Grouped bars: one cluster per category, one bar per model."""
    width = 0.8 / max(len(models), 1)
    for i, name in enumerate(models):
        vals = [value_of(name, c) for c in categories]
        x = [j + i * width for j in range(len(categories))]
        ax.bar(x, vals, width=width, label=name, color=colors[name])
    ax.set_xticks([j + width * (len(models) - 1) / 2 for j in range(len(categories))])
    ax.set_xticklabels(categories)


def _by_metric_figure(title, categories, models, value_of, colors, out_dir, fname,
                      rotate=0, stacked=False):
    """Small multiples, one metric each — side-by-side (1x3) or stacked one-per-row."""
    if stacked:
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, metric in zip(axes, _METRICS, strict=True):
        _grouped_bars(ax, categories, models, lambda n, c, m=metric: value_of(n, c, m), colors)
        ax.set_title(_METRIC_LABEL[metric])
        ax.grid(True, axis="y", alpha=0.25)
        if rotate:
            ax.tick_params(axis="x", rotation=rotate)
    axes[0].legend(fontsize=8)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(fig, out_dir / fname)


def _two_row_by_metric(title, models, draw, out_dir: Path, fname: str) -> Path:
    """2x3 small multiples. Row 0 = all models; row 1 = baseline (naive) excluded, so the close
    models stop being squashed by the naive's much larger error. Columns = the 3 metrics.
    `draw(ax, metric, model_subset, colors)` renders one cell. Colors are stable across rows."""
    colors = _model_colors(models)
    rows = [("with baseline", models),
            ("good models only", [m for m in models if "naive" not in m])]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), squeeze=False)
    for r, (row_label, subset) in enumerate(rows):
        for c, metric in enumerate(_METRICS):
            ax = axes[r][c]
            draw(ax, metric, subset, colors)
            if r == 0:
                ax.set_title(_METRIC_LABEL[metric])
            if c == 0:
                ax.set_ylabel(row_label, fontsize=11, fontweight="bold")
            ax.grid(True, axis="y", alpha=0.25)
        if axes[r][0].get_legend_handles_labels()[1]:
            axes[r][0].legend(fontsize=7)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(fig, out_dir / fname)


def _holdout_scores(report, out_dir: Path) -> Path:
    """The headline: each model's metric on the final holdout (with vs without the naive)."""
    models = list(report.holdout_metrics)

    def draw(ax, metric, subset, colors):
        vals = [report.holdout_metrics[n].get(metric, 0.0) for n in subset]
        ax.bar(range(len(subset)), vals, color=[colors[n] for n in subset])
        ax.set_xticks(range(len(subset)))
        ax.set_xticklabels(subset, rotation=40, ha="right", fontsize=8)

    return _two_row_by_metric(
        f"Final {report.final_horizon}-day holdout — model comparison",
        models, draw, out_dir, "holdout_scores.png",
    )


def _seasonal(report, out_dir: Path) -> Path:
    seasons = ["spring", "summer", "autumn", "winter"]
    models = [n for n, r in report.results.items() if r.by_segment]

    def draw(ax, metric, subset, colors):
        _grouped_bars(
            ax, seasons, subset,
            lambda n, s, m=metric: report.results[n].by_segment.get(s, {}).get(m, 0.0),
            colors,
        )

    return _two_row_by_metric(
        "Error by season (CV) — with vs without the naive baseline",
        models, draw, out_dir, "seasonal.png",
    )


def _by_prefecture(report, out_dir: Path) -> Path:
    models = [n for n, r in report.results.items() if r.by_region]
    if not models:
        return out_dir / "error_by_prefecture.png"  # nothing to draw (no stores table)
    # Prefectures sorted by sample size (busiest first), capped so the bars stay legible.
    counts: dict[str, int] = {}
    for n in models:
        for pref in report.results[n].by_region:
            counts[pref] = counts.get(pref, 0) + 1
    prefs = sorted(counts, key=lambda p: (-counts[p], p))[:12]

    def value_of(name, pref, metric):
        return report.results[name].by_region.get(pref, {}).get(metric, 0.0)

    return _by_metric_figure(
        "Error by prefecture", prefs, models, value_of, _model_colors(models),
        out_dir, "error_by_prefecture.png", rotate=90, stacked=True,
    )


def _error_by_horizon(report, out_dir: Path) -> Path:
    models = [n for n, r in report.results.items() if r.by_horizon is not None]

    def draw(ax, metric, subset, colors):
        for name in subset:
            bh = report.results[name].by_horizon
            if bh is None or bh.empty:
                continue
            ax.plot(bh["horizon_offset"], bh[metric], marker="o", label=name, color=colors[name])
        ax.set_xlabel("days ahead")

    return _two_row_by_metric(
        f"Error by horizon (1..{report.horizon}) — with vs without the naive baseline",
        models, draw, out_dir, "error_by_horizon.png",
    )


def _residuals(report, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = _model_colors(list(report.holdout_preds))
    resids = {n: (df["y_true"] - df["y_pred"]).to_numpy()
              for n, df in report.holdout_preds.items() if not df.empty}
    if resids:
        allr = np.concatenate(list(resids.values()))
        lo, hi = np.percentile(allr, [1, 99])  # drop the long tail so the bulk is visible
        bins = np.linspace(lo, hi, 41)
        for name, r in resids.items():
            ax.hist(r[(r >= lo) & (r <= hi)], bins=bins, histtype="step", label=name,
                    color=colors[name], linewidth=1.4)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("residual (actual − predicted), clipped to 1–99th pct")
    ax.set_ylabel("count")
    ax.set_title("Residual distribution (final holdout)")
    ax.legend(fontsize=8)
    return _save(fig, out_dir / "residuals.png")


def _feature_importance(name: str, imp: dict[str, float], out_dir: Path) -> Path:
    total = sum(imp.values()) or 1.0
    rel = sorted(((f, v / total * 100) for f, v in imp.items()), key=lambda kv: kv[1])[-15:]
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh([f for f, _ in rel], [v for _, v in rel], color="#4878a8")
    ax.bar_label(bars, fmt="%.0f%%", padding=3, fontsize=8)
    ax.set_xlabel("relative gain (% of total)")
    ax.set_title(f"Feature importance — {name}")
    ax.margins(x=0.12)
    return _save(fig, out_dir / f"feature_importance_{name}.png")


_CAPTIONS = {
    "holdout_scores": "Headline comparison: each model's score on the final holdout, for all "
                      "three metrics. Top row includes the naive baseline; bottom row drops it "
                      "so the close models are readable.",
    "pred_vs_real": "Final holdout forecast vs actual, one row per model family (Baseline / "
                    "Trees / TimesFM) so each panel stays readable. Columns: all stores, then "
                    "2 sample stores.",
    "seasonal": "Error by season per model (CV). Top row with the naive baseline, bottom row "
                "without it. xgboost ↔ timesfm_hybrid is the foundation-model ablation.",
    "error_by_horizon": "How error grows with days-ahead — forecasts should degrade with "
                        "distance. Top row with the naive, bottom row without.",
    "error_by_prefecture": "Error by prefecture (the busiest), one metric per row.",
    "residuals": "Residual (actual − predicted) spread on the holdout; long tail trimmed.",
}


def _html_report(paths: list[Path], out_dir: Path, report) -> Path:
    """Bundle the figures into a single scannable index.html (the CV run's headline output)."""
    order = ["holdout_scores", "pred_vs_real", "seasonal", "error_by_horizon",
             "error_by_prefecture", "residuals"]
    by_stem = {p.stem: p for p in paths}
    sections = [by_stem[s] for s in order if s in by_stem]
    sections += [p for p in paths if p.stem.startswith("feature_importance_")]
    models = ", ".join(report.results)
    body = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>Model comparison report</title>",
        "<style>body{font-family:system-ui,sans-serif;max-width:1100px;margin:0 auto;"
        "padding:24px;color:#1a1a1a}h1{font-size:22px}h2{font-size:16px;margin-top:32px}"
        "p{color:#555;line-height:1.5}img{width:100%;border:1px solid #ddd;border-radius:4px}"
        "</style>",
        "<h1>Model comparison report</h1>",
        f"<p>Cross-validation + final {report.final_horizon}-day holdout. "
        f"Models: {models}. CV horizon {report.horizon} days.</p>",
    ]
    for p in sections:
        cap = _CAPTIONS.get(p.stem) or f"{p.stem.replace('_', ' ')}."
        body.append(f"<h2>{p.stem}</h2><p>{cap}</p><img alt='{p.stem}' src='{p.name}'>")
    path = out_dir / "index.html"
    path.write_text("\n".join(body))
    return path


def render_report(report, out_dir: Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        _pred_vs_real(report, out_dir),
        _seasonal(report, out_dir),
        _error_by_horizon(report, out_dir),
        _residuals(report, out_dir),
    ]
    if report.holdout_metrics:
        paths.append(_holdout_scores(report, out_dir))
    if any(r.by_region for r in report.results.values()):
        paths.append(_by_prefecture(report, out_dir))
    for name, imp in report.importances.items():
        if imp:
            paths.append(_feature_importance(name, imp, out_dir))
    paths.append(_html_report(paths, out_dir, report))
    return paths
