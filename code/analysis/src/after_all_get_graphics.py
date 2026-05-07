#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Set, Any, Optional

import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import textwrap


def _wrap_label(s: str, width: int = 16) -> str:
    return "\n".join(textwrap.wrap(s, width=width, break_long_words=False, break_on_hyphens=False))

def _norm_tag(s: str) -> str:
    return (s or "").strip().lower()

MODEL_DARK_GREEN   = "#04261B"  # darker than before
MODEL_BRIGHT_GREEN = "#52B15A"  # clearly brighter, still not neon
MODEL_STRONG_GRAY  = "#595959"  # strong gray for qwen_3b

DARK_PINE     = MODEL_DARK_GREEN          # reference lines, thresholds
FOREST_GREEN  = MODEL_DARK_GREEN          # original bars
MOSS_GREEN    = MODEL_BRIGHT_GREEN        # paraphrase bars / points

LIGHT_GRAY = "#9496B4"
SUPER_DARK_GRAY = "#2D2E39"

MODEL_COLORS = {
    "gemma_2b": MODEL_DARK_GREEN,
    "gemma_9b": MODEL_BRIGHT_GREEN,
    "qwen_3b": MODEL_STRONG_GRAY,
}

plt.rcParams["axes.prop_cycle"] = cycler(color=[MODEL_DARK_GREEN, MODEL_BRIGHT_GREEN, MODEL_STRONG_GRAY])

WARM_YELLOW = "#F2C94C"
MID_LIGHT_GREEN = "#E8F6EA"
GREEN_YELLOW_DIVERGING = LinearSegmentedColormap.from_list(
    "GreenYellowDiverging",
    [MODEL_DARK_GREEN, MID_LIGHT_GREEN, WARM_YELLOW],
    N=256,
)

METRIC_LABELS = [
    "Task Fulfilment/Relevance",   # 0
    "Usefulness/Actionability",    # 1
    "Factual Accuracy/Verifiability",  # 2
    "Efficiency, Depth, & Completeness",  # 3
    "Reasoning Quality & Transparency",    # 4
    "Tone & Likeability",          # 5
    "Adaption to Context",         # 6 (typo preserved for consistency)
    "Safety & Bias Avoidance",     # 7
    "Structuring, Formating, & UX",# 8 (typos preserved)
    "Creativity",                  # 9
]

def prettify_key(key: str) -> str:
    """Remove instruct_/instruction_ prefix. Special-case instruction_original -> Original."""
    if key == "instruction_original":
        return "Original"
    if key.startswith("instruct_"):
        core = key[len("instruct_"):]
    elif key.startswith("instruction_"):
        core = key[len("instruction_"):]
    else:
        core = key
    parts = core.split("_")
    titled = []
    for p in parts:
        titled.append("AAVE" if p.lower() == "aave" else p.capitalize())
    return " ".join(titled)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def parse_kv_list(items: List[str], sep: str = "=") -> Dict[str, str]:
    """Parse ['k=v', ...] into dict."""
    out = {}
    for it in items:
        if sep not in it:
            raise ValueError(f"Expected KEY{sep}VALUE; got: {it}")
        k, v = it.split(sep, 1)
        out[k.strip()] = v.strip()
    return out

def parse_scores_multi(items: List[str]) -> Dict[str, Dict[str, str]]:
    """
    Parse ['dataset:model=path', ...] into nested dict:
      scores[dataset][model] = path
    """
    out: Dict[str, Dict[str, str]] = {}
    for it in items:
        if "=" not in it:
            raise ValueError(f"Expected DATASET:MODEL=PATH; got: {it}")
        left, path = it.split("=", 1)
        if ":" not in left:
            raise ValueError(f"Expected DATASET:MODEL=PATH; got: {it}")
        ds, model = left.split(":", 1)
        ds = ds.strip()
        model = model.strip()
        out.setdefault(ds, {})[model] = path.strip()
    return out


def collect_styles_from_prompts(prompts: List[dict]) -> Set[str]:
    styles = set()
    for obj in prompts:
        for k in obj.keys():
            if k.startswith("instruct_") or k == "instruction_original":
                styles.add(k)
    return styles

def build_prompts_map(prompts: List[dict]) -> Dict[int, dict]:
    out = {}
    for obj in prompts:
        pid = obj.get("prompt_count")
        if pid is None:
            continue
        out[int(pid)] = obj
    return out

def build_cp_maps(cp_list: List[dict]) -> Tuple[Dict[int, Dict[str, int]], Dict[str, int]]:
    """Return: (cp_by_id[pid][style]=score, pass_counts[style]=count of score in {4,5})."""
    cp_by_id: Dict[int, Dict[str, int]] = {}
    pass_counts: Dict[str, int] = {}
    for item in cp_list:
        pid = item.get("prompt_count")
        scores = item.get("scores", {})
        if pid is None:
            continue
        pid = int(pid)
        cp_by_id[pid] = {}
        for style, sc in scores.items():
            try:
                sc_int = int(sc)
            except Exception:
                sc_int = int(round(float(sc)))
            cp_by_id[pid][style] = sc_int
            if sc_int in (4, 5):
                pass_counts[style] = pass_counts.get(style, 0) + 1
    return cp_by_id, pass_counts

def build_scores_map(scores_list: List[dict], src_name: str, src_path: str) -> Dict[int, Dict[str, List[float]]]:
    """
    Expect items like:
      {"prompt_count": 1, "instruction_original": [10 floats], "instruct_xxx": [10 floats], ...}
    """
    out: Dict[int, Dict[str, List[float]]] = {}
    bad_shape_examples = 0
    missing_pid = 0
    for item in scores_list:
        pid = item.get("prompt_count")
        if pid is None:
            missing_pid += 1
            continue
        pid = int(pid)
        out[pid] = {}
        for k, v in item.items():
            if k in ("prompt_count",):
                continue
            if isinstance(v, list) and len(v) == 10 and all(isinstance(x, (int, float)) for x in v):
                out[pid][k] = [float(x) for x in v]
            else:
                bad_shape_examples += 1
    if not out:
        print(
            f"[error] '{src_name}' scores file has no usable numeric 10-metric arrays.\n"
            f"        Path: {src_path}\n",
            flush=True
        )
    else:
        if missing_pid:
            print(f"[warn] {src_name}: {missing_pid} items had no prompt_count and were skipped.", flush=True)
        if bad_shape_examples:
            print(f"[warn] {src_name}: {bad_shape_examples} fields were not 10-float arrays and were skipped.", flush=True)
        cnt = sum(len(v) for v in out.values())
        print(f"[info] Loaded {cnt} style-metric arrays for {src_name}.", flush=True)
    return out

def intersect_prompt_ids(*maps: Iterable[Dict[int, Any]]) -> List[int]:
    sets = [set(m.keys()) for m in maps if m]
    if not sets:
        return []
    inter = set.intersection(*sets)
    return sorted(list(inter))

def choose_prompt_ids(ids: List[int], max_samples: int = None) -> List[int]:
    if max_samples is None or max_samples <= 0 or max_samples >= len(ids):
        return ids
    return ids[:max_samples]

def select_styles(all_styles: Set[str], pass_counts: Dict[str, int], min_ok: int, filter_keys: Set[str]) -> List[str]:
    """Return ordered list with 'instruction_original' first, then other selected styles sorted by pretty name."""
    qualified = {s for s in all_styles if s.startswith("instruct_") and pass_counts.get(s, 0) >= min_ok}
    if filter_keys:
        qualified = {s for s in qualified if s in filter_keys}
    final_styles = ["instruction_original"] + sorted(qualified, key=lambda s: prettify_key(s))
    return final_styles


def tf_from(scores_10: List[float]) -> float:
    return float(scores_10[0]) if scores_10 else float("nan")

def bootstrap_ci_mean(xs: List[float], rng: np.random.Generator, n_boot: int = 2000, alpha: float = 0.05) -> Tuple[float, float]:
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    if len(xs) == 0:
        return (float("nan"), float("nan"))
    arr = np.array(xs, dtype=float)
    n = len(arr)
    if n == 1:
        return (float(arr[0]), float(arr[0]))
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = arr[idx].mean(axis=1)
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return (lo, hi)

def spearman_corr(x: List[float], y: List[float]) -> float:
    x = [float(v) for v in x]
    y = [float(v) for v in y]
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    try:
        from scipy import stats  # type: ignore
        r, _ = stats.spearmanr(x, y)
        return float(r)
    except Exception:
        def rankdata(a):
            tmp = sorted((v, i) for i, v in enumerate(a))
            ranks = [0.0] * len(a)
            i = 0
            while i < len(tmp):
                j = i
                while j < len(tmp) and tmp[j][0] == tmp[i][0]:
                    j += 1
                avg = (i + 1 + j) / 2.0
                for k in range(i, j):
                    ranks[tmp[k][1]] = avg
                i = j
            return ranks
        rx = rankdata(x)
        ry = rankdata(y)
        return float(np.corrcoef(rx, ry)[0, 1])


def compute_core_stats_for_model(
    scores_map: Dict[int, Dict[str, List[float]]],
    prompts_by_id: Dict[int, dict],
    cp_by_id: Dict[int, Dict[str, int]],
    prompt_ids: List[int],
    selected_styles: List[str],
    failure_threshold: float,
    rng: np.random.Generator,
    n_boot: int = 2000,
) -> dict:
    orig_tfs = []
    for pid in prompt_ids:
        sc = scores_map.get(pid, {}).get("instruction_original")
        if sc is None:
            continue
        orig_tfs.append(tf_from(sc))
    orig_tf_mean = float(np.mean(orig_tfs)) if orig_tfs else float("nan")

    par_tfs = []
    par_metrics = []
    total = 0
    fails = 0
    for pid in prompt_ids:
        per = scores_map.get(pid, {})
        for style in selected_styles:
            if style == "instruction_original":
                continue
            cp = cp_by_id.get(pid, {}).get(style)
            if cp not in (4, 5):
                continue
            sc = per.get(style)
            if sc is None:
                continue
            tf = tf_from(sc)
            if not math.isnan(tf):
                par_tfs.append(tf)
                par_metrics.append(sc)
                total += 1
                if tf <= failure_threshold:
                    fails += 1
    par_tf_mean = float(np.mean(par_tfs)) if par_tfs else float("nan")
    par_tf_ci = bootstrap_ci_mean(par_tfs, rng, n_boot=n_boot)
    fail_rate = (fails / total) if total else float("nan")

    style_means = []
    for style in selected_styles:
        if style == "instruction_original":
            continue
        tfs = []
        for pid in prompt_ids:
            per = scores_map.get(pid, {})
            sc = per.get(style)
            if sc is None:
                continue
            cp = cp_by_id.get(pid, {}).get(style)
            if cp not in (4, 5):
                continue
            tf = tf_from(sc)
            if not math.isnan(tf):
                tfs.append(tf)
        if tfs:
            style_means.append((style, float(np.mean(tfs))))
    style_means.sort(key=lambda t: t[1])
    worst_style, worst_tf = (style_means[0] if style_means else ("NA", float("nan")))
    robustness_gap = (orig_tf_mean - worst_tf) if (not math.isnan(orig_tf_mean) and not math.isnan(worst_tf)) else float("nan")

    min_tfs = []
    for pid in prompt_ids:
        per = scores_map.get(pid, {})
        tfs = []
        for style in selected_styles:
            if style == "instruction_original":
                continue
            sc = per.get(style)
            if sc is None:
                continue
            cp = cp_by_id.get(pid, {}).get(style)
            if cp not in (4, 5):
                continue
            tf = tf_from(sc)
            if not math.isnan(tf):
                tfs.append(tf)
        if tfs:
            min_tfs.append(float(np.min(tfs)))

    return {
        "orig_tf_mean": orig_tf_mean,
        "par_tf_mean": par_tf_mean,
        "par_tf_ci": par_tf_ci,
        "fail_rate": fail_rate,
        "worst_style": worst_style,
        "worst_tf": worst_tf,
        "robustness_gap": robustness_gap,
        "min_tfs_per_prompt": min_tfs,
        "par_metrics_rows": par_metrics,  # for metric tradeoff aggregation if needed
    }

def compute_style_delta_tf(
    scores_map: Dict[int, Dict[str, List[float]]],
    cp_by_id: Dict[int, Dict[str, int]],
    prompt_ids: List[int],
    selected_styles: List[str],
) -> Dict[str, float]:
    deltas = {}
    for style in selected_styles:
        if style == "instruction_original":
            continue
        diffs = []
        for pid in prompt_ids:
            per = scores_map.get(pid, {})
            o = per.get("instruction_original")
            s = per.get(style)
            if o is None or s is None:
                continue
            cp = cp_by_id.get(pid, {}).get(style)
            if cp not in (4, 5):
                continue
            diffs.append(tf_from(s) - tf_from(o))
        deltas[style] = float(np.mean(diffs)) if diffs else float("nan")
    return deltas

def compute_tag_deltas_and_metric_deltas(
    scores_map: Dict[int, Dict[str, List[float]]],
    cp_by_id: Dict[int, Dict[str, int]],
    prompt_ids: List[int],
    selected_styles: List[str],
    tags_map: Dict[str, List[str]],
) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
    """
    For each tag:
      ΔTF_tag = mean over prompts of (mean TF over styles in tag - TF_original), with CP>=4 per style.
    For each tag and metric k:
      Δmetric_tag[k] = mean over prompts of (mean metric_k over styles in tag - metric_k_original).
    """
    tag_to_styles: Dict[str, List[str]] = {}
    for style in selected_styles:
        if style == "instruction_original":
            continue
        for tg in tags_map.get(style, []) or []:
            tag_to_styles.setdefault(tg, []).append(style)

    tag_delta_tf: Dict[str, float] = {}
    tag_delta_metrics: Dict[str, List[float]] = {}

    for tg, styles in tag_to_styles.items():
        tf_diffs = []
        metric_diffs = [[] for _ in range(10)]
        for pid in prompt_ids:
            per = scores_map.get(pid, {})
            o = per.get("instruction_original")
            if o is None:
                continue

            vals_tf = []
            vals_m = [[] for _ in range(10)]
            for st in styles:
                sc = per.get(st)
                if sc is None:
                    continue
                cp = cp_by_id.get(pid, {}).get(st)
                if cp not in (4, 5):
                    continue
                vals_tf.append(tf_from(sc))
                for k in range(10):
                    vals_m[k].append(float(sc[k]))

            if not vals_tf:
                continue

            tf_diffs.append(float(np.mean(vals_tf)) - float(o[0]))
            for k in range(10):
                metric_diffs[k].append(float(np.mean(vals_m[k])) - float(o[k]))

        tag_delta_tf[tg] = float(np.mean(tf_diffs)) if tf_diffs else float("nan")
        tag_delta_metrics[tg] = [float(np.mean(metric_diffs[k])) if metric_diffs[k] else float("nan") for k in range(10)]

    return tag_delta_tf, tag_delta_metrics


def savefig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()

def plot_core_orig_vs_par_mean(
    out_path: Path,
    datasets_order: List[str],
    models_order: List[str],
    core: Dict[str, Dict[str, dict]],
):
    n_ds = len(datasets_order)
    fig_w = max(12, 4 * n_ds)
    plt.figure(figsize=(fig_w, 5.5))
    for i, ds in enumerate(datasets_order, start=1):
        ax = plt.subplot(1, n_ds, i)
        x = np.arange(len(models_order))
        width = 0.35

        orig = []
        par = []
        yerr = [[], []]  # lower, upper for par
        for m in models_order:
            st = core.get(ds, {}).get(m, {})
            orig.append(st.get("orig_tf_mean", np.nan))
            par.append(st.get("par_tf_mean", np.nan))
            lo, hi = st.get("par_tf_ci", (np.nan, np.nan))
            mu = st.get("par_tf_mean", np.nan)
            yerr[0].append(mu - lo if not np.isnan(mu) and not np.isnan(lo) else np.nan)
            yerr[1].append(hi - mu if not np.isnan(mu) and not np.isnan(hi) else np.nan)

        ax.bar(x - width/2, orig, width, label="Original", color=FOREST_GREEN)
        ax.bar(x + width/2, par, width, label="Paraphrase (CP≥4)", color=MOSS_GREEN, yerr=np.array(yerr), capsize=3)

        ax.set_title(ds)
        ax.set_xticks(x)
        ax.set_xticklabels(models_order, rotation=20, ha="right")
        ax.set_ylim(0, 10)
        ax.set_ylabel("TF score" if i == 1 else "")
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
        if i == 1:
            ax.legend(frameon=False)

    plt.suptitle("Core robustness: Original TF vs pooled paraphrase TF (CP≥4) with CI", y=1.03, fontsize=14)
    savefig(out_path)

def plot_avg_vs_worstcase(
    out_path: Path,
    datasets_order: List[str],
    models_order: List[str],
    core: Dict[str, Dict[str, dict]],
):
    n_ds = len(datasets_order)
    fig_w = max(12, 4 * n_ds)
    plt.figure(figsize=(fig_w, 5.5))

    for i, ds in enumerate(datasets_order, start=1):
        ax = plt.subplot(1, n_ds, i)
        x = np.arange(len(models_order))

        for j, m in enumerate(models_order):
            st = core.get(ds, {}).get(m, {})
            o = st.get("orig_tf_mean", np.nan)
            p = st.get("par_tf_mean", np.nan)
            w = st.get("worst_tf", np.nan)
            ax.plot(
                [j, j, j],
                [w, p, o],
                marker="o",
                linewidth=2.0,
                alpha=0.9,
                color=MODEL_COLORS.get(m, SUPER_DARK_GRAY),
            )

        ax.set_title(ds)
        ax.set_xticks(x)
        ax.set_xticklabels(models_order, rotation=20, ha="right")
        ax.set_ylim(0, 10)
        ax.set_ylabel("TF score" if i == 1 else "")
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

        if i == 1:
            ax.text(0.02, 0.03, "Points per model: (worst style mean) → (pooled paraphrase mean) → (original mean)",
                    transform=ax.transAxes, fontsize=9, alpha=0.8)

    plt.suptitle("Average vs worst-case robustness (TF): worst-style vs pooled paraphrase vs original", y=1.03, fontsize=14)
    savefig(out_path)

def plot_fragility_cdf_min_tf(
    out_path: Path,
    datasets_order: List[str],
    models_order: List[str],
    core: Dict[str, Dict[str, dict]],
    failure_threshold: float,
):
    n_ds = len(datasets_order)
    fig_w = max(12, 4 * n_ds)
    plt.figure(figsize=(fig_w, 5.5))

    for i, ds in enumerate(datasets_order, start=1):
        ax = plt.subplot(1, n_ds, i)
        for m in models_order:
            mins = core.get(ds, {}).get(m, {}).get("min_tfs_per_prompt", [])
            mins = [float(v) for v in mins if v is not None and not math.isnan(float(v))]
            if not mins:
                continue
            mins_sorted = np.sort(np.array(mins))
            y = np.arange(1, len(mins_sorted) + 1) / len(mins_sorted)
            ax.plot(
                mins_sorted,
                y,
                linewidth=2.0,
                label=m,
                color=MODEL_COLORS.get(m, SUPER_DARK_GRAY),
            )

        ax.axvline(failure_threshold, linestyle="--", linewidth=1.2, alpha=0.8, color=DARK_PINE)
        ax.set_title(ds)
        ax.set_xlabel("min TF across styles (per prompt, CP≥4)")
        ax.set_ylabel("CDF" if i == 1 else "")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
        if i == 1:
            ax.legend(frameon=False)

    plt.suptitle(f"Prompt fragility: CDF of worst-case (min) TF under paraphrasing (CP≥4). Threshold TF≤{failure_threshold:g}", y=1.03, fontsize=14)
    savefig(out_path)

def plot_tag_harm_heatmap(
    out_path: Path,
    datasets_order: List[str],
    top_tags: List[str],
    tag_delta_avg: Dict[str, Dict[str, float]],  # ds -> tag -> delta
):
    mat = np.full((len(top_tags), len(datasets_order)), np.nan, dtype=float)
    for r, tg in enumerate(top_tags):
        for c, ds in enumerate(datasets_order):
            mat[r, c] = tag_delta_avg.get(ds, {}).get(tg, np.nan)

    plt.figure(figsize=(max(8, 2.6 + 1.6 * len(datasets_order)), max(6, 0.35 * len(top_tags) + 2.5)))
    ax = plt.gca()
    vmax = np.nanmax(np.abs(mat)) if np.isfinite(np.nanmax(np.abs(mat))) else 1.0
    im = ax.imshow(mat, aspect="auto", vmin=-vmax, vmax=vmax, cmap=GREEN_YELLOW_DIVERGING)
    plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="ΔTF (tag mean − original)")

    ax.set_xticks(np.arange(len(datasets_order)))
    ax.set_xticklabels(datasets_order)
    ax.set_yticks(np.arange(len(top_tags)))
    ax.set_yticklabels([_wrap_label(tg, 22) for tg in top_tags])

    ax.set_title("Top harmful tag families: ΔTF by dataset (averaged over models)")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Tag family")

    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            v = mat[r, c]
            if not np.isnan(v):
                ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=8)

    savefig(out_path)

def plot_metric_tradeoff_heatmap(
    out_path: Path,
    dataset: str,
    top_tags: List[str],
    tag_metric_delta_avg: Dict[str, Dict[str, List[float]]],  # ds -> tag -> [10 deltas]
):
    mat = np.full((len(top_tags), 10), np.nan, dtype=float)
    for r, tg in enumerate(top_tags):
        vals = tag_metric_delta_avg.get(dataset, {}).get(tg, None)
        if vals is None:
            continue
        for k in range(10):
            mat[r, k] = vals[k]

    plt.figure(figsize=(14, max(6, 0.35 * len(top_tags) + 2.5)))
    ax = plt.gca()
    vmax = np.nanmax(np.abs(mat)) if np.isfinite(np.nanmax(np.abs(mat))) else 1.0
    im = ax.imshow(mat, aspect="auto", vmin=-vmax, vmax=vmax, cmap=GREEN_YELLOW_DIVERGING)
    plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="Δ(metric) (tag mean − original)")

    ax.set_xticks(np.arange(10))
    ax.set_xticklabels([_wrap_label(m, 16) for m in METRIC_LABELS], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(top_tags)))
    ax.set_yticklabels([_wrap_label(tg, 22) for tg in top_tags])

    ax.set_title(f"Multi-metric tradeoffs for top harmful tags — {dataset} (averaged over models)")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Tag family")

    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            v = mat[r, c]
            if not np.isnan(v):
                ax.text(c, r, f"{v:.1f}", ha="center", va="center", fontsize=7)

    savefig(out_path)

def plot_cross_model_agreement_scatter_allpairs(
    out_path: Path,
    dataset: str,
    models_order: List[str],
    style_delta: Dict[str, Dict[str, float]],  # model -> style -> delta
):
    pairs = []
    for i in range(len(models_order)):
        for j in range(i + 1, len(models_order)):
            pairs.append((models_order[i], models_order[j]))

    plt.figure(figsize=(5.5 * max(1, len(pairs)), 5.2))
    for idx, (a, b) in enumerate(pairs, start=1):
        ax = plt.subplot(1, len(pairs), idx)
        shared = sorted(set(style_delta.get(a, {}).keys()) & set(style_delta.get(b, {}).keys()))
        x = [style_delta[a].get(s, np.nan) for s in shared]
        y = [style_delta[b].get(s, np.nan) for s in shared]
        xy = [(xx, yy) for xx, yy in zip(x, y) if not (np.isnan(xx) or np.isnan(yy))]
        if xy:
            x2, y2 = zip(*xy)
        else:
            x2, y2 = [], []

        ax.scatter(x2, y2, s=10, alpha=0.5, color=MOSS_GREEN)
        lim = 3.0
        if x2 and y2:
            lim = max(1.5, float(np.nanmax(np.abs(list(x2) + list(y2)))) * 1.1)
        ax.plot([-lim, lim], [-lim, lim], linestyle="--", linewidth=1.0, alpha=0.8, color=DARK_PINE)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)

        r = spearman_corr(list(x2), list(y2)) if len(x2) >= 2 else float("nan")
        ax.set_title(f"{dataset}\n{a} vs {b}\nSpearman ρ={r:.2f}" if not np.isnan(r) else f"{dataset}\n{a} vs {b}")
        ax.set_xlabel("ΔTF (style−orig)")
        ax.set_ylabel("ΔTF (style−orig)" if idx == 1 else "")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    plt.suptitle("Cross-model agreement: style-level ΔTF scatter (paired mean, CP≥4)", y=1.03, fontsize=14)
    savefig(out_path)


def main():
    parser = argparse.ArgumentParser(description="CALIPER paper graphics (all datasets at once)")
    parser.add_argument("--prompts", nargs="+", required=True, help="Dataset prompts: dataset=path (repeat), e.g., alpaca=... gsm8k=... mmlu=...")
    parser.add_argument("--scores", nargs="+", required=True, help="Scores: dataset:model=path (repeat), e.g., alpaca:gemma_2b=... gsm8k:qwen_3b=...")
    parser.add_argument("--tags-json", required=True, help="Path to tags JSON mapping instruct_* to tags")
    parser.add_argument("--content-preservation", required=True, help="Path to content-preservation JSON")
    parser.add_argument("--output-dir", required=True, help="Directory to save graphics")
    parser.add_argument("--datasets-order", default="alpaca,gsm8k,mmlu", help="Comma-separated dataset order for combined plots")
    parser.add_argument("--filter-keys", default="", help="Comma-separated instruct_* keys to include (optional). Original always included.")
    parser.add_argument("--max-samples", type=int, default=None, help="Max number of prompt_count IDs to use per dataset")
    parser.add_argument("--min-occurrences", type=int, default=200, help="Min # CP>=4 occurrences required for a style in subset")
    parser.add_argument("--failure-threshold", type=float, default=3.0, help="TF failure threshold (default 3.0)")
    parser.add_argument("--top-tags", type=int, default=15, help="How many harmful tags to show in heatmaps (default 15)")
    parser.add_argument("--exclude-tags", default="number_swap", help="Comma-separated tag families to ignore entirely (default: number_swap)")
    parser.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap resamples for CIs (default 2000)")
    parser.add_argument("--seed", type=int, default=7, help="RNG seed (default 7)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets_order = [d.strip() for d in args.datasets_order.split(",") if d.strip()]
    prompts_map = parse_kv_list(args.prompts)
    scores_map = parse_scores_multi(args.scores)

    tags_map_raw = load_json(args.tags_json)
    tags_map = {k: v for k, v in tags_map_raw.items() if isinstance(v, list)}

    excluded_tags = set(_norm_tag(t) for t in (args.exclude_tags or "").split(",") if t.strip())
    if excluded_tags:
        tags_map = {
            style: [tg for tg in tgs if _norm_tag(tg) not in excluded_tags]
            for style, tgs in tags_map.items()
        }
        tags_map = {style: tgs for style, tgs in tags_map.items() if tgs}
        print(f"[info] Excluding tag families: {sorted(excluded_tags)}", flush=True)

    cp_list = load_json(args.content_preservation)
    cp_by_id, _ = build_cp_maps(cp_list)

    rng = np.random.default_rng(args.seed)
    filter_keys = set([s.strip() for s in args.filter_keys.split(",") if s.strip()])

    core: Dict[str, Dict[str, dict]] = {}  # ds -> model -> core stats
    style_delta_all: Dict[str, Dict[str, Dict[str, float]]] = {}  # ds -> model -> style -> delta
    tag_delta_all: Dict[str, Dict[str, Dict[str, float]]] = {}    # ds -> model -> tag -> delta
    tag_metric_delta_all: Dict[str, Dict[str, Dict[str, List[float]]]] = {}  # ds -> model -> tag -> [10]
    models_union: Set[str] = set()

    for ds in datasets_order:
        if ds not in prompts_map:
            print(f"[warn] No prompts provided for dataset '{ds}', skipping dataset.", flush=True)
            continue
        if ds not in scores_map or not scores_map[ds]:
            print(f"[warn] No scores provided for dataset '{ds}', skipping dataset.", flush=True)
            continue

        print(f"[info] === Dataset: {ds} ===", flush=True)

        prompts_list = load_json(prompts_map[ds])
        prompts_by_id = build_prompts_map(prompts_list)
        all_styles = collect_styles_from_prompts(prompts_list)

        if filter_keys:
            unknown = [k for k in filter_keys if k not in all_styles]
            if unknown:
                print(f"[warn] {ds}: some --filter-keys not in prompts styles and will be ignored: {unknown}", flush=True)

        model_to_scores: Dict[str, Dict[int, Dict[str, List[float]]]] = {}
        for model, path in scores_map[ds].items():
            model_to_scores[model] = build_scores_map(load_json(path), f"{ds}:{model}", path)
            models_union.add(model)

        common_ids = intersect_prompt_ids(prompts_by_id, cp_by_id, *(model_to_scores[m] for m in model_to_scores.keys()))
        if not common_ids:
            print(f"[error] {ds}: no overlapping prompt_count IDs across prompts, CP, and all model score files.", flush=True)
            continue
        prompt_ids = choose_prompt_ids(common_ids, args.max_samples)
        print(f"[info] {ds}: Using {len(prompt_ids)} prompt IDs.", flush=True)

        pass_counts_subset: Dict[str, int] = {}
        for pid in prompt_ids:
            per = cp_by_id.get(pid, {})
            for style, sc in per.items():
                if sc in (4, 5):
                    pass_counts_subset[style] = pass_counts_subset.get(style, 0) + 1

        fk = {k for k in filter_keys if k in all_styles} if filter_keys else set()
        selected_styles = select_styles(all_styles, pass_counts_subset, args.min_occurrences, fk)

        core.setdefault(ds, {})
        style_delta_all.setdefault(ds, {})
        tag_delta_all.setdefault(ds, {})
        tag_metric_delta_all.setdefault(ds, {})

        for model, smap in model_to_scores.items():
            cstats = compute_core_stats_for_model(
                scores_map=smap,
                prompts_by_id=prompts_by_id,
                cp_by_id=cp_by_id,
                prompt_ids=prompt_ids,
                selected_styles=selected_styles,
                failure_threshold=args.failure_threshold,
                rng=rng,
                n_boot=args.bootstrap,
            )
            core[ds][model] = cstats

            style_delta_all[ds][model] = compute_style_delta_tf(
                scores_map=smap,
                cp_by_id=cp_by_id,
                prompt_ids=prompt_ids,
                selected_styles=selected_styles,
            )

            tdtf, tdmet = compute_tag_deltas_and_metric_deltas(
                scores_map=smap,
                cp_by_id=cp_by_id,
                prompt_ids=prompt_ids,
                selected_styles=selected_styles,
                tags_map=tags_map,
            )
            tag_delta_all[ds][model] = tdtf
            tag_metric_delta_all[ds][model] = tdmet

    models_order: List[str] = []
    for ds in datasets_order:
        if ds in scores_map:
            for m in scores_map[ds].keys():
                if m not in models_order:
                    models_order.append(m)
    for m in sorted(models_union):
        if m not in models_order:
            models_order.append(m)

    tag_acc: Dict[str, List[float]] = {}
    for ds in datasets_order:
        for m in models_order:
            for tg, v in tag_delta_all.get(ds, {}).get(m, {}).items():
                if _norm_tag(tg) in excluded_tags:
                    continue
                if v is None or math.isnan(v):
                    continue
                tag_acc.setdefault(tg, []).append(float(v))

    tag_mean = {tg: float(np.mean(vs)) for tg, vs in tag_acc.items() if vs}
    top_tags = [tg for tg, _ in sorted(tag_mean.items(), key=lambda kv: kv[1])[:args.top_tags]]

    tag_delta_avg_by_ds: Dict[str, Dict[str, float]] = {}
    tag_metric_delta_avg_by_ds: Dict[str, Dict[str, List[float]]] = {}
    for ds in datasets_order:
        tag_delta_avg_by_ds.setdefault(ds, {})
        tag_metric_delta_avg_by_ds.setdefault(ds, {})
        for tg in top_tags:
            vals = []
            met_vals = [[] for _ in range(10)]
            for m in models_order:
                v = tag_delta_all.get(ds, {}).get(m, {}).get(tg, float("nan"))
                if v is not None and not math.isnan(v):
                    vals.append(float(v))
                mv = tag_metric_delta_all.get(ds, {}).get(m, {}).get(tg, None)
                if mv is not None and any(not math.isnan(float(x)) for x in mv):
                    for k in range(10):
                        if mv[k] is not None and not math.isnan(float(mv[k])):
                            met_vals[k].append(float(mv[k]))
            tag_delta_avg_by_ds[ds][tg] = float(np.mean(vals)) if vals else float("nan")
            tag_metric_delta_avg_by_ds[ds][tg] = [float(np.mean(met_vals[k])) if met_vals[k] else float("nan") for k in range(10)]


    plot_core_orig_vs_par_mean(
        out_path=out_dir / "paper_core_tf_orig_vs_paraphrase.png",
        datasets_order=datasets_order,
        models_order=models_order,
        core=core,
    )

    plot_avg_vs_worstcase(
        out_path=out_dir / "paper_tf_average_vs_worstcase.png",
        datasets_order=datasets_order,
        models_order=models_order,
        core=core,
    )

    plot_fragility_cdf_min_tf(
        out_path=out_dir / "paper_fragility_cdf_min_tf.png",
        datasets_order=datasets_order,
        models_order=models_order,
        core=core,
        failure_threshold=args.failure_threshold,
    )

    plot_tag_harm_heatmap(
        out_path=out_dir / "paper_tag_harm_heatmap_deltaTF.png",
        datasets_order=datasets_order,
        top_tags=top_tags,
        tag_delta_avg=tag_delta_avg_by_ds,
    )

    for ds in datasets_order:
        if ds not in style_delta_all:
            continue
        plot_cross_model_agreement_scatter_allpairs(
            out_path=out_dir / f"paper_agreement_scatter_allpairs_{ds}.png",
            dataset=ds,
            models_order=models_order,
            style_delta=style_delta_all[ds],
        )

    for ds in datasets_order:
        plot_metric_tradeoff_heatmap(
            out_path=out_dir / f"paper_metric_tradeoffs_heatmap_{ds}.png",
            dataset=ds,
            top_tags=top_tags,
            tag_metric_delta_avg=tag_metric_delta_avg_by_ds,
        )

    print("[done] Wrote paper graphics to:", out_dir)
    print("  - paper_core_tf_orig_vs_paraphrase.png")
    print("  - paper_tf_average_vs_worstcase.png")
    print("  - paper_fragility_cdf_min_tf.png")
    print("  - paper_tag_harm_heatmap_deltaTF.png")
    for ds in datasets_order:
        print(f"  - paper_agreement_scatter_allpairs_{ds}.png")
        print(f"  - paper_metric_tradeoffs_heatmap_{ds}.png")

if __name__ == "__main__":
    main()
