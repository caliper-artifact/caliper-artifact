#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Set, Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import textwrap


def _wrap_label(s: str, width: int = 16) -> str:
    return "\n".join(textwrap.wrap(s, width=width, break_long_words=False, break_on_hyphens=False))

MANUAL_LABEL_OVERRIDES = {
    "Usefulness/Actionability": "Usefulness/\nActionability",
    "Safety & Bias Avoidance": "Safety &\nBias Avoidance",
    "Structuring, Formating, & UX": "Structuring,\nFormating, &\nUX",
    "Factual Accuracy/Verifiability": "Factual Accuracy/\nVerifiability",
    "Factural Accuracy/Verifiability": "Factural Accuracy/\nVerifiability",
}

RADIAL_NUDGE = {
    "Safety & Bias Avoidance":  +0.06,
    "Task Fulfilment/Relevance": -0.03,
}

def _norm_tag(s: str) -> str:
    return (s or "").strip().lower()

STRONG_BLUE = "#04129B"
VIOLET = "#704EEB"
LIGHT_GRAY = "#9496B4"
MEDIUM_GRAY = "#5A5B6F"
SUPER_DARK_GRAY = "#2D2E39"

FOREST_GREEN = STRONG_BLUE
DARK_GRAY = SUPER_DARK_GRAY

METRIC_LABELS = [
    "Task Fulfilment/Relevance",   # 0
    "Usefulness/Actionability",    # 1
    "Factual Accuracy/Verifiability",  # 2
    "Efficiency, Depth, & Completeness",  # 3
    "Reasoning Quality & Transparency",    # 4
    "Tone & Likeability",          # 5
    "Adaption to Context",         # 6 (typo preserved for continuity)
    "Safety & Bias Avoidance",     # 7
    "Structuring, Formating, & UX",# 8 (typos preserved for continuity)
    "Creativity",                  # 9
]


def parse_scores_arg(scores_list: List[str]) -> List[Tuple[str, str]]:
    """Parse entries like 'NAME=path/to.json' preserving order."""
    pairs = []
    for item in scores_list:
        if "=" not in item:
            raise ValueError(f"--scores entries must be NAME=PATH; got: {item}")
        name, path = item.split("=", 1)
        pairs.append((name.strip(), path.strip()))
    if len(pairs) != 3:
        print(f"[warn] Expected 3 entries in --scores; got {len(pairs)}. Proceeding anyway.", flush=True)
    return pairs

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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
    if key == "instruct_polite":
        return "Polite"
    if key == "instruct_rude":
        return "Rude"
    parts = core.split("_")
    titled = []
    for p in parts:
        titled.append("AAVE" if p.lower() == "aave" else p.capitalize())
    return " ".join(titled)


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
            f"        Path: {src_path}\n"
            f"        This script expects a metrics JSON with, for each prompt_count, per-style arrays of 10 floats.\n",
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

def build_answers_map(answers_list: List[dict], src_name: str, src_path: str) -> Dict[int, Dict[str, str]]:
    """
    Optional: expect items like:
      {"prompt_count": 1, "instruction_original": "...", "instruct_xxx": "...", ...}
    Values may be strings (model outputs).
    """
    out: Dict[int, Dict[str, str]] = {}
    missing_pid = 0
    for item in answers_list:
        pid = item.get("prompt_count")
        if pid is None:
            missing_pid += 1
            continue
        pid = int(pid)
        out.setdefault(pid, {})
        for k, v in item.items():
            if k == "prompt_count":
                continue
            if isinstance(v, str):
                out[pid][k] = v
    if missing_pid:
        print(f"[warn] {src_name}: {missing_pid} answer items had no prompt_count and were skipped.", flush=True)
    if out:
        print(f"[info] Loaded answers for {src_name}: {len(out)} prompt_count IDs.", flush=True)
    else:
        print(f"[warn] Loaded 0 usable answer strings for {src_name} from {src_path}", flush=True)
    return out

def tf_from(scores_10: List[float]) -> float:
    return float(scores_10[0]) if scores_10 else float("nan")

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

def build_dataset_tf_vectors(
    series_scores: Dict[int, Dict[str, List[float]]],
    selected_styles: List[str],
    cp_by_id: Dict[int, Dict[str, int]],
    prompt_ids: List[int],
) -> Dict[str, List[float]]:
    """Collect TF (index 0) for styles; CP filter applies to non-Original."""
    result: Dict[str, List[float]] = {style: [] for style in selected_styles}
    for pid in prompt_ids:
        per_id = series_scores.get(pid, {})
        for style in selected_styles:
            if style != "instruction_original":
                cp_style_score = cp_by_id.get(pid, {}).get(style, None)
                if cp_style_score not in (4, 5):
                    continue
            scores10 = per_id.get(style)
            if scores10 is None:
                continue
            tf = tf_from(scores10)
            if not math.isnan(tf):
                result[style].append(tf)
    return result


def distinct_gray(i: int, total: int) -> str:
    if total <= 1:
        return SUPER_DARK_GRAY
    t = i / (total - 1)
    def interp_hex(h1, h2, t):
        c1 = tuple(int(h1[i:i+2], 16) for i in (1,3,5))
        c2 = tuple(int(h2[i:i+2], 16) for i in (1,3,5))
        c = tuple(int(round(c1[j] + (c2[j]-c1[j])*t)) for j in range(3))
        return "#" + "".join(f"{v:02X}" for v in c)
    return interp_hex(SUPER_DARK_GRAY, LIGHT_GRAY, t)

def grouped_boxplot_by_styles(
    tf_data_per_series: List[Tuple[str, Dict[str, List[float]]]],
    styles_order: List[str],
    out_path: Path,
    title: str,
):
    """Draw grouped boxplots: x=styles, one box per series (e.g., model)."""
    plt.figure(figsize=(max(12, 1.2 * len(styles_order)), 6))
    ax = plt.gca()

    num_series = len(tf_data_per_series)
    width = 0.18
    positions_base = np.arange(len(styles_order))

    def _series_colors(n: int) -> List[str]:
        if n <= 0:
            return []
        colors = [FOREST_GREEN]
        if n >= 2:
            colors.append(VIOLET)
        if n > 2:
            steps = n - 2
            for i in range(steps):
                colors.append(distinct_gray(i, max(1, steps)))
        return colors

    facecolors = _series_colors(num_series)
    hatches = [None, None] + ["///"] * max(0, num_series - 2)

    means_for_legend = []

    for si, (sname, tf_map) in enumerate(tf_data_per_series):
        pos = positions_base + (si - (num_series-1)/2) * (width + 0.02)
        data = [tf_map.get(style, []) for style in styles_order]
        safe_data = [d if len(d) else [np.nan] for d in data]

        bp = ax.boxplot(
            safe_data,
            positions=pos,
            widths=width,
            patch_artist=True,
            showmeans=True,
            meanline=True,
            whis=1.5,
            manage_ticks=False
        )

        fc = facecolors[si % len(facecolors)]
        for patch in bp['boxes']:
            patch.set_facecolor(fc)
            patch.set_edgecolor("#333333")
            patch.set_linewidth(0.8)
            if hatches[si]:
                patch.set_hatch(hatches[si])

        for med in bp['medians']:
            med.set_color("#222222"); med.set_linewidth(1.2)
        for mean in bp['means']:
            mean.set_color("#111111"); mean.set_linewidth(1.2)
        for w in bp['whiskers']:
            w.set_color("#333333"); w.set_linewidth(0.8)
        for cap in bp['caps']:
            cap.set_color("#333333"); cap.set_linewidth(0.8)
        for fl in bp['fliers']:
            fl.set_markerfacecolor("#666666"); fl.set_markeredgecolor("#666666"); fl.set_alpha(0.5)

        flat = [v for style in styles_order for v in tf_map.get(style, [])]
        mu = np.nan if not flat else float(np.mean(flat))
        means_for_legend.append((sname, mu, fc, hatches[si]))

    ax.set_title(title, fontsize=14, pad=12)
    ax.set_ylabel("TF score", fontsize=12)
    pretty_labels = [prettify_key(s) for s in styles_order]
    ax.set_xticks(positions_base)
    ax.set_xticklabels(pretty_labels, rotation=30, ha="right")

    ax.set_ylim(0, 10)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

    from matplotlib.patches import Patch
    legend_patches = []
    for name, mu, fc, hatch in means_for_legend:
        label = f"{name} (μ={mu:.2f})" if not math.isnan(mu) else f"{name}"
        patch = Patch(facecolor=fc, edgecolor="#333333", hatch=hatch if hatch else None, label=label)
        legend_patches.append(patch)

    ax.legend(
        handles=legend_patches,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False
    )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

def grouped_boxplot_by_tags(
    tf_data_per_series_by_style: List[Tuple[str, Dict[str, List[float]]]],
    styles_order: List[str],
    tags_map: Dict[str, List[str]],
    out_path: Path,
    title: str,
    filter_tags: set | None = None,
):
    canonical_for_norm = {}
    def canon(tag: str) -> str:
        n = _norm_tag(tag)
        if n not in canonical_for_norm:
            canonical_for_norm[n] = tag
        return canonical_for_norm[n]

    norm_filter = set(_norm_tag(t) for t in (filter_tags or set()) if t)

    series_tag_map: List[Tuple[str, Dict[str, List[float]]]] = []
    all_tags = set()
    for sname, style_map in tf_data_per_series_by_style:
        tmap: Dict[str, List[float]] = {}
        tmap.setdefault("Original", [])
        for style in styles_order:
            tfs = style_map.get(style, [])
            if style == "instruction_original":
                tmap["Original"].extend(tfs)
                all_tags.add("Original")
                continue
            for tag in tags_map.get(style, []) or []:
                ctag = canon(tag)
                if norm_filter and _norm_tag(ctag) not in norm_filter:
                    continue
                tmap.setdefault(ctag, []).extend(tfs)
                all_tags.add(ctag)
        series_tag_map.append((sname, tmap))

    if norm_filter:
        all_tags = {t for t in all_tags if (_norm_tag(t) in norm_filter) or (t == "Original")}

    tags_sorted = sorted(all_tags, key=lambda s: ("~" if s == "Original" else "") + s.lower())
    try:
        print(f"[info] Tags in plot: {tags_sorted}", flush=True)
    except Exception:
        pass

    grouped_boxplot_by_styles(series_tag_map, tags_sorted, out_path, title)


def radar_prepare_axes(labels: List[str]):
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_ylim(0, 10)
    ax.grid(color="#AAAAAA", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.set_rlabel_position(0)

    ax.set_thetagrids(np.degrees(angles[:-1]))
    for t in ax.get_xticklabels():
        t.set_visible(False)

    rmax = ax.get_ylim()[1]
    base_frac = 1.12
    for theta_deg, original_txt in zip(np.degrees(angles[:-1]), labels):
        theta = np.deg2rad(theta_deg)

        txt_for_wrap = MANUAL_LABEL_OVERRIDES.get(original_txt, original_txt)

        if "\n" in txt_for_wrap:
            wrapped = txt_for_wrap
        else:
            wrapped = _wrap_label(txt_for_wrap, width=16)
        n_lines = wrapped.count("\n") + 1

        label_r = base_frac + 0.02 * (n_lines - 1)
        label_r += RADIAL_NUDGE.get(original_txt, 0.0)

        ax.text(
            theta, rmax * label_r, wrapped,
            ha="center", va="center",
            fontsize=10, linespacing=1.0, clip_on=False
        )

    return ax, angles

def radar_plot(ax, angles, values: List[float], line_color: str, fill_alpha: float = 0.25, label: str = None):
    vals = list(values) + values[:1]
    ax.plot(angles, vals, linewidth=1.6, color=line_color, label=label)
    ax.fill(angles, vals, color=line_color, alpha=fill_alpha)

def compute_mean_10(scores_map: Dict[int, Dict[str, List[float]]], style: str,
                    cp_by_id: Dict[int, Dict[str, int]], prompt_ids: List[int]) -> List[float]:
    rows = []
    for pid in prompt_ids:
        sc = scores_map.get(pid, {}).get(style)
        if sc is None:
            continue
        if style != "instruction_original":
            cp = cp_by_id.get(pid, {}).get(style)
            if cp not in (4, 5):
                continue
        rows.append(sc)
    if not rows:
        return [float("nan")] * 10
    arr = np.array(rows, dtype=float)
    return list(np.nanmean(arr, axis=0))


def safe_mean(xs: List[float]) -> float:
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    return float(np.mean(xs)) if xs else float("nan")

def safe_std(xs: List[float]) -> float:
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    return float(np.std(xs, ddof=1)) if len(xs) >= 2 else float("nan")

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

def bh_fdr(pvals: List[float]) -> List[float]:
    """Benjamini-Hochberg FDR correction; returns adjusted p-values in original order."""
    n = len(pvals)
    indexed = [(i, p) for i, p in enumerate(pvals)]
    indexed.sort(key=lambda x: (float("inf") if (x[1] is None or math.isnan(x[1])) else x[1]))
    adj = [float("nan")] * n
    prev = 1.0
    for rank, (i, p) in enumerate(indexed, start=1):
        if p is None or math.isnan(p):
            adj[i] = float("nan")
            continue
        val = p * n / rank
        val = min(val, 1.0)
        prev = min(prev, val) if rank > 1 else val
        adj[i] = prev
    pairs = [(i, adj[i], pvals[i]) for i in range(n)]
    pairs.sort(key=lambda t: (float("inf") if (t[2] is None or math.isnan(t[2])) else t[2]))
    min_so_far = 1.0
    for i, a, p in reversed(pairs):
        if p is None or math.isnan(p):
            continue
        min_so_far = min(min_so_far, adj[i])
        adj[i] = min_so_far
    return adj

def paired_diffs_metric(
    scores_map: Dict[int, Dict[str, List[float]]],
    style: str,
    metric_idx: int,
    cp_by_id: Dict[int, Dict[str, int]],
    prompt_ids: List[int],
) -> List[float]:
    diffs = []
    for pid in prompt_ids:
        per = scores_map.get(pid, {})
        o = per.get("instruction_original")
        s = per.get(style)
        if o is None or s is None:
            continue
        if style != "instruction_original":
            cp = cp_by_id.get(pid, {}).get(style)
            if cp not in (4, 5):
                continue
        diffs.append(float(s[metric_idx] - o[metric_idx]))
    return diffs

def paired_ttest_pvalue(diffs: List[float]) -> float:
    """Two-sided paired t-test p-value on diffs vs 0. Uses SciPy if available; else normal approx."""
    diffs = [d for d in diffs if d is not None and not math.isnan(d)]
    n = len(diffs)
    if n < 2:
        return float("nan")
    try:
        from scipy import stats  # type: ignore
        t, p = stats.ttest_1samp(diffs, popmean=0.0, alternative="two-sided")
        return float(p)
    except Exception:
        mu = statistics.mean(diffs)
        sd = statistics.pstdev(diffs) if n > 1 else 0.0
        if sd == 0.0:
            return 1.0 if mu == 0.0 else 0.0
        t = mu / (sd / math.sqrt(n))
        z = abs(t)
        p = math.erfc(z / math.sqrt(2.0))
        return float(p)

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

def corr_matrix(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 2 or arr.shape[0] < 2:
        return np.full((arr.shape[1], arr.shape[1]), np.nan)
    return np.corrcoef(arr, rowvar=False)

def surface_stats(text: str) -> Dict[str, float]:
    if text is None:
        return {"chars": float("nan"), "words": float("nan"), "digits_pct": float("nan"),
                "non_ascii_pct": float("nan"), "punct_pct": float("nan"), "upper_pct": float("nan")}
    s = text
    n = len(s)
    if n == 0:
        return {"chars": 0.0, "words": 0.0, "digits_pct": 0.0, "non_ascii_pct": 0.0, "punct_pct": 0.0, "upper_pct": 0.0}
    words = len(s.split())
    digits = sum(c.isdigit() for c in s)
    non_ascii = sum(ord(c) > 127 for c in s)
    punct = sum(bool(re.match(r"[^\w\s]", c)) for c in s)
    upper = sum(c.isupper() for c in s)
    return {
        "chars": float(n),
        "words": float(words),
        "digits_pct": float(digits / n),
        "non_ascii_pct": float(non_ascii / n),
        "punct_pct": float(punct / n),
        "upper_pct": float(upper / n),
    }

def md_escape(s: str) -> str:
    return (s or "").replace("|", "\\|").replace("\n", "<br>")

def fmt(x: float, nd: int = 2) -> str:
    if x is None or math.isnan(x):
        return "NA"
    return f"{x:.{nd}f}"

def md_table(headers: List[str], rows: List[List[str]]) -> str:
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def is_nan(x: float) -> bool:
    try:
        return x is None or math.isnan(float(x))
    except Exception:
        return True

def trunc(s: str, n: int = 260) -> str:
    s = s or ""
    s = s.replace("\n", " ").strip()
    return (s[:n] + "…") if len(s) > n else s

def pct(x: float, nd: int = 1) -> str:
    if x is None or is_nan(x):
        return "NA"
    return f"{100.0 * float(x):.{nd}f}%"

def top_k_sorted(items: List[dict], key: str, k: int = 5, reverse: bool = False) -> List[dict]:
    def _k(v):
        vv = v.get(key, float("nan"))
        return float("inf") if is_nan(vv) else float(vv)
    return sorted(items, key=_k, reverse=reverse)[:k]

def metric_delta_highlights(orig10: List[float], par10: List[float], topn: int = 3) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    deltas = []
    for i, lab in enumerate(METRIC_LABELS):
        o = orig10[i] if i < len(orig10) else float("nan")
        p = par10[i] if i < len(par10) else float("nan")
        d = (p - o) if (not is_nan(o) and not is_nan(p)) else float("nan")
        deltas.append((lab, d))
    drops = sorted([x for x in deltas if not is_nan(x[1])], key=lambda t: t[1])[:topn]
    gains = sorted([x for x in deltas if not is_nan(x[1])], key=lambda t: t[1], reverse=True)[:topn]
    return drops, gains

def write_csv(path: Path, headers: List[str], rows: List[List[Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for r in rows:
            w.writerow(r)


def main():
    parser = argparse.ArgumentParser(description="CALIPER paraphrase robustness analysis + summary")
    parser.add_argument("--prompts", required=True, help="Path to prompts JSON (paraphrases)")
    parser.add_argument("--scores", nargs="+", required=True, help="Score series as NAME=PATH (e.g., model=metrics.json)")
    parser.add_argument("--answers", nargs="*", default=None, help="Optional answers series as NAME=PATH (e.g., model=answers.json)")
    parser.add_argument("--tags-json", required=True, help="Path to tags JSON mapping instruct_* to tags")
    parser.add_argument("--filter-tags", default="", help="Comma-separated tags; affects tag plot (Original always included)")
    parser.add_argument("--content-preservation", required=True, help="Path to content-preservation JSON")
    parser.add_argument("--output-dir", required=True, help="Directory to save graphics + summary")
    parser.add_argument("--filter-keys", default="", help="Comma-separated instruct_* keys to include (intersected with threshold filter). 'instruction_original' always included.")
    parser.add_argument("--max-samples", type=int, default=None, help="Max number of prompt_count IDs to use")
    parser.add_argument("--min-occurrences", type=int, default=200, help="Min # of CP 4/5 occurrences required for a style (default 200)")
    parser.add_argument("--failure-threshold", type=float, default=3.0, help="TF failure threshold (default: TF<=3.0)")
    parser.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap resamples for CIs (default 2000)")
    parser.add_argument("--seed", type=int, default=7, help="RNG seed for bootstrap (default 7)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts_list = load_json(args.prompts)
    prompts_by_id = build_prompts_map(prompts_list)
    tags_map_raw = load_json(args.tags_json)
    cp_list = load_json(args.content_preservation)

    all_styles = collect_styles_from_prompts(prompts_list)
    cp_by_id, pass_counts_global = build_cp_maps(cp_list)

    filter_keys = set([s.strip() for s in args.filter_keys.split(",") if s.strip()])
    unknown = [k for k in filter_keys if k not in all_styles]
    if unknown:
        print(f"[warn] Some --filter-keys not in prompts styles and will be ignored: {unknown}", flush=True)
        filter_keys = {k for k in filter_keys if k in all_styles}

    series_pairs = parse_scores_arg(args.scores)
    series_scores: List[Tuple[str, Dict[int, Dict[str, List[float]]]]] = []
    for name, path in series_pairs:
        scores_list = load_json(path)
        smap = build_scores_map(scores_list, name, path)
        series_scores.append((name, smap))

    series_answers: Dict[str, Dict[int, Dict[str, str]]] = {}
    if args.answers:
        ans_pairs = parse_scores_arg(args.answers)  # same NAME=PATH format
        for name, path in ans_pairs:
            ans_list = load_json(path)
            amap = build_answers_map(ans_list, name, path)
            series_answers[name] = amap

    if not any(len(smap) for _, smap in series_scores):
        print("[fatal] No usable numeric score arrays found in ANY --scores file.", flush=True)
        return

    common_ids = intersect_prompt_ids(*(dict(m) for _, m in series_scores), cp_by_id, prompts_by_id)
    if not common_ids:
        print("[error] No overlapping prompt_count IDs across scores, CP JSON, and prompts JSON.", flush=True)
        return
    prompt_ids = choose_prompt_ids(common_ids, args.max_samples)
    print(f"[info] Using {len(prompt_ids)} prompt IDs.", flush=True)

    pass_counts_subset: Dict[str, int] = {}
    cp_hist = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0}
    for pid in prompt_ids:
        per = cp_by_id.get(pid, {})
        for style, sc in per.items():
            if isinstance(sc, int) and 0 <= sc <= 5:
                cp_hist[sc] = cp_hist.get(sc, 0) + 1
            if sc in (4, 5):
                pass_counts_subset[style] = pass_counts_subset.get(style, 0) + 1

    selected_styles = select_styles(all_styles, pass_counts_subset, args.min_occurrences, filter_keys)
    if len(selected_styles) <= 1:
        print("[error] After filtering, only 'Original' remained. Consider adjusting --filter-keys or --min-occurrences.", flush=True)

    tags_map = {k: v for k, v in tags_map_raw.items() if isinstance(v, list)}

    tf_maps_per_series: List[Tuple[str, Dict[str, List[float]]]] = []
    for sname, smap in series_scores:
        tf_map = build_dataset_tf_vectors(smap, selected_styles, cp_by_id, prompt_ids)
        tf_maps_per_series.append((sname, tf_map))

    out1 = out_dir / "tf_scores_by_series_styles.png"
    grouped_boxplot_by_styles(tf_maps_per_series, selected_styles, out1, "TF Scores By Series (e.g., Models)")

    out2 = out_dir / "tf_scores_by_series_tags.png"
    filter_tags_raw = [s for s in (args.filter_tags or "").split(",") if s.strip()]
    filter_tags = set(_norm_tag(s) for s in filter_tags_raw)
    known_tags = set()
    for v in tags_map.values():
        if isinstance(v, list):
            known_tags.update(_norm_tag(x) for x in v)
    if filter_tags and not (filter_tags <= known_tags):
        unknown_tags = sorted(list(filter_tags - known_tags))
        if unknown_tags:
            print(f"[warn] Some --filter-tags not recognized in tags JSON (normalized): {unknown_tags}", flush=True)
    grouped_boxplot_by_tags(tf_maps_per_series, selected_styles, tags_map, out2, "TF Scores By Series (by Tags)", filter_tags=filter_tags)

    first_name, first_scores = series_scores[0]
    plt.figure(figsize=(8, 8))
    ax, angles = radar_prepare_axes(METRIC_LABELS)

    any_plotted = False
    for idx, style in enumerate(selected_styles):
        mean10 = compute_mean_10(first_scores, style, cp_by_id, prompt_ids)
        if all(math.isnan(x) for x in mean10):
            print(f"[warn] No numeric data for style {style}; skipping.", flush=True)
            continue
        color = FOREST_GREEN if style == "instruction_original" else distinct_gray(idx-1, max(1, len(selected_styles)-1))
        radar_plot(ax, angles, mean10, line_color=color, fill_alpha=0.25, label=prettify_key(style))
        any_plotted = True

    rmax = 10
    ax.plot([angles[0], angles[0]], [0, rmax], color=FOREST_GREEN, linewidth=2.0, alpha=0.6)
    title_suffix = "" if any_plotted else " (no styles had numeric data)"
    ax.set_title(f"Radar (first series: {first_name}) All Metrics (N={len(prompt_ids)} prompts){title_suffix}", va="bottom", fontsize=14, pad=40)
    if any_plotted:
        ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1), frameon=False)
    plt.tight_layout()
    out3 = out_dir / "radar_first_series_all_styles.png"
    plt.savefig(out3, dpi=200, bbox_inches="tight")
    plt.close()


    rng = np.random.default_rng(args.seed)
    failure_thr = float(args.failure_threshold)

    dataset_label = Path(args.prompts).stem

    tag_to_styles: Dict[str, List[str]] = {}
    for style, tgs in tags_map.items():
        for tg in tgs:
            tag_to_styles.setdefault(tg, []).append(style)

    surface_rows = []
    for style in selected_styles:
        if style == "instruction_original":
            continue
        stats_list = []
        for pid in prompt_ids:
            pobj = prompts_by_id.get(pid, {})
            text = pobj.get(style)
            if not isinstance(text, str):
                continue
            cp = cp_by_id.get(pid, {}).get(style)
            if cp not in (4, 5):
                continue
            stats_list.append(surface_stats(text))
        if not stats_list:
            continue
        keys = list(stats_list[0].keys())
        means = {k: float(np.mean([d[k] for d in stats_list])) for k in keys}
        surface_rows.append([style, prettify_key(style), len(stats_list),
                             means["chars"], means["words"], means["digits_pct"], means["non_ascii_pct"], means["punct_pct"], means["upper_pct"]])

    surface_csv = out_dir / "surface_stats_styles.csv"
    write_csv(surface_csv,
              ["style_key", "style_pretty", "n", "chars_mean", "words_mean", "digits_pct_mean", "non_ascii_pct_mean", "punct_pct_mean", "upper_pct_mean"],
              surface_rows)

    cp_style_rows = []
    for style in selected_styles:
        if style == "instruction_original":
            continue
        vals = []
        pass_45 = 0
        for pid in prompt_ids:
            sc = cp_by_id.get(pid, {}).get(style)
            if sc is None:
                continue
            vals.append(int(sc))
            if sc in (4, 5):
                pass_45 += 1
        if not vals:
            continue
        cp_style_rows.append([
            style, prettify_key(style), len(vals),
            float(np.mean(vals)), float(np.median(vals)),
            pass_45, pass_45 / max(1, len(vals))
        ])

    cp_style_csv = out_dir / "content_preservation_by_style.csv"
    write_csv(cp_style_csv,
              ["style_key", "style_pretty", "n_scored", "cp_mean", "cp_median", "n_cp_ge4", "frac_cp_ge4"],
              cp_style_rows)

    cp_tag_rows = []
    all_tags_present = set()
    for style, tgs in tags_map.items():
        for tg in tgs:
            all_tags_present.add(tg)
    for tg in sorted(all_tags_present, key=lambda s: s.lower()):
        styles = [s for s in tag_to_styles.get(tg, []) if s in all_styles]
        if not styles:
            continue
        per_prompt_means = []
        for pid in prompt_ids:
            cps = []
            for style in styles:
                sc = cp_by_id.get(pid, {}).get(style)
                if sc is None:
                    continue
                cps.append(int(sc))
            if cps:
                per_prompt_means.append(float(np.mean(cps)))
        if not per_prompt_means:
            continue
        cp_tag_rows.append([
            tg, len(styles), len(per_prompt_means),
            float(np.mean(per_prompt_means)), float(np.median(per_prompt_means))
        ])
    cp_tag_csv = out_dir / "content_preservation_by_tag.csv"
    write_csv(cp_tag_csv, ["tag", "n_styles", "n_prompts_with_any_style", "cp_mean", "cp_median"], cp_tag_rows)

    style_stats_all_rows = []
    tag_stats_all_rows = []
    summary_blocks = []
    special_core_rows: List[List[str]] = []          # per-series core robustness table
    special_per_series_blocks: List[str] = []        # compact blocks per series
    special_examples_global: List[str] = []          # short failure examples across series
    special_cross_series_block: str = ""             # correlation / agreement block

    style_drop_by_series: Dict[str, Dict[str, float]] = {}

    for sname, smap in series_scores:
        orig_rows = []
        for pid in prompt_ids:
            sc = smap.get(pid, {}).get("instruction_original")
            if sc is not None:
                orig_rows.append(sc)
        orig_mean10 = list(np.nanmean(np.array(orig_rows, dtype=float), axis=0)) if orig_rows else [float("nan")]*10

        pooled_par = []
        pooled_tf = []
        pooled_fail = 0
        pooled_total = 0

        for pid in prompt_ids:
            per = smap.get(pid, {})
            for style in selected_styles:
                if style == "instruction_original":
                    continue
                cp = cp_by_id.get(pid, {}).get(style)
                if cp not in (4, 5):
                    continue
                sc = per.get(style)
                if sc is None:
                    continue
                pooled_par.append(sc)
                tf = float(sc[0])
                pooled_tf.append(tf)
                pooled_total += 1
                if tf <= failure_thr:
                    pooled_fail += 1

        pooled_mean10 = list(np.nanmean(np.array(pooled_par, dtype=float), axis=0)) if pooled_par else [float("nan")]*10
        pooled_tf_mean = safe_mean(pooled_tf)
        pooled_tf_ci = bootstrap_ci_mean(pooled_tf, rng, n_boot=args.bootstrap)

        style_means_tf = []
        for style in selected_styles:
            if style == "instruction_original":
                continue
            tfs = []
            for pid in prompt_ids:
                per = smap.get(pid, {})
                sc = per.get(style)
                if sc is None:
                    continue
                cp = cp_by_id.get(pid, {}).get(style)
                if cp not in (4, 5):
                    continue
                tfs.append(float(sc[0]))
            if tfs:
                style_means_tf.append((style, float(np.mean(tfs)), len(tfs)))
        style_means_tf.sort(key=lambda t: t[1])
        worst_style, worst_tf, worst_n = style_means_tf[0] if style_means_tf else ("NA", float("nan"), 0)

        pvals = []
        style_rows_for_this_series = []
        for style in selected_styles:
            if style == "instruction_original":
                continue
            diffs = paired_diffs_metric(smap, style, 0, cp_by_id, prompt_ids)
            n = len(diffs)
            mean_delta = safe_mean(diffs)
            ci_lo, ci_hi = bootstrap_ci_mean(diffs, rng, n_boot=args.bootstrap)
            p = paired_ttest_pvalue(diffs)
            pvals.append(p)
            tfs = []
            fails = 0
            for pid in prompt_ids:
                per = smap.get(pid, {})
                sc = per.get(style)
                if sc is None:
                    continue
                cp = cp_by_id.get(pid, {}).get(style)
                if cp not in (4, 5):
                    continue
                tf = float(sc[0])
                tfs.append(tf)
                if tf <= failure_thr:
                    fails += 1
            mean_tf = safe_mean(tfs)
            fail_rate = (fails / len(tfs)) if tfs else float("nan")

            style_rows_for_this_series.append({
                "series": sname,
                "style_key": style,
                "style_pretty": prettify_key(style),
                "n": n,
                "mean_tf": mean_tf,
                "mean_delta_tf": mean_delta,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "p": p,
                "fail_rate": fail_rate,
            })

        p_adj = bh_fdr(pvals)
        for i, row in enumerate(style_rows_for_this_series):
            row["p_adj"] = p_adj[i]

        style_drop_by_series[sname] = {r["style_key"]: r["mean_delta_tf"] for r in style_rows_for_this_series}
        for r in style_rows_for_this_series:
            style_stats_all_rows.append([
                r["series"], r["style_key"], r["style_pretty"], r["n"],
                r["mean_tf"], r["mean_delta_tf"], r["ci_lo"], r["ci_hi"], r["p"], r["p_adj"], r["fail_rate"]
            ])

        tag_rows_for_this_series = []
        tag_pvals = []
        tags_seen = set()
        for style in selected_styles:
            if style == "instruction_original":
                continue
            for tg in tags_map.get(style, []) or []:
                tags_seen.add(tg)

        for tg in sorted(tags_seen, key=lambda s: s.lower()):
            styles_for_tag = [st for st in tag_to_styles.get(tg, []) if st in selected_styles and st != "instruction_original"]
            if not styles_for_tag:
                continue

            diffs = []
            tag_mean_tfs = []
            tag_instance_total = 0
            tag_instance_fails = 0
            prompts_with_any = 0

            for pid in prompt_ids:
                per = smap.get(pid, {})
                orig = per.get("instruction_original")
                if orig is None:
                    continue
                orig_tf = float(orig[0])

                tfs_here = []
                for st in styles_for_tag:
                    sc = per.get(st)
                    if sc is None:
                        continue
                    cp = cp_by_id.get(pid, {}).get(st)
                    if cp not in (4, 5):
                        continue
                    tf = float(sc[0])
                    tfs_here.append(tf)
                    tag_mean_tfs.append(tf)
                    tag_instance_total += 1
                    if tf <= failure_thr:
                        tag_instance_fails += 1

                if tfs_here:
                    prompts_with_any += 1
                    diffs.append(float(np.mean(tfs_here) - orig_tf))

            if not diffs:
                continue

            mean_delta = safe_mean(diffs)
            ci_lo, ci_hi = bootstrap_ci_mean(diffs, rng, n_boot=args.bootstrap)
            p = paired_ttest_pvalue(diffs)
            tag_pvals.append(p)

            mean_tf = safe_mean(tag_mean_tfs)
            fail_rate = (tag_instance_fails / tag_instance_total) if tag_instance_total else float("nan")

            tag_rows_for_this_series.append({
                "series": sname,
                "tag": tg,
                "n_prompts": len(diffs),
                "n_styles": len(styles_for_tag),
                "mean_tf": mean_tf,
                "mean_delta_tf": mean_delta,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "p": p,
                "fail_rate": fail_rate,
                "prompts_with_any": prompts_with_any,
            })

        tag_p_adj = bh_fdr(tag_pvals)
        for i, row in enumerate(tag_rows_for_this_series):
            row["p_adj"] = tag_p_adj[i]

        for r in tag_rows_for_this_series:
            tag_stats_all_rows.append([
                r["series"], r["tag"], r["n_prompts"], r["n_styles"],
                r["mean_tf"], r["mean_delta_tf"], r["ci_lo"], r["ci_hi"],
                r["p"], r["p_adj"], r["fail_rate"], r["prompts_with_any"]
            ])

        prompt_frag_rows = []
        min_tfs = []
        fail_prompts = 0
        any_par_prompts = 0
        for pid in prompt_ids:
            per = smap.get(pid, {})
            orig = per.get("instruction_original")
            if orig is None:
                continue
            orig_tf = float(orig[0])

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
                tfs.append(float(sc[0]))

            if not tfs:
                prompt_frag_rows.append([sname, pid, orig_tf, float("nan"), float("nan"), 0, 0, float("nan")])
                continue

            any_par_prompts += 1
            mn = float(np.min(tfs))
            sd = float(np.std(tfs, ddof=1)) if len(tfs) >= 2 else 0.0
            fails = int(sum(1 for v in tfs if v <= failure_thr))
            total = int(len(tfs))
            frac_fail = float(fails / total) if total else float("nan")
            min_tfs.append(mn)
            if mn <= failure_thr:
                fail_prompts += 1
            prompt_frag_rows.append([sname, pid, orig_tf, mn, sd, fails, total, frac_fail])

        frag_csv = out_dir / f"prompt_fragility_{sname}.csv"
        write_csv(frag_csv,
                  ["series", "prompt_count", "orig_tf", "min_tf", "std_tf", "n_fail_styles", "n_total_styles", "frac_fail_styles"],
                  prompt_frag_rows)

        examples = []
        instances = []
        for pid in prompt_ids:
            per = smap.get(pid, {})
            orig = per.get("instruction_original")
            if orig is None:
                continue
            for style in selected_styles:
                if style == "instruction_original":
                    continue
                cp = cp_by_id.get(pid, {}).get(style)
                if cp not in (4, 5):
                    continue
                sc = per.get(style)
                if sc is None:
                    continue
                instances.append((float(sc[0]), pid, style, sc))
        instances.sort(key=lambda t: t[0])
        for tfv, pid, style, sc in instances[:8]:
            pobj = prompts_by_id.get(pid, {})
            orig_text = pobj.get("instruction_original", "")
            para_text = pobj.get(style, "")
            rec = {
                "pid": pid,
                "style": style,
                "style_pretty": prettify_key(style),
                "tf": tfv,
                "metrics": sc,
                "orig_prompt": orig_text,
                "para_prompt": para_text,
            }
            amap = series_answers.get(sname)
            if amap:
                rec["orig_answer"] = (amap.get(pid, {}).get("instruction_original") or "")
                rec["para_answer"] = (amap.get(pid, {}).get(style) or "")
            examples.append(rec)

        style_rows_sorted = sorted(style_rows_for_this_series, key=lambda r: (float("inf") if math.isnan(r["mean_delta_tf"]) else r["mean_delta_tf"]))
        worst10 = style_rows_sorted[:10]
        best10 = list(reversed(style_rows_sorted[-10:]))

        tag_rows_sorted = sorted(tag_rows_for_this_series, key=lambda r: (float("inf") if math.isnan(r["mean_delta_tf"]) else r["mean_delta_tf"]))
        worst_tags = tag_rows_sorted[:10]
        best_tags = list(reversed(tag_rows_sorted[-10:]))

        worst5_styles = [r["style_key"] for r in style_rows_sorted[:5]]
        multi_metric_rows = []
        for st in worst5_styles:
            mean_st = compute_mean_10(smap, st, cp_by_id, prompt_ids)
            mean_o = orig_mean10
            delta = [(mean_st[i] - mean_o[i]) if (not math.isnan(mean_st[i]) and not math.isnan(mean_o[i])) else float("nan") for i in range(10)]
            multi_metric_rows.append((st, delta))

        sb = []
        sb.append(f"### Series: `{sname}`")
        sb.append("")
        sb.append("**Core performance (Task Fulfilment/Relevance = TF)**")
        sb.append("")
        sb.append(md_table(
            ["Statistic", "Value"],
            [
                ["Original TF mean", fmt(orig_mean10[0])],
                ["Paraphrase TF mean (pooled, CP≥4)", f"{fmt(pooled_tf_mean)}  (CI {fmt(pooled_tf_ci[0])}–{fmt(pooled_tf_ci[1])})"],
                ["Paraphrase failure rate (pooled, TF≤{:.1f})".format(failure_thr), fmt((pooled_fail / pooled_total) if pooled_total else float('nan'))],
                ["Worst style by mean TF (CP≥4)", f"{md_escape(prettify_key(worst_style))}  (TF={fmt(worst_tf)}, n={worst_n})"],
                ["Robustness gap (Original TF − worst-style mean TF)", fmt(orig_mean10[0] - worst_tf if (not math.isnan(orig_mean10[0]) and not math.isnan(worst_tf)) else float('nan'))],
                ["Prompts whose *min* TF across styles is ≤{:.1f}".format(failure_thr), f"{fail_prompts}/{max(1, any_par_prompts)}"],
            ]
        ))
        sb.append("")

        sb.append("**Original vs pooled paraphrases (mean over 10 metrics)**")
        sb.append("")
        rows = []
        for i, lab in enumerate(METRIC_LABELS):
            rows.append([md_escape(lab), fmt(orig_mean10[i]), fmt(pooled_mean10[i]), fmt(pooled_mean10[i] - orig_mean10[i] if (not math.isnan(pooled_mean10[i]) and not math.isnan(orig_mean10[i])) else float("nan"))])
        sb.append(md_table(["Metric", "Original mean", "Paraphrase mean", "Δ (Par − Orig)"], rows))
        sb.append("")

        sb.append("**Worst styles by paired ΔTF (style − original), CP≥4**")
        sb.append("")
        sb.append(md_table(
            ["Style", "n", "Mean TF", "Mean ΔTF", "CI(ΔTF)", "p_adj", "Fail rate"],
            [[
                md_escape(r["style_pretty"]),
                str(r["n"]),
                fmt(r["mean_tf"]),
                fmt(r["mean_delta_tf"]),
                f"{fmt(r['ci_lo'])}–{fmt(r['ci_hi'])}",
                fmt(r.get("p_adj", float("nan")), 4),
                fmt(r["fail_rate"]),
            ] for r in worst10]
        ))
        sb.append("")
        sb.append("**Best styles by paired ΔTF (style − original), CP≥4**")
        sb.append("")
        sb.append(md_table(
            ["Style", "n", "Mean TF", "Mean ΔTF", "CI(ΔTF)", "p_adj", "Fail rate"],
            [[
                md_escape(r["style_pretty"]),
                str(r["n"]),
                fmt(r["mean_tf"]),
                fmt(r["mean_delta_tf"]),
                f"{fmt(r['ci_lo'])}–{fmt(r['ci_hi'])}",
                fmt(r.get("p_adj", float("nan")), 4),
                fmt(r["fail_rate"]),
            ] for r in best10]
        ))
        sb.append("")

        if tag_rows_for_this_series:
            sb.append("**Worst tag families by paired ΔTF (tag mean − original), CP≥4**")
            sb.append("")
            sb.append(md_table(
                ["Tag", "n_prompts", "n_styles", "Mean TF", "Mean ΔTF", "CI(ΔTF)", "p_adj", "Fail rate"],
                [[
                    md_escape(r["tag"]),
                    str(r["n_prompts"]),
                    str(r["n_styles"]),
                    fmt(r["mean_tf"]),
                    fmt(r["mean_delta_tf"]),
                    f"{fmt(r['ci_lo'])}–{fmt(r['ci_hi'])}",
                    fmt(r.get("p_adj", float("nan")), 4),
                    fmt(r["fail_rate"]),
                ] for r in worst_tags]
            ))
            sb.append("")
            sb.append("**Best tag families by paired ΔTF (tag mean − original), CP≥4**")
            sb.append("")
            sb.append(md_table(
                ["Tag", "n_prompts", "n_styles", "Mean TF", "Mean ΔTF", "CI(ΔTF)", "p_adj", "Fail rate"],
                [[
                    md_escape(r["tag"]),
                    str(r["n_prompts"]),
                    str(r["n_styles"]),
                    fmt(r["mean_tf"]),
                    fmt(r["mean_delta_tf"]),
                    f"{fmt(r['ci_lo'])}–{fmt(r['ci_hi'])}",
                    fmt(r.get("p_adj", float("nan")), 4),
                    fmt(r["fail_rate"]),
                ] for r in best_tags]
            ))
            sb.append("")

        sb.append("**Metric deltas for the 5 worst styles (mean metric(style) − mean metric(original))**")
        sb.append("")
        mm_rows = []
        for st, delta in multi_metric_rows:
            mm_rows.append([md_escape(prettify_key(st))] + [fmt(d) for d in delta])
        sb.append(md_table(["Style"] + [md_escape(m) for m in METRIC_LABELS], mm_rows))
        sb.append("")

        sb.append("**Failure-mode examples (lowest TF instances, CP≥4)**")
        sb.append("")
        for ex in examples:
            sb.append(f"- **prompt_count {ex['pid']}** · **{md_escape(ex['style_pretty'])}** · TF={fmt(ex['tf'])}")
            sb.append("")
            sb.append("  - Original prompt:")
            sb.append(f"    - {md_escape(ex['orig_prompt'])}")
            sb.append("  - Paraphrase prompt:")
            sb.append(f"    - {md_escape(ex['para_prompt'])}")
            sb.append("  - Metrics (10): " + ", ".join(fmt(float(v)) for v in ex["metrics"]))
            if "orig_answer" in ex:
                sb.append("  - Original answer (model output):")
                sb.append(f"    - {md_escape(ex['orig_answer'][:800])}{'…' if len(ex['orig_answer'])>800 else ''}")
                sb.append("  - Paraphrase answer (model output):")
                sb.append(f"    - {md_escape(ex['para_answer'][:800])}{'…' if len(ex['para_answer'])>800 else ''}")
            sb.append("")

        summary_blocks.append("\n".join(sb))


        delta_tf = pooled_tf_mean - orig_mean10[0] if (not is_nan(pooled_tf_mean) and not is_nan(orig_mean10[0])) else float("nan")
        pooled_fail_rate = (pooled_fail / pooled_total) if pooled_total else float("nan")
        robust_gap = (orig_mean10[0] - worst_tf) if (not is_nan(orig_mean10[0]) and not is_nan(worst_tf)) else float("nan")

        special_core_rows.append([
            sname,
            fmt(orig_mean10[0]),
            f"{fmt(pooled_tf_mean)} (CI {fmt(pooled_tf_ci[0])}–{fmt(pooled_tf_ci[1])})",
            fmt(delta_tf),
            fmt(pooled_fail_rate),
            f"{prettify_key(worst_style)} (TF={fmt(worst_tf)})",
            fmt(robust_gap),
        ])

        worst_styles_5 = top_k_sorted(style_rows_sorted, key="mean_delta_tf", k=5, reverse=False)
        best_styles_3  = top_k_sorted(style_rows_sorted, key="mean_delta_tf", k=3, reverse=True)

        worst_tags_5 = top_k_sorted(tag_rows_sorted, key="mean_delta_tf", k=5, reverse=False) if tag_rows_sorted else []
        best_tags_3  = top_k_sorted(tag_rows_sorted, key="mean_delta_tf", k=3, reverse=True) if tag_rows_sorted else []

        frag_min_vals = []
        frag_frac_fail_vals = []
        for row in prompt_frag_rows:
            mn = row[3]
            fracf = row[7]
            if mn is not None and not is_nan(mn):
                frag_min_vals.append(float(mn))
            if fracf is not None and not is_nan(fracf):
                frag_frac_fail_vals.append(float(fracf))

        frac_prompts_min_fail = (fail_prompts / max(1, any_par_prompts)) if any_par_prompts else float("nan")
        med_min_tf = float(np.median(frag_min_vals)) if frag_min_vals else float("nan")
        p10_min_tf = float(np.quantile(frag_min_vals, 0.10)) if len(frag_min_vals) >= 5 else float("nan")
        med_frac_fail = float(np.median(frag_frac_fail_vals)) if frag_frac_fail_vals else float("nan")

        drops, gains = metric_delta_highlights(orig_mean10, pooled_mean10, topn=3)

        ex_lines = []
        for ex in examples[:3]:
            pid = ex["pid"]
            style_key = ex["style"]
            cpv = cp_by_id.get(pid, {}).get(style_key, None)
            ex_lines.append(
                f"- prompt_count **{pid}** · **{ex['style_pretty']}** · CP={cpv if cpv is not None else 'NA'} · TF={fmt(ex['tf'])}\n"
                f"  - Original: {trunc(ex['orig_prompt'], 220)}\n"
                f"  - Paraphrase: {trunc(ex['para_prompt'], 220)}"
            )

        block = []
        block.append(f"## Model/Series: `{sname}`")
        block.append("")
        block.append("**Core robustness (TF = Task Fulfilment/Relevance, CP≥4 paraphrases)**")
        block.append("")
        block.append(md_table(
            ["Item", "Value"],
            [
                ["Original TF mean", fmt(orig_mean10[0])],
                ["Pooled paraphrase TF mean", f"{fmt(pooled_tf_mean)} (CI {fmt(pooled_tf_ci[0])}–{fmt(pooled_tf_ci[1])})"],
                ["ΔTF (pooled − original)", fmt(delta_tf)],
                [f"Paraphrase failure rate (TF≤{failure_thr:.1f})", fmt(pooled_fail_rate)],
                ["Worst style (by mean TF)", f"{prettify_key(worst_style)} (TF={fmt(worst_tf)}, n={worst_n})"],
                ["Robustness gap (orig − worst)", fmt(robust_gap)],
                [f"Prompts with min TF across styles ≤{failure_thr:.1f}", f"{fail_prompts}/{max(1, any_par_prompts)} ({pct(frac_prompts_min_fail)})"],
                ["Median min TF across styles (per prompt)", fmt(med_min_tf)],
                ["10th percentile min TF across styles", fmt(p10_min_tf)],
                ["Median frac failing styles per prompt", fmt(med_frac_fail)],
            ]
        ))
        block.append("")

        block.append("**Top harmful styles (paired ΔTF = TF_style − TF_original, CP≥4)**")
        block.append("")
        block.append(md_table(
            ["Style", "Mean ΔTF", "CI(ΔTF)", "p_adj", "Fail rate"],
            [[
                md_escape(r["style_pretty"]),
                fmt(r["mean_delta_tf"]),
                f"{fmt(r['ci_lo'])}–{fmt(r['ci_hi'])}",
                fmt(r.get("p_adj", float("nan")), 4),
                fmt(r["fail_rate"]),
            ] for r in worst_styles_5]
        ))
        block.append("")

        block.append("**Most robust styles (largest paired ΔTF)**")
        block.append("")
        block.append(md_table(
            ["Style", "Mean ΔTF", "CI(ΔTF)", "p_adj"],
            [[
                md_escape(r["style_pretty"]),
                fmt(r["mean_delta_tf"]),
                f"{fmt(r['ci_lo'])}–{fmt(r['ci_hi'])}",
                fmt(r.get("p_adj", float("nan")), 4),
            ] for r in best_styles_3]
        ))
        block.append("")

        if worst_tags_5:
            block.append("**Top harmful tag families (paired ΔTF = tag-mean − original, CP≥4)**")
            block.append("")
            block.append(md_table(
                ["Tag", "Mean ΔTF", "CI(ΔTF)", "p_adj", "Fail rate"],
                [[
                    md_escape(r["tag"]),
                    fmt(r["mean_delta_tf"]),
                    f"{fmt(r['ci_lo'])}–{fmt(r['ci_hi'])}",
                    fmt(r.get("p_adj", float("nan")), 4),
                    fmt(r["fail_rate"]),
                ] for r in worst_tags_5]
            ))
            block.append("")
            block.append("**Most robust tag families**")
            block.append("")
            block.append(md_table(
                ["Tag", "Mean ΔTF", "CI(ΔTF)", "p_adj"],
                [[
                    md_escape(r["tag"]),
                    fmt(r["mean_delta_tf"]),
                    f"{fmt(r['ci_lo'])}–{fmt(r['ci_hi'])}",
                    fmt(r.get("p_adj", float("nan")), 4),
                ] for r in best_tags_3]
            ))
            block.append("")

        block.append("**Multi-metric tradeoffs (pooled paraphrases vs original)**")
        block.append("")
        block.append("- Largest drops:")
        for lab, d in drops:
            block.append(f"  - {lab}: Δ={fmt(d)}")
        block.append("- Largest gains:")
        for lab, d in gains:
            block.append(f"  - {lab}: Δ={fmt(d)}")
        block.append("")

        block.append("**Representative failure examples (lowest TF, CP≥4)**")
        block.append("")
        block.extend(ex_lines)
        block.append("")

        special_per_series_blocks.append("\n".join(block))


    corr_lines = []
    if len(series_scores) >= 2:
        names = [n for n, _ in series_scores]
        shared_styles = [s for s in selected_styles if s != "instruction_original"]
        shared_styles = [s for s in shared_styles if all(s in style_drop_by_series.get(n, {}) for n in names)]
        corr_lines.append("## Cross-series robustness agreement")
        corr_lines.append("")
        corr_lines.append(f"Shared styles considered: **{len(shared_styles)}**")
        corr_lines.append("")
        corr_rows = []
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                a, b = names[i], names[j]
                xa = [style_drop_by_series[a][st] for st in shared_styles]
                xb = [style_drop_by_series[b][st] for st in shared_styles]
                r = spearman_corr(xa, xb)
                corr_rows.append([a, b, fmt(r, 3)])
        corr_lines.append(md_table(["Series A", "Series B", "Spearman ρ (ΔTF ranks)"], corr_rows))
        corr_lines.append("")

    metric_corr_blocks = []
    for sname, smap in series_scores:
        rows = []
        for pid in prompt_ids:
            per = smap.get(pid, {})
            for style in selected_styles:
                sc = per.get(style)
                if sc is None:
                    continue
                if style != "instruction_original":
                    cp = cp_by_id.get(pid, {}).get(style)
                    if cp not in (4, 5):
                        continue
                rows.append(sc)
        if len(rows) < 3:
            continue
        arr = np.array(rows, dtype=float)
        C = corr_matrix(arr)  # 10x10
        corr_csv = out_dir / f"metric_corr_{sname}.csv"
        corr_rows = []
        for i in range(10):
            corr_rows.append([METRIC_LABELS[i]] + [C[i, j] for j in range(10)])
        write_csv(corr_csv, ["metric"] + METRIC_LABELS, corr_rows)

        tf_corrs = [C[0, j] for j in range(10)]
        metric_corr_blocks.append("### Metric correlations (pooled instances, CP≥4 for paraphrases)")
        metric_corr_blocks.append("")
        metric_corr_blocks.append(f"Series: `{sname}` · saved full matrix: `{corr_csv.name}`")
        metric_corr_blocks.append("")
        metric_corr_blocks.append(md_table(
            ["Metric", "Corr with TF"],
            [[md_escape(METRIC_LABELS[j]), fmt(tf_corrs[j], 3)] for j in range(10)]
        ))
        metric_corr_blocks.append("")

    styles_csv = out_dir / "style_stats_all_series.csv"
    write_csv(styles_csv,
              ["series","style_key","style_pretty","n_paired","mean_tf","mean_delta_tf","ci_lo","ci_hi","p","p_adj","fail_rate"],
              style_stats_all_rows)

    tags_csv = out_dir / "tag_stats_all_series.csv"
    write_csv(tags_csv,
              ["series","tag","n_prompts_paired","n_styles","mean_tf","mean_delta_tf","ci_lo","ci_hi","p","p_adj","fail_rate","prompts_with_any"],
              tag_stats_all_rows)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    md = []
    md.append(f"# CALIPER evaluation summary — `{dataset_label}`")
    md.append("")
    md.append(f"- Generated: **{now}**")
    md.append(f"- Prompts file: `{Path(args.prompts).name}`")
    md.append(f"- Output dir: `{out_dir}`")
    md.append(f"- Prompt IDs used: **{len(prompt_ids)}** (max_samples={args.max_samples})")
    md.append(f"- CP filter for paraphrases: **CP ∈ {{4,5}}**")
    md.append(f"- Style inclusion threshold: **min_occurrences={args.min_occurrences}** (within selected prompt IDs)")
    md.append(f"- Failure threshold: **TF ≤ {failure_thr:.1f}**")
    md.append("")
    md.append("## Generated figures")
    md.append("")
    md.append(f"- `{out1.name}`")
    md.append(f"- `{out2.name}`")
    md.append(f"- `{out3.name}`")
    md.append("")
    md.append("## Content preservation summary (this run subset)")
    md.append("")
    md.append(md_table(
        ["CP score", "Count (over all styles+prompts in subset CP table)"],
        [[str(k), str(cp_hist.get(k, 0))] for k in [0,1,2,3,4,5]]
    ))
    md.append("")
    md.append("Saved full breakdowns:")
    md.append("")
    md.append(f"- By style: `{cp_style_csv.name}`")
    md.append(f"- By tag: `{cp_tag_csv.name}`")
    md.append("")
    md.append("## Surface statistics for paraphrase prompts (CP≥4)")
    md.append("")
    md.append(f"Saved: `{surface_csv.name}`")
    md.append("")

    md.append("## Core evaluation results")
    md.append("")
    md.append(f"Saved full tables:")
    md.append("")
    md.append(f"- Style stats (all series): `{styles_csv.name}`")
    md.append(f"- Tag stats (all series): `{tags_csv.name}`")
    md.append("")
    if args.answers:
        md.append("Answers were provided via `--answers`, so the failure-mode examples include model outputs.")
    else:
        md.append("No `--answers` provided; failure-mode examples include prompts + metric vectors only.")
    md.append("")
    md.append("\n\n".join(summary_blocks))
    md.append("")
    if corr_lines:
        md.append("\n".join(corr_lines))
        md.append("")
    if metric_corr_blocks:
        md.append("## Metric correlation notes")
        md.append("")
        md.append("\n".join(metric_corr_blocks))
        md.append("")

    md.append("## Files written by this script")
    md.append("")
    md.append("- Figures: `tf_scores_by_series_styles.png`, `tf_scores_by_series_tags.png`, `radar_first_series_all_styles.png`")
    md.append("- Tables: `style_stats_all_series.csv`, `tag_stats_all_series.csv`")
    md.append("- Content-preservation: `content_preservation_by_style.csv`, `content_preservation_by_tag.csv`")
    md.append("- Surface stats: `surface_stats_styles.csv`")
    md.append("- Prompt fragility: `prompt_fragility_<series>.csv`")
    md.append("- Metric correlations: `metric_corr_<series>.csv`")
    md.append("")
    md.append("## Notes on interpretation")
    md.append("")
    md.append("- **Paired ΔTF** is computed as *(TF_style − TF_original)* per prompt_count, using only prompts where the style exists and passes CP≥4.")
    md.append("- **Tag ΔTF** is computed per prompt as: *(mean TF over styles in tag − TF_original)*, then averaged over prompts.")
    md.append("- **p_adj** uses Benjamini–Hochberg correction over styles/tags *within the series* (for the reported paired tests).")
    md.append("- Bootstrap CIs are percentile CIs over paired diffs.")
    md.append("")

    summary_path = out_dir / "summary.md"
    summary_path.write_text("\n".join(md), encoding="utf-8")


    total_cp = sum(cp_hist.values())
    cp_pass = (cp_hist.get(4, 0) + cp_hist.get(5, 0))
    cp_pass_rate = (cp_pass / total_cp) if total_cp else float("nan")

    special_cross = []
    if len(series_scores) >= 2:
        names = [n for n, _ in series_scores]
        shared_styles = [s for s in selected_styles if s != "instruction_original"]
        shared_styles = [s for s in shared_styles if all(s in style_drop_by_series.get(n, {}) for n in names)]
        special_cross.append("## Cross-model agreement (style harm ranking)")
        special_cross.append("")
        special_cross.append(f"Shared styles used for correlation: **{len(shared_styles)}**")
        special_cross.append("")
        corr_rows = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                xa = [style_drop_by_series[a][st] for st in shared_styles]
                xb = [style_drop_by_series[b][st] for st in shared_styles]
                r = spearman_corr(xa, xb)
                corr_rows.append([a, b, fmt(r, 3)])
        special_cross.append(md_table(["Model A", "Model B", "Spearman ρ (ΔTF ranks)"], corr_rows))
        special_cross.append("")
    special_cross_series_block = "\n".join(special_cross)

    core_table_md = md_table(
        ["Series", "Orig TF", "Par TF (CI)", "ΔTF", f"Fail rate (TF≤{failure_thr:.1f})", "Worst style", "Gap (orig-worst)"],
        special_core_rows
    )

    special_md = []
    special_md.append(f"# SPECIAL SUMMARY — `{dataset_label}`")
    special_md.append("")
    special_md.append(f"- Generated: **{now}**")
    special_md.append(f"- Prompts: `{Path(args.prompts).name}`")
    special_md.append(f"- Prompt IDs used: **{len(prompt_ids)}** (max_samples={args.max_samples})")
    special_md.append(f"- Paraphrase validity filter: **CP ∈ {{4,5}}**")
    special_md.append(f"- Style inclusion threshold (for style-level stats): **min_occurrences={args.min_occurrences}**")
    special_md.append(f"- Failure threshold: **TF ≤ {failure_thr:.1f}**")
    special_md.append("")
    special_md.append("## Content preservation validity (all CP entries in this run subset)")
    special_md.append("")
    special_md.append(md_table(
        ["CP score", "Count"],
        [[str(k), str(cp_hist.get(k, 0))] for k in [0, 1, 2, 3, 4, 5]]
    ))
    special_md.append("")
    special_md.append(f"- CP≥4 pass rate: **{pct(cp_pass_rate)}** ({cp_pass}/{total_cp})")
    special_md.append("")
    special_md.append("## Core robustness results (upload-friendly)")
    special_md.append("")
    special_md.append(core_table_md)
    special_md.append("")
    special_md.append("## Per-model details for paper writing")
    special_md.append("")
    special_md.append("\n\n".join(special_per_series_blocks))
    special_md.append("")
    if special_cross_series_block:
        special_md.append(special_cross_series_block)
        special_md.append("")
    special_md.append("## Pointers to full artifacts (if you want deeper tables/plots)")
    special_md.append("")
    special_md.append("- Full human summary: `summary.md`")
    special_md.append("- Full style table: `style_stats_all_series.csv`")
    special_md.append("- Full tag table: `tag_stats_all_series.csv`")
    special_md.append("- Prompt fragility: `prompt_fragility_<series>.csv`")
    special_md.append("- CP breakdowns: `content_preservation_by_style.csv`, `content_preservation_by_tag.csv`")
    special_md.append("- Surface stats: `surface_stats_styles.csv`")
    special_md.append("- Figures: `tf_scores_by_series_styles.png`, `tf_scores_by_series_tags.png`, `radar_first_series_all_styles.png`")
    special_md.append("")

    special_path = out_dir / "special_summary.md"
    special_path.write_text("\n".join(special_md), encoding="utf-8")

    print("[done] Saved:")
    print(f" - {out1}")
    print(f" - {out2}")
    print(f" - {out3}")
    print(f" - {styles_csv}")
    print(f" - {tags_csv}")
    print(f" - {cp_style_csv}")
    print(f" - {cp_tag_csv}")
    print(f" - {surface_csv}")
    print(f" - {summary_path}")
    print(f" - {special_path}")
    print(f" - {out_dir} / prompt_fragility_*.csv")
    print(f" - {out_dir} / metric_corr_*.csv")

if __name__ == "__main__":
    main()
