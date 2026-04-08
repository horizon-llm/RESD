#!/usr/bin/env python3
"""Visualize token-level losses dumped by DataParallelPPOActor.

Supports both standard PG loss dumps and SDPO self-distillation loss dumps.
The script auto-detects the mode from the dump file's ``loss_mode`` field.

Usage:
    python visualize_token_loss.py <dump_path> --tokenizer <name_or_path> [--field <field>] [--out <output.html>]

Examples:
    python visualize_token_loss.py token_loss_dumps/step_10.pt --tokenizer Qwen/Qwen3-4B
    python visualize_token_loss.py token_loss_dumps/step_10.pt --field token_distill_loss
    python visualize_token_loss.py token_loss_dumps/ --tokenizer Qwen/Qwen3-4B
"""

import argparse
import html
import json
import os
from pathlib import Path

import torch


# ── Field registry ──────────────────────────────────────────────────────────
# Each entry maps a dump key to display metadata.
# The "modes" list controls which loss_mode(s) the field is relevant for.

FIELD_REGISTRY = {
    "token_distill_loss": {"label": "Distillation Loss (per token)", "cmap": "sequential"},
    "log_ratio": {"label": "Advantage log p_teacher/p_student", "cmap": "category"},
    "entropy_category": {"label": "Entropy & Top-1 Category", "cmap": "category4", "renderer": "category"},
    "student_entropy": {"label": "Entropy (student)", "cmap": "sequential"},
    "teacher_entropy": {"label": "Entropy (teacher)", "cmap": "sequential"},
    "topk_log_ratio_dist": {"label": "Top-K Log Ratio Density", "cmap": "diverging", "renderer": "density"},
}


def load_dump(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


def _compute_entropy_from_topk(topk_logps: torch.Tensor) -> torch.Tensor:
    """Compute approximate per-token entropy from top-k log probs.

    Args:
        topk_logps: Tensor of shape (n, seq_len, k) containing log probs of top-k tokens.
    Returns:
        Tensor of shape (n, seq_len) with approximate entropy at each position.
    """
    # Renormalize over the top-k subset so probabilities sum to 1
    logps = topk_logps - torch.logsumexp(topk_logps, dim=-1, keepdim=True)
    probs = torch.exp(logps)
    return -(probs * logps).sum(dim=-1)


def _preprocess_dump(dump: dict) -> dict:
    """Return a new dump dict with derived fields added.

    - Converts ``log_prob`` and ``teacher_log_prob`` from log-space to actual
      probabilities via exp().
    - Adds ``student_entropy`` / ``teacher_entropy`` computed from top-k logps
      when available.
    """
    dump = dict(dump)  # shallow copy — avoid mutating the original
    # Compute log ratio before converting to prob space
    if "teacher_log_prob" in dump and "log_prob" in dump:
        dump["log_ratio"] = dump["teacher_log_prob"] - dump["log_prob"]
    # Convert log probs to probs for use in tooltips
    for key in ("log_prob", "teacher_log_prob"):
        if key in dump:
            dump[key] = torch.exp(dump[key])
    for prefix, topk_key in [("student", "student_topk_logps"), ("teacher", "teacher_topk_logps")]:
        if topk_key in dump:
            dump[f"{prefix}_entropy"] = _compute_entropy_from_topk(dump[topk_key])
    # Compute top-k log ratio distribution: (bs, response_len, k)
    if "teacher_topk_logps" in dump and "student_topk_logps" in dump:
        dump["topk_log_ratio"] = dump["teacher_topk_logps"] - dump["student_topk_logps"]
        # Marker so this field shows up in get_default_fields
        dump["topk_log_ratio_dist"] = True
    # Compute entropy category: 5 classes based on entropy comparison + top-1 match
    # Category 4 = "tail": teacher's top-1 is likely outside the student's top-k
    # (detected when the teacher's cumulative prob within student's top-k < 0.5)
    if ("student_entropy" in dump and "teacher_entropy" in dump
            and "student_topk_logps" in dump and "teacher_topk_logps" in dump
            and "topk_indices" in dump):
        s_top1 = dump["topk_indices"].gather(-1, dump["student_topk_logps"].argmax(dim=-1, keepdim=True)).squeeze(-1)
        t_top1 = dump["topk_indices"].gather(-1, dump["teacher_topk_logps"].argmax(dim=-1, keepdim=True)).squeeze(-1)
        top1_match = (s_top1 == t_top1)  # (bs, response_len)
        teacher_higher = dump["teacher_entropy"] > dump["student_entropy"]  # (bs, response_len)
        # Detect tail: teacher's probability mass within student's top-k is low
        teacher_topk_probs = torch.exp(dump["teacher_topk_logps"])  # (bs, response_len, k)
        teacher_topk_mass = teacher_topk_probs.sum(dim=-1)  # (bs, response_len)
        is_tail = teacher_topk_mass < 0.5  # teacher's true top-1 likely outside student's top-k
        # Categories: 0=T>S+match, 1=S>T+match, 2=T>S+nomatch, 3=S>T+nomatch, 4=tail
        category = torch.zeros_like(dump["student_entropy"], dtype=torch.long)
        category[teacher_higher & top1_match] = 0
        category[~teacher_higher & top1_match] = 1
        category[teacher_higher & ~top1_match] = 2
        category[~teacher_higher & ~top1_match] = 3
        category[is_tail] = 4  # override: tail takes precedence
        dump["entropy_category"] = category
    return dump


def get_default_fields(dump: dict) -> list[str]:
    return [k for k in FIELD_REGISTRY if k in dump]


def decode_tokens(token_ids: torch.Tensor, tokenizer) -> list[str]:
    """Decode each token id individually, returning a list of token strings."""
    tokens = []
    for tid in token_ids.tolist():
        t = tokenizer.decode([tid], skip_special_tokens=False)
        tokens.append(t)
    return tokens


# ── Color mapping ───────────────────────────────────────────────────────────

CATEGORY_LABELS = [
    "T_entropy > S_entropy, top-1 match",
    "S_entropy > T_entropy, top-1 match",
    "T_entropy > S_entropy, top-1 mismatch",
    "S_entropy > T_entropy, top-1 mismatch",
    "Tail (teacher top-1 outside top-k)",
]
CATEGORY_COLORS = [
    "rgba(76,175,80,0.7)",    # 0: green — teacher more uncertain, but agree
    "rgba(33,150,243,0.7)",   # 1: blue — student more uncertain, but agree
    "rgba(255,152,0,0.7)",    # 2: orange — teacher more uncertain, disagree
    "rgba(244,67,54,0.7)",    # 3: red — student more uncertain, disagree
    "rgba(158,158,158,0.7)",  # 4: grey — tail dominated
]
CATEGORY_PIE_COLORS = ["#4CAF50", "#2196F3", "#FF9800", "#F44336", "#9E9E9E"]
NUM_CATEGORIES = len(CATEGORY_LABELS)


def value_to_color(value: float, vmin: float, vmax: float, cmap: str = "diverging") -> str:
    """Map a scalar value to an RGB color string.

    cmap options:
        'diverging'  - blue (negative) -> white (zero) -> red (positive)
        'sequential' - white (low) -> red (high)
    """
    if vmax == vmin:
        return "rgba(255,255,255,0)"

    if cmap == "category4":
        # 4 discrete categories with distinct colors
        cat = int(value)
        return CATEGORY_COLORS[cat]

    if cmap == "category":
        # Promotion (green, positive) vs Suppression (red, negative)
        abs_max = max(abs(vmin), abs(vmax))
        if abs_max == 0:
            return "rgba(255,255,255,0)"
        t = value / abs_max
        t = max(-1.0, min(1.0, t))
        # Sqrt scaling to make moderate values more visible with long-tailed distributions
        intensity = abs(t) ** 0.5
        if t < 0:  # suppression (red)
            r, g, b = 255, int(255 * (1 - 0.6 * intensity)), int(255 * (1 - 0.7 * intensity))
        else:  # promotion (green)
            r, g, b = int(255 * (1 - 0.7 * intensity)), 255, int(255 * (1 - 0.6 * intensity))
        alpha = 0.2 + 0.7 * intensity
        return f"rgba({r},{g},{b},{alpha:.2f})"
    elif cmap == "diverging":
        abs_max = max(abs(vmin), abs(vmax))
        if abs_max == 0:
            return "rgba(255,255,255,0)"
        t = value / abs_max
        t = max(-1.0, min(1.0, t))
        if t < 0:
            intensity = abs(t)
            r, g, b = int(255 * (1 - 0.6 * intensity)), int(255 * (1 - 0.4 * intensity)), 255
        else:
            intensity = t
            r, g, b = 255, int(255 * (1 - 0.5 * intensity)), int(255 * (1 - 0.6 * intensity))
        alpha = 0.15 + 0.75 * abs(t)
        return f"rgba({r},{g},{b},{alpha:.2f})"
    else:  # sequential
        t = (value - vmin) / (vmax - vmin)
        t = max(0.0, min(1.0, t))
        r = 255
        g = int(255 * (1 - 0.7 * t))
        b = int(255 * (1 - 0.85 * t))
        alpha = 0.1 + 0.8 * t
        return f"rgba({r},{g},{b},{alpha:.2f})"


# ── Per-sample HTML rendering ──────────────────────────────────────────────

def render_sample_html(
    sample_idx: int,
    prompt_tokens: list[str],
    response_tokens: list[str],
    values: torch.Tensor,
    mask: torch.Tensor,
    field_name: str,
    extra_tooltip: dict[str, torch.Tensor] | None = None,
    cmap: str = "diverging",
    teacher_prompt_tokens: list[str] | None = None,
    topk_info: list[list[tuple[str, float, float | None]]] | None = None,
) -> str:
    """Render one sample as an HTML block with colored response tokens.

    Args:
        extra_tooltip: optional dict mapping label -> per-token tensor to show on hover.
        teacher_prompt_tokens: if provided, show teacher prompt as a separate section.
        topk_info: per-position list of (token_str, student_lp, teacher_lp) tuples.
    """
    masked_values = values[mask.bool()]
    if masked_values.numel() == 0:
        return ""
    vmin, vmax = masked_values.min().item(), masked_values.max().item()

    parts = []
    parts.append(f'<div class="sample">')
    parts.append(f'<h3>Sample {sample_idx}</h3>')
    stats = (
        f"<b>{field_name}</b>: min={vmin:.4f}, max={vmax:.4f}, "
        f"mean={masked_values.mean().item():.4f}, "
        f"num_tokens={mask.sum().item():.0f}"
    )
    parts.append(f'<div class="stats">{stats}</div>')

    # Prompt sections (collapsible, separate blocks)
    student_label = "STUDENT PROMPT" if teacher_prompt_tokens is not None else "PROMPT"
    prompt_text = "".join(html.escape(tok) for tok in prompt_tokens)
    parts.append(
        f'<details class="prompt-section">'
        f'<summary class="section-label">{student_label} ({len(prompt_tokens)} tokens)</summary>'
        f'<div class="text-block"><span class="prompt">{prompt_text}</span></div>'
        f'</details>'
    )
    if teacher_prompt_tokens is not None:
        teacher_text = "".join(html.escape(tok) for tok in teacher_prompt_tokens)
        parts.append(
            f'<details class="prompt-section">'
            f'<summary class="section-label">TEACHER PROMPT ({len(teacher_prompt_tokens)} tokens)</summary>'
            f'<div class="text-block"><span class="prompt">{teacher_text}</span></div>'
            f'</details>'
        )

    # Response section (colored by values)
    parts.append('<div class="text-block">')
    parts.append('<span class="section-label">RESPONSE</span> ')
    token_spans = []
    for i, tok in enumerate(response_tokens):
        if mask[i].item() == 0:
            token_spans.append(f'<span class="masked-token">{html.escape(tok)}</span>')
        else:
            v = values[i].item()
            color = value_to_color(v, vmin, vmax, cmap=cmap)
            escaped = html.escape(tok)
            title = f"{field_name}={v:.4f}"
            if extra_tooltip:
                for lbl, tensor in extra_tooltip.items():
                    title += f", {lbl}={tensor[i].item():.4f}"
            # Encode top-k data as attribute if available
            topk_attr = ""
            if topk_info is not None and i < len(topk_info):
                topk_json = json.dumps(topk_info[i], ensure_ascii=False)
                topk_attr = f' data-topk="{html.escape(topk_json, quote=True)}"'
            cls = "token topk-clickable" if topk_attr else "token"
            token_spans.append(
                f'<span class="{cls}" style="background-color:{color}" title="{title}"{topk_attr}>'
                f"{escaped}</span>"
            )
    parts.append("".join(token_spans))
    parts.append("</div>")

    # Color legend
    parts.append('<div class="legend">')
    n_legend = 11
    for j in range(n_legend):
        t = vmin + (vmax - vmin) * j / (n_legend - 1)
        c = value_to_color(t, vmin, vmax, cmap=cmap)
        parts.append(
            f'<span class="legend-cell" style="background-color:{c}">{t:.3f}</span>'
        )
    parts.append("</div>")
    parts.append("</div>")
    return "\n".join(parts)



def _svg_density_single(flat: torch.Tensor, n_bins: int = 60) -> str:
    """Render a single-color SVG density chart for all top-k log ratios."""
    if flat.numel() == 0:
        return ""
    min_v = flat.min().item()
    max_v = flat.max().item()
    if max_v == min_v:
        max_v = min_v + 1.0
    bin_width = (max_v - min_v) / n_bins

    counts = torch.histc(flat.float(), bins=n_bins, min=min_v, max=max_v).tolist()
    total = flat.numel()
    density = [c / (total * bin_width) for c in counts]
    max_density = max(density) if density else 1.0
    if max_density == 0:
        max_density = 1.0

    svg_w, svg_h = 700, 260
    margin_l, margin_r, margin_t, margin_b = 60, 15, 20, 45
    plot_w = svg_w - margin_l - margin_r
    plot_h = svg_h - margin_t - margin_b

    parts = [
        f'<svg width="{svg_w}" height="{svg_h}" xmlns="http://www.w3.org/2000/svg" '
        f'style="background:#12122a;border-radius:6px;">'
    ]

    # Filled density curve — blue for negative bins, red for positive
    for i, d in enumerate(density):
        if d == 0:
            continue
        bin_center = min_v + (i + 0.5) * bin_width
        bar_w = plot_w / n_bins
        bar_h = (d / max_density) * plot_h
        x = margin_l + i * bar_w
        y = margin_t + plot_h - bar_h
        fill = "rgba(100,150,255,0.7)" if bin_center < 0 else "rgba(255,100,100,0.7)"
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.2f}" height="{bar_h:.1f}" fill="{fill}">'
            f'<title>[{min_v + i * bin_width:.3f}, {min_v + (i+1) * bin_width:.3f}): {d:.4f}</title></rect>'
        )

    # Zero line
    if min_v < 0 < max_v:
        zero_x = margin_l + ((0 - min_v) / (max_v - min_v)) * plot_w
        parts.append(
            f'<line x1="{zero_x:.1f}" y1="{margin_t}" x2="{zero_x:.1f}" '
            f'y2="{margin_t + plot_h}" stroke="#ffcc00" stroke-width="1" stroke-dasharray="4,3"/>'
        )

    # Axes
    parts.append(
        f'<line x1="{margin_l}" y1="{margin_t + plot_h}" x2="{margin_l + plot_w}" '
        f'y2="{margin_t + plot_h}" stroke="#666" stroke-width="1"/>'
    )
    parts.append(
        f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" '
        f'y2="{margin_t + plot_h}" stroke="#666" stroke-width="1"/>'
    )

    # X ticks
    for j in range(6):
        frac = j / 5
        val = min_v + frac * (max_v - min_v)
        tx = margin_l + frac * plot_w
        parts.append(
            f'<text x="{tx:.1f}" y="{margin_t + plot_h + 15}" fill="#999" font-size="10" '
            f'text-anchor="middle" font-family="monospace">{val:.2f}</text>'
        )
    # Y ticks
    parts.append(
        f'<text x="{margin_l - 6}" y="{margin_t + 4}" fill="#999" font-size="9" '
        f'text-anchor="end" font-family="monospace">{max_density:.2f}</text>'
    )
    parts.append(
        f'<text x="{margin_l - 6}" y="{margin_t + plot_h}" fill="#999" font-size="9" '
        f'text-anchor="end" font-family="monospace">0</text>'
    )
    # Axis labels
    parts.append(
        f'<text x="{margin_l + plot_w / 2}" y="{svg_h - 3}" fill="#aaa" font-size="11" '
        f'text-anchor="middle" font-family="monospace">log(p_teacher / p_student)</text>'
    )
    parts.append(
        f'<text x="{margin_l - 6}" y="{margin_t + plot_h / 2}" fill="#aaa" font-size="10" '
        f'text-anchor="middle" font-family="monospace" '
        f'transform="rotate(-90,{margin_l - 40},{margin_t + plot_h / 2})">density</text>'
    )

    parts.append("</svg>")
    return "\n".join(parts)



def _render_density_section(
    sample_idx: int,
    topk_log_ratios: torch.Tensor,
    field_label: str,
) -> str:
    """Render a single-color density chart for one sample (all top-k log ratios).

    Computes over the entire top-k at every position (not just masked/generated tokens).
    """
    all_flat = topk_log_ratios.reshape(-1)
    if all_flat.numel() == 0:
        return ""
    mean_v = all_flat.mean().item()
    std_v = all_flat.std().item() if all_flat.numel() > 1 else 0.0
    median_v = all_flat.median().item()
    min_v = all_flat.min().item()
    max_v = all_flat.max().item()
    parts = [f'<div class="sample">', f'<h3>Sample {sample_idx}</h3>']
    parts.append(f'<div class="stats"><b>{field_label}</b>: mean={mean_v:.4f}, std={std_v:.4f}, '
                 f'median={median_v:.4f}, min={min_v:.4f}, max={max_v:.4f}, '
                 f'num_positions={topk_log_ratios.shape[0]}, k={topk_log_ratios.shape[1]}</div>')
    parts.append('<div class="dist-chart">')
    parts.append(_svg_density_single(all_flat))
    parts.append('</div></div>')
    return "\n".join(parts)



def _svg_pie_chart(counts: list[int], labels: list[str], colors: list[str], size: int = 220) -> str:
    """Render an SVG pie chart with legend."""
    import math
    total = sum(counts)
    if total == 0:
        return ""
    cx, cy, r = size // 2, size // 2, size // 2 - 10
    legend_w = 280
    svg_w = size + legend_w + 20
    parts = [
        f'<svg width="{svg_w}" height="{max(size, len(labels) * 28 + 20)}" '
        f'xmlns="http://www.w3.org/2000/svg" style="background:#12122a;border-radius:6px;">'
    ]
    start_angle = -math.pi / 2
    for i, (cnt, label, color) in enumerate(zip(counts, labels, colors)):
        if cnt == 0:
            continue
        frac = cnt / total
        end_angle = start_angle + 2 * math.pi * frac
        large_arc = 1 if frac > 0.5 else 0
        x1, y1 = cx + r * math.cos(start_angle), cy + r * math.sin(start_angle)
        x2, y2 = cx + r * math.cos(end_angle), cy + r * math.sin(end_angle)
        if frac >= 1.0 - 1e-9:
            # Full circle
            parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{color}"/>')
        else:
            parts.append(
                f'<path d="M{cx},{cy} L{x1:.2f},{y1:.2f} '
                f'A{r},{r} 0 {large_arc},1 {x2:.2f},{y2:.2f} Z" fill="{color}"/>'
            )
        start_angle = end_angle

    # Legend
    lx = size + 15
    for i, (cnt, label, color) in enumerate(zip(counts, labels, colors)):
        ly = 20 + i * 28
        pct = cnt / total * 100 if total > 0 else 0
        parts.append(f'<rect x="{lx}" y="{ly}" width="14" height="14" rx="3" fill="{color}"/>')
        parts.append(
            f'<text x="{lx + 20}" y="{ly + 12}" fill="#ccc" font-size="11" '
            f'font-family="monospace">{label}: {cnt} ({pct:.1f}%)</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


def _render_category_section(
    sample_idx: int,
    response_tokens: list[str],
    categories: torch.Tensor,
    mask: torch.Tensor,
    prompt_tokens: list[str],
    teacher_prompt_tokens: list[str] | None = None,
) -> str:
    """Render a sample with tokens colored by entropy category + pie chart."""
    parts = [f'<div class="sample">', f'<h3>Sample {sample_idx}</h3>']

    # Count categories (only for masked/valid tokens)
    valid_cats = categories[mask.bool()]
    counts = [(valid_cats == c).sum().item() for c in range(NUM_CATEGORIES)]
    total = sum(counts)
    stats_parts = [f"<b>Entropy & Top-1 Category</b>: num_tokens={total}"]
    for c in range(NUM_CATEGORIES):
        pct = counts[c] / total * 100 if total > 0 else 0
        stats_parts.append(f", {CATEGORY_LABELS[c]}={counts[c]} ({pct:.1f}%)")
    parts.append(f'<div class="stats">{"".join(stats_parts)}</div>')

    # Prompt sections
    student_label = "STUDENT PROMPT" if teacher_prompt_tokens is not None else "PROMPT"
    prompt_text = "".join(html.escape(tok) for tok in prompt_tokens)
    parts.append(
        f'<details class="prompt-section">'
        f'<summary class="section-label">{student_label} ({len(prompt_tokens)} tokens)</summary>'
        f'<div class="text-block"><span class="prompt">{prompt_text}</span></div>'
        f'</details>'
    )
    if teacher_prompt_tokens is not None:
        teacher_text = "".join(html.escape(tok) for tok in teacher_prompt_tokens)
        parts.append(
            f'<details class="prompt-section">'
            f'<summary class="section-label">TEACHER PROMPT ({len(teacher_prompt_tokens)} tokens)</summary>'
            f'<div class="text-block"><span class="prompt">{teacher_text}</span></div>'
            f'</details>'
        )

    # Response tokens colored by category
    parts.append('<div class="text-block">')
    parts.append('<span class="section-label">RESPONSE</span> ')
    token_spans = []
    for i, tok in enumerate(response_tokens):
        if mask[i].item() == 0:
            token_spans.append(f'<span class="masked-token">{html.escape(tok)}</span>')
        else:
            cat = categories[i].item()
            color = CATEGORY_COLORS[cat]
            title = CATEGORY_LABELS[cat]
            token_spans.append(
                f'<span class="token" style="background-color:{color}" title="{title}">'
                f'{html.escape(tok)}</span>'
            )
    parts.append("".join(token_spans))
    parts.append("</div>")

    # Pie chart
    parts.append('<div class="dist-chart" style="margin-top:12px;">')
    parts.append(_svg_pie_chart(counts, CATEGORY_LABELS, CATEGORY_PIE_COLORS))
    parts.append("</div>")

    # Category legend
    parts.append('<div class="legend">')
    for c in range(NUM_CATEGORIES):
        parts.append(
            f'<span class="legend-cell" style="background-color:{CATEGORY_COLORS[c]};color:#fff;">'
            f'{CATEGORY_LABELS[c]}</span>'
        )
    parts.append("</div>")

    parts.append("</div>")
    return "\n".join(parts)


# ── HTML template ───────────────────────────────────────────────────────────

HTML_HEADER = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Token Loss Visualization</title>
<style>
body {{
    font-family: 'SF Mono', 'Menlo', 'Consolas', monospace;
    font-size: 13px;
    background: #1a1a2e;
    color: #e0e0e0;
    margin: 20px;
    line-height: 1.6;
}}
h1 {{
    color: #e0e0e0;
    border-bottom: 2px solid #444;
    padding-bottom: 8px;
}}
h2 {{
    color: #b0b0c0;
    margin-top: 30px;
}}
h3 {{
    color: #a0a0b8;
    margin: 8px 0 4px 0;
}}
.field-section {{
    margin-bottom: 40px;
}}
.sample {{
    background: #16213e;
    border: 1px solid #333;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 20px;
}}
.stats {{
    color: #8888aa;
    font-size: 12px;
    margin-bottom: 10px;
}}
.text-block {{
    word-wrap: break-word;
    white-space: pre-wrap;
    line-height: 1.8;
}}
.section-label {{
    display: inline-block;
    background: #333;
    color: #aaa;
    font-size: 10px;
    font-weight: bold;
    padding: 1px 6px;
    border-radius: 3px;
    margin: 4px 4px 4px 0;
    vertical-align: middle;
}}
.prompt-section {{
    margin-bottom: 10px;
    border: 1px solid #2a2a4a;
    border-radius: 6px;
    padding: 4px 8px;
}}
.prompt-section summary {{
    cursor: pointer;
    user-select: none;
}}
.prompt-section .text-block {{
    margin-top: 6px;
}}
.prompt {{
    color: #777;
}}
.token {{
    border-radius: 3px;
    padding: 1px 0;
    cursor: default;
    position: relative;
}}
.token:hover {{
    outline: 2px solid #ffcc00;
    z-index: 10;
}}
.masked-token {{
    color: #444;
}}
.legend {{
    margin-top: 8px;
    display: flex;
    gap: 2px;
    font-size: 10px;
    align-items: center;
}}
.legend-cell {{
    padding: 2px 6px;
    border-radius: 3px;
    text-align: center;
    color: #222;
    font-weight: bold;
}}
.toc {{
    background: #16213e;
    border: 1px solid #333;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 20px;
}}
.toc a {{
    color: #6699cc;
    text-decoration: none;
}}
.toc a:hover {{
    text-decoration: underline;
}}
.dist-layout {{
    display: flex;
    gap: 16px;
    align-items: flex-start;
    margin: 8px 0;
}}
.dist-chart {{
    flex: 1;
}}
.topk-clickable {{
    cursor: pointer;
    border-bottom: 1px dotted #888;
}}
.topk-popup {{
    position: fixed;
    background: #1e1e3a;
    border: 1px solid #555;
    border-radius: 8px;
    padding: 10px;
    z-index: 1000;
    max-height: 400px;
    overflow-y: auto;
    box-shadow: 0 4px 20px rgba(0,0,0,0.5);
    font-size: 12px;
    min-width: 320px;
}}
.topk-popup table {{
    border-collapse: collapse;
    width: 100%;
}}
.topk-popup th {{
    text-align: left;
    color: #aaa;
    border-bottom: 1px solid #444;
    padding: 3px 8px;
    font-size: 10px;
}}
.topk-popup td {{
    padding: 2px 8px;
    white-space: pre;
    font-family: 'SF Mono', 'Menlo', 'Consolas', monospace;
}}
.topk-popup tr.topk-selected {{
    background: rgba(255, 204, 0, 0.15);
    font-weight: bold;
}}
.topk-popup .topk-title {{
    color: #ccc;
    font-weight: bold;
    margin-bottom: 6px;
}}
</style>
<script>
document.addEventListener('click', function(e) {{
    // Close any open popup when clicking outside
    var existing = document.querySelector('.topk-popup');
    if (existing && !existing.contains(e.target) && !e.target.classList.contains('topk-clickable')) {{
        existing.remove();
    }}
    var el = e.target.closest('.topk-clickable');
    if (!el || !el.dataset.topk) return;
    if (existing) existing.remove();
    var data = JSON.parse(el.dataset.topk);
    var popup = document.createElement('div');
    popup.className = 'topk-popup';
    var selectedTok = el.textContent;
    var h = '<div class="topk-title">Top-' + data.length + ' tokens</div>';
    h += '<table><tr><th>#</th><th>Token</th><th>Student Prob</th>';
    if (data[0] && data[0][2] !== null) h += '<th>Teacher Prob</th>';
    h += '</tr>';
    for (var i = 0; i < data.length; i++) {{
        var tok = data[i][0], slp = data[i][1], tlp = data[i][2];
        var cls = (tok === selectedTok) ? ' class="topk-selected"' : '';
        h += '<tr' + cls + '><td>' + (i+1) + '</td>';
        h += '<td>' + tok.replace(/</g,'&lt;').replace(/>/g,'&gt;') + '</td>';
        h += '<td>' + Math.exp(slp).toFixed(6) + '</td>';
        if (tlp !== null) h += '<td>' + Math.exp(tlp).toFixed(6) + '</td>';
        h += '</tr>';
    }}
    h += '</table>';
    popup.innerHTML = h;
    document.body.appendChild(popup);
    var rect = el.getBoundingClientRect();
    var popRect = popup.getBoundingClientRect();
    var left = Math.min(rect.left, window.innerWidth - popRect.width - 10);
    var top = rect.bottom + 4;
    if (top + popRect.height > window.innerHeight) top = rect.top - popRect.height - 4;
    popup.style.left = Math.max(0, left) + 'px';
    popup.style.top = Math.max(0, top) + 'px';
}});
</script>
</head>
<body>
<h1>{title}</h1>
"""

HTML_FOOTER = """\
</body>
</html>
"""


# ── Tooltip helpers (contextual extra info on hover) ────────────────────────

def _extra_tooltip_for(field: str, dump: dict, sample_idx: int) -> dict[str, torch.Tensor]:
    """Return extra per-token tensors to show on hover for a given field."""
    extra = {}
    # For distillation loss and log ratio, show student and teacher probs (already converted from log-space)
    if field in ("token_distill_loss", "log_ratio"):
        if "log_prob" in dump:
            extra["student_prob"] = dump["log_prob"][sample_idx]
        if "teacher_log_prob" in dump:
            extra["teacher_prob"] = dump["teacher_log_prob"][sample_idx]
    return extra


# ── Main HTML builder ───────────────────────────────────────────────────────

def build_html(dump: dict, tokenizer, fields: list[str] | None = None, title: str = "") -> str:
    """Build the full HTML visualization for a single dump file."""
    dump = _preprocess_dump(dump)
    n_samples = dump["response_ids"].shape[0]
    response_len = dump["response_ids"].shape[1]
    input_ids = dump["input_ids"]
    response_ids = dump["response_ids"]
    response_mask = dump["response_mask"]
    prompt_len = input_ids.shape[1] - response_len
    teacher_input_ids = dump.get("teacher_input_ids", None)
    teacher_prompt_len = (teacher_input_ids.shape[1] - response_len) if teacher_input_ids is not None else None
    if fields is None:
        fields = get_default_fields(dump)

    html_parts = [HTML_HEADER.format(title=title or "Token Loss Visualization")]

    # Table of contents
    html_parts.append('<div class="toc"><b>Fields:</b> ')
    for f in fields:
        label = FIELD_REGISTRY.get(f, {}).get("label", f)
        html_parts.append(f'<a href="#{f}">{label}</a> &nbsp;|&nbsp; ')
    html_parts.append("</div>")

    for field in fields:
        if field not in dump:
            continue
        meta = FIELD_REGISTRY.get(field, {"label": field, "cmap": "diverging"})
        renderer = meta.get("renderer", "token")
        html_parts.append(f'<div class="field-section" id="{field}">')
        html_parts.append(f'<h2>{meta["label"]}</h2>')

        # Topk-based density renderer
        if renderer == "density":
            topk_ratios = dump.get("topk_log_ratio")
            if topk_ratios is None:
                html_parts.append("<p>No top-k data available.</p>")
                html_parts.append("</div>")
                continue
            for i in range(n_samples):
                resp_ids = response_ids[i]
                content_end = _find_content_end(resp_ids, tokenizer)
                sample_ratios = topk_ratios[i, :content_end]
                html_parts.append(_render_density_section(
                    i, sample_ratios, meta["label"],
                ))
            html_parts.append("</div>")
            continue

        # Category renderer (entropy + top-1 match)
        if renderer == "category":
            categories = dump[field]  # (n, response_len) int tensor
            for i in range(n_samples):
                prompt_ids = input_ids[i, :prompt_len]
                resp_ids = response_ids[i]
                content_end = _find_content_end(resp_ids, tokenizer)
                resp_ids = resp_ids[:content_end]
                sample_cats = categories[i, :content_end]
                sample_mask = response_mask[i, :content_end]

                if tokenizer.pad_token_id is not None:
                    prompt_start = 0
                    while prompt_start < len(prompt_ids) and prompt_ids[prompt_start].item() == tokenizer.pad_token_id:
                        prompt_start += 1
                    prompt_ids = prompt_ids[prompt_start:]

                prompt_tokens = decode_tokens(prompt_ids, tokenizer)
                resp_tokens = decode_tokens(resp_ids, tokenizer)

                t_prompt_tokens = None
                if teacher_input_ids is not None:
                    t_prompt_ids = teacher_input_ids[i, :teacher_prompt_len]
                    if tokenizer.pad_token_id is not None:
                        t_start = 0
                        while t_start < len(t_prompt_ids) and t_prompt_ids[t_start].item() == tokenizer.pad_token_id:
                            t_start += 1
                        t_prompt_ids = t_prompt_ids[t_start:]
                    t_prompt_tokens = decode_tokens(t_prompt_ids, tokenizer)

                html_parts.append(_render_category_section(
                    i, resp_tokens, sample_cats, sample_mask,
                    prompt_tokens, t_prompt_tokens,
                ))
            html_parts.append("</div>")
            continue

        values = dump[field]  # (n, response_len)

        for i in range(n_samples):
            prompt_ids = input_ids[i, :prompt_len]
            resp_ids = response_ids[i]

            # Remove trailing pad/eos tokens from response
            content_end = _find_content_end(resp_ids, tokenizer)
            resp_ids = resp_ids[:content_end]
            sample_values = values[i, :content_end]
            sample_mask = response_mask[i, :content_end]

            # Remove leading pad tokens from prompt
            if tokenizer.pad_token_id is not None:
                prompt_start = 0
                while prompt_start < len(prompt_ids) and prompt_ids[prompt_start].item() == tokenizer.pad_token_id:
                    prompt_start += 1
                prompt_ids = prompt_ids[prompt_start:]

            prompt_tokens = decode_tokens(prompt_ids, tokenizer)
            resp_tokens = decode_tokens(resp_ids, tokenizer)

            # Extract teacher prompt if available
            t_prompt_tokens = None
            if teacher_input_ids is not None:
                t_prompt_ids = teacher_input_ids[i, :teacher_prompt_len]
                if tokenizer.pad_token_id is not None:
                    t_start = 0
                    while t_start < len(t_prompt_ids) and t_prompt_ids[t_start].item() == tokenizer.pad_token_id:
                        t_start += 1
                    t_prompt_ids = t_prompt_ids[t_start:]
                t_prompt_tokens = decode_tokens(t_prompt_ids, tokenizer)

            extra_tooltip = _extra_tooltip_for(field, dump, i)
            extra_tooltip = {k: v[:content_end] for k, v in extra_tooltip.items()}

            # Build top-k info for this sample if available
            sample_topk_info = None
            if "topk_indices" in dump:
                topk_idx = dump["topk_indices"][i, :content_end]  # (content_end, k)
                s_logps = dump.get("student_topk_logps")
                t_logps = dump.get("teacher_topk_logps")
                s_lp = s_logps[i, :content_end] if s_logps is not None else None
                t_lp = t_logps[i, :content_end] if t_logps is not None else None
                sample_topk_info = []
                for pos in range(content_end):
                    entries = []
                    for j in range(topk_idx.shape[1]):
                        tid = topk_idx[pos, j].item()
                        tok_str = tokenizer.decode([tid], skip_special_tokens=False)
                        s_val = s_lp[pos, j].item() if s_lp is not None else 0.0
                        t_val = t_lp[pos, j].item() if t_lp is not None else None
                        entries.append((tok_str, s_val, t_val))
                    sample_topk_info.append(entries)

            sample_html = render_sample_html(
                sample_idx=i,
                prompt_tokens=prompt_tokens,
                response_tokens=resp_tokens,
                values=sample_values,
                mask=sample_mask,
                field_name=meta["label"],
                extra_tooltip=extra_tooltip,
                cmap=meta["cmap"],
                teacher_prompt_tokens=t_prompt_tokens,
                topk_info=sample_topk_info,
            )
            html_parts.append(sample_html)
        html_parts.append("</div>")

    html_parts.append(HTML_FOOTER)
    return "\n".join(html_parts)


# ── CLI ─────────────────────────────────────────────────────────────────────

def _trim_prompt(dump: dict, max_prompt_tokens: int) -> dict:
    if max_prompt_tokens <= 0:
        return dump
    response_len = dump["response_ids"].shape[1]
    prompt_len = dump["input_ids"].shape[1] - response_len
    if prompt_len > max_prompt_tokens:
        trim = prompt_len - max_prompt_tokens
        dump["input_ids"] = dump["input_ids"][:, trim:]
    return dump


def _find_content_end(token_ids: torch.Tensor, tokenizer) -> int:
    """Find the end of meaningful content, excluding trailing pad/eos tokens."""
    special_ids = set()
    if tokenizer.pad_token_id is not None:
        special_ids.add(tokenizer.pad_token_id)
    if tokenizer.eos_token_id is not None:
        special_ids.add(tokenizer.eos_token_id)
    if not special_ids:
        return len(token_ids)
    end = len(token_ids)
    while end > 0 and token_ids[end - 1].item() in special_ids:
        end -= 1
    return end


def main():
    parser = argparse.ArgumentParser(description="Visualize token-level loss dumps (PG or SDPO)")
    parser.add_argument("dump_path", type=str, help="Path to .pt file or directory of .pt files")
    parser.add_argument("--tokenizer", type=str, required=True, help="HuggingFace tokenizer name or path")
    parser.add_argument(
        "--field",
        type=str,
        default=None,
        help="Field to visualize (token_distill_loss, log_ratio, student_entropy, teacher_entropy, topk_log_ratio_dist). "
        "Default: all available fields.",
    )
    parser.add_argument("--out", type=str, default=None, help="Output HTML path (default: auto)")
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=0,
        help="Max prompt tokens to show (truncate from left). 0 = show all.",
    )
    args = parser.parse_args()

    from transformers import AutoTokenizer

    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    fields = [args.field] if args.field else None

    dump_path = Path(args.dump_path)
    if dump_path.is_dir():
        pt_files = sorted(dump_path.glob("*.pt"))
        if not pt_files:
            print(f"No .pt files found in {dump_path}")
            return
        print(f"Found {len(pt_files)} dump files")
        all_html_parts = [HTML_HEADER.format(title=f"Token Loss Visualization — {dump_path}")]
        for pt_file in pt_files:
            print(f"  Processing {pt_file.name}...")
            dump = _trim_prompt(load_dump(str(pt_file)), args.max_prompt_tokens)
            all_html_parts.append(f"<h2>{pt_file.stem}</h2>")
            inner = build_html(dump, tokenizer, fields=fields, title="")
            inner = inner.replace(HTML_HEADER.format(title=""), "").replace(HTML_FOOTER, "")
            all_html_parts.append(inner)
        all_html_parts.append(HTML_FOOTER)
        combined_html = "\n".join(all_html_parts)
        out_path = args.out or str(dump_path / "visualization.html")
        Path(out_path).write_text(combined_html, encoding="utf-8")
        print(f"Saved to {out_path}")
    else:
        print(f"Loading dump: {dump_path}")
        dump = _trim_prompt(load_dump(str(dump_path)), args.max_prompt_tokens)
        full_html = build_html(
            dump, tokenizer, fields=fields, title=f"Token Loss — {dump_path.name}"
        )
        out_path = args.out or str(dump_path.with_suffix(".html"))
        Path(out_path).write_text(full_html, encoding="utf-8")
        print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
