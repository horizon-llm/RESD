"""
Generate a paper figure comparing token-level distillation loss between two methods.
Renders response tokens with background color proportional to their distillation loss.

Usage:
    python figs/plot_token_loss_case_study.py \
        --method_dir <path_to_method_token_loss_dumps> \
        --baseline_dir <path_to_baseline_token_loss_dumps> \
        --step 20 \
        --output figs/token_loss_paper_figure.pdf
"""

import re
import os
import html
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, Normalize


def extract_all_tokens_from_section(filepath, section_name="Distillation Loss (per token)"):
    with open(filepath) as f:
        content = f.read()
    section_start = content.find(f'<h2>{section_name}</h2>')
    if section_start == -1:
        return []
    next_section = content.find('<div class="field-section"', section_start + 10)
    if next_section == -1:
        section_content = content[section_start:]
    else:
        section_content = content[section_start:next_section]
    resp_idx = section_content.find('RESPONSE</span>')
    if resp_idx == -1:
        return []
    response_content = section_content[resp_idx:]
    token_pattern = r'title="Distillation Loss \(per token\)=([0-9eE.+-]+),([^"]*)"[^>]*>(.*?)</span>'
    tokens = re.findall(token_pattern, response_content, re.DOTALL)
    result = []
    for loss_str, extra_info, text in tokens:
        try:
            loss = float(loss_str)
            decoded_text = html.unescape(text)
            result.append({'loss': loss, 'text': decoded_text})
        except ValueError:
            continue
    return result


def wrap_tokens_into_lines(tokens, max_chars_per_line=78):
    lines = []
    current_line = []
    current_len = 0
    for t in tokens:
        text = t['text']
        if '\n' in text:
            parts = text.split('\n')
            for i, part in enumerate(parts):
                if i > 0:
                    lines.append(current_line)
                    current_line = []
                    current_len = 0
                if part:
                    current_line.append({'text': part, 'loss': t['loss']})
                    current_len += len(part)
        else:
            if current_len + len(text) > max_chars_per_line and current_len > 0:
                lines.append(current_line)
                current_line = []
                current_len = 0
            current_line.append(t)
            current_len += len(text)
    if current_line:
        lines.append(current_line)
    return lines


def render_panel(ax, tokens, title, stats_text, cmap, norm, max_chars=78):
    lines = wrap_tokens_into_lines(tokens, max_chars_per_line=max_chars)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    ax.text(0.0, 0.97, title, fontsize=9.5, fontweight='bold',
            verticalalignment='top', transform=ax.transAxes)
    ax.text(1.0, 0.97, stats_text, fontsize=7.5,
            verticalalignment='top', horizontalalignment='right',
            transform=ax.transAxes, color='#555', style='italic')

    line_height = 0.125
    y_start = 0.86
    fontsize = 7.8
    char_width = 0.0094

    for line_idx, line_tokens in enumerate(lines):
        y = y_start - line_idx * line_height
        x = 0.01

        for tok in line_tokens:
            text = tok['text']
            loss = tok['loss']
            color = cmap(norm(loss))

            text_width = len(text) * char_width

            rect = mpatches.FancyBboxPatch(
                (x - 0.001, y - line_height * 0.36),
                text_width + 0.002,
                line_height * 0.7,
                boxstyle="round,pad=0.002",
                facecolor=color,
                edgecolor=(0.7, 0.7, 0.7, 0.4) if loss > 0.5 else 'none',
                linewidth=0.4,
                alpha=0.92
            )
            ax.add_patch(rect)

            text_color = 'white' if loss > 1.3 else 'black'
            ax.text(x, y, text, fontsize=fontsize, fontfamily='monospace',
                    verticalalignment='center', color=text_color,
                    fontweight='bold' if loss > 1.0 else 'normal')

            x += text_width


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method_dir', required=True, help='Path to method token_loss_dumps directory')
    parser.add_argument('--baseline_dir', required=True, help='Path to baseline token_loss_dumps directory')
    parser.add_argument('--step', type=int, default=20, help='Training step to visualize')
    parser.add_argument('--num_tokens', type=int, default=80, help='Number of tokens to show')
    parser.add_argument('--max_chars', type=int, default=78, help='Max characters per line')
    parser.add_argument('--output', default='figs/token_loss_paper_figure.pdf', help='Output file path')
    parser.add_argument('--method_name', default='RESD (Ours)', help='Display name for method')
    parser.add_argument('--baseline_name', default='SDPO', help='Display name for baseline')
    parser.add_argument('--task_label', default=None, help='Task description for subtitle (auto-detected if not set)')
    args = parser.parse_args()

    m_file = os.path.join(args.method_dir, f'step_{args.step}_rank0.html')
    b_file = os.path.join(args.baseline_dir, f'step_{args.step}_rank0.html')

    m_tokens_full = extract_all_tokens_from_section(m_file)
    b_tokens_full = extract_all_tokens_from_section(b_file)

    if not m_tokens_full or not b_tokens_full:
        print(f"Error: Could not extract tokens from step {args.step}")
        return

    m_tokens = m_tokens_full[:args.num_tokens]
    b_tokens = b_tokens_full[:args.num_tokens]

    # Auto-detect task if not provided
    if args.task_label is None:
        with open(m_file) as f:
            content = f.read()
        task_match = re.search(r'Accept if ([^<]+?)(?:&lt;|<)', content)
        args.task_label = f'Accept if {html.unescape(task_match.group(1)).strip()}' if task_match else ''

    # Colormap: white -> light orange -> deep red
    colors_list = [
        (0.98, 0.98, 0.98),
        (1.0, 0.95, 0.8),
        (1.0, 0.8, 0.4),
        (0.95, 0.45, 0.15),
        (0.75, 0.1, 0.1),
    ]
    cmap = LinearSegmentedColormap.from_list('loss_cmap', colors_list, N=256)
    norm = Normalize(vmin=0, vmax=1.8)

    # Create figure
    fig = plt.figure(figsize=(10, 4.5))
    gs = fig.add_gridspec(2, 1, hspace=0.2, left=0.02, right=0.98, top=0.92, bottom=0.04)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    render_panel(ax1, m_tokens, f'(a) {args.method_name}',
                '',
                cmap, norm, max_chars=args.max_chars)
    render_panel(ax2, b_tokens, f'(b) {args.baseline_name}',
                '',
                cmap, norm, max_chars=args.max_chars)


    plt.savefig(args.output, bbox_inches='tight', dpi=300)
    # Also save PNG if output is PDF
    if args.output.endswith('.pdf'):
        plt.savefig(args.output.replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
    print(f"Saved: {args.output}")


if __name__ == '__main__':
    main()
