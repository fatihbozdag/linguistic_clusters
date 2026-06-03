#!/usr/bin/env python3
"""
Regenerate Figure 2: sensitivity-analysis heatmap.

Produces figures/Figure2_Sensitivity_Heatmap.png from hardcoded values
verified against the OFAT sensitivity-analysis output. Values are asserted
against the expected overall-stability column at script start; a mismatch
fails loudly rather than producing a silently-wrong figure.

Output: 600 DPI PNG suitable for direct submission to publisher (no need
for downstream upscaling or metadata re-tagging).

Usage:
    python scripts/plot_sensitivity_heatmap.py
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ── cell values verified against OFAT sensitivity output ──
SCHEMAS = [
    'modal-be,pp_up_NP',
    'modal-be,pp_above_NP',
    'modal-be,pp_towards_NP',
    'perf-be,pp_within_NP',
    'modal-be,pp_over_NP',
    'past-be,pp_out_NP',
    'modal-be,pp_around_NP',
    'modal-be,pp_outside_NP',
]
THRESHOLDS = ['ATP', r'$\Delta P_{\mathrm{backward}}$', r'$H_r$', 'NPMI', 'Hslot']
DATA = np.array([
    [100, 100, 100, 100, 100],
    [100, 100, 100,  80, 100],
    [100, 100,  80, 100,  80],
    [100,  80, 100, 100,  80],
    [100,  80, 100,  80,  80],
    [ 80, 100,  80,  80,  80],
    [ 80,  80,  80,  60,  80],
    [ 80,  80,  60,  60,  80],
], dtype=float)

EXPECTED_OVERALL = [100, 96, 92, 92, 88, 84, 76, 72]


def main():
    overall = DATA.mean(axis=1)
    for i, (got, exp) in enumerate(zip(overall, EXPECTED_OVERALL)):
        assert abs(got - exp) < 0.01, f'row {i} ({SCHEMAS[i]}): got {got}, expected {exp}'
    print('verified: overall-stability column matches expected values')

    mpl.rcParams['font.family'] = 'serif'
    fig, ax = plt.subplots(figsize=(11, 6.5))

    im = ax.imshow(DATA / 100.0, cmap='Greens', aspect='auto', vmin=0.0, vmax=1.0)

    for i in range(DATA.shape[0]):
        for j in range(DATA.shape[1]):
            ax.text(j, i, f'{int(DATA[i, j])}%',
                    ha='center', va='center',
                    color='white', fontweight='bold', fontsize=16)

    ax.set_xticks(np.arange(len(THRESHOLDS)))
    ax.set_xticklabels(THRESHOLDS, fontsize=16)
    ax.set_yticks(np.arange(len(SCHEMAS)))
    ax.set_yticklabels(SCHEMAS, fontsize=14)
    ax.set_xlabel('Parameter Varied', fontsize=16, labelpad=12)
    ax.set_ylabel('Schema', fontsize=16, labelpad=12)
    ax.tick_params(top=False, bottom=True, labelbottom=True)

    ax_right = ax.twinx()
    ax_right.set_ylim(ax.get_ylim())
    ax_right.set_yticks(np.arange(len(SCHEMAS)))
    ax_right.set_yticklabels([f'{int(v)}%' for v in overall], fontsize=14)
    ax_right.set_ylabel('Overall Stability', fontsize=16, labelpad=18, rotation=270)

    cbar = fig.colorbar(im, ax=ax_right, pad=0.18, shrink=0.85)
    cbar.set_label('Proportion of Configurations', fontsize=14, labelpad=10)
    cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.ax.tick_params(labelsize=12)

    ax.axhline(y=5.5, color='red', linestyle='--', linewidth=2.0, alpha=0.9)

    plt.tight_layout()

    out = Path(__file__).parent.parent / 'figures' / 'Figure2_Sensitivity_Heatmap.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=600, bbox_inches='tight', facecolor='white')
    print(f'saved: {out} ({out.stat().st_size:,} bytes)')


if __name__ == '__main__':
    main()
