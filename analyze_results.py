#!/usr/bin/env python3
"""
Analysis and Figure Generation for Information-Theoretic Prime Gap Experiments
==============================================================================
Tobias Canavesi, April 2026

Reads CSV output from compute_experiments.py and produces publication-quality
figures summarizing all five experiments.

Figures:
  1. fig_block_entropy.pdf     — Block entropy H_k and entropy rate
  2. fig_mutual_information.pdf — Mutual information vs lag
  3. fig_lz_complexity.pdf     — Lempel-Ziv complexity ratios
  4. fig_exponential_fit.pdf   — Goodness-of-fit to Exp(1)
  5. fig_conditional_entropy.pdf — Entropy by residue class
  6. fig_overview.pdf          — 4-panel composite for publication
"""

import numpy as np
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

data_dir = Path(__file__).parent / 'data'

# Consistent color scheme across figures
SCALE_COLORS = {
    10**4: '#1f77b4',
    10**5: '#ff7f0e',
    10**6: '#2ca02c',
    10**7: '#d62728',
    10**8: '#9467bd',
}
SCALE_LABELS = {s: f'$10^{{{int(np.log10(s))}}}$' for s in SCALE_COLORS}


def read_csv(filename):
    """Read a CSV file into a list of dicts with numeric conversion."""
    rows = []
    with open(data_dir / filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            converted = {}
            for k, v in row.items():
                try:
                    converted[k] = int(v)
                except ValueError:
                    try:
                        converted[k] = float(v)
                    except ValueError:
                        converted[k] = v
            rows.append(converted)
    return rows


def group_by(rows, key):
    """Group rows by a key field."""
    groups = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    return groups


# ============================================================================
# Figure 1: Block Entropy and Entropy Rate
# ============================================================================

def fig_block_entropy():
    rows = read_csv('block_entropy.csv')
    by_scale = group_by(rows, 'scale')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left panel: H_k vs k
    for scale in sorted(by_scale.keys()):
        data = sorted(by_scale[scale], key=lambda r: r['k'])
        ks = [d['k'] for d in data]
        Hks = [d['H_k'] for d in data]
        ax1.plot(ks, Hks, 'o-', color=SCALE_COLORS.get(scale, 'gray'),
                 label=SCALE_LABELS.get(scale, str(scale)), markersize=4)

    # i.i.d. baseline (from largest scale)
    largest = max(by_scale.keys())
    data = sorted(by_scale[largest], key=lambda r: r['k'])
    ks = [d['k'] for d in data]
    Hk_iid = [d['H_k_iid'] for d in data]
    ax1.plot(ks, Hk_iid, 'k--', alpha=0.5, linewidth=1.5, label='i.i.d. baseline')

    ax1.set_xlabel('Block size $k$')
    ax1.set_ylabel('Block entropy $H_k$ (bits)')
    ax1.set_title('(a) Block Entropy')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Right panel: Entropy rate h_k vs k
    for scale in sorted(by_scale.keys()):
        data = sorted(by_scale[scale], key=lambda r: r['k'])
        ks = [d['k'] for d in data]
        hks = [d['h_k'] for d in data]
        ax2.plot(ks, hks, 'o-', color=SCALE_COLORS.get(scale, 'gray'),
                 label=SCALE_LABELS.get(scale, str(scale)), markersize=4)

    ax2.set_xlabel('Block size $k$')
    ax2.set_ylabel('Entropy rate $h_k = H_k - H_{k-1}$ (bits)')
    ax2.set_title('(b) Entropy Rate Convergence')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(data_dir / 'fig_block_entropy.pdf', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved fig_block_entropy.pdf")


# ============================================================================
# Figure 2: Mutual Information vs Lag
# ============================================================================

def fig_mutual_information():
    rows = read_csv('mutual_information.csv')
    by_scale = group_by(rows, 'scale')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left panel: MI vs lag (log y-axis)
    for scale in sorted(by_scale.keys()):
        data = sorted(by_scale[scale], key=lambda r: r['lag'])
        lags = [d['lag'] for d in data]
        mi = [max(d['MI_corrected'], 1e-10) for d in data]
        ax1.semilogy(lags, mi, '-', color=SCALE_COLORS.get(scale, 'gray'),
                     label=SCALE_LABELS.get(scale, str(scale)), linewidth=1)

    ax1.set_xlabel('Lag')
    ax1.set_ylabel('Mutual Information (bits)')
    ax1.set_title('(a) MI$(g_n; g_{n+\\ell})$ vs Lag')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Right panel: MI vs |autocorrelation|^2 at largest scale
    largest = max(by_scale.keys())
    data = sorted(by_scale[largest], key=lambda r: r['lag'])
    mi = [d['MI_corrected'] for d in data]
    ac2 = [d['autocorrelation']**2 for d in data]
    lags = [d['lag'] for d in data]

    ax2.scatter(ac2, mi, c=lags, cmap='viridis', s=15, alpha=0.7)
    # Reference line: MI ~ r^2 / (2 ln 2) for Gaussian
    ac2_sorted = sorted(ac2)
    mi_gaussian = [r2 / (2 * np.log(2)) for r2 in ac2_sorted]
    ax2.plot(ac2_sorted, mi_gaussian, 'r--', alpha=0.5, label='Gaussian: $r^2/(2\\ln 2)$')

    ax2.set_xlabel('$|\\rho|^2$ (squared autocorrelation)')
    ax2.set_ylabel('Mutual Information (bits)')
    ax2.set_title(f'(b) MI vs $|\\rho|^2$ at scale {SCALE_LABELS.get(largest, str(largest))}')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    cbar = fig.colorbar(ax2.collections[0], ax=ax2, label='Lag')

    fig.tight_layout()
    fig.savefig(data_dir / 'fig_mutual_information.pdf', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved fig_mutual_information.pdf")


# ============================================================================
# Figure 3: Lempel-Ziv Complexity
# ============================================================================

def fig_lz_complexity():
    rows = read_csv('lz_complexity.csv')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    scales = [r['scale'] for r in rows]
    scale_labels = [f'$10^{{{int(np.log10(s))}}}$' for s in scales]
    x = np.arange(len(scales))
    width = 0.25

    # Left panel: Normalized complexity comparison
    C_norm = [r['C_normalized'] for r in rows]
    C_shuf = [r['C_shuffled_mean'] / (r['n_symbols'] / (np.log(r['n_symbols']) / np.log(r['alphabet_size'])))
              for r in rows]
    C_pois = [r['C_poisson_mean'] / (r['n_symbols'] / (np.log(r['n_symbols']) / np.log(r['alphabet_size'])))
              for r in rows]

    ax1.bar(x - width, C_norm, width, label='Prime gaps', color='#1f77b4')
    ax1.bar(x, C_shuf, width, label='Shuffled', color='#ff7f0e')
    ax1.bar(x + width, C_pois, width, label='Poisson synthetic', color='#2ca02c')
    ax1.set_xticks(x)
    ax1.set_xticklabels(scale_labels)
    ax1.set_xlabel('Scale (primes near)')
    ax1.set_ylabel('Normalized LZ76 Complexity')
    ax1.set_title('(a) LZ Complexity Comparison')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, axis='y')

    # Right panel: Ratio to shuffled
    ratios = [r['ratio_to_shuffled'] for r in rows]
    ratios_p = [r['ratio_to_poisson'] for r in rows]
    ax2.plot(scales, ratios, 'o-', color='#1f77b4', label='C(gaps) / C(shuffled)')
    ax2.plot(scales, ratios_p, 's--', color='#2ca02c', label='C(gaps) / C(Poisson)')
    ax2.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    ax2.set_xscale('log')
    ax2.set_xlabel('Scale')
    ax2.set_ylabel('Complexity Ratio')
    ax2.set_title('(b) Compressibility Ratio')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(data_dir / 'fig_lz_complexity.pdf', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved fig_lz_complexity.pdf")


# ============================================================================
# Figure 4: Goodness-of-Fit to Exponential
# ============================================================================

def fig_exponential_fit():
    rows = read_csv('exponential_fit.csv')

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Left panel: QQ plot at largest scale
    largest_row = max(rows, key=lambda r: r['scale'])
    # We need the actual data for QQ plot -- regenerate normalized gaps
    # Instead, just show the test statistics across scales
    scales = [r['scale'] for r in rows]
    ks_stats = [r['ks_statistic'] for r in rows]
    ad_stats = [r['ad_statistic'] for r in rows]
    chi2_stats = [r['chi2_statistic'] for r in rows]
    ks_pvals = [r['ks_pvalue'] for r in rows]
    chi2_pvals = [r['chi2_pvalue'] for r in rows]

    # Panel 1: Test statistics vs scale
    ax1.plot(scales, ks_stats, 'o-', color='#1f77b4', label='KS statistic')
    ax1.set_xscale('log')
    ax1.set_xlabel('Scale')
    ax1.set_ylabel('KS Statistic')
    ax1.set_title('(a) KS Statistic vs Scale')
    ax1.grid(True, alpha=0.3)

    # Panel 2: AD statistic vs scale
    ax2.plot(scales, ad_stats, 's-', color='#d62728', label='AD statistic')
    ax2.set_xscale('log')
    ax2.set_xlabel('Scale')
    ax2.set_ylabel('Anderson-Darling Statistic')
    ax2.set_title('(b) Anderson-Darling vs Scale')
    ax2.grid(True, alpha=0.3)

    # Panel 3: Test statistics summary (p-values are all ~0, so show statistics)
    # Normalize each statistic to its value at the smallest scale to show trend
    ks_norm = [k / ks_stats[0] for k in ks_stats]
    ad_norm = [a / ad_stats[0] for a in ad_stats]
    chi2_norm = [c / chi2_stats[0] for c in chi2_stats]
    ax3.plot(scales, ks_norm, 'o-', color='#1f77b4', label='KS (normalized)')
    ax3.plot(scales, ad_norm, 's-', color='#d62728', label='AD (normalized)')
    ax3.plot(scales, chi2_norm, 'D-', color='#2ca02c', label='$\\chi^2$ (normalized)')
    ax3.set_xscale('log')
    ax3.set_xlabel('Scale')
    ax3.set_ylabel('Statistic / Statistic($10^4$)')
    ax3.set_title('(c) Normalized Test Statistics')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(data_dir / 'fig_exponential_fit.pdf', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved fig_exponential_fit.pdf")


# ============================================================================
# Figure 5: Conditional Entropy by Residue Class
# ============================================================================

def fig_conditional_entropy():
    rows = read_csv('conditional_entropy.csv')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left panel: H_1 by residue class at each scale
    by_class = group_by(rows, 'residue_class')
    class_colors = {
        '1_mod_6': '#1f77b4',
        '5_mod_6': '#ff7f0e',
        'all': '#2ca02c',
    }

    for cls in ['1_mod_6', '5_mod_6', 'all']:
        if cls not in by_class:
            continue
        data = [r for r in by_class[cls] if r['k'] == 1]
        data.sort(key=lambda r: r['scale'])
        scales = [r['scale'] for r in data]
        H1s = [r['H_k'] for r in data]
        label = {'1_mod_6': '$p \\equiv 1$ mod 6',
                 '5_mod_6': '$p \\equiv 5$ mod 6',
                 'all': 'All primes'}[cls]
        ax1.plot(scales, H1s, 'o-', color=class_colors[cls], label=label)

    ax1.set_xscale('log')
    ax1.set_xlabel('Scale')
    ax1.set_ylabel('$H_1$ (bits)')
    ax1.set_title('(a) Single-Gap Entropy by Residue Class')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Right panel: Block entropy H_k at largest scale, comparing classes
    largest_scale = max(r['scale'] for r in rows)
    for cls in ['1_mod_6', '5_mod_6', 'all']:
        if cls not in by_class:
            continue
        data = [r for r in by_class[cls] if r['scale'] == largest_scale]
        data.sort(key=lambda r: r['k'])
        ks = [r['k'] for r in data]
        Hks = [r['H_k'] for r in data]
        label = {'1_mod_6': '$p \\equiv 1$ mod 6',
                 '5_mod_6': '$p \\equiv 5$ mod 6',
                 'all': 'All primes'}[cls]
        ax2.plot(ks, Hks, 'o-', color=class_colors[cls], label=label)

    ax2.set_xlabel('Block size $k$')
    ax2.set_ylabel('$H_k$ (bits)')
    ax2.set_title(f'(b) Block Entropy at Scale {SCALE_LABELS.get(largest_scale, str(largest_scale))}')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(data_dir / 'fig_conditional_entropy.pdf', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved fig_conditional_entropy.pdf")


# ============================================================================
# Figure 6: Overview Composite (Publication Figure)
# ============================================================================

def fig_overview():
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel (a): Entropy rate convergence
    ax = axes[0, 0]
    rows = read_csv('block_entropy.csv')
    by_scale = group_by(rows, 'scale')
    for scale in sorted(by_scale.keys()):
        data = sorted(by_scale[scale], key=lambda r: r['k'])
        ks = [d['k'] for d in data]
        hks = [d['h_k'] for d in data]
        ax.plot(ks, hks, 'o-', color=SCALE_COLORS.get(scale, 'gray'),
                label=SCALE_LABELS.get(scale, str(scale)), markersize=3)
    ax.set_xlabel('Block size $k$')
    ax.set_ylabel('$h_k$ (bits)')
    ax.set_title('(a) Entropy Rate Convergence')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Panel (b): MI decay at largest scale
    ax = axes[0, 1]
    mi_rows = read_csv('mutual_information.csv')
    mi_by_scale = group_by(mi_rows, 'scale')
    largest = max(mi_by_scale.keys())
    data = sorted(mi_by_scale[largest], key=lambda r: r['lag'])
    lags = [d['lag'] for d in data]
    mi = [max(d['MI_corrected'], 1e-10) for d in data]
    ax.semilogy(lags, mi, '-', color='#1f77b4', linewidth=1.2)
    ax.set_xlabel('Lag $\\ell$')
    ax.set_ylabel('MI (bits)')
    ax.set_title(f'(b) Mutual Information Decay (scale {SCALE_LABELS.get(largest)})')
    ax.grid(True, alpha=0.3)

    # Panel (c): LZ complexity ratio
    ax = axes[1, 0]
    lz_rows = read_csv('lz_complexity.csv')
    scales = [r['scale'] for r in lz_rows]
    ratios_s = [r['ratio_to_shuffled'] for r in lz_rows]
    ratios_p = [r['ratio_to_poisson'] for r in lz_rows]
    ax.plot(scales, ratios_s, 'o-', color='#1f77b4', label='vs. shuffled')
    ax.plot(scales, ratios_p, 's--', color='#2ca02c', label='vs. Poisson')
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xscale('log')
    ax.set_xlabel('Scale')
    ax.set_ylabel('$C_{\\mathrm{gaps}} / C_{\\mathrm{reference}}$')
    ax.set_title('(c) LZ76 Compressibility Ratio')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel (d): KS statistic vs scale
    ax = axes[1, 1]
    exp_rows = read_csv('exponential_fit.csv')
    scales_e = [r['scale'] for r in exp_rows]
    ks_stats = [r['ks_statistic'] for r in exp_rows]
    ax.plot(scales_e, ks_stats, 'o-', color='#d62728', linewidth=1.5)
    ax.set_xscale('log')
    ax.set_xlabel('Scale')
    ax.set_ylabel('KS Statistic')
    ax.set_title('(d) Exp(1) Fit: KS Statistic')
    ax.grid(True, alpha=0.3)

    fig.suptitle('Information-Theoretic Properties of Prime Gap Sequences',
                 fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(data_dir / 'fig_overview.pdf', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved fig_overview.pdf")


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print("Generating figures from CSV data...")
    print(f"Data directory: {data_dir}\n")

    fig_block_entropy()
    fig_mutual_information()
    fig_lz_complexity()
    fig_exponential_fit()
    fig_conditional_entropy()
    fig_overview()

    print(f"\nAll figures saved to {data_dir}/")
