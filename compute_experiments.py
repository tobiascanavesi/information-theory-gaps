#!/usr/bin/env python3
"""
Information-Theoretic Characterization of Prime Gap Sequences
=============================================================
Tobias Canavesi, April 2026

Systematic measurement of information-theoretic properties of the
prime gap sequence g_n = p_{n+1} - p_n:

  Experiment 1: Block entropy (Shannon) and entropy rate
  Experiment 2: Mutual information at multiple lags
  Experiment 3: Lempel-Ziv complexity / compressibility
  Experiment 4: Goodness-of-fit to exponential / Poisson model
  Experiment 5: Conditional entropy by residue class

All results saved as CSV files in data/ for analysis and figure generation.
"""

import numpy as np
import csv
import time
from pathlib import Path
from collections import Counter
from math import log2, log, e as EULER_E
from sympy import nextprime
from scipy.stats import kstest, anderson, chisquare, expon, poisson

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_PRIMES = 200_000  # primes per scale
SCALES = [10**4, 10**5, 10**6, 10**7, 10**8]
MAX_BLOCK_K = 10
MAX_MI_LAG = 100
N_SURROGATES = 20

output_dir = Path(__file__).parent / 'data'
output_dir.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def banner(text):
    w = max(len(text) + 4, 80)
    print(f"\n{'=' * w}")
    print(f"  {text}")
    print(f"{'=' * w}")


def generate_primes_near(target, count):
    """Generate `count` consecutive primes starting near `target`."""
    primes = []
    p = nextprime(target - 1)
    for _ in range(count):
        primes.append(p)
        p = nextprime(p)
    return np.array(primes)


def shannon_entropy(counts_dict, total):
    """Compute Shannon entropy in bits from a frequency dictionary."""
    H = 0.0
    for c in counts_dict.values():
        if c > 0:
            p = c / total
            H -= p * log2(p)
    return H


# ---------------------------------------------------------------------------
# Prime generation (cached for all experiments)
# ---------------------------------------------------------------------------

banner("INFORMATION-THEORETIC CHARACTERIZATION OF PRIME GAP SEQUENCES")
print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Scales: {SCALES}")
print(f"Primes per scale: {N_PRIMES:,}")
total_start = time.time()

prime_data = {}  # scale -> (primes_array, gaps_array)

banner("PHASE 0: Generating primes and gaps at all scales")
for scale in SCALES:
    t0 = time.time()
    primes = generate_primes_near(scale, N_PRIMES)
    gaps = np.diff(primes)
    prime_data[scale] = (primes, gaps)
    elapsed = time.time() - t0
    print(f"  Scale {scale:.0e}: {N_PRIMES:,} primes in {elapsed:.1f}s  "
          f"(range [{primes[0]:,} .. {primes[-1]:,}], "
          f"mean gap = {gaps.mean():.2f})")


# ============================================================================
# EXPERIMENT 1: Block Entropy (Shannon) and Entropy Rate
# ============================================================================

banner("EXPERIMENT 1: Block Entropy and Entropy Rate")
t_exp = time.time()

block_entropy_rows = []

for scale in SCALES:
    primes, gaps = prime_data[scale]
    gaps_list = gaps.tolist()
    n_gaps = len(gaps_list)

    H_prev = 0.0
    H1 = None  # store single-gap entropy for i.i.d. baseline

    for k in range(1, MAX_BLOCK_K + 1):
        # Build overlapping k-tuples
        tuples = []
        for i in range(n_gaps - k + 1):
            tuples.append(tuple(gaps_list[i:i + k]))

        counts = Counter(tuples)
        n_samples = len(tuples)
        n_distinct = len(counts)

        H_k = shannon_entropy(counts, n_samples)

        if k == 1:
            H1 = H_k
            h_k = H_k
        else:
            h_k = H_k - H_prev

        H_k_iid = k * H1

        block_entropy_rows.append({
            'scale': scale,
            'k': k,
            'H_k': H_k,
            'h_k': h_k,
            'H_k_iid': H_k_iid,
            'redundancy': H_k_iid - H_k,
            'n_distinct': n_distinct,
            'n_samples': n_samples,
        })

        H_prev = H_k

    print(f"  Scale {scale:.0e}: H_1 = {H1:.4f} bits, "
          f"h_10 = {block_entropy_rows[-1]['h_k']:.4f} bits, "
          f"redundancy at k=10: {block_entropy_rows[-1]['redundancy']:.4f} bits")

# Write CSV
csv_path = output_dir / 'block_entropy.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'scale', 'k', 'H_k', 'h_k', 'H_k_iid', 'redundancy',
        'n_distinct', 'n_samples'])
    writer.writeheader()
    for row in block_entropy_rows:
        writer.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v)
                         for k, v in row.items()})
print(f"  Saved: {csv_path}")
print(f"  Time: {time.time() - t_exp:.1f}s")


# ============================================================================
# EXPERIMENT 2: Mutual Information at Multiple Lags
# ============================================================================

banner("EXPERIMENT 2: Mutual Information vs Lag")
t_exp = time.time()

mi_rows = []

for scale in SCALES:
    primes, gaps = prime_data[scale]
    gaps_list = gaps.tolist()
    n_gaps = len(gaps_list)

    # Marginal entropy (full sequence)
    marginal_counts = Counter(gaps_list)
    H_marginal = shannon_entropy(marginal_counts, n_gaps)
    alphabet_size = len(marginal_counts)

    for lag in range(1, MAX_MI_LAG + 1):
        n_pairs = n_gaps - lag

        # Joint distribution of (g_n, g_{n+lag})
        joint_counts = Counter()
        for i in range(n_pairs):
            joint_counts[(gaps_list[i], gaps_list[i + lag])] += 1

        H_joint = shannon_entropy(joint_counts, n_pairs)

        # Marginal entropies from the paired subsequences (for consistency)
        marginal_x_counts = Counter(gaps_list[:n_pairs])
        marginal_y_counts = Counter(gaps_list[lag:lag + n_pairs])
        H_x = shannon_entropy(marginal_x_counts, n_pairs)
        H_y = shannon_entropy(marginal_y_counts, n_pairs)

        MI_raw = H_x + H_y - H_joint

        # Miller-Madow bias correction
        alpha_x = len(marginal_x_counts)
        alpha_y = len(marginal_y_counts)
        alpha_joint = len(joint_counts)
        bias_correction = ((alpha_joint - alpha_x - alpha_y + 1)
                           / (2 * n_pairs * log(2)))
        MI_corrected = max(0.0, MI_raw - bias_correction)

        NMI = MI_corrected / H_marginal if H_marginal > 0 else 0.0

        # Autocorrelation for comparison
        autocorr = float(np.corrcoef(gaps[:n_pairs], gaps[lag:lag + n_pairs])[0, 1])

        mi_rows.append({
            'scale': scale,
            'lag': lag,
            'MI_bits': MI_raw,
            'MI_corrected': MI_corrected,
            'NMI': NMI,
            'autocorrelation': autocorr,
            'bias_correction': bias_correction,
            'n_samples': n_pairs,
        })

    print(f"  Scale {scale:.0e}: MI(lag=1) = {mi_rows[-MAX_MI_LAG]['MI_corrected']:.6f} bits, "
          f"MI(lag=10) = {mi_rows[-MAX_MI_LAG + 9]['MI_corrected']:.6f} bits, "
          f"autocorr(lag=1) = {mi_rows[-MAX_MI_LAG]['autocorrelation']:.4f}")

# Write CSV
csv_path = output_dir / 'mutual_information.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'scale', 'lag', 'MI_bits', 'MI_corrected', 'NMI',
        'autocorrelation', 'bias_correction', 'n_samples'])
    writer.writeheader()
    for row in mi_rows:
        writer.writerow({k: (f"{v:.8f}" if isinstance(v, float) else v)
                         for k, v in row.items()})
print(f"  Saved: {csv_path}")
print(f"  Time: {time.time() - t_exp:.1f}s")


# ============================================================================
# EXPERIMENT 3: Lempel-Ziv Complexity
# ============================================================================

banner("EXPERIMENT 3: Lempel-Ziv Complexity / Compressibility")
t_exp = time.time()


def lz76_complexity(sequence):
    """
    LZ76 factorization complexity: count distinct phrases when parsing
    the sequence left-to-right into shortest substrings not previously seen.
    """
    n = len(sequence)
    if n == 0:
        return 0

    phrases = set()
    current = []
    complexity = 0

    for symbol in sequence:
        current.append(symbol)
        key = tuple(current)
        if key not in phrases:
            phrases.add(key)
            complexity += 1
            current = []

    if current:
        complexity += 1

    return complexity


def generate_poisson_gaps(mean_gap, n, rng):
    """Generate synthetic gaps from discretized exponential distribution."""
    raw = rng.exponential(scale=mean_gap, size=n)
    # Round to nearest even integer >= 2 (prime gaps are even for p > 2)
    gaps = np.maximum(2, 2 * np.round(raw / 2).astype(int))
    return gaps.tolist()


lz_rows = []
rng = np.random.default_rng(42)

for scale in SCALES:
    primes, gaps = prime_data[scale]
    gaps_list = gaps.tolist()
    n = len(gaps_list)
    mean_gap = float(gaps.mean())
    alphabet_size = len(set(gaps_list))

    # LZ complexity of actual gap sequence
    C_actual = lz76_complexity(gaps_list)

    # Theoretical normalization: for random i.i.d. over alphabet k,
    # C ~ n / log_k(n)
    if alphabet_size > 1:
        C_random_expected = n / (log(n) / log(alphabet_size))
        C_normalized = C_actual / C_random_expected
    else:
        C_normalized = 0.0

    # Shuffled surrogates
    C_shuffled_list = []
    for _ in range(N_SURROGATES):
        shuffled = gaps_list.copy()
        rng.shuffle(shuffled)
        C_shuffled_list.append(lz76_complexity(shuffled))
    C_shuffled_mean = np.mean(C_shuffled_list)
    C_shuffled_std = np.std(C_shuffled_list)

    # Poisson surrogates (discretized exponential gaps)
    C_poisson_list = []
    for _ in range(N_SURROGATES):
        synth = generate_poisson_gaps(mean_gap, n, rng)
        C_poisson_list.append(lz76_complexity(synth))
    C_poisson_mean = np.mean(C_poisson_list)

    ratio_to_shuffled = C_actual / C_shuffled_mean if C_shuffled_mean > 0 else 0.0
    ratio_to_poisson = C_actual / C_poisson_mean if C_poisson_mean > 0 else 0.0

    lz_rows.append({
        'scale': scale,
        'C_raw': C_actual,
        'C_normalized': C_normalized,
        'C_shuffled_mean': C_shuffled_mean,
        'C_shuffled_std': C_shuffled_std,
        'C_poisson_mean': C_poisson_mean,
        'ratio_to_shuffled': ratio_to_shuffled,
        'ratio_to_poisson': ratio_to_poisson,
        'alphabet_size': alphabet_size,
        'n_symbols': n,
    })

    print(f"  Scale {scale:.0e}: C = {C_actual}, "
          f"C/C_shuffled = {ratio_to_shuffled:.4f}, "
          f"C/C_poisson = {ratio_to_poisson:.4f}, "
          f"alphabet = {alphabet_size}")

# Write CSV
csv_path = output_dir / 'lz_complexity.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'scale', 'C_raw', 'C_normalized', 'C_shuffled_mean', 'C_shuffled_std',
        'C_poisson_mean', 'ratio_to_shuffled', 'ratio_to_poisson',
        'alphabet_size', 'n_symbols'])
    writer.writeheader()
    for row in lz_rows:
        writer.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v)
                         for k, v in row.items()})
print(f"  Saved: {csv_path}")
print(f"  Time: {time.time() - t_exp:.1f}s")


# ============================================================================
# EXPERIMENT 4: Goodness-of-Fit to Exponential / Poisson Model
# ============================================================================

banner("EXPERIMENT 4: Goodness-of-Fit to Gallagher/Poisson Model")
t_exp = time.time()

# --- Part A: Exponential fit for normalized gaps ---

exp_fit_rows = []

for scale in SCALES:
    primes, gaps = prime_data[scale]
    mean_gap = float(gaps.mean())
    normalized = gaps / mean_gap

    # KS test against Exp(1)
    ks_stat, ks_pvalue = kstest(normalized, 'expon', args=(0, 1))

    # Anderson-Darling test against exponential
    ad_result = anderson(normalized, dist='expon')
    ad_stat = ad_result.statistic

    # Chi-squared test with 20 equal-probability bins under Exp(1)
    n_bins = 20
    bin_edges = [expon.ppf(i / n_bins) for i in range(n_bins)] + [np.inf]
    observed, _ = np.histogram(normalized, bins=bin_edges)
    expected_counts = np.full(n_bins, len(normalized) / n_bins)
    chi2_stat, chi2_pvalue = chisquare(observed, f_exp=expected_counts)

    exp_fit_rows.append({
        'scale': scale,
        'mean_gap': mean_gap,
        'ks_statistic': ks_stat,
        'ks_pvalue': ks_pvalue,
        'ad_statistic': ad_stat,
        'chi2_statistic': chi2_stat,
        'chi2_pvalue': chi2_pvalue,
        'n_gaps': len(gaps),
    })

    print(f"  Scale {scale:.0e}: KS = {ks_stat:.6f} (p={ks_pvalue:.2e}), "
          f"AD = {ad_stat:.4f}, chi2 = {chi2_stat:.1f} (p={chi2_pvalue:.2e})")

csv_path = output_dir / 'exponential_fit.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'scale', 'mean_gap', 'ks_statistic', 'ks_pvalue',
        'ad_statistic', 'chi2_statistic', 'chi2_pvalue', 'n_gaps'])
    writer.writeheader()
    for row in exp_fit_rows:
        writer.writerow({k: (f"{v:.8f}" if isinstance(v, float) else v)
                         for k, v in row.items()})
print(f"  Saved: {csv_path}")

# --- Part B: Poisson counting in windows ---

poisson_rows = []
lambdas = [1.0, 2.0, 5.0]

for scale in SCALES:
    primes, gaps = prime_data[scale]
    x_mid = float(np.median(primes))
    log_x = log(x_mid)

    for lam in lambdas:
        window_size = lam * log_x
        # Count primes in each window
        p_min, p_max = float(primes[0]), float(primes[-1])
        n_windows = int((p_max - p_min) / window_size)
        if n_windows < 20:
            continue

        window_starts = p_min + np.arange(n_windows) * window_size
        window_ends = window_starts + window_size

        counts_arr = np.zeros(n_windows, dtype=int)
        primes_sorted = primes.astype(float)
        for w in range(n_windows):
            counts_arr[w] = np.sum(
                (primes_sorted >= window_starts[w]) &
                (primes_sorted < window_ends[w]))

        mean_count = float(counts_arr.mean())
        var_count = float(counts_arr.var())

        # Chi-squared test against Poisson(lam)
        max_k = int(np.max(counts_arr)) + 1
        observed_hist = np.bincount(counts_arr, minlength=max_k + 1)

        # Group bins with expected count < 5 (standard rule)
        expected_probs = np.array([poisson.pmf(k, lam) for k in range(max_k + 1)])
        # Merge tail into last bin
        tail_prob = 1.0 - sum(expected_probs[:-1])
        expected_probs[-1] = max(tail_prob, 0.0)

        # Group small-expected bins from the right
        obs_grouped = []
        exp_grouped = []
        acc_obs = 0
        acc_exp = 0.0
        for i in range(len(observed_hist)):
            acc_obs += observed_hist[i]
            acc_exp += expected_probs[i] * n_windows
            if acc_exp >= 5:
                obs_grouped.append(acc_obs)
                exp_grouped.append(acc_exp)
                acc_obs = 0
                acc_exp = 0.0
        if acc_obs > 0:
            if exp_grouped:
                obs_grouped[-1] += acc_obs
                exp_grouped[-1] += acc_exp
            else:
                obs_grouped.append(acc_obs)
                exp_grouped.append(acc_exp)

        if len(obs_grouped) >= 2:
            chi2_p, chi2_pval = chisquare(obs_grouped, f_exp=exp_grouped)
        else:
            chi2_p, chi2_pval = float('nan'), float('nan')

        poisson_rows.append({
            'scale': scale,
            'lambda': lam,
            'n_windows': n_windows,
            'mean_count': mean_count,
            'var_count': var_count,
            'var_over_mean': var_count / mean_count if mean_count > 0 else 0,
            'chi2_statistic': chi2_p,
            'chi2_pvalue': chi2_pval,
        })

        print(f"  Scale {scale:.0e}, lam={lam}: mean={mean_count:.2f}, "
              f"var/mean={var_count / mean_count:.4f}, "
              f"chi2 p={chi2_pval:.2e}")

csv_path = output_dir / 'poisson_fit.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'scale', 'lambda', 'n_windows', 'mean_count', 'var_count',
        'var_over_mean', 'chi2_statistic', 'chi2_pvalue'])
    writer.writeheader()
    for row in poisson_rows:
        writer.writerow({k: (f"{v:.8f}" if isinstance(v, float) else v)
                         for k, v in row.items()})
print(f"  Saved: {csv_path}")
print(f"  Time: {time.time() - t_exp:.1f}s")


# ============================================================================
# EXPERIMENT 5: Conditional Entropy by Residue Class
# ============================================================================

banner("EXPERIMENT 5: Conditional Entropy by Residue Class")
t_exp = time.time()

cond_entropy_rows = []

for scale in SCALES:
    primes, gaps = prime_data[scale]
    gaps_list = gaps.tolist()
    primes_list = primes.tolist()
    n_gaps = len(gaps_list)

    # --- Part A: Block entropy for primes in each residue class mod 6 ---
    for residue in [1, 5]:
        # Select gaps originating from primes in this class
        # gap[i] = prime[i+1] - prime[i], so the "starting prime" is prime[i]
        class_gaps = [gaps_list[i] for i in range(n_gaps)
                      if primes_list[i] % 6 == residue]
        n_class = len(class_gaps)
        pct = 100.0 * n_class / n_gaps

        for k in range(1, min(6, len(class_gaps))):  # k=1..5 for class-restricted
            tuples = []
            for i in range(n_class - k + 1):
                tuples.append(tuple(class_gaps[i:i + k]))

            counts = Counter(tuples)
            n_samples = len(tuples)
            H_k = shannon_entropy(counts, n_samples)

            cond_entropy_rows.append({
                'scale': scale,
                'residue_class': f'{residue}_mod_6',
                'k': k,
                'H_k': H_k,
                'n_gaps': n_class,
                'pct_of_total': pct,
            })

        print(f"  Scale {scale:.0e}, p={residue} mod 6: "
              f"n={n_class} ({pct:.1f}%), "
              f"H_1={cond_entropy_rows[-min(5, len(class_gaps)) + 1]['H_k']:.4f} bits")

    # --- Part B: Block entropy for ALL primes (reference) ---
    for k in range(1, 6):
        tuples = [tuple(gaps_list[i:i + k]) for i in range(n_gaps - k + 1)]
        counts = Counter(tuples)
        H_k = shannon_entropy(counts, len(tuples))
        cond_entropy_rows.append({
            'scale': scale,
            'residue_class': 'all',
            'k': k,
            'H_k': H_k,
            'n_gaps': n_gaps,
            'pct_of_total': 100.0,
        })

    # --- Part C: Conditional entropy H(g_n | g_{n-1} mod 6) ---
    for prev_mod in [0, 2, 4]:
        cond_gaps = [gaps_list[i] for i in range(1, n_gaps)
                     if gaps_list[i - 1] % 6 == prev_mod]
        if len(cond_gaps) < 10:
            continue
        counts = Counter(cond_gaps)
        H_cond = shannon_entropy(counts, len(cond_gaps))
        cond_entropy_rows.append({
            'scale': scale,
            'residue_class': f'prev_gap_{prev_mod}_mod_6',
            'k': 1,
            'H_k': H_cond,
            'n_gaps': len(cond_gaps),
            'pct_of_total': 100.0 * len(cond_gaps) / n_gaps,
        })

    print(f"  Scale {scale:.0e}: conditional entropies computed")

# Write CSV
csv_path = output_dir / 'conditional_entropy.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'scale', 'residue_class', 'k', 'H_k', 'n_gaps', 'pct_of_total'])
    writer.writeheader()
    for row in cond_entropy_rows:
        writer.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v)
                         for k, v in row.items()})
print(f"  Saved: {csv_path}")
print(f"  Time: {time.time() - t_exp:.1f}s")


# ============================================================================
# SUMMARY
# ============================================================================

banner("COMPLETE")
total_elapsed = time.time() - total_start
print(f"Total runtime: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")
print(f"Output directory: {output_dir}")
print(f"Files generated:")
for f in sorted(output_dir.glob('*.csv')):
    print(f"  {f.name} ({f.stat().st_size:,} bytes)")
print(f"\nEnd: {time.strftime('%Y-%m-%d %H:%M:%S')}")
