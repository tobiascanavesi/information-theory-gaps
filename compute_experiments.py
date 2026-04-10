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
  Experiment 6: Cramer model MI (theoretical prediction)
  Experiment 7: Permutation tests for MI significance
  Experiment 8: LZ-based entropy rate estimator and H_1 scaling

All results saved as CSV files in data/ for analysis and figure generation.
"""

import numpy as np
import csv
import time
from pathlib import Path
from collections import Counter
from math import log2, log, e as EULER_E
from sympy import nextprime, factorint
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
# EXPERIMENT 6: Cramer Model MI (Theoretical Prediction)
# ============================================================================

banner("EXPERIMENT 6: Cramer Model MI — Theoretical vs Empirical")
t_exp = time.time()

cramer_rows = []

for scale in SCALES:
    primes, gaps = prime_data[scale]
    gaps_list = gaps.tolist()
    n_gaps = len(gaps_list)
    mean_gap = float(gaps.mean())
    L = log(float(np.median(primes)))  # log(x)

    # --- Cramer model: geometric gaps with parameter p = 1/L ---
    # In the Cramer model, each integer after a prime is independently prime
    # with probability ~1/log(x). For even gaps among odd numbers, g/2 ~ Geom(2/L).
    # Consecutive gaps are INDEPENDENT (memoryless property), so MI_Cramer = 0.
    MI_cramer = 0.0

    # --- Hardy-Littlewood model: compute predicted MI from singular series ---
    # Under H-L, P(gap = 2k) ~ S({0,2k}) * (1/L) * exp(-2k/L)
    # where S({0,h}) = 2*C2 * prod_{p|h, p>2} (p-1)/(p-2)
    # The joint P(g_n=2j, g_{n+1}=2k) involves the 3-tuple singular series
    # S({0, 2j, 2j+2k}).
    #
    # Precompute small primes for singular series (up to 500 covers all gaps).
    _small_primes = [p for p in range(3, 500)
                     if all(p % d != 0 for d in range(2, int(p**0.5) + 1))]

    def singular_series_pair(h):
        """S({0,h}) relative weight: prod_{p|h, p>2} (p-1)/(p-2).
        Uses sympy factorint for exact factorization."""
        if h <= 0 or h % 2 != 0:
            return 0.0
        product = 1.0
        for p in factorint(h):
            if p > 2:
                product *= (p - 1) / (p - 2)
        return product

    def singular_series_triple(h1, h2):
        """S({0,h1,h2}) relative weight (same normalization as pair^2).
        Product over primes p of local density factor for 3-tuple.
        For admissibility: need nu_p < p for all p.
        We compute prod_{p} f_3(p) / f_2(p)^2 where
          f_k(p) = (1 - nu_p/p) / (1-1/p)^k
        is the local factor for a k-tuple with nu_p distinct residues mod p.
        The ratio f_3(p)/f_2(p)^2 simplifies to:
          (1 - nu3/p)(1 - 1/p) / [(1 - nu_h1/p)(1 - nu_h2/p)]
        where nu3 = |{0,h1,h2} mod p|, nu_h1 = |{0,h1} mod p|, etc."""
        if h1 <= 0 or h2 <= 0 or h1 % 2 != 0 or h2 % 2 != 0 or h1 == h2:
            return 0.0
        product = 1.0
        for p in _small_primes:
            if p > max(h1, h2):
                break
            residues_3 = len({0 % p, h1 % p, h2 % p})
            if residues_3 >= p:
                return 0.0  # not admissible
            residues_h1 = len({0 % p, h1 % p})
            residues_h2 = len({0 % p, h2 % p})
            num = (1.0 - residues_3 / p) * (1.0 - 1.0 / p)
            den = (1.0 - residues_h1 / p) * (1.0 - residues_h2 / p)
            if den > 0 and num >= 0:
                product *= num / den
            elif num == 0:
                return 0.0
        return product

    # Build marginal P(g=h) ~ S_pair(h) * exp(-h/L)
    max_gap_val = int(min(8 * L, 300))
    gap_values = list(range(2, max_gap_val + 1, 2))

    marginal_weights = {g: singular_series_pair(g) * np.exp(-g / L)
                        for g in gap_values}
    Z_m = sum(marginal_weights.values())
    marginal_probs = {g: w / Z_m for g, w in marginal_weights.items()}
    H_hl_marginal = -sum(p * log2(p) for p in marginal_probs.values() if p > 0)

    # Build joint P(g1, g2) ~ S_pair(g1)*S_pair(g2)*coupling(g1,g1+g2)*exp(-(g1+g2)/L)
    # The coupling factor captures the non-independence from the triple singular series.
    joint_weights = {}
    for g1 in gap_values:
        sp1 = marginal_weights[g1]  # already includes S_pair and exponential
        for g2 in gap_values:
            sp2 = marginal_weights[g2]
            coupling = singular_series_triple(g1, g1 + g2)
            w = sp1 * sp2 * coupling
            if w > 0:
                joint_weights[(g1, g2)] = w
    Z_j = sum(joint_weights.values())

    if Z_j > 0:
        joint_probs = {k: w / Z_j for k, w in joint_weights.items()}
        H_hl_joint = -sum(p * log2(p) for p in joint_probs.values() if p > 0)
        # MI from marginals of the joint (not the standalone marginals)
        joint_marg1 = {}
        joint_marg2 = {}
        for (g1, g2), p in joint_probs.items():
            joint_marg1[g1] = joint_marg1.get(g1, 0) + p
            joint_marg2[g2] = joint_marg2.get(g2, 0) + p
        H_jm1 = -sum(p * log2(p) for p in joint_marg1.values() if p > 0)
        H_jm2 = -sum(p * log2(p) for p in joint_marg2.values() if p > 0)
        MI_hl = H_jm1 + H_jm2 - H_hl_joint
    else:
        MI_hl = 0.0

    # Empirical MI at lag 1 (from Experiment 2)
    mi_lag1_row = [r for r in mi_rows if r['scale'] == scale and r['lag'] == 1][0]
    MI_empirical = mi_lag1_row['MI_corrected']

    # Sieve contribution = empirical MI - Cramer MI
    MI_sieve = MI_empirical - MI_cramer

    cramer_rows.append({
        'scale': scale,
        'log_x': L,
        'MI_cramer': MI_cramer,
        'MI_hardy_littlewood': MI_hl,
        'MI_empirical': MI_empirical,
        'MI_sieve_contribution': MI_sieve,
        'H1_empirical': [r for r in block_entropy_rows
                         if r['scale'] == scale and r['k'] == 1][0]['H_k'],
        'H1_predicted': log2(L),  # log2(log(x))
    })

    print(f"  Scale {scale:.0e}: MI_Cramer = {MI_cramer:.6f}, "
          f"MI_H-L = {MI_hl:.6f}, MI_empirical = {MI_empirical:.6f}, "
          f"sieve contribution = {MI_sieve:.6f} bits")
    print(f"    H_1 empirical = {cramer_rows[-1]['H1_empirical']:.4f}, "
          f"H_1 predicted log2(log x) = {cramer_rows[-1]['H1_predicted']:.4f}")

csv_path = output_dir / 'cramer_model.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'scale', 'log_x', 'MI_cramer', 'MI_hardy_littlewood', 'MI_empirical',
        'MI_sieve_contribution', 'H1_empirical', 'H1_predicted'])
    writer.writeheader()
    for row in cramer_rows:
        writer.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v)
                         for k, v in row.items()})
print(f"  Saved: {csv_path}")
print(f"  Time: {time.time() - t_exp:.1f}s")


# ============================================================================
# EXPERIMENT 7: Permutation Tests for MI Significance
# ============================================================================

banner("EXPERIMENT 7: Permutation Tests for MI Significance")
t_exp = time.time()

N_PERMUTATIONS = 200
perm_rows = []
rng_perm = np.random.default_rng(123)

# Test at selected lags across all scales
test_lags = [1, 2, 3, 5, 10, 20, 50, 100]

for scale in SCALES:
    primes, gaps = prime_data[scale]
    gaps_list = gaps.tolist()
    n_gaps = len(gaps_list)

    for lag in test_lags:
        n_pairs = n_gaps - lag

        # Empirical MI (from Experiment 2)
        mi_row = [r for r in mi_rows
                   if r['scale'] == scale and r['lag'] == lag][0]
        MI_obs = mi_row['MI_bits']  # raw MI (before correction)

        # Null distribution: shuffle g_n (the first variable) while keeping
        # g_{n+lag} fixed. This tests the null hypothesis of independence
        # while preserving both marginal distributions.
        null_mi = []
        for _ in range(N_PERMUTATIONS):
            shuffled = gaps_list[:n_pairs].copy()
            rng_perm.shuffle(shuffled)
            joint_counts = Counter()
            for i in range(n_pairs):
                joint_counts[(shuffled[i], gaps_list[i + lag])] += 1
            H_joint = shannon_entropy(joint_counts, n_pairs)

            marginal_x_counts = Counter(shuffled)
            marginal_y_counts = Counter(gaps_list[lag:lag + n_pairs])
            H_x = shannon_entropy(marginal_x_counts, n_pairs)
            H_y = shannon_entropy(marginal_y_counts, n_pairs)

            null_mi.append(H_x + H_y - H_joint)

        null_mean = float(np.mean(null_mi))
        null_std = float(np.std(null_mi))
        z_score = (MI_obs - null_mean) / null_std if null_std > 0 else 0.0
        p_value = float(np.mean([m >= MI_obs for m in null_mi]))
        significant = MI_obs > np.percentile(null_mi, 99)

        perm_rows.append({
            'scale': scale,
            'lag': lag,
            'MI_observed': MI_obs,
            'null_mean': null_mean,
            'null_std': null_std,
            'z_score': z_score,
            'p_value_permutation': p_value,
            'significant_99': int(significant),
        })

    sig_count = sum(1 for r in perm_rows[-len(test_lags):]
                    if r['significant_99'])
    print(f"  Scale {scale:.0e}: {sig_count}/{len(test_lags)} lags significant at 99%")

csv_path = output_dir / 'permutation_tests.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'scale', 'lag', 'MI_observed', 'null_mean', 'null_std',
        'z_score', 'p_value_permutation', 'significant_99'])
    writer.writeheader()
    for row in perm_rows:
        writer.writerow({k: (f"{v:.8f}" if isinstance(v, float) else v)
                         for k, v in row.items()})
print(f"  Saved: {csv_path}")
print(f"  Time: {time.time() - t_exp:.1f}s")


# ============================================================================
# EXPERIMENT 8: LZ-Based Entropy Rate Estimator and H_1 Scaling
# ============================================================================

banner("EXPERIMENT 8: LZ Entropy Rate Estimator and H_1 Scaling")
t_exp = time.time()

lz_rate_rows = []

for scale in SCALES:
    primes, gaps = prime_data[scale]
    gaps_list = gaps.tolist()
    n = len(gaps_list)
    alphabet_size = len(set(gaps_list))
    L = log(float(np.median(primes)))

    # LZ-based entropy rate: h_LZ = C(s) * log_alpha(n) / n
    # where alpha = alphabet size, and C(s) is the LZ76 complexity
    C = lz76_complexity(gaps_list)
    if alphabet_size > 1:
        h_lz = C * (log(n) / log(alphabet_size)) / n
    else:
        h_lz = 0.0

    # Same for shuffled (gives h of the i.i.d. marginal)
    C_shuffled_list = []
    for _ in range(20):
        s = gaps_list.copy()
        rng.shuffle(s)
        C_shuffled_list.append(lz76_complexity(s))
    C_shuffled_mean = float(np.mean(C_shuffled_list))
    if alphabet_size > 1:
        h_lz_shuffled = C_shuffled_mean * (log(n) / log(alphabet_size)) / n
    else:
        h_lz_shuffled = 0.0

    # Ratio of entropy rates
    h_ratio = h_lz / h_lz_shuffled if h_lz_shuffled > 0 else 0.0

    # H_1 scaling: empirical vs log2(log(x))
    H1_emp = [r for r in block_entropy_rows
              if r['scale'] == scale and r['k'] == 1][0]['H_k']
    H1_pred = log2(L)
    H1_residual = H1_emp - H1_pred

    lz_rate_rows.append({
        'scale': scale,
        'log_x': L,
        'C_gaps': C,
        'h_lz_gaps': h_lz,
        'h_lz_shuffled': h_lz_shuffled,
        'h_ratio': h_ratio,
        'H1_empirical': H1_emp,
        'H1_log2logx': H1_pred,
        'H1_residual': H1_residual,
    })

    print(f"  Scale {scale:.0e}: h_LZ(gaps) = {h_lz:.4f}, "
          f"h_LZ(shuffled) = {h_lz_shuffled:.4f}, "
          f"ratio = {h_ratio:.4f}")
    print(f"    H_1 = {H1_emp:.4f}, log2(log x) = {H1_pred:.4f}, "
          f"residual = {H1_residual:.4f}")

csv_path = output_dir / 'lz_entropy_rate.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'scale', 'log_x', 'C_gaps', 'h_lz_gaps', 'h_lz_shuffled',
        'h_ratio', 'H1_empirical', 'H1_log2logx', 'H1_residual'])
    writer.writeheader()
    for row in lz_rate_rows:
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
