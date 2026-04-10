# Information-Theoretic Properties of Prime Gap Sequences

Code, data, and paper for the first systematic information-theoretic analysis of the prime gap sequence $g_n = p_{n+1} - p_n$.

> **How Random Are Prime Gaps?**
> Tobias Canavesi, April 2026

## Overview

Prime gaps are widely believed to converge toward a Poisson process at large scales (Gallagher, 1976). But *how random* are they, really? We apply five information-theoretic measures — Shannon block entropy, mutual information, Lempel–Ziv complexity, distributional goodness-of-fit tests, and conditional entropy by residue class — to quantify the structure hidden in the gap sequence.

### Key Findings

1. **Massive memory**: 10 consecutive gaps carry only 18 bits of information — an i.i.d. model predicts 45. The gap sequence is 2.5× more predictable than random.

2. **Nonlinear dependence dwarfs linear correlation**: Mutual information between consecutive gaps is 0.32 bits — **300× more** than the autocorrelation (r ≈ −0.04) would predict. Standard models miss the dominant dependence structure.

3. **Compressible, but converging**: Lempel–Ziv complexity is 7% below shuffled surrogates at scale 10⁴, closing to 5.7% at scale 10⁸. Gaps are becoming "more random."

4. **Poisson is the destination, but we're not there yet**: Variance-to-mean ratio in Poisson windows climbs from 0.72 toward 1.0. All test statistics decrease monotonically with scale.

5. **Residue class asymmetry**: Primes ≡ 5 (mod 6) have ~0.06 bits more gap entropy than primes ≡ 1 (mod 6), linked to the Hardy–Littlewood singular series.

## Interactive Guide

Open **[interactive.html](interactive.html)** in any browser for an accessible, visual explanation with live charts and a prime gap explorer.

## Repository Structure

```
├── compute_experiments.py     # Main computation (5 experiments, ~6 min)
├── analyze_results.py         # Figure generation (6 PDF figures)
├── paper.tex                  # LaTeX paper (amsart)
├── paper.pdf                  # Compiled paper (8 pages)
├── interactive.html           # Interactive guide for general audience
├── data/
│   ├── block_entropy.csv          # Experiment 1: H_k at 5 scales, k=1..10
│   ├── mutual_information.csv     # Experiment 2: MI at lags 1–100
│   ├── lz_complexity.csv          # Experiment 3: LZ76 + surrogates
│   ├── exponential_fit.csv        # Experiment 4a: KS, AD, chi² tests
│   ├── poisson_fit.csv            # Experiment 4b: Poisson window counts
│   ├── conditional_entropy.csv    # Experiment 5: entropy by residue class
│   ├── fig_block_entropy.pdf      # Figure 1
│   ├── fig_mutual_information.pdf # Figure 2
│   ├── fig_lz_complexity.pdf      # Figure 3
│   ├── fig_exponential_fit.pdf    # Figure 4
│   ├── fig_conditional_entropy.pdf# Figure 5
│   └── fig_overview.pdf           # Figure 6 (composite)
```

## Reproducing the Results

```bash
# Install dependencies (numpy, sympy, scipy, matplotlib)
pip install numpy sympy scipy matplotlib

# Run all 5 experiments (~6 minutes)
python compute_experiments.py

# Generate figures
python analyze_results.py

# Compile paper
pdflatex paper.tex && pdflatex paper.tex
```

**Scales:** Primes near 10⁴, 10⁵, 10⁶, 10⁷, 10⁸ with 200,000 primes per scale.

**Dependencies:** Python 3.8+, numpy, sympy, scipy, matplotlib.

## The Five Experiments

| # | Experiment | Measures | Key result |
|---|-----------|----------|------------|
| 1 | Block entropy | H_k for k=1..10 | Redundancy = 27 bits at k=10 |
| 2 | Mutual information | MI(g_n; g_{n+ℓ}) for ℓ=1..100 | MI(lag=1) = 0.32 bits ≫ r² |
| 3 | Lempel–Ziv complexity | LZ76 vs shuffled/Poisson | C/C_shuffled = 0.92–0.94 |
| 4 | Goodness-of-fit | KS, AD, χ² vs Exp(1); Poisson windows | KS decreasing; σ²/μ → 1 |
| 5 | Conditional entropy | H_1 by prime mod 6 | Δ ≈ 0.06 bits (5 mod 6 > 1 mod 6) |

## Citation

```bibtex
@article{canavesi2026infogaps,
  title={Information-Theoretic Properties of Prime Gap Sequences},
  author={Canavesi, Tobias},
  year={2026}
}
```

## Related Work

- Canavesi (2026), *Toward Gilbreath's Conjecture via XOR Bias Absorption* — [gilbreath-conjecture](https://github.com/tobiascanavesi/gilbreath-conjecture)
- Gallagher (1976), *On the distribution of primes in short intervals*
- Cohen, Iyer, Manack (2024), *Gaps Between Consecutive Primes and the Exponential Distribution*, Experimental Mathematics
- Lempel & Ziv (1976), *On the complexity of finite sequences*, IEEE Trans. IT

## License

MIT
