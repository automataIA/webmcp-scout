//! Percentile bootstrap and latency percentiles, deterministic by seed.

pub const RESAMPLES: usize = 2000;
pub const SEED: u64 = 0x5C07_0E7A_1B00_0002;

/// Ratio metric with a 95% percentile-bootstrap interval.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Estimate {
    pub value: f64,
    pub low: f64,
    pub high: f64,
    /// Resampling units (scenarios) with a non-zero denominator.
    pub n: usize,
}

/// `Σ num / Σ den` over units, resampling units with replacement. Units with
/// `den == 0` do not count toward the metric. `None` when no unit counts.
pub fn bootstrap_ratio(units: &[(f64, f64)], resamples: usize, seed: u64) -> Option<Estimate> {
    let units: Vec<(f64, f64)> = units.iter().copied().filter(|u| u.1 > 0.0).collect();
    let value = ratio(units.iter())?;
    let mut rng = SplitMix64(seed);
    let mut samples: Vec<f64> = (0..resamples)
        .filter_map(|_| ratio((0..units.len()).map(|_| &units[rng.below(units.len())])))
        .collect();
    samples.sort_by(f64::total_cmp);
    Some(Estimate {
        value,
        low: quantile(&samples, 0.025),
        high: quantile(&samples, 0.975),
        n: units.len(),
    })
}

fn ratio<'a>(units: impl Iterator<Item = &'a (f64, f64)>) -> Option<f64> {
    let (num, den) = units.fold((0.0, 0.0), |(n, d), u| (n + u.0, d + u.1));
    (den > 0.0).then(|| num / den)
}

/// Nearest-rank percentile of `values` (`p` in 0..=1).
pub fn percentile(values: &[u64], p: f64) -> Option<u64> {
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let rank = (p * sorted.len() as f64).ceil() as usize;
    sorted.get(rank.clamp(1, sorted.len().max(1)) - 1).copied()
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    sorted[((sorted.len() - 1) as f64 * q).round() as usize]
}

/// Two-sided exact McNemar test on the discordant pairs: `b` pairs where only
/// the first system is right, `c` where only the second is. The p-value of a
/// fair coin giving a split at least this uneven.
pub fn mcnemar_exact(b: usize, c: usize) -> f64 {
    let n = b + c;
    if n == 0 {
        return 1.0;
    }
    // P(X <= k) for X ~ Binomial(n, 1/2), built term by term.
    let mut term = 0.5f64.powi(n as i32);
    let mut tail = term;
    for i in 1..=b.min(c) {
        term *= (n + 1 - i) as f64 / i as f64;
        tail += term;
    }
    (2.0 * tail).min(1.0)
}

/// SplitMix64: tiny, well-distributed, and stable across platforms.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bootstrap_brackets_the_point_estimate_and_is_deterministic() {
        let units: Vec<(f64, f64)> = (0..40).map(|i| (f64::from(i % 4 != 0), 1.0)).collect();
        let a = bootstrap_ratio(&units, RESAMPLES, SEED).unwrap();
        assert_eq!(a.value, 0.75);
        assert!(a.low < 0.75 && a.high > 0.75 && a.low > 0.5 && a.high <= 1.0);
        assert_eq!(Some(a), bootstrap_ratio(&units, RESAMPLES, SEED));
        assert_eq!(a.n, 40);
    }

    #[test]
    fn degenerate_inputs() {
        assert_eq!(bootstrap_ratio(&[(0.0, 0.0)], RESAMPLES, SEED), None);
        let all = bootstrap_ratio(&[(1.0, 1.0); 5], RESAMPLES, SEED).unwrap();
        assert_eq!((all.value, all.low, all.high), (1.0, 1.0, 1.0));
    }

    #[test]
    fn mcnemar_exact_matches_the_binomial_tail() {
        assert_eq!(mcnemar_exact(0, 0), 1.0);
        assert_eq!(mcnemar_exact(3, 3), 1.0);
        // 0 vs 6: 2 * 0.5^6.
        assert!((mcnemar_exact(0, 6) - 0.03125).abs() < 1e-12);
        // 1 vs 9: 2 * (1 + 10) / 1024.
        assert!((mcnemar_exact(9, 1) - 22.0 / 1024.0).abs() < 1e-12);
    }

    #[test]
    fn nearest_rank_percentiles() {
        let v = [50, 10, 40, 20, 30];
        assert_eq!(percentile(&v, 0.5), Some(30));
        assert_eq!(percentile(&v, 0.95), Some(50));
        assert_eq!(percentile(&[], 0.5), None);
    }
}
