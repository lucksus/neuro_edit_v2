//! Deterministic seeded RNG for network construction.
//!
//! SplitMix64 seed expansion into xoshiro256**. Reproducible across platforms
//! and runs — important for preregistered experiments. Not a CSPRNG.

/// Seeded RNG used by builders and by the stimulus generators in `sim`.
#[derive(Clone, Debug)]
pub struct Rng {
    s: [u64; 4],
}

fn rotl(x: u64, k: u32) -> u64 {
    x.rotate_left(k)
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        // SplitMix64 to fill the state.
        let mut z = seed;
        let mut next = || {
            z = z.wrapping_add(0x9E3779B97F4A7C15);
            let mut r = z;
            r = (r ^ (r >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            r = (r ^ (r >> 27)).wrapping_mul(0x94D049BB133111EB);
            r ^ (r >> 31)
        };
        Rng {
            s: [next(), next(), next(), next()],
        }
    }

    fn next_u64(&mut self) -> u64 {
        let result = rotl(self.s[1].wrapping_mul(5), 7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = rotl(self.s[3], 45);
        result
    }

    /// Uniform in [0,1).
    pub fn f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform in [0,1).
    pub fn f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 * (1.0 / (1u64 << 24) as f32)
    }

    /// Uniform in [lo, hi).
    pub fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.f64()
    }

    /// Uniform integer in [0, n).
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n.max(1) as u64) as usize
    }

    /// Standard normal (Box-Muller).
    pub fn gauss(&mut self, mean: f64, stddev: f64) -> f64 {
        let u1 = (1.0 - self.f64()).max(1e-300);
        let u2 = self.f64();
        mean + stddev * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Bernoulli(p).
    pub fn bernoulli(&mut self, p: f64) -> bool {
        self.f64() < p
    }

    /// Fisher–Yates shuffle.
    pub fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            v.swap(i, self.below(i + 1));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..1000 {
            assert_eq!(a.f64(), b.f64());
        }
    }

    #[test]
    fn gauss_moments() {
        let mut r = Rng::new(7);
        let mut sum = 0.0;
        let mut sq = 0.0;
        let n = 200_000;
        for _ in 0..n {
            let x = r.gauss(3.0, 2.0);
            sum += x;
            sq += x * x;
        }
        let mean = sum / n as f64;
        let var = sq / n as f64 - mean * mean;
        assert!((mean - 3.0).abs() < 0.02);
        assert!((var - 4.0).abs() < 0.1);
    }
}
