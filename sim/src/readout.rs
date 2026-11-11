//! Readouts, in Rust only (no Python):
//!
//! * `DeltaReadout` — the original's `LSMReadOutNeuron`: a sigmoid unit over
//!   exponentially filtered spike traces, trained online with the delta rule.
//!   Reproduces the original's quirk: the "input" to the weight update is the
//!   weighted trace `w_i * f_i` (its `active_potential`), not the bare `f_i`.
//! * `fit_ridge` — closed-form ridge regression readout, the baseline.
//! * metrics: accuracy, AUC.

/// Original's online delta-rule sigmoid readout.
#[derive(Clone, Debug)]
pub struct DeltaReadout {
    pub weights: Vec<f32>,
    pub bias: f32,
    pub bias_weight: f32,
}

impl DeltaReadout {
    pub fn new(n_units: usize) -> Self {
        DeltaReadout {
            weights: vec![0.0; n_units],
            bias: 1.0,
            bias_weight: 1.0,
        }
    }

    pub fn init_weights_random(&mut self, lo: f32, hi: f32, rng: &mut net_format::Rng) {
        for w in &mut self.weights {
            *w = rng.uniform(lo as f64, hi as f64) as f32;
        }
        self.bias_weight = rng.uniform(lo as f64, hi as f64) as f32;
    }

    /// Membrane potential = Σ w_i·f_i + bias·bias_weight.
    pub fn membrane(&self, trace: &[f32]) -> f32 {
        let mut m = self.bias * self.bias_weight;
        for (w, f) in self.weights.iter().zip(trace) {
            m += w * f;
        }
        m
    }

    pub fn output(&self, trace: &[f32]) -> f32 {
        sigmoid(self.membrane(trace))
    }

    /// One delta-rule step. `learn(target, rate)` in the original.
    /// d = (target - output) * σ'(membrane); w_i += rate·d·(w_i·f_i).
    pub fn learn(&mut self, trace: &[f32], target: f32, rate: f32) {
        let m = self.membrane(trace);
        let d = (target - self.output(trace)) * sigmoid_deriv(m);
        for (w, f) in self.weights.iter_mut().zip(trace) {
            *w += rate * d * (*w * f);
        }
        self.bias_weight += rate * d * self.bias;
    }
}

/// Proper delta-rule variant: input is the bare trace f_i, not w_i·f_i.
/// (Correction of the original's quirk; kept separate for comparison.)
pub fn learn_proper(
    w: &mut [f32],
    bias_w: &mut f32,
    bias: f32,
    trace: &[f32],
    target: f32,
    rate: f32,
) {
    let mut m = bias * *bias_w;
    for (wi, f) in w.iter().zip(trace) {
        m += wi * f;
    }
    let d = (target - sigmoid(m)) * sigmoid_deriv(m);
    for (wi, f) in w.iter_mut().zip(trace) {
        *wi += rate * d * f;
    }
    *bias_w += rate * d * bias;
}

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn sigmoid_deriv(x: f32) -> f32 {
    let s = sigmoid(x);
    s * (1.0 - s)
}

/// Closed-form ridge regression readout.
///
/// `x`: T×N design matrix (row-major, N = n_features).
/// `y`: T targets (e.g. 0/1 or ±1).
/// Returns (weights incl. trailing bias weight, rmse). Solves
/// (XᵀX + λI) w = Xᵀ y on the N×N normal-equation system by Cholesky.
/// A constant-1 feature column is appended for the bias.
pub fn fit_ridge(x: &[f32], n_rows: usize, n_features: usize, y: &[f32], lambda: f32) -> Vec<f64> {
    let nf = n_features + 1; // + bias column
    let mut a = vec![0.0f64; nf * nf];
    let mut b = vec![0.0f64; nf];
    for r in 0..n_rows {
        let row = &x[r * n_features..(r + 1) * n_features];
        for i in 0..n_features {
            let xi = row[i] as f64;
            b[i] += xi * y[r] as f64;
            for j in i..n_features {
                a[i * nf + j] += xi * row[j] as f64;
            }
        }
        b[n_features] += y[r] as f64;
        for j in 0..n_features {
            // bias cross-terms in the UPPER triangle so symmetrize fills them in
            a[j * nf + n_features] += row[j] as f64;
        }
        a[n_features * nf + n_features] += 1.0;
    }
    // symmetrize, add ridge (not to the bias diagonal)
    for i in 0..nf {
        for j in 0..i {
            a[i * nf + j] = a[j * nf + i];
        }
        a[i * nf + i] += lambda as f64;
    }
    a[n_features * nf + n_features] -= lambda as f64; // don't regularize bias
    match cholesky_solve(&a, &b, nf) {
        Some(w) => w,
        // fall back to extra regularization if not PD
        None => {
            for i in 0..nf {
                a[i * nf + i] += 1e-3 + lambda as f64;
            }
            cholesky_solve(&a, &b, nf).unwrap_or_else(|| vec![0.0; nf])
        }
    }
}

/// Cholesky solve A w = b for symmetric positive definite A (row-major n×n).
pub fn cholesky_solve(a: &[f64], b: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i * n + j];
            for k in 0..j {
                sum -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if sum <= 0.0 {
                    return None;
                }
                l[i * n + j] = sum.sqrt();
            } else {
                l[i * n + j] = sum / l[j * n + j];
            }
        }
    }
    // forward L z = b
    let mut z = vec![0.0f64; n];
    for i in 0..n {
        let mut sum = b[i];
        for k in 0..i {
            sum -= l[i * n + k] * z[k];
        }
        z[i] = sum / l[i * n + i];
    }
    // back L^T w = z
    let mut w = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut sum = z[i];
        for k in i + 1..n {
            sum -= l[k * n + i] * w[k];
        }
        w[i] = sum / l[i * n + i];
    }
    Some(w)
}

/// Apply a readout `w` (from `fit_ridge`) to one feature row; last weight is bias.
pub fn ridge_output(w: &[f64], features: &[f32]) -> f64 {
    let n = features.len();
    debug_assert_eq!(w.len(), n + 1);
    let mut s = w[n];
    for i in 0..n {
        s += w[i] * features[i] as f64;
    }
    s
}

/// Accuracy at a 0.5 threshold on scores (labels are 0/1).
pub fn accuracy(scores: &[f64], labels: &[u8]) -> f64 {
    if labels.is_empty() {
        return 0.0;
    }
    let correct = scores
        .iter()
        .zip(labels)
        .filter(|(&s, &l)| (s >= 0.5) == (l == 1))
        .count();
    correct as f64 / labels.len() as f64
}

/// Mann–Whitney AUC.
pub fn auc(scores: &[f64], labels: &[u8]) -> f64 {
    let mut pos: Vec<f64> = Vec::new();
    let mut neg: Vec<f64> = Vec::new();
    for (s, l) in scores.iter().zip(labels) {
        if *l == 1 {
            pos.push(*s);
        } else {
            neg.push(*s);
        }
    }
    if pos.is_empty() || neg.is_empty() {
        return f64::NAN;
    }
    let mut wins = 0.0;
    for &p in &pos {
        for &n in &neg {
            wins += if p > n {
                1.0
            } else if p == n {
                0.5
            } else {
                0.0
            };
        }
    }
    wins / (pos.len() * neg.len()) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ridge_fits_linear() {
        // y = 2*x0 - x1 + 0.5 exactly, 6 samples 2 features
        let x: Vec<f32> = vec![
            0.0, 0.0, //
            1.0, 0.0, //
            0.0, 1.0, //
            1.0, 1.0, //
            2.0, 1.0, //
            -1.0, 2.0,
        ];
        let y: Vec<f32> = x.chunks(2).map(|r| 2.0 * r[0] - r[1] + 0.5).collect();
        let w = fit_ridge(&x, 6, 2, &y, 1e-9);
        assert!((w[0] - 2.0).abs() < 1e-3, "w0={}", w[0]);
        assert!((w[1] + 1.0).abs() < 1e-3, "w1={}", w[1]);
        assert!((w[2] - 0.5).abs() < 1e-3, "bias={}", w[2]);
    }

    #[test]
    fn auc_perfect_and_chance() {
        let labels = vec![0u8, 0, 1, 1];
        let good = vec![0.1, 0.2, 0.8, 0.9];
        assert_eq!(auc(&good, &labels), 1.0);
        let flat = vec![0.5; 4];
        assert_eq!(auc(&flat, &labels), 0.5);
        assert_eq!(accuracy(&good, &labels), 1.0);
    }

    #[test]
    fn delta_learns() {
        // separable: unit0 on -> 1, off -> 0
        let mut ro = DeltaReadout::new(2);
        let mut rng = net_format::Rng::new(1);
        ro.init_weights_random(-1.0, 1.0, &mut rng);
        for _ in 0..200 {
            ro.learn(&[1.0, 0.0], 1.0, 0.1);
            ro.learn(&[0.0, 1.0], 0.0, 0.1);
        }
        assert!(ro.output(&[1.0, 0.0]) > 0.5);
        assert!(ro.output(&[0.0, 1.0]) < 0.5);
    }
}
