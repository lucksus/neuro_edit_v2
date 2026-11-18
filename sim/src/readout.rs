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
    /// false = the original's update (input is w_i·f_i — vanishes at small
    /// weights); true = the corrected delta rule (input is f_i).
    pub proper: bool,
}

impl DeltaReadout {
    pub fn new(n_units: usize) -> Self {
        DeltaReadout {
            weights: vec![0.0; n_units],
            bias: 1.0,
            bias_weight: 1.0,
            proper: false,
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
    /// With `proper` set, the weight sees the bare trace f_i instead.
    pub fn learn(&mut self, trace: &[f32], target: f32, rate: f32) {
        let m = self.membrane(trace);
        let d = (target - self.output(trace)) * sigmoid_deriv(m);
        for (w, f) in self.weights.iter_mut().zip(trace) {
            *w += rate * d * if self.proper { *f } else { *w * f };
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

/// Multi-class ridge (one-hot targets, argmax at inference) solved in the
/// DUAL: with T rows ≪ F features the T×T Gram system replaces the F×F
/// normal equations (`fit_ridge` on 8000 features is an 8000² Cholesky).
/// Columns and targets are centred on their means so the bias stays
/// unregularized — the same minimizer as primal ridge with a free bias.
/// Returns K weight vectors, each `F + 1` long with the bias trailing
/// (`ridge_output` applies one).
pub fn fit_ridge_multi(
    x: &[f32],
    n_rows: usize,
    n_features: usize,
    labels: &[usize],
    k: usize,
    lambda: f64,
) -> Vec<Vec<f64>> {
    let (t, f) = (n_rows, n_features);
    let mut mu = vec![0.0f64; f];
    for r in 0..t {
        for (m, &v) in mu.iter_mut().zip(&x[r * f..(r + 1) * f]) {
            *m += v as f64;
        }
    }
    mu.iter_mut().for_each(|m| *m /= t as f64);
    let xc: Vec<f64> = (0..t * f).map(|i| x[i] as f64 - mu[i % f]).collect();
    let mut ybar = vec![0.0f64; k];
    for &l in labels {
        ybar[l] += 1.0 / t as f64;
    }
    // Gram G = Xc Xcᵀ + λI (4 accumulators per dot: vectorizes).
    let mut g = vec![0.0f64; t * t];
    for i in 0..t {
        let ri = &xc[i * f..(i + 1) * f];
        for j in 0..=i {
            let rj = &xc[j * f..(j + 1) * f];
            let mut acc = [0.0f64; 4];
            let (ci, ti) = ri.as_chunks::<4>();
            let (cj, tj) = rj.as_chunks::<4>();
            for (a, b) in ci.iter().zip(cj) {
                for q in 0..4 {
                    acc[q] += a[q] * b[q];
                }
            }
            let mut s: f64 = ti.iter().zip(tj).map(|(a, b)| a * b).sum();
            s += acc.iter().sum::<f64>();
            g[i * t + j] = s;
            g[j * t + i] = s;
        }
        g[i * t + i] += lambda;
    }
    (0..k)
        .map(|c| {
            let yc: Vec<f64> = labels
                .iter()
                .map(|&l| (l == c) as u8 as f64 - ybar[c])
                .collect();
            let a = cholesky_solve(&g, &yc, t).unwrap_or_else(|| vec![0.0; t]);
            let mut w = vec![0.0f64; f + 1];
            for r in 0..t {
                let ar = a[r];
                for (wv, &xv) in w[..f].iter_mut().zip(&xc[r * f..(r + 1) * f]) {
                    *wv += ar * xv;
                }
            }
            let dot: f64 = w[..f].iter().zip(&mu).map(|(a, b)| a * b).sum();
            w[f] = ybar[c] - dot;
            w
        })
        .collect()
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

    /// Dual multi-class ridge = primal ridge per class (free bias):
    /// one-vs-rest scores from `fit_ridge` and `fit_ridge_multi` agree.
    #[test]
    fn ridge_multi_dual_matches_primal() {
        let mut rng = net_format::Rng::new(3);
        let (t, f, k) = (40, 7, 3);
        let x: Vec<f32> = (0..t * f).map(|_| rng.uniform(-1.0, 1.0) as f32).collect();
        let labels: Vec<usize> = (0..t).map(|r| r % k).collect();
        let ws = fit_ridge_multi(&x, t, f, &labels, k, 0.7);
        for c in 0..k {
            let y: Vec<f32> = labels.iter().map(|&l| (l == c) as u8 as f32).collect();
            let wp = fit_ridge(&x, t, f, &y, 0.7);
            for r in 0..t {
                let row = &x[r * f..(r + 1) * f];
                let (a, b) = (ridge_output(&wp, row), ridge_output(&ws[c], row));
                assert!(
                    (a - b).abs() < 1e-4,
                    "class {c} row {r}: primal {a} dual {b}"
                );
            }
        }
    }
}
