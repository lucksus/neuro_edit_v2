//! Non-spiking multilayer-perceptron readout, in Rust only.
//!
//! One hidden tanh layer, linear output, trained by SGD on reservoir
//! feature rows (rates or filtered traces). Deterministic for a seed:
//! init and minibatch order come from `net_format::Rng`. Training
//! holds out a validation tail and keeps the best-val weights (early
//! stopping). Features are standardized inside the fit; `mu`/`sd`
//! travel with the trained net in `MlpDef` so `mlp_output` applies the
//! same transform at inference.

use net_format::MlpDef;
use net_format::Rng;

/// Fit hyperparameters. `Default` is the sane starting point for the
/// examples; every field is reachable from script args.
#[derive(Clone, Debug)]
pub struct MlpParams {
    /// Hidden units.
    pub hidden: usize,
    /// Max training epochs (full passes over the train split).
    pub epochs: usize,
    /// SGD learning rate.
    pub lr: f32,
    /// L2 weight-decay coefficient (not applied to biases).
    pub wd: f32,
    /// Fraction of rows held out for early stopping (tail of the data).
    pub val_frac: f32,
    /// Epochs without val improvement before stopping.
    pub patience: usize,
    /// RNG seed — fixes init and row order.
    pub seed: u64,
}

impl Default for MlpParams {
    fn default() -> Self {
        MlpParams {
            hidden: 32,
            epochs: 300,
            lr: 0.02,
            wd: 1e-4,
            val_frac: 0.2,
            patience: 25,
            seed: 1,
        }
    }
}

#[inline]
fn tanh_act(x: f32) -> f32 {
    x.tanh()
}

/// Per-row forward pass on already-standardized features. Returns the
/// linear output score.
fn forward(sizes: &[u32], w: &[Vec<f32>], x: &[f32], acts: &mut Vec<Vec<f32>>) -> f32 {
    acts.clear();
    acts.push(x.to_vec());
    for l in 1..sizes.len() {
        let (n_in, n_out) = (sizes[l - 1] as usize, sizes[l] as usize);
        let wl = &w[l - 1];
        let prev = &acts[l - 1];
        let last = l == sizes.len() - 1;
        let mut a = Vec::with_capacity(n_out);
        for o in 0..n_out {
            let base = o * (n_in + 1);
            let mut s = wl[base + n_in]; // bias column
            for (i, &v) in prev.iter().enumerate() {
                s += wl[base + i] * v;
            }
            a.push(if last { s } else { tanh_act(s) });
        }
        acts.push(a);
    }
    acts.last().unwrap()[0]
}

/// Score one raw (unstandardized) feature row.
pub fn mlp_output(m: &MlpDef, features: &[f32]) -> f32 {
    let n_in = m.sizes[0] as usize;
    debug_assert_eq!(features.len(), n_in);
    let mut x = vec![0.0f32; n_in];
    for i in 0..n_in {
        x[i] = (features[i] - m.mu[i]) / m.sd[i];
    }
    let mut acts = Vec::new();
    forward(&m.sizes, &m.w, &x, &mut acts)
}

/// Train an MLP on `x` (n_rows × n_feat, row-major) → `y`.
///
/// Layout: `sizes = [n_feat, hidden, 1]`; each layer's weight block is
/// `n_out * (n_in + 1)` row-major floats with a trailing bias column.
/// Deterministic for `p.seed`. Returns the trained `MlpDef` ready to
/// store on a `Readout`.
pub fn fit_mlp(x: &[f32], n_rows: usize, n_feat: usize, y: &[f32], p: &MlpParams) -> MlpDef {
    assert_eq!(x.len(), n_rows * n_feat, "fit_mlp: x size mismatch");
    assert_eq!(y.len(), n_rows, "fit_mlp: y size mismatch");
    assert!(n_rows >= 4, "fit_mlp: need at least 4 rows");

    // Column standardization over ALL rows — matches what the script
    // would see at inference time (per-example feature distributions).
    let mut mu = vec![0.0f32; n_feat];
    for r in 0..n_rows {
        for j in 0..n_feat {
            mu[j] += x[r * n_feat + j];
        }
    }
    for m in mu.iter_mut() {
        *m /= n_rows as f32;
    }
    let mut sd = vec![0.0f32; n_feat];
    for r in 0..n_rows {
        for j in 0..n_feat {
            let d = x[r * n_feat + j] - mu[j];
            sd[j] += d * d;
        }
    }
    for s in sd.iter_mut() {
        *s = (*s / n_rows as f32).sqrt().max(1e-6);
    }
    let mut xs = vec![0.0f32; x.len()];
    for r in 0..n_rows {
        for j in 0..n_feat {
            xs[r * n_feat + j] = (x[r * n_feat + j] - mu[j]) / sd[j];
        }
    }

    // Tail validation split for early stopping.
    let n_val = ((n_rows as f32) * p.val_frac) as usize;
    let n_val = n_val.clamp(1, n_rows - 2);
    let n_tr = n_rows - n_val;

    // Deterministic init: He-ish scale per layer from the seeded Rng.
    let mut rng = Rng::new(p.seed);
    let sizes = vec![n_feat as u32, p.hidden as u32, 1];
    let mut w: Vec<Vec<f32>> = Vec::new();
    for l in 1..sizes.len() {
        let (n_in, n_out) = (sizes[l - 1] as usize, sizes[l] as usize);
        let scale = (2.0 / n_in as f32).sqrt();
        let mut wl = vec![0.0f32; n_out * (n_in + 1)];
        for o in 0..n_out {
            for i in 0..n_in {
                wl[o * (n_in + 1) + i] = rng.uniform(-scale as f64, scale as f64) as f32;
            }
        }
        w.push(wl);
    }

    // Row visit order, shuffled once per epoch from the same Rng.
    let mut order: Vec<usize> = (0..n_tr).collect();
    let mut grad = vec![0.0f32; n_feat + 2]; // scratch (hidden delta ≤ hidden+out)
    let mut acts: Vec<Vec<f32>> = Vec::new();
    let mut deltas: Vec<Vec<f32>> = Vec::new();
    let mut best_w: Option<Vec<Vec<f32>>> = None;
    let mut best_val = f32::INFINITY;
    let mut stale = 0usize;

    for _epoch in 0..p.epochs {
        rng.shuffle(&mut order);
        for &r in &order {
            let row = &xs[r * n_feat..(r + 1) * n_feat];
            let out = forward(&sizes, &w, row, &mut acts);
            // dL/dout = 2*(out - y); absorb constants into lr.
            let err = out - y[r];
            deltas.clear();
            deltas.resize_with(sizes.len(), Vec::new);
            deltas[sizes.len() - 1] = vec![err];
            // Backprop into hidden layer l (only one today, but the loop
            // is shape-correct for deeper stacks): delta_i =
            // tanh'(a_i) * Σ_o w[l][o,i] * delta_o.
            for l in (1..sizes.len() - 1).rev() {
                let n_cur = sizes[l] as usize;
                let wn = &w[l]; // sizes[l] → sizes[l+1], stride n_cur+1
                let dn = &deltas[l + 1];
                grad.clear();
                grad.resize(n_cur, 0.0);
                for i in 0..n_cur {
                    let a = acts[l][i];
                    let dt = 1.0 - a * a; // tanh'
                    let mut s = 0.0;
                    for o in 0..dn.len() {
                        s += wn[o * (n_cur + 1) + i] * dn[o];
                    }
                    grad[i] = s * dt;
                }
                deltas[l] = grad.clone();
            }
            // Gradient step on every layer.
            for l in 1..sizes.len() {
                let (n_in, n_out) = (sizes[l - 1] as usize, sizes[l] as usize);
                for o in 0..n_out {
                    let base = o * (n_in + 1);
                    for i in 0..n_in {
                        let g = deltas[l][o] * acts[l - 1][i] + p.wd * w[l - 1][base + i];
                        w[l - 1][base + i] -= p.lr * g;
                    }
                    w[l - 1][base + n_in] -= p.lr * deltas[l][o]; // bias: no wd
                }
            }
        }
        // Val MSE → early stopping bookkeeping.
        let mut se = 0.0f32;
        for r in n_tr..n_rows {
            let row = &xs[r * n_feat..(r + 1) * n_feat];
            let e = forward(&sizes, &w, row, &mut acts) - y[r];
            se += e * e;
        }
        let val = se / n_val as f32;
        if val < best_val - 1e-7 {
            best_val = val;
            best_w = Some(w.clone());
            stale = 0;
        } else {
            stale += 1;
            if stale >= p.patience {
                break;
            }
        }
    }
    if let Some(bw) = best_w {
        w = bw;
    }
    MlpDef { sizes, w, mu, sd }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::readout::{auc, fit_ridge, ridge_output};

    fn mse(m: &MlpDef, x: &[f32], n_feat: usize, y: &[f32]) -> f32 {
        let mut se = 0.0;
        for r in 0..y.len() {
            let o = mlp_output(m, &x[r * n_feat..(r + 1) * n_feat]);
            se += (o - y[r]).powi(2);
        }
        se / y.len() as f32
    }

    /// XOR must be learnable — a linear readout cannot do this, so it is
    /// the canary that the hidden layer actually trains.
    /// Sabotage: `hidden=0` or removing tanh would fail.
    #[test]
    fn learns_xor() {
        let x: Vec<f32> = [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.1, 0.9],
            [0.9, 0.1],
            [0.1, 0.1],
            [0.9, 0.9],
        ]
        .concat();
        let y = vec![0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0];
        // repeat the batch so the val tail has all four cases
        let mut xa = x.clone();
        xa.extend_from_slice(&x);
        let mut ya = y.clone();
        ya.extend_from_slice(&y);
        let m = fit_mlp(
            &xa,
            ya.len(),
            2,
            &ya,
            &MlpParams {
                hidden: 8,
                epochs: 400,
                lr: 0.1,
                seed: 7,
                ..Default::default()
            },
        );
        assert!(
            mse(&m, &xa, 2, &ya) < 0.05,
            "xor mse {}",
            mse(&m, &xa, 2, &ya)
        );
        // classes separated
        let neg = mlp_output(&m, &[0.05, 0.05]);
        let pos = mlp_output(&m, &[0.95, 0.05]);
        assert!(pos - neg > 0.6, "xor separation {} vs {}", neg, pos);
    }

    /// On a linear task the MLP should be competitive with ridge —
    /// proves the optimizer is sane, not just expressive.
    /// Sabotage: lr=0 (no learning) fails.
    #[test]
    fn matches_ridge_on_linear_task() {
        let mut rng = Rng::new(11);
        let n = 300;
        let nf = 6;
        let w_true = [0.8, -0.5, 0.3, 0.0, 1.2, -0.9];
        let mut x = vec![0.0f32; n * nf];
        let mut y = vec![0.0f32; n];
        for r in 0..n {
            let mut s = 0.3f32;
            for j in 0..nf {
                let v = rng.uniform(-1.0, 1.0) as f32;
                x[r * nf + j] = v;
                s += v * w_true[j];
            }
            y[r] = if s + rng.gauss(0.0, 0.15) as f32 > 0.5 {
                1.0
            } else {
                0.0
            };
        }
        let rw = fit_ridge(&x, n, nf, &y, 1.0);
        let m = fit_mlp(
            &x,
            n,
            nf,
            &y,
            &MlpParams {
                hidden: 8,
                epochs: 150,
                seed: 3,
                ..Default::default()
            },
        );
        let r_scores: Vec<f64> = (0..n)
            .map(|r| ridge_output(&rw, &x[r * nf..(r + 1) * nf]))
            .collect();
        let m_scores: Vec<f64> = (0..n)
            .map(|r| mlp_output(&m, &x[r * nf..(r + 1) * nf]) as f64)
            .collect();
        let yb: Vec<u8> = y.iter().map(|v| *v as u8).collect();
        let ra = auc(&r_scores, &yb);
        let ma = auc(&m_scores, &yb);
        assert!(
            (ma - ra).abs() < 0.05,
            "mlp auc {ma:.3} vs ridge auc {ra:.3}"
        );
    }

    /// Same seed → bit-identical weights; different seed → different.
    /// Sabotage: thread_rng or time-based init fails the second half.
    #[test]
    fn deterministic_for_seed() {
        let x: Vec<f32> = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]].concat();
        let y = vec![0.0, 1.0, 1.0, 0.0];
        let p = MlpParams {
            hidden: 4,
            epochs: 30,
            seed: 42,
            ..Default::default()
        };
        let a = fit_mlp(&x, 4, 2, &y, &p);
        let b = fit_mlp(&x, 4, 2, &y, &p);
        assert_eq!(a.w, b.w, "same seed must give same weights");
        let c = fit_mlp(&x, 4, 2, &y, &MlpParams { seed: 43, ..p });
        assert_ne!(a.w, c.w, "different seed should differ");
    }

    /// `mlp_output` on unstandardized input must use mu/sd — check the
    /// transform path directly.
    #[test]
    fn output_applies_standardization() {
        let m = MlpDef {
            sizes: vec![2, 2, 1],
            w: vec![
                vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0], // 2×(2+1): identity h
                vec![1.0, 1.0, 0.0],                // 1×(2+1)
            ],
            mu: vec![10.0, 10.0],
            sd: vec![2.0, 2.0],
        };
        // x=(12,10) → z=(1,0) → h=(tanh1, tanh0) → out = tanh(1)+0
        let o = mlp_output(&m, &[12.0, 10.0]);
        assert!((o - 1.0f32.tanh()).abs() < 1e-5, "{o}");
    }
}
