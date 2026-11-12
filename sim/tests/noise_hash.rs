//! P2-9: `noise_hash` is splitmix64 over the (neuron, step, seed) tuple.
//! Locks the algorithm (a regression back to the linear-pre-mix version
//! breaks the golden values) and checks neighbouring (i, t, seed) inputs
//! produce uncorrelated fire decisions.

use sim::state::noise_hash;

/// Reference splitmix64 finaliser — independent of the implementation.
fn splitmix64(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x ^ (x >> 31)
}

#[test]
fn noise_hash_is_splitmix64() {
    for &(i, t, seed) in &[
        (0u32, 0u64, 0u64),
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (7, 4096, 42),
        (u32::MAX, u64::MAX / 2, 0xDEADBEEF),
    ] {
        let expect = (splitmix64(
            (i as u64)
                .wrapping_mul(0x9E3779B97F4A7C15)
                .wrapping_add(t.wrapping_mul(0xBF58476D1CE4E5B9))
                .wrapping_add(seed.wrapping_mul(0x94D049BB133111EB)),
        ) >> 32) as u32;
        assert_eq!(noise_hash(i, t, seed), expect, "i={i} t={t} seed={seed}");
    }
}

/// Fire decisions must look independent across (i, t, seed) neighbours.
/// Expected agreement for two p=0.1 streams: p² + (1−p)² = 0.82.
#[test]
fn noise_streams_are_uncorrelated() {
    let p = 0.1f32;
    let fires = |i: u32, t: u64, s: u64| (noise_hash(i, t, s) as f32) * 2.3283064e-10 < p;
    let mut n = 0u64;
    let mut agree_seed = 0u64;
    let mut agree_neuron = 0u64;
    let mut agree_step = 0u64;
    let mut total = 0u64;
    for i in 0..100u32 {
        for t in 0..500u64 {
            let a = fires(i, t, 7);
            if a == fires(i, t, 8) {
                agree_seed += 1;
            }
            if a == fires(i + 1, t, 7) {
                agree_neuron += 1;
            }
            if a == fires(i, t + 1, 7) {
                agree_step += 1;
            }
            total += a as u64;
            n += 1;
        }
    }
    for (name, agree) in [
        ("seed", agree_seed),
        ("neuron", agree_neuron),
        ("step", agree_step),
    ] {
        let r = agree as f64 / n as f64;
        assert!(
            (0.74..=0.90).contains(&r),
            "{name}-neighbour agreement should be ~0.82 (independent): {r:.3}"
        );
    }
    let rate = total as f64 / n as f64;
    assert!((0.07..=0.13).contains(&rate), "fire rate ~p: {rate:.3}");
}
