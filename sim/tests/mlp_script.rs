//! Stage 26: the deep-MLP script API — `layers`/`classes`/`act`/`opt`
//! options, `mlp_scores`/`mlp_predict`, and the readout layer config
//! (`set_readout_layers`, `readout_layers`, `train_readout_mlp`).

use net_format::Network;
use sim::script::Env;
use sim::BackendKind;

fn env() -> Env {
    Env::new(Network::new(), BackendKind::Cpu, 1).unwrap()
}

/// 4-class XOR-ish task data as a Rhai prelude: X (flat), Y, T=200, F=2.
const DATA: &str = r#"
seed(5);
let X = []; let Y = [];
for i in 0..200 {
    let a = rand(-1.0, 1.0); let b = rand(-1.0, 1.0);
    X.push(a); X.push(b);
    let xr = if (a > 0.0) != (b > 0.0) { 1 } else { 0 };
    let ring = if a * a + b * b > 0.5 { 1 } else { 0 };
    Y.push(xr * 2 + ring);
}
"#;

#[test]
fn layers_classes_scores_predict() {
    let mut e = env();
    let code = format!(
        "{DATA}
        let m = fit_mlp(X, 200, 2, Y, #{{layers: [12, 8], classes: 4, opt: \"adam\", lr: 0.02, epochs: 200, seed: 2}});
        let s = mlp_scores(m, [0.5, -0.5]);
        let tot = 0.0; for v in s {{ tot += v; }}
        let pred = []; for i in 0..200 {{ pred.push(mlp_predict(m, [X[2*i], X[2*i+1]])); }}
        [mlp_sizes(m), s.len(), tot, class_accuracy(pred, Y), mlp_act(m), mlp_classes(m)]"
    );
    let r = e.eval_str(&code).unwrap().cast::<rhai::Array>();
    let sizes: Vec<i64> = r[0]
        .clone()
        .cast::<rhai::Array>()
        .iter()
        .map(|d| d.as_int().unwrap())
        .collect();
    assert_eq!(sizes, vec![2, 12, 8, 4]);
    assert_eq!(r[1].as_int().unwrap(), 4);
    assert!((r[2].as_float().unwrap() - 1.0).abs() < 1e-5);
    assert!(r[3].as_float().unwrap() > 0.8, "train acc {}", r[3]);
    assert_eq!(r[4].clone().into_string().unwrap(), "tanh");
    assert_eq!(r[5].as_int().unwrap(), 4);
}

#[test]
fn option_errors() {
    let mut e = env();
    let both = format!("{DATA} fit_mlp(X, 200, 2, Y, #{{hidden: 4, layers: [4]}})");
    assert!(e.eval_str(&both).unwrap_err().contains("not both"));
    let bad_label = format!("{DATA} fit_mlp(X, 200, 2, Y, #{{classes: 3}})");
    assert!(e.eval_str(&bad_label).unwrap_err().contains("class index"));
    let bad_act = format!("{DATA} fit_mlp(X, 200, 2, Y, #{{act: \"sigmoid\"}})");
    assert!(e.eval_str(&bad_act).is_err());
    let zero = format!("{DATA} fit_mlp(X, 200, 2, Y, #{{layers: [8, 0]}})");
    assert!(e.eval_str(&zero).unwrap_err().contains("out of range"));
    // row of the wrong width is an error, not a panic
    let wide = format!(
        "{DATA} let m = fit_mlp(X, 200, 2, Y, #{{layers: [3], epochs: 2}}); mlp_scores(m, [1.0])"
    );
    assert!(e.eval_str(&wide).unwrap_err().contains("features"));
}

/// The readout carries a layer config; `train_readout_mlp` uses it,
/// opts override it, and the trained net is attached to the readout.
#[test]
fn readout_layer_config_drives_training() {
    let mut e = env();
    let _ = e.eval_str("net.add_neuron(0.0, 0.0, 0.0); net.add_neuron(1.0, 0.0, 0.0); net.add_readout(\"ro\", [0, 1]);")
        .unwrap();
    assert!(e.eval_str("net.readout_layers(\"ro\")").unwrap().is_unit());
    assert!(e
        .eval_str("net.readout_layers(\"nope\")")
        .unwrap()
        .is_unit());
    assert!(e.eval_str("net.set_readout_layers(\"nope\", [4])").is_err());
    let _ = e
        .eval_str("net.set_readout_layers(\"ro\", [7, 5, 3])")
        .unwrap();
    let l = e
        .eval_str("net.readout_layers(\"ro\")")
        .unwrap()
        .cast::<rhai::Array>();
    assert_eq!(
        l.iter().map(|d| d.as_int().unwrap()).collect::<Vec<_>>(),
        vec![7, 5, 3]
    );
    let code = format!(
        "{DATA} let m = net.train_readout_mlp(\"ro\", X, 200, 2, Y, #{{classes: 4, opt: \"adam\", epochs: 20}}); mlp_sizes(m)"
    );
    let _ = e.eval_str(&code).unwrap();
    let m = e.net.borrow().readout_mlp("ro").cloned().expect("attached");
    assert_eq!(m.sizes, vec![2, 7, 5, 3, 4]);
    // opts override the config
    let code = format!(
        "{DATA} net.train_readout_mlp(\"ro\", X, 200, 2, Y, #{{layers: [6], classes: 4, opt: \"adam\", epochs: 5}})"
    );
    let _ = e.eval_str(&code).unwrap();
    assert_eq!(
        e.net.borrow().readout_mlp("ro").unwrap().sizes,
        vec![2, 6, 4]
    );
    // the config survives (an empty array clears it)
    let _ = e.eval_str("net.set_readout_layers(\"ro\", [])").unwrap();
    assert!(e.eval_str("net.readout_layers(\"ro\")").unwrap().is_unit());
}
