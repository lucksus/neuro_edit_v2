//! `SimCtl::connect` over a real `server` process's WebSocket: the same
//! `Shared` read model hydrates and folds deltas as the local transport.
//! Also covers the remote Drop path — a hang here means the socket
//! shutdown regressed.

use editor::simctl::SimCtl;
use net_format::Network;
use proto::ClientMsg;
use sim::{serve::ServeOpts, BackendKind, SimConfig};
use std::net::TcpListener;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::{Duration, Instant};

fn wait_for<T>(timeout: Duration, mut f: impl FnMut() -> Option<T>) -> T {
    let t0 = Instant::now();
    loop {
        if let Some(v) = f() {
            return v;
        }
        assert!(
            t0.elapsed() < timeout,
            "timed out waiting for remote worker"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
}

#[test]
fn remote_ctl_hydrates_and_folds() {
    let mut net = Network::new();
    net.add_neuron([0.0; 3], net_format::NeuronKind::Excitatory);
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    let srv = server::Server::start(
        net,
        ServeOpts {
            backend: BackendKind::Cpu,
            seed: 1,
            cfg: SimConfig::default(),
            data_dir: None,
            abort: Arc::new(AtomicBool::new(false)),
        },
    );
    srv.spawn_acceptor(listener);

    let ctl = SimCtl::connect(&addr).unwrap();

    // hydration: the server pushes Geo + Status on Hello
    wait_for(Duration::from_secs(10), || {
        let g = ctl.shared.read().ok()?.geo.clone()?;
        (g.pos.len() == 1).then_some(g)
    });

    // an edit over the wire lands as a versioned fold on Shared
    ctl.send(ClientMsg::Eval {
        code: r#"net.add_neuron(1.0, 2.0, 3.0);"#.to_string(),
    });
    wait_for(Duration::from_secs(10), || {
        let g = ctl.shared.read().ok()?.geo.clone()?;
        (g.pos.len() == 2).then_some(())
    });

    drop(ctl); // must not hang: socket shutdown unblocks the reader thread
    srv.shutdown();
}
