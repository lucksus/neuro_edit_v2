//! Tiny remote control: `ctl <addr> <script-file|->` sends Eval and
//! prints console lines for a few seconds.
use proto::{ClientMsg, ServerMsg};
use std::time::Duration;

fn main() -> anyhow::Result<()> {
    let addr = std::env::args().nth(1).expect("addr");
    let code = std::env::args().nth(2).expect("script path or -");
    let code = if code == "-" {
        std::env::args().nth(3).expect("code")
    } else {
        std::fs::read_to_string(code)?
    };
    let mut c = server::ClientConn::connect(&addr)?;
    c.send(&ClientMsg::Hello {
        proto: proto::PROTO_VERSION,
    })?;
    c.send(&ClientMsg::Eval { code })?;
    let end = std::time::Instant::now() + Duration::from_secs(15);
    while std::time::Instant::now() < end {
        if let Ok(Some(ServerMsg::Console(s))) = c.recv_opt(Duration::from_millis(500)) {
            println!("{s}");
            if s == "done" {
                break;
            }
        }
    }
    Ok(())
}
