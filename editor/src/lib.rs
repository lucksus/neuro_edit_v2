//! Library surface for the neuro_edit v2 editor (modules shared with the binary
//! and integration tests).

pub mod app;
pub mod camera;
pub mod gpu;
pub mod icons;
pub mod plane;
pub mod scene;
#[cfg(feature = "native")]
pub mod shot;
pub mod simctl;
pub mod snap;
pub mod syntax;
pub mod view;
