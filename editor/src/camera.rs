//! Orbit camera, matching the original's controls (right-drag orbit,
//! wheel zoom, WASD pan, double-click to re-target).

use glam::{Mat4, Vec3};

pub struct Camera {
    pub target: Vec3,
    pub yaw_deg: f32,
    pub pitch_deg: f32,
    pub dist: f32,
    /// Smoothed target (double-click ease).
    anim: Option<(Vec3, f32)>,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            target: Vec3::ZERO,
            yaw_deg: -35.0,
            pitch_deg: -20.0,
            dist: 700.0,
            anim: None,
        }
    }
}

impl Camera {
    pub fn eye(&self) -> Vec3 {
        let (ya, pa) = (self.yaw_deg.to_radians(), self.pitch_deg.to_radians());
        self.target
            + Vec3::new(
                self.dist * pa.cos() * ya.sin(),
                -self.dist * pa.sin(),
                self.dist * pa.cos() * ya.cos(),
            )
    }

    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.yaw_deg += dx * 0.25;
        self.pitch_deg = (self.pitch_deg + dy * 0.25).clamp(-89.0, 89.0);
    }

    pub fn zoom(&mut self, wheel: f32) {
        self.dist = (self.dist * (1.0 - wheel * 0.001)).clamp(70.0, 10_000.0);
    }

    /// Pan in the camera's right/up plane (WASD + QE), scaled by distance.
    pub fn pan(&mut self, right: f32, fwd: f32, up: f32, dt: f32) {
        let fwd_v = (self.target - self.eye()).normalize();
        let right_v = fwd_v.cross(Vec3::Y).normalize_or(Vec3::X);
        let s = self.dist * 0.6 * dt;
        self.target += right_v * (right * s) + fwd_v * (fwd * s) + Vec3::Y * (up * s);
        self.anim = None;
    }

    /// Double-click: glide the target toward `p`.
    pub fn focus(&mut self, p: Vec3) {
        self.anim = Some((p, 0.0));
    }

    pub fn tick(&mut self, dt: f32) {
        if let Some((goal, mut t)) = self.anim {
            t += dt * 3.0;
            let k = (t.min(1.0) * std::f32::consts::FRAC_PI_2).sin();
            let prev_goal = goal;
            self.target = self.target.lerp(prev_goal, k);
            if t >= 1.0 {
                self.target = goal;
                self.anim = None;
            } else {
                self.anim = Some((goal, t));
            }
        }
    }

    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        let proj = Mat4::perspective_rh(60f32.to_radians(), aspect, 1.0, 20_000.0);
        let view = Mat4::look_at_rh(self.eye(), self.target, Vec3::Y);
        proj * view
    }

    /// Centre on a bounding box (used after net load).
    pub fn frame(&mut self, min: Vec3, max: Vec3) {
        let c = (min + max) * 0.5;
        let r = (max - min).length().max(1.0);
        self.target = c;
        self.dist = (r * 0.9).clamp(70.0, 10_000.0);
        self.anim = None;
    }
}
