//! Grid activity planes: host-side lattice math for the plane view —
//! texel geometry, the 3D -> 2D reduction (slice or projection), the
//! texel->member pick map, and the zoom LOD that chooses cubes vs plane.
//! Kept renderer-free so the logic is unit-testable; the GPU pieces are
//! in `gpu.rs`/`shader.wgsl`.

use crate::snap::GridSnap;

/// Toolbar toggle: how grid groups render.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GridView {
    /// Per-grid zoom LOD — the plane takes over once texels shrink below
    /// `LOD_PX` on screen; zooming in brings the cubes back for editing.
    Auto,
    /// Always instanced cubes (original renderer).
    Cubes,
    /// Always activity planes where a group has grid metadata.
    Planes,
}

/// How a 3D grid's collapsed lattice dim is displayed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Proj {
    /// Single slice at this depth index along `axis`.
    Slice(u32),
    /// Per-texel maximum along `axis` (brightest member wins).
    Max,
    /// Per-texel mean along `axis`.
    Mean,
}

/// Per-grid display setting. A 2D grid (`dims[2] == 1`) ignores it — the
/// single slice is always shown.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PlaneCfg {
    /// Collapsed lattice dim (0, 1 or 2).
    pub axis: u8,
    pub proj: Proj,
}

impl Default for PlaneCfg {
    fn default() -> Self {
        Self {
            axis: 2,
            proj: Proj::Max,
        }
    }
}

/// Which lattice dims the displayed (u, v) axes map to: the two dims that
/// are not collapsed, in order. 2D grids always show dims (0, 1).
pub fn plane_dims(gs: &GridSnap, cfg: &PlaneCfg) -> (usize, usize) {
    if gs.dims[2] <= 1 {
        return (0, 1);
    }
    let axis = cfg.axis.min(2) as usize;
    let mut it = (0..3).filter(move |&k| k != axis);
    (it.next().unwrap(), it.next().unwrap())
}

/// The collapsed lattice dim actually in effect (forced to 2 for 2D grids).
pub fn collapse_axis(gs: &GridSnap, cfg: &PlaneCfg) -> usize {
    if gs.dims[2] <= 1 {
        2
    } else {
        cfg.axis.min(2) as usize
    }
}

/// Texel index inside the member/trace buffers: lattice position of
/// member-list entry k is `k = (i2 * dims0 + i0) * dims1 + i1`.
pub fn lattice_index(gs: &GridSnap, l: [usize; 3]) -> usize {
    let (d0, d1) = (gs.dims[0] as usize, gs.dims[1] as usize);
    (l[2] * d0 + l[0]) * d1 + l[1]
}

/// Reduce a grid's per-texel trace (w*h*d f32, texel order) to the
/// displayed 2D image plus the texel->neuron map the pick shader uses.
///
/// Returns `(w, h, pixels, member_ids)`. For `Proj::Slice` the pick map
/// points at that slice's members; for projections it points at the
/// argmax member along the collapsed axis (the texel you actually see).
pub fn reduce(gs: &GridSnap, trace: &[f32], cfg: &PlaneCfg) -> (u32, u32, Vec<f32>, Vec<u32>) {
    let (ua, va) = plane_dims(gs, cfg);
    let axis = collapse_axis(gs, cfg);
    let (w, h) = (gs.dims[ua] as usize, gs.dims[va] as usize);
    let depth = gs.dims[axis] as usize;
    let tr = |k: usize| trace.get(k).copied().unwrap_or(0.0);
    let mem = |k: usize| gs.members.get(k).copied().unwrap_or(0);

    let mut pix = vec![0f32; w * h];
    let mut pick = vec![0u32; w * h];
    for v in 0..h {
        for u in 0..w {
            let mut l = [0usize; 3];
            l[ua] = u;
            l[va] = v;
            let out = v * w + u;
            match cfg.proj {
                Proj::Slice(k) if depth > 1 => {
                    l[axis] = (k as usize).min(depth - 1);
                    let ki = lattice_index(gs, l);
                    pix[out] = tr(ki);
                    pick[out] = mem(ki);
                }
                _ => {
                    // Max / Mean / 2D (depth == 1 collapses to itself).
                    let mut best = 0usize;
                    let mut best_v = f32::MIN;
                    let mut sum = 0f32;
                    for z in 0..depth {
                        l[axis] = z;
                        let ki = lattice_index(gs, l);
                        let t = tr(ki);
                        sum += t;
                        if t > best_v {
                            best_v = t;
                            best = ki;
                        }
                    }
                    pix[out] = match cfg.proj {
                        Proj::Mean if depth > 1 => sum / depth as f32,
                        _ => best_v.max(0.0),
                    };
                    pick[out] = mem(best);
                }
            }
        }
    }
    (w as u32, h as u32, pix, pick)
}

/// Quad corners for the plane, matching `vs_plane`: the rectangle is
/// `origin + u*du + v*dv` for u in 0..w, v in 0..h, so each texel covers
/// its own -0.5..+0.5 cell. A projected plane sits at the middle of the
/// collapsed axis; a slice sits at its depth.
pub fn quad(gs: &GridSnap, cfg: &PlaneCfg) -> ([f32; 4], [f32; 4], [f32; 4]) {
    let (ua, va) = plane_dims(gs, cfg);
    let axis = collapse_axis(gs, cfg);
    let axis_pos = match cfg.proj {
        Proj::Slice(k) if gs.dims[axis] > 1 => k.min(gs.dims[axis] - 1) as f32,
        _ => (gs.dims[axis].saturating_sub(1)) as f32 * 0.5,
    };
    let mut o = gs.origin;
    o[gs.axes[axis] as usize] += gs.spacing * axis_pos;
    // texel-centre position -> cell corner: shift -0.5 texel on both axes
    o[gs.axes[ua] as usize] -= gs.spacing * 0.5;
    o[gs.axes[va] as usize] -= gs.spacing * 0.5;
    let mut du = [0f32; 4];
    let mut dv = [0f32; 4];
    du[gs.axes[ua] as usize] = gs.spacing;
    dv[gs.axes[va] as usize] = gs.spacing;
    ([o[0], o[1], o[2], 0.0], du, dv)
}

/// World-space centre of the lattice — used for the LOD distance and the
/// "frame group" camera move.
pub fn center(gs: &GridSnap) -> [f32; 3] {
    let mut c = gs.origin;
    for k in 0..3 {
        c[gs.axes[k] as usize] += gs.spacing * (gs.dims[k].saturating_sub(1)) as f32 * 0.5;
    }
    c
}

/// Apparent texel size in screen pixels — the auto LOD switches a group
/// to its plane once this drops below `LOD_PX`.
pub fn texel_px(gs: &GridSnap, eye_dist: f32, vp_h_px: f32) -> f32 {
    let px_per_unit = vp_h_px * 0.5 / (eye_dist.max(1.0) * 0.577_350_3);
    gs.spacing * px_per_unit
}

/// Auto-LOD threshold in pixels: below this a grid's cubes are too small
/// to read or edit, so the plane takes over.
pub const LOD_PX: f32 = 4.0;

/// In `Auto` mode a selection this small or smaller that touches a grid's
/// members flips that grid back to cubes, so the selected neurons stay
/// directly editable. Larger selections (e.g. box-selecting the whole
/// sheet) keep the plane.
pub const SEL_CUBE_MAX: usize = 256;

#[cfg(test)]
mod tests {
    use super::*;

    fn gs(dims: [u32; 3]) -> GridSnap {
        GridSnap {
            group: 0,
            dims,
            axes: [0, 1, 2],
            origin: [10.0, 20.0, 30.0],
            spacing: 4.0,
            members: (0..dims[0] * dims[1] * dims[2]).collect(),
        }
    }

    #[test]
    fn reduce_2d_reorders_lattice_to_raster() {
        let g = gs([3, 2, 1]);
        let tr: Vec<f32> = (0..6).map(|k| k as f32).collect();
        let (w, h, pix, pick) = reduce(&g, &tr, &PlaneCfg::default());
        assert_eq!((w, h), (3, 2));
        // Texture order is u-fastest (dim0); the lattice member order is
        // dim1-fastest, so the plane is the transpose of the trace.
        let want: Vec<f32> = (0..2)
            .flat_map(|v| {
                (0..3)
                    .map(|u| tr[lattice_index(&g, [u, v, 0])])
                    .collect::<Vec<_>>()
            })
            .collect();
        assert_eq!(pix, want);
        assert_eq!(pix, vec![0.0, 2.0, 4.0, 1.0, 3.0, 5.0]);
        let want_m: Vec<u32> = (0..2)
            .flat_map(|v| {
                (0..3)
                    .map(|u| g.members[lattice_index(&g, [u, v, 0])])
                    .collect::<Vec<_>>()
            })
            .collect();
        assert_eq!(pick, want_m);
    }

    #[test]
    fn slice_picks_its_members() {
        let g = gs([3, 2, 2]); // 12 texels, member == index
        let tr: Vec<f32> = (0..12).map(|k| k as f32).collect();
        let cfg = PlaneCfg {
            axis: 2,
            proj: Proj::Slice(1),
        };
        let (w, h, pix, pick) = reduce(&g, &tr, &cfg);
        assert_eq!((w, h), (3, 2));
        // lattice (u,v,1): idx = (1*3 + u)*2 + v = 6 + 2u + v
        assert_eq!(pix[0], tr[6]);
        assert_eq!(pick[1 * 3 + 2], g.members[lattice_index(&g, [2, 1, 1])]);
    }

    #[test]
    fn max_proj_picks_argmax() {
        let g = gs([2, 2, 3]);
        let mut tr = vec![0f32; 12];
        // make lattice (1,0,2) the brightest in column (u=1,v=0)
        tr[lattice_index(&g, [1, 0, 2])] = 9.0;
        let cfg = PlaneCfg {
            axis: 2,
            proj: Proj::Max,
        };
        let (w, h, pix, pick) = reduce(&g, &tr, &cfg);
        assert_eq!((w, h), (2, 2));
        assert_eq!(pix[1], 9.0);
        assert_eq!(pick[1], g.members[lattice_index(&g, [1, 0, 2])]);
    }

    #[test]
    fn mean_proj_averages() {
        let g = gs([1, 1, 4]);
        let tr = vec![0.0, 1.0, 0.0, 1.0];
        let cfg = PlaneCfg {
            axis: 2,
            proj: Proj::Mean,
        };
        let (_, _, pix, _) = reduce(&g, &tr, &cfg);
        assert_eq!(pix[0], 0.5);
    }

    #[test]
    fn quad_covers_texel_cells() {
        let g = gs([4, 3, 1]);
        let (o, du, dv) = quad(&g, &PlaneCfg::default());
        // axes [0,1,2], dims0 -> x, dims1 -> y, normal z at origin z=30
        assert_eq!(o, [10.0 - 2.0, 20.0 - 2.0, 30.0, 0.0]);
        assert_eq!(du, [4.0, 0.0, 0.0, 0.0]);
        assert_eq!(dv, [0.0, 4.0, 0.0, 0.0]);
    }

    #[test]
    fn quad_slice_sits_at_its_depth() {
        let g = gs([4, 3, 2]);
        let cfg = PlaneCfg {
            axis: 2,
            proj: Proj::Slice(1),
        };
        let (o, _, _) = quad(&g, &cfg);
        assert_eq!(o[2], 30.0 + 4.0); // z = origin + k*spacing
    }
}
