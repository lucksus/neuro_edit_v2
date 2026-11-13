//! Stage 23: per-group interaction/display modes — one control family,
//! not two toggles. A group's view IS its interaction mode:
//!
//! - `Whole` (default): members render; clicking any member selects the
//!   whole group and dragging moves every member by the same offset.
//! - `Inside`: members render; clicks/drags act on a single member.
//! - `Block`: the members collapse to a labelled block (hide-bit
//!   collapsed cubes + a screen-space glyph). A block is always
//!   whole-group — there are no members left to grab.
//!
//! The mode lives in the app (it's a view/interaction concern, not net
//! data), keyed by group name so it survives group-index reordering in
//! `Geo.groups`.

use std::collections::HashSet;

use glam::Vec3;

use crate::snap::GeoSnap;

use super::EditorApp;

/// Per-group interaction/display mode (see module docs).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GroupMode {
    /// Expanded + atomic: the group acts as one object.
    Whole,
    /// Expanded + per-member: clicks/edits affect single members.
    Inside,
    /// Collapsed block glyph — members hidden, always whole-group.
    Block,
}

/// Padding (world units) a group's bounding box grows by — shared by the
/// 3D box lines and the collapsed-block glyph so they agree on the
/// group's footprint.
pub(crate) const GROUP_PAD: f32 = 12.0;

impl EditorApp {
    /// The effective mode of group `name`: the per-group override when
    /// set, else the app-wide default (Whole).
    pub(crate) fn group_mode(&self, name: &str) -> GroupMode {
        self.group_modes
            .get(name)
            .copied()
            .unwrap_or(self.group_mode_default)
    }

    /// Does `name`'s mode treat the group as one object? Block counts —
    /// a collapsed group is always whole-group.
    fn atomic(&self, name: &str) -> bool {
        matches!(self.group_mode(name), GroupMode::Whole | GroupMode::Block)
    }

    /// The instances a click/drag on `inst` acts on. For a neuron in
    /// atomic-mode group(s) it's the union of those groups' members
    /// (overlapping groups unite — both boxes highlight); anything else
    /// is just `inst` itself. The picked instance is always included.
    ///
    /// Only neuron instances group — inputs/readouts pass through.
    pub(crate) fn group_expand(&self, inst: u32) -> HashSet<u32> {
        let mut set = HashSet::from([inst]);
        let Some(g) = &self.geo else { return set };
        for (name, members) in &g.groups {
            if self.atomic(name) && members.contains(&inst) {
                set.extend(members.iter().copied());
            }
        }
        set
    }

    /// Click-select `inst` — whole-group modes take the member's whole
    /// group (overlap unions too). `additive` is the Shift case: union
    /// into the existing selection instead of replacing it.
    pub(crate) fn click_select(&mut self, inst: u32, additive: bool) {
        if !additive {
            self.selection.clear();
        }
        self.selection.extend(self.group_expand(inst));
    }

    /// Box-select expansion: a box that catches any member of an atomic
    /// group catches the whole group. Runs to a fixpoint — overlapping
    /// groups chain (a shared member completes both).
    ///
    /// Field-disjoint form: callers can pass `&mut app.selection` while
    /// `app.geo`/`app.group_modes` are read through field borrows.
    pub(crate) fn expand_sel_groups_in(
        modes: &std::collections::HashMap<String, GroupMode>,
        default: GroupMode,
        g: &GeoSnap,
        sel: &mut HashSet<u32>,
    ) {
        let atomic = |name: &str| {
            matches!(
                modes.get(name).copied().unwrap_or(default),
                GroupMode::Whole | GroupMode::Block
            )
        };
        loop {
            let mut added = false;
            for (name, members) in &g.groups {
                if !atomic(name) {
                    continue;
                }
                if members.iter().any(|m| sel.contains(m))
                    && members.iter().any(|m| !sel.contains(m))
                {
                    sel.extend(members.iter().copied());
                    added = true;
                }
            }
            if !added {
                return;
            }
        }
    }

    /// Indices into `Geo.groups` whose effective mode is `mode`.
    pub(crate) fn group_ids_mode(&self, mode: GroupMode) -> Vec<usize> {
        let Some(g) = &self.geo else {
            return Vec::new();
        };
        g.groups
            .iter()
            .enumerate()
            .filter(|(_, (name, _))| self.group_mode(name) == mode)
            .map(|(i, _)| i)
            .collect()
    }

    /// Neuron ids hidden from direct picking because their group renders
    /// as a block — ray tests skip them (the block glyph is the handle).
    pub(crate) fn block_members(&self) -> HashSet<u32> {
        let mut out = HashSet::new();
        for gi in self.group_ids_mode(GroupMode::Block) {
            if let Some(g) = &self.geo {
                out.extend(g.groups[gi].1.iter().copied());
            }
        }
        out
    }

    /// Padded member bounding box — the block glyph's volume, or None
    /// for an empty/unknown member list. Shared by the 3D box lines, the
    /// screen-space glyph and the block hit test, so all three agree.
    pub(crate) fn group_aabb(g: &GeoSnap, members: &[u32]) -> Option<(Vec3, Vec3)> {
        let mut mn = Vec3::splat(f32::MAX);
        let mut mx = Vec3::splat(f32::MIN);
        for &m in members {
            if let Some(p) = g.pos.get(m as usize) {
                let v = Vec3::from(*p);
                mn = mn.min(v);
                mx = mx.max(v);
            }
        }
        (mn.x <= mx.x).then(|| (mn - Vec3::splat(GROUP_PAD), mx + Vec3::splat(GROUP_PAD)))
    }

    /// Is `inst` a neuron whose whole group is selected? (box highlight).
    pub(crate) fn group_fully_selected(&self, members: &[u32]) -> bool {
        !members.is_empty() && members.iter().all(|m| self.selection.contains(m))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simctl::SimCtl;

    /// geo: 2 groups — "g" members 0..4 (a row on x, spacing 10), "h"
    /// members 2..6 overlapping on 2,3 — plus loner id 9 and input inst.
    fn grouped_app() -> EditorApp {
        let mut a = EditorApp::new(SimCtl::offline(), None);
        let mut g = GeoSnap::default();
        g.pos = (0..10).map(|i| [i as f32 * 10.0, 0.0, 0.0]).collect();
        g.groups = vec![
            ("g".into(), vec![0, 1, 2, 3]),
            ("h".into(), vec![2, 3, 4, 5]),
        ];
        g.inputs = vec![crate::snap::InputSnap {
            name: "in".into(),
            targets: vec![0],
            pos: [0.0; 3],
        }];
        a.geo = Some(std::sync::Arc::new(g));
        a
    }

    /// Whole mode: a member click expands to every member; an
    /// overlapping whole group unites. Inside mode: the member alone.
    /// Sabotage: drop the `atomic` check → inside-group clicks select
    /// the group too.
    #[test]
    fn group_expand_modes() {
        let mut a = grouped_app();
        // default Whole: member 0 of g → all of g
        assert_eq!(a.group_expand(0), HashSet::from([0, 1, 2, 3]));
        // member 2 is in BOTH g and h → the union
        assert_eq!(a.group_expand(2), HashSet::from([0, 1, 2, 3, 4, 5]));
        // loner → itself
        assert_eq!(a.group_expand(9), HashSet::from([9]));
        // input instance (10 = 10 pos + 0 inputs… inst 10 is input 0)
        assert_eq!(a.group_expand(10), HashSet::from([10]));
        // g → Inside: member click stays single
        a.group_modes.insert("g".into(), GroupMode::Inside);
        assert_eq!(a.group_expand(0), HashSet::from([0]));
        // shared member 2: h is still Whole → h's members + inst
        assert_eq!(a.group_expand(2), HashSet::from([2, 3, 4, 5]));
        // g → Block: whole again, and h is still whole → union
        a.group_modes.insert("g".into(), GroupMode::Block);
        assert_eq!(a.group_expand(2), HashSet::from([0, 1, 2, 3, 4, 5]));
    }

    /// Box select: catching one member of an atomic group catches them
    /// all — including through the overlap chain (member of h pulls in
    /// g because they share 2/3). Inside-mode groups keep only what the
    /// box caught. Sabotage: a single-pass (non-fixpoint) expansion
    /// misses the chained group — `sel` ends with h but not g.
    #[test]
    fn box_expansion_chains_overlaps() {
        let mut a = grouped_app();
        // box "caught" only member 5 — inside h; h whole → pulls 2,3,4;
        // 2/3 also in g → g completes too.
        let g = a.geo.clone().unwrap();
        let expand = |a: &EditorApp, sel: &mut HashSet<u32>| {
            EditorApp::expand_sel_groups_in(&a.group_modes, a.group_mode_default, &g, sel);
        };
        let mut sel = HashSet::from([5u32]);
        expand(&a, &mut sel);
        assert_eq!(sel, HashSet::from([0, 1, 2, 3, 4, 5]));
        // h Inside: the box's catch stays what it was.
        a.group_modes.insert("h".into(), GroupMode::Inside);
        let mut sel = HashSet::from([5u32]);
        expand(&a, &mut sel);
        assert_eq!(sel, HashSet::from([5]));
        // but catching a g member still completes g.
        let mut sel = HashSet::from([0u32]);
        expand(&a, &mut sel);
        assert_eq!(sel, HashSet::from([0, 1, 2, 3]));
    }

    /// Block mode hides its members from ray tests; Whole/Inside don't.
    /// Sabotage: hiding by name instead of mode → a renamed default-group
    /// member stays pickable.
    #[test]
    fn block_members_hidden() {
        let mut a = grouped_app();
        assert!(a.block_members().is_empty());
        a.group_modes.insert("g".into(), GroupMode::Block);
        assert_eq!(a.block_members(), HashSet::from([0, 1, 2, 3]));
        // h members stay visible even where they overlap g's block.
        assert_eq!(a.group_ids_mode(GroupMode::Block), vec![0]);
    }

    /// The highlight rule: a group box lights when EVERY member is
    /// selected — a partial inside-mode selection doesn't.
    #[test]
    fn full_selection_highlight() {
        let a = {
            let mut a = grouped_app();
            a.selection = HashSet::from([0, 1, 2, 3]);
            a
        };
        assert!(a.group_fully_selected(&[0, 1, 2, 3]));
        assert!(!a.group_fully_selected(&[2, 3, 4, 5]));
        assert!(!a.group_fully_selected(&[]));
    }

    // ---- end-to-end: real worker, whole-group drag lands as one edit --
    //
    // Sabotage: if `click_select`/`group_expand` forgot to widen a member
    // pick to its group, `apply_move` below would emit a one-item
    // MoveNeurons and the position asserts would fail on the other 15.

    use crate::app::MoveDrag;
    use crate::view::DragPlane;
    use std::time::{Duration, Instant};

    fn wait_for<T>(mut f: impl FnMut() -> Option<T>) -> T {
        let t0 = Instant::now();
        loop {
            if let Some(v) = f() {
                return v;
            }
            assert!(t0.elapsed() < Duration::from_secs(10), "worker timeout");
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    /// A 4×4 "sheet" grid on a real worker; `app.geo` folded in by hand —
    /// `sync_shared` needs a GPU the test harness doesn't have.
    fn sheet_app() -> EditorApp {
        let mut net = net_format::Network::new();
        net_format::builders::create_2d_grid(&mut net, "sheet", 4, 4, 30.0, [0.0; 3]);
        let mut a = EditorApp::new(SimCtl::start(net, sim::BackendKind::Cpu, 1), None);
        a.geo = Some(wait_for(|| {
            let s = a.ctl.shared.read().ok()?;
            let g = s.geo.clone()?;
            (g.pos.len() == 16 && g.groups.len() == 1).then_some(g)
        }));
        a
    }

    /// The `MoveDrag` `begin_move` builds for the current selection;
    /// `apply_move` turns it into one NetEdit on the wire.
    fn drag_sel(a: &mut EditorApp, delta: glam::Vec3) {
        let g = a.geo.clone().unwrap();
        let ids: Vec<u32> = a.selection.iter().copied().collect();
        let md = MoveDrag {
            anchor: glam::Vec3::ZERO,
            plane_pt: glam::Vec3::ZERO,
            plane: DragPlane::Work(crate::scene::Plane::XZ),
            axis: None,
            start_pos: ids
                .iter()
                .map(|&i| glam::Vec3::from(g.pos[i as usize]))
                .collect(),
            ids,
        };
        a.apply_move(&md, delta);
    }

    fn geo_at(a: &EditorApp, version: u64) -> std::sync::Arc<GeoSnap> {
        wait_for(|| {
            let s = a.ctl.shared.read().ok()?;
            let g = s.geo.clone()?;
            (g.version >= version).then_some(g)
        })
    }

    #[test]
    fn whole_group_click_and_drag_moves_every_member() {
        let mut a = sheet_app();
        let before: Vec<[f32; 3]> = a.geo.as_ref().unwrap().pos.clone();

        a.click_select(3, false);
        assert_eq!(a.selection, (0..16).collect::<HashSet<u32>>());

        let d = glam::vec3(12.0, 0.0, 7.0);
        let v0 = a.geo.as_ref().unwrap().version;
        drag_sel(&mut a, d);
        let g = geo_at(&a, v0 + 1);
        for (i, p) in g.pos.iter().enumerate() {
            let moved = glam::Vec3::from(*p) - glam::Vec3::from(before[i]);
            assert!(
                (moved - d).length() < 0.05,
                "member {i} moved {moved:?}, expected {d:?}"
            );
        }

        // The worker labels the step — the Edit menu shows it.
        let lbl = wait_for(|| a.ctl.shared.read().ok()?.undo_label.clone());
        assert_eq!(lbl, "move 16 neurons");

        // Undo is one step — every member snaps back together.
        a.ctl.send(proto::ClientMsg::Undo);
        let g2 = geo_at(&a, g.version + 1);
        assert_eq!(g2.pos, before, "undo must restore the whole group");
    }

    #[test]
    fn inside_mode_drag_moves_one_member() {
        let mut a = sheet_app();
        a.group_modes.insert("sheet".into(), GroupMode::Inside);
        let before: Vec<[f32; 3]> = a.geo.as_ref().unwrap().pos.clone();

        a.click_select(3, false);
        assert_eq!(a.selection, HashSet::from([3]));

        let d = glam::vec3(-9.0, 0.0, 4.0);
        let v0 = a.geo.as_ref().unwrap().version;
        drag_sel(&mut a, d);
        let g = geo_at(&a, v0 + 1);
        for (i, p) in g.pos.iter().enumerate() {
            let moved = glam::Vec3::from(*p) - glam::Vec3::from(before[i]);
            if i == 3 {
                assert!((moved - d).length() < 0.05);
            } else {
                assert!(moved.length() < 0.01, "member {i} moved inside-mode");
            }
        }
    }

    /// Block click path: the glyph's members land in the selection and
    /// `block_members` is what hide-bits + ray-skips consume.
    #[test]
    fn block_mode_is_always_whole() {
        let mut a = sheet_app();
        a.group_modes.insert("sheet".into(), GroupMode::Block);
        assert_eq!(a.block_members(), (0..16).collect::<HashSet<u32>>());
        a.click_select(7, false);
        assert_eq!(a.selection, (0..16).collect::<HashSet<u32>>());
    }
}
