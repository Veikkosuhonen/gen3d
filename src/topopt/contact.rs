//! A ground that only pushes (`Ground::Slide`, `Ground::Grip`): frictionless unilateral contact
//! (Signorini) at the bottom layer's nodes, solved per load case by a primal–dual active set.
//!
//! Each case keeps the set of contact nodes that touch down. A touching node is held along its
//! contact axes (Z for slide, all three for grip); the others are free. After a solve:
//! - a touching node whose vertical reaction pulls the part down lifts off (released);
//! - a released node that sank below the ground touches down again.
//!
//! and the case is solved again, until the set settles. Every case starts with all nodes touching,
//! and each optimizer iteration starts from the previous iteration's sets, so once the design has
//! formed a solve rarely changes them. With the set settled the solution is that of a linear
//! problem with those supports, and the compliance sensitivities are the usual ones (the reactions
//! do no work: touching nodes don't move, free ones carry none).

use super::Problem;

/// At most this many solves per case and optimizer iteration. The sets settle in a few rounds
/// from the uniform start; later the design moves them little, and an unsettled set is carried
/// on to the next iteration.
pub const MAX_ROUNDS: usize = 8;

/// A reaction pulls when below −`PULL_TOL` × the case's mean reaction per touching node: CG's
/// tolerance leaves noise on the reactions, and nodes at the edge of a contact patch carry none.
const PULL_TOL: f64 = 1e-3;
/// A free node sank when below −`SINK_TOL` × the case's largest displacement.
const SINK_TOL: f64 = 1e-6;

pub struct Contact {
    /// the contact nodes and the axes each is held along while touching
    pub nodes: Vec<usize>,
    pub axes: Vec<u8>,
    /// per load case, per contact node: touching down
    pub touch: Vec<Vec<bool>>,
}

/// What one update changed.
#[derive(Clone, Copy, Debug, Default)]
pub struct Change {
    pub lifted: usize,
    pub landed: usize,
}

impl Change {
    pub fn any(self) -> bool {
        self.lifted + self.landed > 0
    }
}

impl Contact {
    /// The contact nodes of `p`, all touching (none without a contact ground).
    pub fn new(p: &Problem) -> Option<Contact> {
        if !p.has_contact() {
            return None;
        }
        let nodes: Vec<usize> = (0..p.contact.len()).filter(|&i| p.contact[i] != 0).collect();
        let axes = nodes.iter().map(|&i| p.contact[i]).collect();
        let touch = vec![vec![true; nodes.len()]; p.loads.len()];
        Some(Contact { nodes, axes, touch })
    }

    /// Case `c`'s free mask: `base` (the problem's own) less the touching nodes' contact axes.
    pub fn free(&self, base: &[u8], c: usize) -> Vec<u8> {
        let mut free = base.to_vec();
        for (k, &i) in self.nodes.iter().enumerate() {
            if self.touch[c][k] {
                free[i] &= !self.axes[k];
            }
        }
        free
    }

    /// Update case `c`'s set from the vertical reaction on each contact node (`rz`, +Z pushes the
    /// part up; read on touching nodes) and its vertical displacement (`uz`, read on free ones);
    /// `umax` is the case's largest displacement.
    pub fn update(&mut self, c: usize, rz: &[f64], uz: &[f64], umax: f64) -> Change {
        let before = self.touch[c].clone();
        let touch = &mut self.touch[c];
        let (sum, n) = touch
            .iter()
            .zip(rz)
            .filter(|(t, _)| **t)
            .fold((0.0, 0usize), |(s, n), (_, r)| (s + r.abs(), n + 1));
        let pull = -PULL_TOL * sum / n.max(1) as f64;
        let sink = -SINK_TOL * umax;
        let mut ch = Change::default();
        for k in 0..touch.len() {
            if touch[k] && rz[k] < pull {
                touch[k] = false;
                ch.lifted += 1;
            } else if !touch[k] && uz[k] < sink {
                touch[k] = true;
                ch.landed += 1;
            }
        }
        // a case that would lift off everywhere (the tipping check rules this out for a part on
        // the ground alone): keep the set, so the problem stays solvable
        if !touch.iter().any(|t| *t) {
            *touch = before;
            return Change::default();
        }
        ch
    }

    /// Case `c`'s touching nodes.
    pub fn touching(&self, c: usize) -> usize {
        self.touch[c].iter().filter(|t| **t).count()
    }
}
