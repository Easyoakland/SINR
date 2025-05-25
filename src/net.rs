//! Organization of the [`Node`]s of the net.

use crate::{
    left_right::LeftRight,
    macros::{uassert, uunreachable},
    node::{Node, Ptr, PtrTag, SharedNode, Slot},
    redex::{Redex, RedexTy, Redexes},
    unsafe_vec::UnsafeVec,
};
use core::mem::ManuallyDrop;
use parking_lot::RwLock;
use rmw_free_barrier::Waiter;

pub type Nodes = UnsafeVec<SharedNode>;
pub type FreeList = UnsafeVec<Slot>;
#[derive(Debug, Clone)]
pub struct Net {
    pub nodes: Nodes,
    pub redexes: Redexes,
    pub free_list: FreeList,
}
impl Default for Net {
    fn default() -> Self {
        #[allow(unused_mut)]
        let mut net = Self {
            // The first slot can't be pointed towards since slot 0 is an invalid slot id.
            // Can, however, use slot 0 to point towards something, e.g., the root of the net.
            nodes: UnsafeVec(vec![SharedNode::new(Node::EMP())]),
            redexes: Default::default(),
            free_list: UnsafeVec(Vec::new()),
        };
        #[cfg(feature = "prealloc")]
        {
            net.nodes.0.reserve(10000000);
            for redex in &mut net.redexes.regular {
                redex.0.reserve(10000000)
            }
            net.free_list.0.reserve(10000000);
            net.redexes.erase.0.reserve(1000000);
        }
        net
    }
}

#[derive(Debug, Default)]
pub struct GlobalState {
    pub nodes: RwLock<Nodes>,
    pub redexes: RwLock<Redexes>,
    pub free_list: RwLock<FreeList>,
}

pub enum Either<A, B> {
    A(A),
    B(B),
}

#[inline]
pub fn add_redex(redexes: &mut Redexes, left: Ptr, right: Ptr) {
    uassert!(left != Ptr::EMP());
    uassert!(right != Ptr::EMP());
    uassert!(left != Ptr::IN());
    uassert!(right != Ptr::IN());
    let lt = left.tag();
    let rt = right.tag();
    match (lt, rt) {
        // If right is a follow.
        // Safety: check in match that value is `< 4` which is `< RedexTy::LEN`.
        _ if (rt as u8) < 4 => {
            let (redex, redex_ty) = (Redex::new(left, right), unsafe {
                RedexTy::from_u8(rt as u8)
            });
            redexes.regular[redex_ty as usize].push(redex);
        }
        // If left is a follow.
        // Safety: check in match that value is `< 4` which is `< RedexTy::LEN`.
        _ if (lt as u8) < 4 => {
            let (redex, redex_ty) = (Redex::new(right, left), unsafe {
                RedexTy::from_u8(lt as u8)
            });
            redexes.regular[redex_ty as usize].push(redex);
        }
        (PtrTag::Era, PtrTag::Era) => (),
        (PtrTag::Era, _) => redexes.erase.push(right),
        (_, PtrTag::Era) => redexes.erase.push(left),
        _ if lt == rt => {
            let (redex, redex_ty) = (Redex::new(left, right), RedexTy::Ann);
            redexes.regular[redex_ty as usize].push(redex);
        }
        _ => {
            let (redex, redex_ty) = (Redex::new(left, right), RedexTy::Com);
            redexes.regular[redex_ty as usize].push(redex);
        }
    }
}

/// `ptr_to_fst` should point to `fst` with stage 1.
/// # Note
/// Either `fst` or `snd` may have been set to `EMP`. To free memory the node containing both should be checked.
#[inline]
pub fn link_aux_ports(redexes: &mut Redexes, fst: &mut Ptr, snd: &mut Ptr, ptr_to_fst: Ptr) {
    uassert!(ptr_to_fst.tag() == PtrTag::LeftAux1 || ptr_to_fst.tag() == PtrTag::RightAux1);
    let (fst_start_out, snd_start_out) = (*fst != Ptr::IN(), *snd != Ptr::IN());
    // All cases either swap (heterogenous cases) or or don't care that they swap (homogenous cases).
    // TODO perf: see if this is an anti-optimization.
    core::mem::swap(fst, snd);
    match (*fst == Ptr::IN(), *snd == Ptr::IN()) {
        (true, true) => {
            // This is the only reason there needs to be two stages for left and right instead of only 1 per left and right.
            // The snd ptr should not be an `IN` anymore, and instead redirect (point) to the fst aux with stage1 to avoid a race.
            *snd = ptr_to_fst;
        }
        (true, false) | (false, true) => {
            // Already swapped
        }
        (false, false) => add_redex(redexes, *fst, *snd),
    }
    if fst_start_out {
        *fst = Ptr::EMP();
    }
    if snd_start_out {
        *snd = Ptr::EMP()
    }
}

pub fn interact_ann(
    redexes: &mut Redexes,
    free_list: &mut FreeList,
    nodes: &Nodes,
    left_ptr: Ptr,
    right_ptr: Ptr,
) {
    let l_idx = left_ptr.slot_usize();
    let r_idx = right_ptr.slot_usize();
    uassert!(l_idx != r_idx);
    let [left, right] = [&nodes[l_idx], &nodes[r_idx]];
    let [left, right] = [&mut *left.get(), &mut *right.get()];
    let (ll, lr) = (&mut left.left, &mut left.right);
    let (rl, rr) = (&mut right.left, &mut right.right);

    link_aux_ports(redexes, ll, rl, {
        let mut out = left_ptr;
        out.set_tag(PtrTag::LeftAux1);
        out
    });
    link_aux_ports(redexes, lr, rr, {
        let mut out = left_ptr;
        out.set_tag(PtrTag::RightAux1);
        out
    });
    if *left == Node::EMP() {
        free_list.push(left_ptr.slot());
    }
    if *right == Node::EMP() {
        free_list.push(right_ptr.slot());
    }
}

/// Follow target which is a &mut to an auxiliary port.
/// Either it redirects to another port or it is a [`Ptr::IN`]
/// After following `source` is connected again.
/// # Note
/// `target` port might be `PtrTag::EMP()` after this. Check the node containing it to maybe free memory.
#[inline]
pub fn follow_target(redexes: &mut Redexes, source: Ptr, target: &mut Ptr) {
    if *target == Ptr::IN() {
        *target = source // redirect to the new port
    } else {
        add_redex(redexes, source, *target);
        *target = Ptr::EMP()
    }
}

/// Interact a redex where the `right` `Ptr`'s target is not a primary port and instead is either a redirector or an auxiliary port.
pub fn interact_follow(
    free_list: &mut FreeList,
    redexes: &mut Redexes,
    nodes: &Nodes,
    left: Ptr,
    right: Ptr,
) {
    uassert!(right.tag().is_aux());

    let right_node = &mut *nodes[right.slot_usize()].get();
    {
        let target = match right.tag().aux_side() {
            LeftRight::Left => &mut right_node.left,
            LeftRight::Right => &mut right_node.right,
        };

        follow_target(redexes, left, target);
    }
    if *right_node == Node::EMP() {
        free_list.push(right.slot());
    }
}

/// Erase the node pointed to by `Ptr`
///
/// Erasers need a unique interaction distinct from annihilate because vicious circles of wires can be created if using the annihilate interaction to erase things.
/// On the positive side, this might speed things up and means that many erase nodes don't actually have to be stored in new slots of the net.
pub fn interact_era(redexes: &mut Redexes, free_list: &mut FreeList, nodes: &Nodes, ptr: Ptr) {
    let node_idx = ptr.slot_usize();
    let node = &mut *nodes[node_idx].get();

    if node.left != Ptr::EMP() {
        follow_target(redexes, Ptr::ERA_0(), &mut node.left);
    };
    if node.right != Ptr::EMP() {
        follow_target(redexes, Ptr::ERA_0(), &mut node.right);
    };
    if *node == Node::EMP() {
        free_list.push(ptr.slot());
    }
}

// perf: I have no idea why but for some reason this function spends significant time on the prologue and epilogue and `inline(always)` (inline insufficient) noticeably improves performance.
#[inline(always)]
pub fn interact_com(
    redexes: &mut Redexes,
    free_list: &mut FreeList,
    nodes: &Nodes,
    allocated_slots: (Slot, Slot, Slot, Slot),
    left_ptr: Ptr,
    right_ptr: Ptr,
) {
    let (ll2, lr2, rl2, rr2) = allocated_slots;

    {
        let left_idx = left_ptr.slot_usize();
        let right_idx = right_ptr.slot_usize();
        uassert!(left_idx != right_idx);
        let [left, right] = [&nodes[left_idx], &nodes[right_idx]];
        let [left, right] = [&mut *left.get(), &mut *right.get()];
        let (ll, lr, lt) = (&mut left.left, &mut left.right, left_ptr.tag());
        let (rl, rr, rt) = (&mut right.left, &mut right.right, right_ptr.tag());

        // ll2 and lr2 are now of type rt
        // rl2 and rr2 are now of type lt
        // Using the old auxiliary as the target and the new nodes' principal ports as sources, follow the targets.
        uassert!(!rt.is_aux());
        uassert!(!lt.is_aux());
        follow_target(redexes, Ptr::new(rt, ll2), ll);
        follow_target(redexes, Ptr::new(rt, lr2), lr);
        follow_target(redexes, Ptr::new(lt, rl2), rl);
        follow_target(redexes, Ptr::new(lt, rr2), rr);
        if *left == Node::EMP() {
            free_list.push(left_ptr.slot());
        }
        if *right == Node::EMP() {
            free_list.push(right_ptr.slot());
        }
    }

    // Make new nodes and link their aux together so each has 1 out and 1 in. No particular reason for 1 out and 1 in. Could be something else. I picked it because it's just nice and symmetric looking.
    // All nodes start at stage 0 since handling left and right in separate stages is sufficient to avoid races here since no *port* has 2 incoming pointers, i.e., no node with 2 incoming pointers to same aux.
    *nodes[ll2.value() as usize].get() = Node {
        left: Ptr::new(PtrTag::LeftAux0, rl2),
        right: Ptr::IN(),
    };
    *nodes[lr2.value() as usize].get() = Node {
        left: Ptr::IN(),
        right: Ptr::new(PtrTag::RightAux0, rr2),
    };
    *nodes[rl2.value() as usize].get() = Node {
        left: Ptr::IN(),
        right: Ptr::new(PtrTag::LeftAux0, lr2),
    };
    *nodes[rr2.value() as usize].get() = Node {
        left: Ptr::new(PtrTag::RightAux0, ll2),
        right: Ptr::IN(),
    };
}

pub mod nodes {
    use super::*;

    /// Read the target of this pointer.
    // Not used by the runtime, so this doesn't need to be performant.
    pub fn read(nodes: &Nodes, ptr: Ptr) -> Either<Node, Ptr> {
        match ptr.tag() {
            PtrTag::LeftAux0 | PtrTag::LeftAux1 => Either::B(nodes[ptr.slot_usize()].get().left),
            PtrTag::RightAux0 | PtrTag::RightAux1 => Either::B(nodes[ptr.slot_usize()].get().right),
            PtrTag::Era | PtrTag::Con | PtrTag::Dup => Either::A(*nodes[ptr.slot_usize()].get()),
            PtrTag::_Unused => uunreachable!(),
        }
    }
    #[inline]
    pub fn alloc_node(nodes: &mut Nodes, free_list: &mut FreeList) -> Slot {
        free_list.pop().unwrap_or_else(|| {
            let res = nodes.len();
            nodes.push(SharedNode::new(Node::default()));
            uassert!(res <= u64::MAX as usize); // prevent check on feature=unsafe
            uassert!(res <= <Slot as bilge::Bitsized>::MAX.value() as usize);
            Slot::new(res.try_into().unwrap())
        })
    }
    #[inline]
    pub fn alloc_node2(nodes: &mut Nodes, free_list: &mut FreeList) -> (Slot, Slot) {
        (alloc_node(nodes, free_list), alloc_node(nodes, free_list))
    }
    #[inline]
    pub fn alloc_node4(nodes: &mut Nodes, free_list: &mut FreeList) -> (Slot, Slot, Slot, Slot) {
        let ((a, b), (c, d)) = (alloc_node2(nodes, free_list), alloc_node2(nodes, free_list));
        (a, b, c, d)
    }
}

impl Net {
    /// Read the target of this pointer.
    // Not used by the runtime, so this doesn't need to be performant.
    pub fn read(&self, ptr: Ptr) -> Either<Node, Ptr> {
        match ptr.tag() {
            PtrTag::LeftAux0 | PtrTag::LeftAux1 => {
                Either::B(self.nodes[ptr.slot_usize()].get().left)
            }
            PtrTag::RightAux0 | PtrTag::RightAux1 => {
                Either::B(self.nodes[ptr.slot_usize()].get().right)
            }
            PtrTag::Era | PtrTag::Con | PtrTag::Dup => {
                Either::A(*self.nodes[ptr.slot_usize()].get())
            }
            PtrTag::_Unused => uunreachable!(),
        }
    }
    #[inline]
    pub fn alloc_node(&mut self) -> Slot {
        self.free_list.pop().unwrap_or_else(|| {
            let res = self.nodes.len();
            self.nodes.push(SharedNode::new(Node::default()));
            uassert!(res <= u64::MAX as usize); // prevent check on feature=unsafe
            uassert!(res <= <Slot as bilge::Bitsized>::MAX.value() as usize);
            Slot::new(res.try_into().unwrap())
        })
    }
    #[inline]
    pub fn alloc_node2(&mut self) -> (Slot, Slot) {
        (self.alloc_node(), self.alloc_node())
    }
    #[inline]
    pub fn alloc_node4(&mut self) -> (Slot, Slot, Slot, Slot) {
        let ((a, b), (c, d)) = (self.alloc_node2(), self.alloc_node2());
        (a, b, c, d)
    }

    /// Convenience method using &mut access to the net to perform an annihilate interaction.
    pub fn interact_ann(&mut self, left_ptr: Ptr, right_ptr: Ptr) {
        interact_ann(
            &mut self.redexes,
            &mut self.free_list,
            &self.nodes,
            left_ptr,
            right_ptr,
        );
    }

    /// Convenience method using &mut access to the net to perform a commute interaction.
    #[inline(always)]
    pub fn interact_com(&mut self, left_ptr: Ptr, right_ptr: Ptr) {
        let slots = self.alloc_node4();
        interact_com(
            &mut self.redexes,
            &mut self.free_list,
            &self.nodes,
            slots,
            left_ptr,
            right_ptr,
        );
    }

    /// Convenience method using &mut access to the net to perform a follow interaction.
    pub fn interact_follow(&mut self, left: Ptr, right: Ptr) {
        interact_follow(
            &mut self.free_list,
            &mut self.redexes,
            &self.nodes,
            left,
            right,
        );
    }

    /// Convenience method using &mut access to the net to perform an erase interaction.
    pub fn interact_era(&mut self, ptr: Ptr) {
        interact_era(&mut self.redexes, &mut self.free_list, &self.nodes, ptr);
    }

    pub fn active_redexes(&self) -> u64 {
        active_redexes(&self.redexes)
    }
}
pub fn active_redexes(redexes: &Redexes) -> u64 {
    redexes
        .regular
        .iter()
        .map(|x| x.len())
        .chain([redexes.erase.len()])
        .sum::<usize>()
        .try_into()
        .unwrap()
}

/// State per thread performing reduction.
#[derive(Debug, Default)]
pub struct ThreadState {
    pub interactions_fol: u64,
    pub interactions_ann: u64,
    pub interactions_com: u64,
    pub interactions_era: u64,
    pub nodes_max: u64,
    pub redexes_max: u64,
    pub local_net: Net,
    pub waiter: Waiter,
}
impl ThreadState {
    pub fn new(waiter: Waiter) -> Self {
        Self {
            interactions_fol: Default::default(),
            interactions_ann: Default::default(),
            interactions_com: Default::default(),
            interactions_era: Default::default(),
            nodes_max: Default::default(),
            redexes_max: Default::default(),
            local_net: Default::default(),
            waiter,
        }
    }
    /// Reduce available redexes of each type
    #[inline(always)] // perf: *massive* (over 50%) performance loss when this is not inlined.
    pub fn run_once(&mut self, global_net: &GlobalState) {
        // Limit the number of interactions per stage for all redex types which might allocate new nodes. This way we limit parallelism to avoid overflowing the data cache.
        // TODO perf: tune this value. Bigger means less stage syncs and more friendly to branch prediction, less means better cache locality.
        // Looks like single-threaded perf gets worse for big nets at < 2^3 and > 2^7 with a 64KB/core L1 cache. I suspect the optimal number will depend on cache sizes.
        // It might be worth changing this dynamically depending on current number of redexes.
        // It would be pretty cool to have this auto-tune if that turns out to be case case.
        const MAX_ITER_PER_ALLOC_TY: usize = 2usize.pow(7);

        // Threshold below which redexes should be injected from the global net into the local net.
        const STEAL_GLOBAL_THRESH: u64 = 2u64.pow(10);

        // Threshold above which redexes should be injected from the local net into the global net.
        const PUSH_GLOBAL_THRESH: u64 = 2u64.pow(12);

        self.nodes_max = self
            .nodes_max
            .max(self.local_net.nodes.len().try_into().unwrap());
        self.redexes_max = self.redexes_max.max(self.local_net.active_redexes());

        // TODO it should be possible to be lock-free when allocating new slots in the global net.
        // It should also be possible to be lock-free when sending or receiving new redexes.
        // TODO benchmark if contention becomes an issue in which case that might help.

        // Take redexes from the global net if local net doesn't have enough.
        if self.local_net.active_redexes() < STEAL_GLOBAL_THRESH {
            let mut global_net_redexes = global_net.redexes.write();
            let mut to_steal = STEAL_GLOBAL_THRESH as usize;
            for (local, global) in core::iter::zip(
                &mut self.local_net.redexes.regular,
                &mut global_net_redexes.regular,
            ) {
                local.0.extend(global.0.drain(..global.len().min(to_steal)));
                to_steal -= global.len().min(to_steal);
            }
            self.local_net.redexes.erase.0.extend({
                let global = &mut global_net_redexes.erase;
                global.0.drain(..global.len().min(to_steal))
            });
        }
        // Push redexes to the global net.
        else if self.local_net.active_redexes() > PUSH_GLOBAL_THRESH {
            let mut global_net_redexes = global_net.redexes.write();
            // TODO perf: test pushing commute first

            for (local, global) in core::iter::zip(
                &mut self.local_net.redexes.regular,
                &mut global_net_redexes.regular,
            ) {
                global.0.extend(local.0.drain(..(local.len() / 2)));
            }
            global_net_redexes.erase.0.extend({
                let local = &mut self.local_net.redexes.erase;
                local.0.drain(..(local.len() / 2))
            })
        }

        // Make sure that the local net has sufficient free nodes.
        if self.local_net.free_list.len()
            < self.local_net.redexes.regular[RedexTy::Com as usize]
                .len()
                .min(MAX_ITER_PER_ALLOC_TY)
                * 4
        {
            while self.local_net.free_list.len() < MAX_ITER_PER_ALLOC_TY * 4 {
                self.local_net.free_list.push(nodes::alloc_node(
                    &mut global_net.nodes.write(),
                    &mut global_net.free_list.write(),
                ));
            }
        }

        self.waiter.wait(); // exit redex move stage and enter interaction stage

        // Safety: In this stage threads only read from `global_net.nodes` and none write.
        // This is `ManuallyDrop` to avoid the guard messing with the lock on `Drop`.
        let global_net_nodes =
            unsafe { &ManuallyDrop::new(global_net.nodes.make_read_guard_unchecked()) };
        // let global_net_nodes = &global_net.nodes.read();

        while let Some(ptr) = self.local_net.redexes.erase.pop() {
            interact_era(
                &mut self.local_net.redexes,
                &mut self.local_net.free_list,
                &global_net_nodes,
                ptr,
            );
            self.interactions_era += 1;
        }

        for _ in 0..MAX_ITER_PER_ALLOC_TY {
            let Some(Redex(l, r)) = self.local_net.redexes.regular[RedexTy::Com as usize].pop()
            else {
                break;
            };
            {
                // Made sure that there are sufficient nodes above.
                // TODO check if this prevents bound-check code gen.
                uassert!(self.local_net.free_list.len() >= 4);
                let slots = (
                    self.local_net.free_list.pop().unwrap(),
                    self.local_net.free_list.pop().unwrap(),
                    self.local_net.free_list.pop().unwrap(),
                    self.local_net.free_list.pop().unwrap(),
                );
                interact_com(
                    &mut self.local_net.redexes,
                    &mut self.local_net.free_list,
                    &global_net_nodes,
                    slots,
                    l,
                    r,
                );
                self.interactions_com += 1;
            }
        }

        while let Some(Redex(l, r)) = self.local_net.redexes.regular[RedexTy::Ann as usize].pop() {
            interact_ann(
                &mut self.local_net.redexes,
                &mut self.local_net.free_list,
                &global_net_nodes,
                l,
                r,
            );
            self.interactions_ann += 1;
        }

        self.waiter.wait();
        while let Some(Redex(l, r)) = self.local_net.redexes.regular[RedexTy::FolL0 as usize].pop()
        {
            interact_follow(
                &mut self.local_net.free_list,
                &mut self.local_net.redexes,
                &global_net_nodes,
                l,
                r,
            );
            self.interactions_fol += 1;
        }

        self.waiter.wait();
        while let Some(Redex(l, r)) = self.local_net.redexes.regular[RedexTy::FolR0 as usize].pop()
        {
            interact_follow(
                &mut self.local_net.free_list,
                &mut self.local_net.redexes,
                &global_net_nodes,
                l,
                r,
            );
            self.interactions_fol += 1;
        }

        self.waiter.wait();
        while let Some(Redex(l, r)) = self.local_net.redexes.regular[RedexTy::FolL1 as usize].pop()
        {
            interact_follow(
                &mut self.local_net.free_list,
                &mut self.local_net.redexes,
                &global_net_nodes,
                l,
                r,
            );
            self.interactions_fol += 1;
        }

        self.waiter.wait();
        while let Some(Redex(l, r)) = self.local_net.redexes.regular[RedexTy::FolR1 as usize].pop()
        {
            interact_follow(
                &mut self.local_net.free_list,
                &mut self.local_net.redexes,
                &global_net_nodes,
                l,
                r,
            );
            self.interactions_fol += 1;
        }

        self.waiter.wait(); // make sure node readguards are all dropped.
    }

    pub fn interactions(&self) -> u64 {
        self.interactions_com
            + self.interactions_ann
            + self.interactions_era
            + self.interactions_fol
    }
    pub fn non_follow_interactions(&self) -> u64 {
        self.interactions_com + self.interactions_ann + self.interactions_era
    }
}
