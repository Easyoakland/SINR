//! # SINR: Staged Interaction Net Runtime
//!
//! Staged, Parallel, and SIMD capable Interpreter of Symmetric Interaction Combinators (SIC) + extensions.
//!
//! # Layout
//! ## Nodes
//! Nodes are represented as 2 tagged pointers and redexes as 2 tagged pointers. Nodes may have 0, 1, or 2 auxiliary ports. Although all currently implemented nodes are 2-ary others should work.
//! ## Node tagged pointers
//! Tagged pointers contain a target and the type of the target they point to.
//! A Pointer is either "out" in which case its target is one of the [`PtrTag`] variants, `IN`, in which case it has no target, or `EMP` in which case it represents unallocated memory. The [`PtrTag`] represents the different port types, and the 4 auxiliary port types.
//!
//! If you are familiar with inets you may wonder why 2-ary nodes have 4 separate auxiliary types instead of 2. The difference here is that auxiliary nodes have an associated stage, which dictates the synchronization of interactions. This makes it possible to handle annihilation interactions and re-use the annihilating nodes to redirect incoming pointers from the rest of the net in the case that two incoming pointers are to be linked. In this particular case one of the two auxiliary ports points to the other with a stage of 1 instead of 0, and therefore avoids racing when performing follows.
//!
//! The reason that left and right follow interactions must be separated is to deallocate nodes only once both pointers in the node are no longer in use. This simplifies memory deallocation (TODO) by preventing any fragmentation (holes) less than a node in size. If this turns out to not be sufficiently useful, then left and right auxiliary follows can be performed simultaneously, only synchronizing between stage 0, 1, and principal interaction.
//!
//! # Reduction
//! All redexes are stored into one of several redex buffers based upon the redex's interaction type, [`RedexTy`]. The regular commute and annihilate are two interactions. So are the 4 possible follow operations depending on the type of the auxiliary target. By separating the operations into these 2+4 types it becomes possible to perform all operations of the same type with minimal branching (for SIMD) and without atomic synchronization (for CPU SIMD and general perf improvements). All threads and SIMD lanes operate to reduce the same interaction type at the same time. When one of the threads runs out of reductions of that type (or some other signal such as number of reductions) all the threads synchronize their memory and start simultaneously reducing operations of a new type.
//! # Performance
//! See `Benchmarks.md` for performance measurements.
//! - On x86 single-threaded: Counting only commute and annihilate interactions appears to be ~145 million interactions per second (MIPS): 34.48 cycles per interaction.
//! - On x86 single-threaded: Counting commute, annihilate, and linking auxiliary ports appears to be ~330 MIPS: 15.15 cycles per interaction.
//! # Future possibilities
//! ## Amb nodes
//! Since this design allows for synchronizing multiple pointers to a single port without atomics, implementing amb (a 2 principal port node) should be as simple as an FollowAmb0 and FollowAmb1 interactions.
//! ## Global ref nodes
//! Global nets can be supported simply by adding the interaction type as usual. To enable SIMD, the nets should be stored with offsets local to 0. Then, when instantiating the net, allocate a continuous block the size of the global subnet and add the start of the block to all the offsets of pointers in the global subnet.
//! # Goals
//! - [x] Basic interactions
//! - [x] Single-threaded scalar implementation
//! - [ ] Memory deallocation and reclamation.
//! - [x] Minimize branching using lookup tables
//!     - This does not improve performance, and instead appears to decrease it. Remaining branching is very minimal.
//! - [x] SIMD
//!     - SIMD seems useless. Profiling indicates most of the program time is spent in `ptr::write`, `ptr::read`, and checking `len==0`, inside `Vec::push` and `Vec::pop`.
//!     Consequently, SIMD is unlikely to be useful since that part is not SIMD-able. Attempting to implement some parts with SIMD appear to only serve to slow things down by increasing front-end load and performing unnecessary extra work to swap values inside registers. It's possible that the SIMD code was poor and could have been improved. See the `SIMD` branch for details.
//! - [ ] Multiple threads
//! - [ ] Parse net from text

#![feature(get_disjoint_mut_helpers)]
// TODO remove these
#![allow(dead_code)]

mod builder;
mod left_right;
mod macros;
mod unsafe_vec;
mod viz;

use bilge::prelude::*;
use core::u32;
use left_right::LeftRight;
use macros::{uassert, uunreachable};
use unsafe_vec::UnsafeVec;

#[bilge::bitsize(3)]
#[derive(
    Debug,
    Default,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    bilge::BinaryBits,
    bilge::FromBits,
)]
#[repr(u8)]
pub enum PtrTag {
    // pri=0,1; l=0,r=1
    #[default]
    LeftAux0 = 0b00,
    RightAux0 = 0b01,
    LeftAux1 = 0b10,
    RightAux1 = 0b11,
    Era = 0b100,
    Con = 0b101,
    Dup = 0b110,
    /// Don't use this. Free bit pattern. Maybe repurpose for packages?
    _Unused = 0b111,
}
impl core::fmt::Display for PtrTag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PtrTag::LeftAux0 => "L0",
            PtrTag::LeftAux1 => "L1",
            PtrTag::RightAux0 => "R0",
            PtrTag::RightAux1 => "R1",
            PtrTag::Era => "ε",
            PtrTag::Con => "ζ",
            PtrTag::Dup => "δ",
            PtrTag::_Unused => "_",
        }
        .fmt(f)
    }
}
impl PtrTag {
    pub const LEN: usize = 7;
    pub const fn is_aux(self) -> bool {
        self as u8 <= Self::RightAux1 as u8
    }
    /// # Safety
    /// Assumes this is an aux
    pub fn aux_side(&self) -> LeftRight {
        // match self {
        //     PtrTag::LeftAux0 | PtrTag::LeftAux1 => LeftRight::Left,
        //     PtrTag::RightAux0 | PtrTag::RightAux1 => LeftRight::Right,
        //     _ => uunreachable!(),
        // }
        // perf: `match` doesn't optimize as well as manual bitfiddling
        uassert!((*self as u8) < 4);
        uassert!(LeftRight::from(u1::new(PtrTag::LeftAux0 as u8 & 1)) == LeftRight::Left);
        uassert!(LeftRight::from(u1::new(PtrTag::LeftAux1 as u8 & 1)) == LeftRight::Left);
        uassert!(LeftRight::from(u1::new(PtrTag::RightAux0 as u8 & 1)) == LeftRight::Right);
        uassert!(LeftRight::from(u1::new(PtrTag::RightAux1 as u8 & 1)) == LeftRight::Right);
        LeftRight::from(u1::new((*self as u8) & 1))
    }
}

type Slot = u61;

#[bilge::bitsize(64)]
#[derive(
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    bilge::BinaryBits,
    bilge::DebugBits,
    bilge::FromBits,
)]
#[repr(transparent)]
struct Ptr {
    tag: PtrTag,
    // A 32bit `Ptr` means we can address 2^29 nodes or 2^32 bytes since each node is u64 in size.
    // A 64bit `Ptr` means we can address 2^61 nodes or 2^67 bytes since each node is u128 in size, or make PtrTag 6 bits to get 2^64 bytes.
    slot: Slot,
}

impl Default for Ptr {
    fn default() -> Self {
        Self::EMP()
    }
}
impl Ptr {
    // ---
    // The first slot is not valid target of `Ptr` because it represents these special values
    // ---
    /// An empty, unused ptr slot. Everything after a calloc should be EMP.
    pub const EMP: fn() -> Ptr = || Ptr { value: 0 };
    /// Auxiliary which is pointed *to* but doesn't point out.
    pub const IN: fn() -> Ptr = || Ptr::new(PtrTag::from(u3::new(1)), Slot::new(0));
    /// An eraser doesn't need to be allocated since it contains no information (no data or auxiliary ports).\
    /// As such, we can create new erasers by using slot 0 since the slot doesn't matter (or mean anything).\
    /// We can also change only the tag and leave the slot the same if that turns out to be faster.
    pub const ERA_0: fn() -> Ptr = || Ptr::new(PtrTag::Era, Slot::new(0));
    // Only used by `viz` so okay to be slow.
    #[inline]
    pub fn slot_u32(self) -> u32 {
        self.slot().value().try_into().unwrap()
    }
    #[inline]
    pub fn slot_usize(self) -> usize {
        const {
            assert!(
                size_of::<u32>() <= size_of::<usize>(),
                "u32 larger than usize"
            )
        };
        self.slot().value() as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Hash)]
#[repr(C)]
struct Node {
    left: Ptr,
    right: Ptr,
}
impl Node {
    pub const EMP: fn() -> Node = || Node {
        left: Ptr::EMP(),
        right: Ptr::EMP(),
    };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Hash)]
struct Redex(Ptr, Ptr);
impl Redex {
    #[inline]
    pub fn new(left: Ptr, right: Ptr) -> Self {
        Self(left, right)
    }
}

/// - Follow l0: ?? <> LeftAux0
/// - Follow r0: ?? <> RightAux0
/// - Follow l1: ?? <> LeftAux1
/// - Follow r1: ?? <> RightAux1
/// - Annihilate: Combinator i <> Combinator i (same label)
/// - Commute: Combinator i <> Combinator j (different label)
///
/// Note to avoid races only one of the `Fol` queues or any of principal redex queue can be operated on simultaneously.
/// On the positive side, the queue being operated on can do so without any atomics whatsoever (including the follow wires rules).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
enum RedexTy {
    // Note Fol* variants have same discriminant as corresponding `PtrTag` variant
    #[default]
    FolL0 = 0,
    FolR0 = 1,
    FolL1 = 2,
    FolR1 = 3,
    Ann,
    Com,
}
impl RedexTy {
    pub const LEN: usize = 6;
    /// # Safety
    /// `val` < RedexTy::LEN
    pub unsafe fn from_u8(val: u8) -> Self {
        uassert!({ val as usize } < Self::LEN);
        unsafe { core::mem::transmute(val) }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct Redexes {
    regular: [UnsafeVec<Redex>; RedexTy::LEN],
    erase: UnsafeVec<Ptr>,
}
type Nodes = UnsafeVec<Node>;
type FreeList = UnsafeVec<Slot>;
#[derive(Debug, Clone)]
struct Net {
    nodes: Nodes,
    redexes: Redexes,
    free_list: FreeList,
}
impl Default for Net {
    fn default() -> Self {
        #[allow(unused_mut)]
        let mut net = Self {
            // The first slot can't be pointed towards since slot 0 is an invalid slot id.
            // Can, however, use slot 0 to point towards something, e.g., the root of the net.
            nodes: UnsafeVec(vec![Node::EMP()]),
            redexes: Default::default(),
            free_list: UnsafeVec(Vec::new()),
        };
        #[cfg(feature = "prealloc")]
        {
            net.nodes.0.reserve(100000000);
            for redex in &mut net.redexes.regular {
                redex.0.reserve(100000000)
            }
            net.free_list.0.reserve(100000000);
            net.redexes.erase.0.reserve(10000000);
        }
        net
    }
}

enum Either<A, B> {
    A(A),
    B(B),
}

impl Net {
    /// Read the target of this pointer.
    // Not used by the runtime, so this doesn't need to be performant.
    pub fn read(&self, ptr: Ptr) -> Either<Node, Ptr> {
        match ptr.tag() {
            PtrTag::LeftAux0 | PtrTag::LeftAux1 => Either::B(self.nodes[ptr.slot_usize()].left),
            PtrTag::RightAux0 | PtrTag::RightAux1 => Either::B(self.nodes[ptr.slot_usize()].right),
            PtrTag::Era | PtrTag::Con | PtrTag::Dup => Either::A(self.nodes[ptr.slot_usize()]),
            PtrTag::_Unused => uunreachable!(),
        }
    }
    #[inline]
    pub fn alloc_node(&mut self) -> Slot {
        self.free_list.pop().unwrap_or_else(|| {
            let res = self.nodes.len();
            self.nodes.push(Node::default());
            uassert!(res <= u32::MAX as usize); // prevent check on feature=unsafe
            uassert!(res <= <Slot as Bitsized>::MAX.value() as usize);
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
    #[inline]
    pub fn add_redex(redexes: &mut Redexes, left: Ptr, right: Ptr) {
        uassert!(left.tag() != PtrTag::_Unused);
        uassert!(right.tag() != PtrTag::_Unused);
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
            (false, false) => Self::add_redex(redexes, *fst, *snd),
        }
        if fst_start_out {
            *fst = Ptr::EMP();
        }
        if snd_start_out {
            *snd = Ptr::EMP()
        }
    }

    pub fn interact_ann(&mut self, left_ptr: Ptr, right_ptr: Ptr) {
        let l_idx = left_ptr.slot_usize();
        let r_idx = right_ptr.slot_usize();
        let [left, right] = self.nodes.get_disjoint_mut([l_idx, r_idx]);
        let (ll, lr) = (&mut left.left, &mut left.right);
        let (rl, rr) = (&mut right.left, &mut right.right);

        Self::link_aux_ports(&mut self.redexes, ll, rl, {
            let mut out = left_ptr;
            out.set_tag(PtrTag::LeftAux1);
            out
        });
        Self::link_aux_ports(&mut self.redexes, lr, rr, {
            let mut out = left_ptr;
            out.set_tag(PtrTag::RightAux1);
            out
        });
        if *left == Node::EMP() {
            self.free_list.push(left_ptr.slot());
        }
        if *right == Node::EMP() {
            self.free_list.push(right_ptr.slot());
        }
    }

    /// Follow target which is a &mut to an auxiliary port.
    /// Either it redirects to another port or it is a [`Ptr::IN()`]
    /// After following `source` is connected again.
    /// # Note
    /// `target` port might be `PtrTag::EMP()` after this. Check the node containing it to maybe free memory.
    #[inline]
    pub fn follow_target(redexes: &mut Redexes, source: Ptr, target: &mut Ptr) {
        if *target == Ptr::IN() {
            *target = source // redirect to the new port
        } else {
            Self::add_redex(redexes, source, *target);
            *target = Ptr::EMP()
        }
    }

    // perf: I have no idea why but for some reason this function spends significant time on the prologue and epilogue and `inline(always)` (inline insufficient) noticeably improves performance.
    #[inline(always)]
    pub fn interact_com(&mut self, left_ptr: Ptr, right_ptr: Ptr) {
        let (ll2, lr2, rl2, rr2) = self.alloc_node4();

        let left_idx = left_ptr.slot_usize();
        let right_idx = right_ptr.slot_usize();
        let [left, right] = self.nodes.get_disjoint_mut([left_idx, right_idx]);
        let (ll, lr, lt) = (&mut left.left, &mut left.right, left_ptr.tag());
        let (rl, rr, rt) = (&mut right.left, &mut right.right, right_ptr.tag());

        // ll2 and lr2 are now of type rt
        // rl2 and rr2 are now of type lt
        // Using the old auxiliary as the target and the new nodes' principal ports as sources, follow the targets.
        uassert!(!rt.is_aux());
        uassert!(!lt.is_aux());
        Self::follow_target(&mut self.redexes, Ptr::new(rt, ll2), ll);
        Self::follow_target(&mut self.redexes, Ptr::new(rt, lr2), lr);
        Self::follow_target(&mut self.redexes, Ptr::new(lt, rl2), rl);
        Self::follow_target(&mut self.redexes, Ptr::new(lt, rr2), rr);
        if *left == Node::EMP() {
            self.free_list.push(left_ptr.slot());
        }
        if *right == Node::EMP() {
            self.free_list.push(right_ptr.slot());
        }

        // Make new nodes and link their aux together so each has 1 out and 1 in.
        // All nodes start at stage 0 since handling left and right in separate stages is sufficient to avoid races here since no *port* has 2 incoming pointers, i.e., no node with 2 incoming pointers to same aux.
        self.nodes[ll2.value() as usize] = Node {
            left: Ptr::new(PtrTag::LeftAux0, rl2),
            right: Ptr::IN(),
        };
        self.nodes[lr2.value() as usize] = Node {
            left: Ptr::IN(),
            right: Ptr::new(PtrTag::RightAux0, rr2),
        };
        self.nodes[rl2.value() as usize] = Node {
            left: Ptr::IN(),
            right: Ptr::new(PtrTag::LeftAux0, lr2),
        };
        self.nodes[rr2.value() as usize] = Node {
            left: Ptr::new(PtrTag::RightAux0, ll2),
            right: Ptr::IN(),
        };
    }

    /// Interact a redex where the `right` `Ptr`'s target is not a primary port and instead is either a redirector or an auxiliary port.
    pub fn interact_follow(&mut self, left: Ptr, right: Ptr) {
        uassert!(right.tag().is_aux());

        let right_node = &mut self.nodes[right.slot_usize()];
        let target = match right.tag().aux_side() {
            LeftRight::Left => &mut right_node.left,
            LeftRight::Right => &mut right_node.right,
        };

        Self::follow_target(&mut self.redexes, left, target);
        if *right_node == Node::EMP() {
            self.free_list.push(right.slot());
        }
    }

    /// Erase the node pointed to by `Ptr`
    ///
    /// Erasers need a unique interaction distinct from annihilate because vicious circles of wires can be created if using the annihilate interaction to erase things.
    /// On the positive side, this might speed things up and means that many erase nodes don't actually have to be stored in new slots of the net.
    pub fn interact_era(&mut self, ptr: Ptr) {
        let node_idx = ptr.slot_usize();
        let node = &mut self.nodes[node_idx];

        if node.left != Ptr::EMP() {
            Self::follow_target(&mut self.redexes, Ptr::ERA_0(), &mut node.left);
        };
        if node.right != Ptr::EMP() {
            Self::follow_target(&mut self.redexes, Ptr::ERA_0(), &mut node.right);
        };
        if *node == Node::EMP() {
            self.free_list.push(ptr.slot());
        }
    }
}

/// State per thread performing reduction.
#[derive(Debug, Default, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct ThreadState {
    interactions_fol: u64,
    interactions_ann: u64,
    interactions_com: u64,
    interactions_era: u64,
    nodes_max: u64,
    redexes_max: u64,
}
impl ThreadState {
    /// Reduce available redexes of each type
    pub fn run_once(&mut self, net: &mut Net) {
        // Limit the number of interactions per stage for all redex types which might allocate new nodes. This way we limit parallelism to avoid overflowing the data cache.
        const MAX_ITER_PER_ALLOC_TY: usize = 2usize.pow(9);

        self.nodes_max = self.nodes_max.max(net.nodes.len().try_into().unwrap());
        self.redexes_max = self.redexes_max.max(
            u64::try_from(
                net.redexes
                    .regular
                    .iter()
                    .map(|x| x.len())
                    .chain([net.redexes.erase.len()])
                    .sum::<usize>(),
            )
            .unwrap(),
        );

        for _ in 0..MAX_ITER_PER_ALLOC_TY {
            let Some(Redex(l, r)) = net.redexes.regular[RedexTy::Com as usize].pop() else {
                break;
            };
            {
                net.interact_com(l, r);
                self.interactions_com += 1;
            }
        }

        while let Some(Redex(l, r)) = net.redexes.regular[RedexTy::Ann as usize].pop() {
            net.interact_ann(l, r);
            self.interactions_ann += 1;
        }

        while let Some(Redex(l, r)) = net.redexes.regular[RedexTy::FolL0 as usize].pop() {
            net.interact_follow(l, r);
            self.interactions_fol += 1;
        }
        while let Some(Redex(l, r)) = net.redexes.regular[RedexTy::FolR0 as usize].pop() {
            net.interact_follow(l, r);
            self.interactions_fol += 1;
        }
        while let Some(Redex(l, r)) = net.redexes.regular[RedexTy::FolL1 as usize].pop() {
            net.interact_follow(l, r);
            self.interactions_fol += 1;
        }
        while let Some(Redex(l, r)) = net.redexes.regular[RedexTy::FolR1 as usize].pop() {
            net.interact_follow(l, r);
            self.interactions_fol += 1;
        }

        while let Some(ptr) = net.redexes.erase.pop() {
            net.interact_era(ptr);
            self.interactions_era += 1;
        }
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

// Make asm generate for the function.
/*
#[used]
static INTERACT_ANN: fn(&mut Net, left_ptr: [Ptr; 256], right_ptr: [Ptr; 256]) = |n, l, r| {
    for (l, r) in core::iter::zip(l, r) {
        Net::interact_ann(n, l, r);
    }
    core::hint::black_box(n);
};
#[used]
static INTERACT_COM: fn(&mut Net, left_ptr: [Ptr; 256], right_ptr: [Ptr; 256]) = {
    fn interact_com_batch<const N: usize>(n: &mut Net, l: [Ptr; N], r: [Ptr; N]) {
        for (l, r) in core::iter::zip(l, r) {
            Net::interact_com(n, l, r);
        }
    }
    interact_com_batch::<256>
};
#[used]
static INTERACT_ERA: fn(&mut Net, ptr: Ptr) = |mut net, ptr| {
    for _ in 0..2usize.pow(14) {
        Net::interact_era(&mut net, ptr);
    }
};
#[used]
static RUN_ONCE: fn(&mut ThreadState, &mut Net) = |ts, net| {
    for _ in 0..2usize.pow(14) {
        ThreadState::run_once(ts, net)
    }
};
*/

#[cfg(test)]
mod tests {
    use super::*;
    use crate::macros::trace;
    use core::sync::atomic::AtomicU32;

    #[test]
    fn test_viz() {
        let mut net = Net::default();
        net.nodes.push(Node {
            left: Ptr::IN(),
            right: Ptr::new(PtrTag::LeftAux0, Slot::new(1)),
        });
        net.nodes.push(Node {
            left: Ptr::IN(),
            right: Ptr::new(PtrTag::LeftAux0, Slot::new(2)),
        });
        net.redexes.regular[RedexTy::Ann as usize].push(Redex(
            Ptr::new(PtrTag::Con, Slot::new(1)),
            Ptr::new(PtrTag::Con, Slot::new(2)),
        ));
        trace!(file "end.dot",; viz::mem_to_dot(&net));
    }

    fn _2layer_con_net() -> Net {
        let mut net = Net::default();
        let make_id = |net: &mut Net| {
            let slot = net.nodes.len();
            net.nodes.push(Node {
                left: Ptr::IN(),
                right: Ptr::new(PtrTag::LeftAux0, Slot::new(slot.try_into().unwrap())),
            });
        };
        make_id(&mut net);
        make_id(&mut net);
        make_id(&mut net);
        make_id(&mut net);
        net.nodes.push(Node {
            left: Ptr::new(PtrTag::Con, Slot::new(1)),
            right: Ptr::new(PtrTag::Con, Slot::new(2)),
        });
        net.nodes.push(Node {
            left: Ptr::new(PtrTag::Con, Slot::new(3)),
            right: Ptr::new(PtrTag::Con, Slot::new(4)),
        });
        net.redexes.regular[RedexTy::Ann as usize].push(Redex(
            Ptr::new(PtrTag::Con, Slot::new(5)),
            Ptr::new(PtrTag::Con, Slot::new(6)),
        ));
        net
    }

    #[test]
    fn test_ann() {
        let mut net = _2layer_con_net();
        trace!(file "0.dot",; viz::mem_to_dot(&net));
        let Redex(l, r) = net.redexes.regular[RedexTy::Ann as usize].pop().unwrap();
        net.interact_ann(l, r);
        trace!(file "1.dot",; viz::mem_to_dot(&net));
        let Redex(l, r) = net.redexes.regular[RedexTy::Ann as usize].pop().unwrap();
        net.interact_ann(l, r);
        trace!(file "2.dot",; viz::mem_to_dot(&net));
        let Redex(l, r) = net.redexes.regular[RedexTy::FolL0 as usize].pop().unwrap();
        dbg!(&net.redexes);
        net.interact_follow(l, r);
        trace!(file "3.dot",; viz::mem_to_dot(&net));
    }

    fn _2layer_con_dup_net() -> Net {
        let mut net = Net::default();
        let make_id = |net: &mut Net| {
            let slot = net.nodes.len();
            net.nodes.push(Node {
                left: Ptr::IN(),
                right: Ptr::new(PtrTag::LeftAux0, Slot::new(slot.try_into().unwrap())),
            });
        };
        make_id(&mut net);
        make_id(&mut net);
        make_id(&mut net);
        net.nodes.push(Node {
            left: Ptr::ERA_0(),
            right: Ptr::new(PtrTag::Con, Slot::new(1)),
        });
        net.nodes.push(Node {
            left: Ptr::new(PtrTag::Con, Slot::new(2)),
            right: Ptr::new(PtrTag::Con, Slot::new(3)),
        });
        net.redexes.regular[RedexTy::Com as usize].push(Redex(
            Ptr::new(PtrTag::Con, Slot::new(4)),
            Ptr::new(PtrTag::Dup, Slot::new(5)),
        ));
        net
    }

    #[test]
    fn test_com() {
        let mut net = _2layer_con_dup_net();
        trace!(file "0.dot",; viz::mem_to_dot(&net));
        eprintln!("0 {:?}", net.free_list);
        let Redex(l, r) = net.redexes.regular[RedexTy::Com as usize].pop().unwrap();
        net.interact_com(l, r);
        trace!(file "1.dot",; viz::mem_to_dot(&net));
        eprintln!("1 {:?}", net.free_list);
        let Redex(l, r) = net.redexes.regular[RedexTy::Ann as usize].pop().unwrap();
        net.interact_ann(l, r);
        trace!(file "2.dot",; viz::mem_to_dot(&net));
        eprintln!("2 {:?}", net.free_list);
        let Redex(l, r) = net.redexes.regular[RedexTy::Com as usize].pop().unwrap();
        net.interact_com(l, r);
        trace!(file "3.dot",; viz::mem_to_dot(&net));
        eprintln!("3 {:?}", net.free_list);
        let Redex(l, r) = net.redexes.regular[RedexTy::FolL0 as usize].pop().unwrap();
        net.interact_follow(l, r);
        trace!(file "4.dot",; viz::mem_to_dot(&net));
        eprintln!("4 {:?}", net.free_list);
        let Redex(l, r) = net.redexes.regular[RedexTy::FolR0 as usize].pop().unwrap();
        net.interact_follow(l, r);
        trace!(file "5.dot",; viz::mem_to_dot(&net));
        eprintln!("5 {:?}", net.free_list);
        let Redex(l, r) = net.redexes.regular[RedexTy::FolL0 as usize].pop().unwrap();
        net.interact_follow(l, r);
        trace!(file "6.dot",; viz::mem_to_dot(&net));
        eprintln!("6 {:?}", net.free_list);
        let Redex(l, r) = net.redexes.regular[RedexTy::FolR0 as usize].pop().unwrap();
        net.interact_follow(l, r);
        trace!(file "7.dot",; viz::mem_to_dot(&net));
        eprintln!("7 {:?}", net.free_list);
        let p = net.redexes.erase.pop().unwrap();
        net.interact_era(p);
        trace!(file "8.dot",; viz::mem_to_dot(&net));
        eprintln!("8 {:?}", net.free_list);
        let Redex(l, r) = net.redexes.regular[RedexTy::Ann as usize].pop().unwrap();
        net.interact_ann(l, r);
        trace!(file "9.dot",; viz::mem_to_dot(&net));
        eprintln!("9 {:?}", net.free_list);
        let Redex(l, r) = net.redexes.regular[RedexTy::Ann as usize].pop().unwrap();
        net.interact_ann(l, r);
        trace!(file "10.dot",; viz::mem_to_dot(&net));
        eprintln!("10 {:?}", net.free_list);
        let Redex(l, r) = net.redexes.regular[RedexTy::FolL0 as usize].pop().unwrap();
        net.interact_follow(l, r);
        trace!(file "11.dot",; viz::mem_to_dot(&net));
        eprintln!("11 {:?}", net.free_list);
        let Redex(l, r) = net.redexes.regular[RedexTy::FolL0 as usize].pop().unwrap();
        net.interact_follow(l, r);
        trace!(file "12.dot",; viz::mem_to_dot(&net));
        eprintln!("12 {:?}", net.free_list);
        // Now race is resolved by existence of stages.
        let Redex(l, r) = net.redexes.regular[RedexTy::FolL1 as usize].pop().unwrap();
        net.interact_follow(l, r);
        trace!(file "13.dot",; viz::mem_to_dot(&net));
        eprintln!("13 {:?}", net.free_list);
        let p = net.redexes.erase.pop().unwrap();
        net.interact_era(dbg!(p));
        trace!(file "14.dot",; viz::mem_to_dot(&net));
        eprintln!("14 {:?}", net.free_list);
        let Redex(l, r) = net.redexes.regular[RedexTy::FolR0 as usize].pop().unwrap();
        net.interact_follow(l, r);
        trace!(file "15.dot",; viz::mem_to_dot(&net));
        eprintln!("15 {:?}", net.free_list);
        let Redex(l, r) = net.redexes.regular[RedexTy::FolL0 as usize].pop().unwrap();
        net.interact_follow(l, r);
        trace!(file "16.dot",; viz::mem_to_dot(&net));
        eprintln!("16 {:?}", net.free_list);
        let Redex(l, r) = net.redexes.regular[RedexTy::FolL0 as usize].pop().unwrap();
        net.interact_follow(l, r);
        trace!(file "17.dot",; viz::mem_to_dot(&net));
        eprintln!("17 {:?}", net.free_list);
        let p = net.redexes.erase.pop().unwrap();
        net.interact_era(p);
        trace!(file "18.dot",; viz::mem_to_dot(&net));
        eprintln!("18 {:?}", net.free_list);
        let Redex(l, r) = net.redexes.regular[RedexTy::FolL0 as usize].pop().unwrap();
        net.interact_follow(l, r);
        trace!(file "19.dot",; viz::mem_to_dot(&net));
        eprintln!("19 {:?}", net.free_list);
        let Redex(l, r) = net.redexes.regular[RedexTy::FolR0 as usize].pop().unwrap();
        net.interact_follow(l, r);
        trace!(file "20.dot",; viz::mem_to_dot(&net));
        eprintln!("20 {:?}", net.free_list);
    }

    fn infinite_reduction_net(net: &mut Net) {
        let (n1, n2) = net.alloc_node2();
        net.nodes[n1.value() as usize] = Node {
            left: Ptr::ERA_0(),
            right: Ptr::new(PtrTag::RightAux0, n2),
        };
        net.nodes[n2.value() as usize] = Node {
            left: Ptr::ERA_0(),
            right: Ptr::IN(),
        };
        net.redexes.regular[RedexTy::Com as usize]
            .push(Redex(Ptr::new(PtrTag::Dup, n1), Ptr::new(PtrTag::Con, n2)));
    }

    #[test]
    #[ignore = "bench"]
    fn speed_test() {
        let mut net = Net::default();

        // Force page faults now so they don't happen while benchmarking.
        for _ in 0..100000000 {
            net.nodes.push(Node::default());
        }
        for _ in 0..100000000 {
            net.nodes.pop();
        }
        for redex in &mut net.redexes.regular {
            for _ in 0..1000000 {
                redex.push(Redex::default());
            }
            for _ in 0..1000000 {
                redex.pop();
            }
        }
        eprintln!("page fault warmup finished\n");

        for _ in 0..100000 {
            infinite_reduction_net(&mut net);
        }
        trace!(file "start.dot",;viz::mem_to_dot(&net));
        let start = std::time::Instant::now();
        let mut thread_state = ThreadState::default();
        for _ in 0..400000 {
            thread_state.run_once(&mut net);
        }
        let end = std::time::Instant::now();
        eprintln!("Max redexes: {}", thread_state.redexes_max);
        eprintln!("Nodes max: {}", thread_state.nodes_max);
        eprintln!("Final free_list length: {}", net.free_list.len(),);
        eprintln!("Total time: {:?}", end - start);
        eprintln!(
            "---\n\
            total: {}\n\
            commute: {}\n\
            annihilate: {}\n\
            erase: {}\n\
            follow: {}",
            thread_state.interactions(),
            thread_state.interactions_com,
            thread_state.interactions_ann,
            thread_state.interactions_era,
            thread_state.interactions_fol,
        );
        eprintln!(
            "---\n\
            All MIPS: {}\n\
            Non-follow MIPS: {}",
            thread_state.interactions() as f32 / (end.duration_since(start)).as_micros() as f32,
            thread_state.non_follow_interactions() as f32
                / (end.duration_since(start)).as_micros() as f32
        );
    }

    #[test]
    #[ignore = "bench"]
    fn spinlock_latency() {
        static A: AtomicU32 = AtomicU32::new(0);
        const STORES: u32 = 10_000_000;
        let spawn_thread = move |mod_: u32, incr_on: u32| {
            let mut it = 0;
            std::thread::spawn(move || loop {
                let a = A.load(core::sync::atomic::Ordering::Relaxed);
                if a > STORES {
                    break it;
                } else if a % mod_ == incr_on {
                    // A.store(a + 1, core::sync::atomic::Ordering::Relaxed);
                    A.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
                }
                it += 1;
                core::hint::spin_loop();
            })
        };
        for n in 0..=(usize::from(std::thread::available_parallelism().unwrap())) {
            let start = std::time::Instant::now();
            let mut i = 0;
            let incrs: Vec<std::thread::JoinHandle<u32>> = core::iter::repeat_with(|| {
                let thread = spawn_thread(n.try_into().unwrap(), i.try_into().unwrap());
                i += 1;
                thread
            })
            .take(n)
            .collect();
            let incrs = incrs
                .into_iter()
                .map(|x| x.join().unwrap())
                .collect::<Vec<_>>();
            let elapsed = std::time::Instant::now().duration_since(start);
            eprintln!(
                "{n} incrs: {:?}, time: {:?}, sec_per_store: {:?}",
                incrs,
                elapsed,
                elapsed.as_secs_f32() / (STORES as f32)
            );
            A.store(0, core::sync::atomic::Ordering::Relaxed)
        }
    }
}
