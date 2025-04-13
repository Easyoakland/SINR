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

type Slot = u29;

#[bilge::bitsize(32)]
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
    // This could be bigger. Chose smaller number to give best opportunity for performance to not suffer.
    // This means we can address 2^29 nodes or 2^32 bytes since each node is u64 in size.
    // If `Ptr` was a u64 then 2^61 nodes or 2^67 bytes since each node is u128 in size, or make PtrTag 6 bits to get 2^64 bytes.
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
    pub const ZERO: Self = RedexTy::Ann;
    /// # Safety
    /// `val` < RedexTy::LEN
    pub unsafe fn from_u8(val: u8) -> Self {
        uassert!({ val as usize } < Self::LEN);
        unsafe { core::mem::transmute(val) }
    }
}

type Redexes = [UnsafeVec<Redex>; RedexTy::LEN];
type Nodes = UnsafeVec<Node>;
#[derive(Debug, Clone)]
struct Net {
    nodes: Nodes,
    redex: Redexes,
}
impl Default for Net {
    fn default() -> Self {
        #[allow(unused_mut)]
        let mut net = Self {
            // The first slot can't be pointed towards since slot 0 is an invalid slot id.
            // Can, however, use slot 0 to point towards something, e.g., the root of the net.
            nodes: UnsafeVec(vec![Node::EMP()]),
            redex: Default::default(),
        };
        #[cfg(feature = "prealloc")]
        {
            net.nodes.0.reserve(1000000000);
            for redex in &mut net.redex {
                redex.0.reserve(1000000000)
            }
        }
        net
    }
}

enum Either<A, B> {
    A(A),
    B(B),
}

impl Net {
    pub fn root(&self) -> Ptr {
        self.nodes[0].left
    }
    pub fn set_root(&mut self, root: Ptr) {
        self.nodes[0].left = root
    }
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
    pub fn free_node(&mut self, idx: Ptr) {
        // TODO add to a stack of free slot addresses.
        self.nodes[idx.slot_usize()] = Node::EMP();
    }
    #[inline]
    pub fn alloc_node(&mut self) -> Slot {
        let res = self.nodes.len();
        self.nodes.push(Node::default());
        uassert!(res <= u32::MAX as usize); // prevent check on feature=unsafe
        uassert!(res <= <Slot as Bitsized>::MAX.value() as usize);
        Slot::new(res.try_into().unwrap())
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
        let (redex, redex_ty) = Self::new_redex(left, right);
        redexes[redex_ty as usize].push(redex);
    }
    #[inline]
    pub fn new_redex(left: Ptr, right: Ptr) -> (Redex, RedexTy) {
        uassert!(left.tag() != PtrTag::_Unused);
        uassert!(right.tag() != PtrTag::_Unused);
        uassert!(left != Ptr::EMP());
        uassert!(right != Ptr::EMP());
        uassert!(left != Ptr::IN());
        uassert!(right != Ptr::IN());
        let lt = left.tag();
        let rt = right.tag();
        let (redex, redex_ty) = match () {
            // If right is a follow.
            // Safety: check in match that value is `< 4` which is `< RedexTy::LEN`.
            _ if (rt as u8) < 4 => (Redex::new(left, right), unsafe {
                RedexTy::from_u8(rt as u8)
            }),
            // If left is a follow.
            // Safety: check in match that value is `< 4` which is `< RedexTy::LEN`.
            _ if (lt as u8) < 4 => (Redex::new(right, left), unsafe {
                RedexTy::from_u8(lt as u8)
            }),
            _ if lt == rt => (Redex::new(left, right), RedexTy::Ann),
            _ => (Redex::new(left, right), RedexTy::Com),
        };
        (redex, redex_ty)
    }
    /// `ptr_to_fst` should point to `fst` with stage 1.
    #[inline]
    pub fn link_aux_ports(redexes: &mut Redexes, fst: &mut Ptr, snd: &mut Ptr, ptr_to_fst: Ptr) {
        uassert!(ptr_to_fst.tag() == PtrTag::LeftAux1 || ptr_to_fst.tag() == PtrTag::RightAux1);
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
                // TODO dealloc the false (out) side ports
            }
            (false, false) => {
                // TODO dealloc both sides ports
                Self::add_redex(redexes, *fst, *snd)
            }
        }
    }

    pub fn interact_ann(&mut self, left_ptr: Ptr, right_ptr: Ptr) {
        let l_idx = left_ptr.slot_usize();
        let r_idx = right_ptr.slot_usize();
        let [left, right] = self.nodes.get_disjoint_mut([l_idx, r_idx]);
        let (ll, lr) = (&mut left.left, &mut left.right);
        let (rl, rr) = (&mut right.left, &mut right.right);

        Self::link_aux_ports(&mut self.redex, ll, rl, {
            let mut out = left_ptr;
            out.set_tag(PtrTag::LeftAux1);
            out
        });
        Self::link_aux_ports(&mut self.redex, lr, rr, {
            let mut out = left_ptr;
            out.set_tag(PtrTag::RightAux1);
            out
        });
    }

    /// Follow target which is an auxiliary.
    /// Either it redirects to another port or it is a [`Ptr::IN()`]
    /// After following `source` is connected again.
    #[inline]
    pub fn follow_target(redexes: &mut Redexes, source: Ptr, target: &mut Ptr) {
        if *target == Ptr::IN() {
            *target = source // redirect to the new port
        } else {
            Self::add_redex(redexes, source, *target);
            // TODO free port a's original location
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

        // Leave redirect from old nodes to new node's primary port for each of ll,lr,rl,rr that was Ptr::IN(). Rest are new redexes.
        // `a` is the aux
        // `b` is the new primary port of the new node
        // ll and lr are now of type rt
        // rl and rr are now of type lt
        for (a, b) in [
            (ll, Ptr::new(rt, ll2)),
            (lr, Ptr::new(rt, lr2)),
            (rl, Ptr::new(lt, rl2)),
            (rr, Ptr::new(lt, rr2)),
        ] {
            Self::follow_target(&mut self.redex, b, a);
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

    /// Interact an redex where the `right` `Ptr`'s target is not a primary port and instead is either a redirector or an auxiliary port.
    pub fn interact_follow(&mut self, left: Ptr, right: Ptr) {
        uassert!(!matches!(
            right.tag(),
            PtrTag::Con | PtrTag::Dup | PtrTag::Era
        ));

        let target = match right.tag().aux_side() {
            LeftRight::Left => &mut self.nodes[right.slot_usize()].left,
            LeftRight::Right => &mut self.nodes[right.slot_usize()].right,
        };

        Self::follow_target(&mut self.redex, left, target);
    }
}

// Make asm generate for the function.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::macros::trace;

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
        net.redex[RedexTy::Ann as usize].push(Redex(
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
        net.redex[RedexTy::Ann as usize].push(Redex(
            Ptr::new(PtrTag::Con, Slot::new(5)),
            Ptr::new(PtrTag::Con, Slot::new(6)),
        ));
        net.set_root(Ptr::new(PtrTag::Con, Slot::new(1)));
        net
    }

    #[test]
    fn test_ann() {
        let mut net = _2layer_con_net();
        trace!(file "0.dot",; viz::mem_to_dot(&net));
        let Redex(l, r) = net.redex[RedexTy::Ann as usize].pop().unwrap();
        net.interact_ann(l, r);
        trace!(file "1.dot",; viz::mem_to_dot(&net));
        let Redex(l, r) = net.redex[RedexTy::Ann as usize].pop().unwrap();
        net.interact_ann(l, r);
        trace!(file "2.dot",; viz::mem_to_dot(&net));
        let Redex(l, r) = net.redex[RedexTy::FolL0 as usize].pop().unwrap();
        dbg!(&net.redex);
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
        make_id(&mut net);
        net.nodes.push(Node {
            left: Ptr::new(PtrTag::Con, Slot::new(1)),
            right: Ptr::new(PtrTag::Con, Slot::new(2)),
        });
        net.nodes.push(Node {
            left: Ptr::new(PtrTag::Con, Slot::new(3)),
            right: Ptr::new(PtrTag::Con, Slot::new(4)),
        });
        net.redex[RedexTy::Com as usize].push(Redex(
            Ptr::new(PtrTag::Con, Slot::new(5)),
            Ptr::new(PtrTag::Dup, Slot::new(6)),
        ));
        net.set_root(Ptr::new(PtrTag::Con, Slot::new(1)));
        net
    }

    #[test]
    fn test_com() {
        let mut net = _2layer_con_dup_net();
        trace!(file "0.dot",; viz::mem_to_dot(&net));
        let Redex(l, r) = net.redex[RedexTy::Com as usize].pop().unwrap();
        net.interact_com(l, r);
        trace!(file "1.dot",; viz::mem_to_dot(&net));
        let Redex(l, r) = net.redex[RedexTy::Ann as usize].pop().unwrap();
        net.interact_ann(l, r);
        trace!(file "2.dot",; viz::mem_to_dot(&net));
        let Redex(l, r) = net.redex[RedexTy::Com as usize].pop().unwrap();
        net.interact_com(l, r);
        trace!(file "3.dot",; viz::mem_to_dot(&net));
        let Redex(l, r) = net.redex[RedexTy::FolL0 as usize].pop().unwrap();
        net.interact_follow(l, r);
        trace!(file "4.dot",; viz::mem_to_dot(&net));
        let Redex(l, r) = net.redex[RedexTy::FolR0 as usize].pop().unwrap();
        net.interact_follow(l, r);
        trace!(file "5.dot",; viz::mem_to_dot(&net));
        let Redex(l, r) = net.redex[RedexTy::FolL0 as usize].pop().unwrap();
        net.interact_follow(l, r);
        trace!(file "6.dot",; viz::mem_to_dot(&net));
        let Redex(l, r) = net.redex[RedexTy::FolR0 as usize].pop().unwrap();
        net.interact_follow(l, r);
        trace!(file "7.dot",; viz::mem_to_dot(&net));
        let Redex(l, r) = net.redex[RedexTy::Com as usize].pop().unwrap();
        net.interact_com(l, r);
        trace!(file "8.dot",; viz::mem_to_dot(&net));
        let Redex(l, r) = net.redex[RedexTy::FolL0 as usize].pop().unwrap();
        net.interact_follow(l, r);
        trace!(file "9.dot",; viz::mem_to_dot(&net));
        let Redex(l, r) = net.redex[RedexTy::FolL0 as usize].pop().unwrap();
        net.interact_follow(l, r);
        trace!(file "10.dot",; viz::mem_to_dot(&net));
        let Redex(l, r) = net.redex[RedexTy::Ann as usize].pop().unwrap();
        net.interact_ann(l, r);
        trace!(file "11.dot",; viz::mem_to_dot(&net));
        let Redex(l, r) = net.redex[RedexTy::Ann as usize].pop().unwrap();
        net.interact_ann(l, r);
        trace!(file "12.dot",; viz::mem_to_dot(&net)); // note LeftAux1 used here in 12L->16L1
        let Redex(l, r) = net.redex[RedexTy::Ann as usize].pop().unwrap();
        net.interact_ann(l, r);
        trace!(file "13.dot",; viz::mem_to_dot(&net));
        let Redex(l, r) = net.redex[RedexTy::Ann as usize].pop().unwrap();
        net.interact_ann(l, r);
        trace!(file "14.dot",; viz::mem_to_dot(&net));
        let Redex(l, r) = net.redex[RedexTy::FolR0 as usize].pop().unwrap();
        net.interact_follow(l, r);
        trace!(file "15.dot",; viz::mem_to_dot(&net));
        let Redex(l, r) = net.redex[RedexTy::FolL0 as usize].pop().unwrap();
        // Note this will never happen
        // but this is the easiest way to confirm that the stage0,1 technique will prevent the race.
        // If this doesn't happen the bad case will trivially not occur, but I want to try the bad case.
        // This also doesn't break anything since the redex that's being flipped is R0, L0 so interact_follow should work anyway.
        let (r, l) = (l, r);
        net.interact_follow(l, r);
        trace!(file "16.dot",; viz::mem_to_dot(&net));
        let Redex(l, r) = net.redex[RedexTy::FolL0 as usize].pop().unwrap();
        net.interact_follow(l, r);
        trace!(file "17.dot",; viz::mem_to_dot(&net));
        // And here we see the stage method will prevent a race. Either RedexTy::FolL0 or RedexTy::FolL1 will run next, but not both simultaneously.
        // Admittedly, this isn't a great example because of the circular path of the node onto itself, and the fact that the node is only in 1 redex, but the hopefully the idea is clear.
        // If it were in two redexes, one that wanted to follow L0 and one that wanted to follow L1, then they wouldn't conflict because they are in different stages.
        let Redex(l, r) = net.redex[RedexTy::FolL1 as usize].pop().unwrap();
        net.interact_follow(l, r);
        trace!(file "18.dot",; viz::mem_to_dot(&net));
        let Redex(l, r) = net.redex[RedexTy::FolL0 as usize].pop().unwrap();
        net.interact_follow(l, r);
        trace!(file "19.dot",; viz::mem_to_dot(&net));
        let Redex(l, r) = net.redex[RedexTy::FolL0 as usize].pop().unwrap();
        net.interact_follow(l, r);
        trace!(file "20.dot",; viz::mem_to_dot(&net));
        let Redex(l, r) = net.redex[RedexTy::Ann as usize].pop().unwrap();
        net.interact_ann(l, r);
        trace!(file "21.dot",; viz::mem_to_dot(&net)); // and here's an R1 generated
        let Redex(l, r) = net.redex[RedexTy::FolL0 as usize].pop().unwrap();
        net.interact_ann(l, r);
        trace!(file "22.dot",; viz::mem_to_dot(&net));
        let Redex(l, r) = net.redex[RedexTy::FolR0 as usize].pop().unwrap();
        net.interact_follow(l, r);
        trace!(file "23.dot",; viz::mem_to_dot(&net));
        let Redex(l, r) = net.redex[RedexTy::FolR1 as usize].pop().unwrap();
        net.interact_follow(l, r);
        trace!(file "24.dot",; viz::mem_to_dot(&net));
    }

    fn infinite_reduction_net(net: &mut Net) {
        let (n1, n2, e1, e2) = net.alloc_node4();
        net.nodes[n1.value() as usize] = Node {
            left: Ptr::new(PtrTag::Era, e1),
            right: Ptr::new(PtrTag::RightAux0, n2),
        };
        net.nodes[n2.value() as usize] = Node {
            left: Ptr::new(PtrTag::Era, e2),
            right: Ptr::IN(),
        };
        net.nodes[e1.value() as usize] = Node {
            left: Ptr::IN(),
            right: Ptr::new(PtrTag::LeftAux0, e1),
        };
        net.nodes[e2.value() as usize] = Node {
            left: Ptr::IN(),
            right: Ptr::new(PtrTag::LeftAux0, e2),
        };
        net.redex[RedexTy::Com as usize]
            .push(Redex(Ptr::new(PtrTag::Dup, n1), Ptr::new(PtrTag::Con, n2)));
    }

    #[test]
    fn speed_test() {
        let mut net = Net::default();

        // Force page faults now so they don't happen while benchmarking.
        for _ in 0..1000000000 {
            net.nodes.push(Node::default());
        }
        for _ in 0..1000000000 {
            net.nodes.pop();
        }
        for redex in &mut net.redex {
            for _ in 0..1000000 {
                redex.push(Redex::default());
            }
            for _ in 0..1000000 {
                redex.pop();
            }
        }
        eprintln!("page fault warmup finished");

        for _ in 0..111 {
            infinite_reduction_net(&mut net);
        }
        let mut interactions = 0;
        let mut interactions_com = 0;
        let mut interactions_ann = 0;
        let mut interactions_fol = 0;
        let mut redexes_max = 0usize;
        let mut nodes_max = 0usize;
        let start = std::time::Instant::now();
        const ITERS: usize = 400000;
        for _ in 0..ITERS {
            nodes_max = nodes_max.max(net.nodes.len());
            redexes_max = redexes_max.max(net.redex.iter().flat_map(|x| &x.0).count());
            // eprintln!(
            //     "{:0>2?}",
            //     net.redex.iter().map(|x| x.len()).collect::<Vec<_>>()
            // );
            while let Some(Redex(l, r)) = net.redex[RedexTy::Ann as usize].0.pop() {
                net.interact_ann(l, r);
                interactions += 1;
                interactions_ann += 1;
            }
            while let Some(Redex(l, r)) = net.redex[RedexTy::Com as usize].0.pop() {
                net.interact_com(l, r);
                interactions += 1;
                interactions_com += 1;
            }
            while let Some(Redex(l, r)) = net.redex[RedexTy::FolL0 as usize].0.pop() {
                net.interact_follow(l, r);
                interactions += 1;
                interactions_fol += 1;
            }
            while let Some(Redex(l, r)) = net.redex[RedexTy::FolR0 as usize].0.pop() {
                net.interact_follow(l, r);
                interactions += 1;
                interactions_fol += 1;
            }
            while let Some(Redex(l, r)) = net.redex[RedexTy::FolL1 as usize].0.pop() {
                net.interact_follow(l, r);
                interactions += 1;
                interactions_fol += 1;
            }
            while let Some(Redex(l, r)) = net.redex[RedexTy::FolR1 as usize].0.pop() {
                net.interact_follow(l, r);
                interactions += 1;
                interactions_fol += 1;
            }
        }
        let end = std::time::Instant::now();
        eprintln!("Max redexes {}", redexes_max);
        eprintln!("Nodes max {}", nodes_max);
        eprintln!("Total time: {:?}", end - start);
        eprintln!(
            "---\n\
            total: {interactions}\n\
            commute: {interactions_com}\n\
            annihilate: {interactions_ann}\n\
            follow: {interactions_fol}",
        );
        eprintln!(
            "---\n\
            All MIPS: {}\n\
            Non-follow MIPS: {}",
            interactions as f32 / (end.duration_since(start)).as_micros() as f32,
            (interactions - interactions_fol) as f32
                / (end.duration_since(start)).as_micros() as f32
        );
    }
}
