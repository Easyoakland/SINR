//! # SINR: Staged Interaction Net Runtime
//!
//! Staged, Parallel, and SIMD capable Interpreter of Symmetric Interaction Combinators (SIC) + extensions.
//!
//! # Layout
//! ## Nodes
//! Nodes are represented as 2 tagged pointers and redexes as 2 tagged pointers. Nodes may have 0, 1, or 2 auxiliary ports. Although all currently implemented nodes are 2-ary others should work.
//! ## Node tagged pointers
//! Tagged pointers contain a target and the type of the target they point to.
//! A Pointer is either "out" in which case its target is one of the [`PtrTag`] variants, `IN`, in which case it has no target, or `EMP` in which case it represents unallocated memory. The [`PtrTag`] represents the different principal port types, and the 4 auxiliary port types.
//!
//! If you are familiar with inets you may wonder why 2-ary nodes have 4 separate auxiliary types instead of 2. The difference here is that auxiliary nodes have an associated stage, which dictates the synchronization of interactions. This makes it possible to handle annihilation interactions and re-use the annihilating nodes to redirect incoming pointers from the rest of the net in the case that two incoming pointers are to be linked. In this particular case one of the two auxiliary ports points to the other with a stage of 1 instead of 0, and therefore avoids racing when performing follows.
//!
//! The reason that left and right follow interactions must be separated is to deallocate nodes only once both pointers in the node are no longer in use. This simplifies memory deallocation (TODO) by preventing any fragmentation (holes) less than a node in size. If this turns out to not be sufficiently useful, then left and right auxiliary follows can be performed simultaneously, only synchronizing between stage 0, 1, and principal interaction.
//!
//! # Reduction
//! All redexes are stored into one of several redex buffers based upon the redex's interaction type, [`RedexTy`]. The regular commute and annihilate are two interactions. So are the 4 possible follow operations depending on the type of the auxiliary target. By separating the operations into these 2+4 types it becomes possible to perform all operations of the same type with minimal branching (for SIMD) and without atomic synchronization (for CPU SIMD and general perf improvements). All threads and SIMD lanes operate reduce the same interaction type at the same time. When one of the threads runs out of reductions of that type (or some other signal such as number of reductions) all the threads synchronize their memory and start simultaneously reducing operations of a new type.
//! # Future possibilities
//! ## Amb nodes
//! Since this design allows for synchronizing multiple pointers to a single port without atomics, implementing amb (a 2 principal port node) should be as simple as an FollowAmb0 and FollowAmb1 interactions.
//! ## Global ref nodes
//! Global nets can be supported simply by adding the interaction type as usual. To enable SIMD, the nets should be stored with offsets local to 0. Then, when instantiating the net, allocate a continuous block the size of the global subnet and add the start of the block to all the offsets of pointers in the global subnet.
//! # TODO
//! - [x] Basic interactions
//! - [ ] Single-threaded scalar implementation with branching
//! - [ ] Memory deallocation and reclamation.
//! - [ ] Minimize branching using lookup tables
//! - [ ] SIMD
//! - [ ] Multiple threads

#![feature(portable_simd)]
#![feature(get_disjoint_mut_helpers)]
// TODO remove these
#![allow(dead_code)]
#![allow(unused_imports)]

mod left_right;
mod macros;
mod unsafe_vec;
mod viz;

use bilge::prelude::*;
use core::{array, u32};
use left_right::LeftRight;
use macros::{trace, uassert, uunreachable};
use std::simd::{
    self, cmp::SimdPartialEq, num::SimdUint, ptr::SimdConstPtr, LaneCount, Mask, Simd,
    SupportedLaneCount,
};
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
    /// If currently stage 0 makes stage 1, and vice versa
    // TODO perf: use `val ^ 0b10` since this doesn't optimize out branches on its own.
    // # Safety
    // Assumes this is an aux.
    pub fn aux_swap_stage(self) -> Self {
        match self {
            PtrTag::LeftAux0 => PtrTag::LeftAux1,
            PtrTag::LeftAux1 => PtrTag::LeftAux0,
            PtrTag::RightAux0 => PtrTag::RightAux1,
            PtrTag::RightAux1 => PtrTag::RightAux0,
            _ => uunreachable!(),
        }
    }
    // left -> right and right -> left
    // TODO perf: use `val ^ 0b01` since this doesn't optimize out branches on its own.
    // # Safety
    // Assumes this is an aux.
    pub fn aux_swap_side(self) -> Self {
        match self {
            PtrTag::LeftAux0 => PtrTag::RightAux0,
            PtrTag::LeftAux1 => PtrTag::RightAux1,
            PtrTag::RightAux0 => PtrTag::LeftAux0,
            PtrTag::RightAux1 => PtrTag::LeftAux1,
            _ => uunreachable!(),
        }
    }
    /// Keep side, set stage to 1
    // TODO perf: use `val & 0b10`
    // # Safety
    // Assumes this is an aux
    pub fn set_s1(self) -> Self {
        match self {
            PtrTag::LeftAux0 | PtrTag::LeftAux1 => PtrTag::LeftAux1,
            PtrTag::RightAux0 | PtrTag::RightAux1 => PtrTag::RightAux1,
            _ => uunreachable!(),
        }
    }
    // TODO perf: use `if val & 1 != 0 {Right} else {Left}` or `LeftRight::from(val & 1 as bool)`
    // # Safety
    // Assumes this is an aux
    pub fn aux_side(&self) -> LeftRight {
        match self {
            PtrTag::LeftAux0 | PtrTag::LeftAux1 => LeftRight::Left,
            PtrTag::RightAux0 | PtrTag::RightAux1 => LeftRight::Right,
            _ => uunreachable!(),
        }
    }
}

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
    slot: u29,
}

impl Default for Ptr {
    fn default() -> Self {
        Self {
            value: Default::default(),
        }
    }
}
impl Ptr {
    // ---
    // The first slot is not valid target of `Ptr` because it represents these special values
    // ---
    /// An empty, unused ptr slot. Everything after a calloc should be EMP.
    pub const EMP: fn() -> Ptr = || Ptr { value: 0 };
    /// Auxiliary which is pointed *to* but doesn't point out.
    pub const IN: fn() -> Ptr = || Ptr::new(PtrTag::from(u3::new(1)), u29::new(0));
    #[inline]
    pub fn slot_u32(self) -> u32 {
        self.slot().value()
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

/// - Commute: Combinator i <> Combinator j (different label)
/// - Annihilate: Combinator i <> Combinator i (same label)
/// - Follow l0: ?? <> LeftAux0
/// - Follow l1: ?? <> LeftAux1
/// - Follow r0: ?? <> RightAux0
/// - Follow r1: ?? <> RightAux1
/// Note to avoid races only one of these redex queues can be operated on simultaneously.
/// On the positive side, the queue being operated on can do so without any atomics whatsoever (including the follow wires rules).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
enum RedexTy {
    #[default]
    Ann,
    Com,
    FolL0,
    FolL1,
    FolR0,
    FolR1,
}
impl RedexTy {
    pub const LEN: usize = 6;
    pub const ZERO: Self = RedexTy::Ann;
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
    pub fn read(&self, ptr: Ptr) -> Either<Node, Ptr> {
        match ptr.tag() {
            PtrTag::LeftAux0 | PtrTag::LeftAux1 => Either::B(self.nodes[ptr.slot_usize()].left),
            PtrTag::RightAux0 | PtrTag::RightAux1 => Either::B(self.nodes[ptr.slot_usize()].right),
            PtrTag::Era | PtrTag::Con | PtrTag::Dup => Either::A(self.nodes[ptr.slot_usize()]),
            PtrTag::_Unused => uunreachable!(),
        }
    }
    // The other aux of a `Node`
    pub fn other_aux(&self, ptr: Ptr) -> Ptr {
        let node = self.nodes[ptr.slot_usize()];
        uassert!(matches!(
            ptr.tag(),
            PtrTag::RightAux0 | PtrTag::RightAux1 | PtrTag::LeftAux0 | PtrTag::LeftAux1
        ));
        // TODO xor instead of branch.
        if matches!(ptr.tag(), PtrTag::RightAux0 | PtrTag::RightAux1) {
            node.left
        } else {
            node.right
        }
    }
    pub fn free_node(&mut self, idx: Ptr) {
        // TODO add to a stack of free slot addresses.
        self.nodes[idx.slot_usize()] = Node::EMP();
    }
    #[inline]
    pub fn alloc_node(&mut self) -> u29 {
        let res = self.nodes.len();
        self.nodes.push(Node::default());
        uassert!(res <= u32::MAX as usize); // prevent check on feature=unsafe
        uassert!(res <= <u29 as Bitsized>::MAX.value() as usize);
        u29::new(res.try_into().unwrap())
    }
    #[inline]
    pub fn alloc_node2(&mut self) -> (u29, u29) {
        (self.alloc_node(), self.alloc_node())
    }
    #[inline]
    pub fn alloc_node4(&mut self) -> (u29, u29, u29, u29) {
        let ((a, b), (c, d)) = (self.alloc_node2(), self.alloc_node2());
        (a, b, c, d)
    }
    #[inline]
    pub fn add_redex(redexes: &mut Redexes, left: Ptr, right: Ptr) {
        // TODO: find a non-branching algorithm for this. Probably going to be a LUT.
        uassert!(left.tag() != PtrTag::_Unused);
        uassert!(right.tag() != PtrTag::_Unused);
        uassert!(left != Ptr::EMP());
        uassert!(right != Ptr::EMP());
        uassert!(left != Ptr::IN());
        uassert!(right != Ptr::IN());
        let (redex, redex_ty) = match (left.tag(), right.tag()) {
            (_, PtrTag::LeftAux0) => (Redex::new(left, right), RedexTy::FolL0),
            (_, PtrTag::LeftAux1) => (Redex::new(left, right), RedexTy::FolL1),
            (_, PtrTag::RightAux0) => (Redex::new(left, right), RedexTy::FolR0),
            (_, PtrTag::RightAux1) => (Redex::new(left, right), RedexTy::FolR1),
            (PtrTag::LeftAux0, _) => (Redex::new(right, left), RedexTy::FolL0),
            (PtrTag::LeftAux1, _) => (Redex::new(right, left), RedexTy::FolL1),
            (PtrTag::RightAux0, _) => (Redex::new(right, left), RedexTy::FolR0),
            (PtrTag::RightAux1, _) => (Redex::new(right, left), RedexTy::FolR1),
            (x, y) if x == y => (Redex::new(left, right), RedexTy::Ann),
            (_, _) => (Redex::new(left, right), RedexTy::Com),
        };
        redexes[redex_ty as usize].push(redex);
    }
    #[inline]
    pub fn redex_lut_batch<const N: usize>(
        left: [Ptr; N],
        right: [Ptr; N],
    ) -> ([Redex; N], [RedexTy; N]) {
        const LUT: [[RedexTy; PtrTag::LEN]; PtrTag::LEN] = {
            let mut out = [[RedexTy::ZERO; PtrTag::LEN]; PtrTag::LEN];
            use PtrTag::*;
            use RedexTy::*;
            out[Con as usize][Con as usize] = Ann;
            out[Con as usize][Dup as usize] = Com;
            out[Con as usize][Era as usize] = Com;
            out[Con as usize][LeftAux0 as usize] = FolL0;
            out[Con as usize][LeftAux1 as usize] = FolL1;
            out[Con as usize][RightAux0 as usize] = FolR0;
            out[Con as usize][RightAux1 as usize] = FolR1;

            out[Dup as usize][Con as usize] = Com;
            out[Dup as usize][Dup as usize] = Ann;
            out[Dup as usize][Era as usize] = Com;
            out[Dup as usize][LeftAux0 as usize] = FolL0;
            out[Dup as usize][LeftAux1 as usize] = FolL1;
            out[Dup as usize][RightAux0 as usize] = FolR0;
            out[Dup as usize][RightAux1 as usize] = FolR1;

            out[Era as usize][Con as usize] = Com;
            out[Era as usize][Dup as usize] = Com;
            out[Era as usize][Era as usize] = Ann;
            out[Era as usize][LeftAux0 as usize] = FolL0;
            out[Era as usize][LeftAux1 as usize] = FolL1;
            out[Era as usize][RightAux0 as usize] = FolR0;
            out[Era as usize][RightAux1 as usize] = FolR1;

            out[LeftAux0 as usize][Con as usize] = RedexTy::ZERO;
            out[LeftAux0 as usize][Dup as usize] = RedexTy::ZERO;
            out[LeftAux0 as usize][Era as usize] = RedexTy::ZERO;
            out[LeftAux0 as usize][LeftAux0 as usize] = FolL0;
            out[LeftAux0 as usize][LeftAux1 as usize] = FolL1;
            out[LeftAux0 as usize][RightAux0 as usize] = FolR0;
            out[LeftAux0 as usize][RightAux1 as usize] = FolR1;

            out[LeftAux1 as usize][Con as usize] = RedexTy::ZERO;
            out[LeftAux1 as usize][Dup as usize] = RedexTy::ZERO;
            out[LeftAux1 as usize][Era as usize] = RedexTy::ZERO;
            out[LeftAux1 as usize][LeftAux0 as usize] = FolL0;
            out[LeftAux1 as usize][LeftAux1 as usize] = FolL1;
            out[LeftAux1 as usize][RightAux0 as usize] = FolR0;
            out[LeftAux1 as usize][RightAux1 as usize] = FolR1;

            out[RightAux0 as usize][Con as usize] = RedexTy::ZERO;
            out[RightAux0 as usize][Dup as usize] = RedexTy::ZERO;
            out[RightAux0 as usize][Era as usize] = RedexTy::ZERO;
            out[RightAux0 as usize][LeftAux0 as usize] = FolL0;
            out[RightAux0 as usize][LeftAux1 as usize] = FolL1;
            out[RightAux0 as usize][RightAux0 as usize] = FolR0;
            out[RightAux0 as usize][RightAux1 as usize] = FolR1;

            out[RightAux1 as usize][Con as usize] = RedexTy::ZERO;
            out[RightAux1 as usize][Dup as usize] = RedexTy::ZERO;
            out[RightAux1 as usize][Era as usize] = RedexTy::ZERO;
            out[RightAux1 as usize][LeftAux0 as usize] = FolL0;
            out[RightAux1 as usize][LeftAux1 as usize] = FolL1;
            out[RightAux1 as usize][RightAux0 as usize] = FolR0;
            out[RightAux1 as usize][RightAux1 as usize] = FolR1;
            out
        };
        let redexes = core::array::from_fn(|i| {
            if !right[i].tag().is_aux() {
                Redex(right[i], left[i])
            } else {
                Redex(left[i], right[i])
            }
        });
        let ty = core::array::from_fn(|i| {
            let l_idx = left[i].tag() as usize;
            let r_idx = right[i].tag() as usize;
            uassert!(l_idx < RedexTy::LEN);
            uassert!(r_idx < RedexTy::LEN);
            LUT[l_idx][r_idx]
        });
        (redexes, ty)
    }
    #[inline]
    pub fn redex_lut_batch_manual<const N: usize>(
        left: [Ptr; N],
        right: [Ptr; N],
    ) -> ([Redex; N], [RedexTy; N])
    where
        LaneCount<N>: SupportedLaneCount,
    {
        const LUT: [[RedexTy; PtrTag::LEN]; PtrTag::LEN] = {
            let mut out = [[RedexTy::ZERO; PtrTag::LEN]; PtrTag::LEN];
            use PtrTag::*;
            use RedexTy::*;
            out[Con as usize][Con as usize] = Ann;
            out[Con as usize][Dup as usize] = Com;
            out[Con as usize][Era as usize] = Com;
            out[Con as usize][LeftAux0 as usize] = FolL0;
            out[Con as usize][LeftAux1 as usize] = FolL1;
            out[Con as usize][RightAux0 as usize] = FolR0;
            out[Con as usize][RightAux1 as usize] = FolR1;

            out[Dup as usize][Con as usize] = Com;
            out[Dup as usize][Dup as usize] = Ann;
            out[Dup as usize][Era as usize] = Com;
            out[Dup as usize][LeftAux0 as usize] = FolL0;
            out[Dup as usize][LeftAux1 as usize] = FolL1;
            out[Dup as usize][RightAux0 as usize] = FolR0;
            out[Dup as usize][RightAux1 as usize] = FolR1;

            out[Era as usize][Con as usize] = Com;
            out[Era as usize][Dup as usize] = Com;
            out[Era as usize][Era as usize] = Ann;
            out[Era as usize][LeftAux0 as usize] = FolL0;
            out[Era as usize][LeftAux1 as usize] = FolL1;
            out[Era as usize][RightAux0 as usize] = FolR0;
            out[Era as usize][RightAux1 as usize] = FolR1;

            out[LeftAux0 as usize][Con as usize] = RedexTy::ZERO;
            out[LeftAux0 as usize][Dup as usize] = RedexTy::ZERO;
            out[LeftAux0 as usize][Era as usize] = RedexTy::ZERO;
            out[LeftAux0 as usize][LeftAux0 as usize] = FolL0;
            out[LeftAux0 as usize][LeftAux1 as usize] = FolL1;
            out[LeftAux0 as usize][RightAux0 as usize] = FolR0;
            out[LeftAux0 as usize][RightAux1 as usize] = FolR1;

            out[LeftAux1 as usize][Con as usize] = RedexTy::ZERO;
            out[LeftAux1 as usize][Dup as usize] = RedexTy::ZERO;
            out[LeftAux1 as usize][Era as usize] = RedexTy::ZERO;
            out[LeftAux1 as usize][LeftAux0 as usize] = FolL0;
            out[LeftAux1 as usize][LeftAux1 as usize] = FolL1;
            out[LeftAux1 as usize][RightAux0 as usize] = FolR0;
            out[LeftAux1 as usize][RightAux1 as usize] = FolR1;

            out[RightAux0 as usize][Con as usize] = RedexTy::ZERO;
            out[RightAux0 as usize][Dup as usize] = RedexTy::ZERO;
            out[RightAux0 as usize][Era as usize] = RedexTy::ZERO;
            out[RightAux0 as usize][LeftAux0 as usize] = FolL0;
            out[RightAux0 as usize][LeftAux1 as usize] = FolL1;
            out[RightAux0 as usize][RightAux0 as usize] = FolR0;
            out[RightAux0 as usize][RightAux1 as usize] = FolR1;

            out[RightAux1 as usize][Con as usize] = RedexTy::ZERO;
            out[RightAux1 as usize][Dup as usize] = RedexTy::ZERO;
            out[RightAux1 as usize][Era as usize] = RedexTy::ZERO;
            out[RightAux1 as usize][LeftAux0 as usize] = FolL0;
            out[RightAux1 as usize][LeftAux1 as usize] = FolL1;
            out[RightAux1 as usize][RightAux0 as usize] = FolR0;
            out[RightAux1 as usize][RightAux1 as usize] = FolR1;
            out
        };
        let redexes = core::array::from_fn(|i| {
            if !right[i].tag().is_aux() {
                Redex(right[i], left[i])
            } else {
                Redex(left[i], right[i])
            }
        });
        // Safety: Ptr is repr(transparent) of u32.
        let left: [u32; N] = unsafe { core::mem::transmute_copy(&left) };
        let right: [u32; N] = unsafe { core::mem::transmute_copy(&right) };
        let left = Simd::from_array(left);
        let right = Simd::from_array(right);
        let idxs = left.cast::<usize>()
            & Simd::splat(PtrTag::BITS) * Simd::splat(PtrTag::LEN) + right.cast::<usize>()
            & Simd::splat(PtrTag::BITS);
        let lut: &[RedexTy] = LUT.as_flattened();
        let lut: &[u8] = unsafe { core::mem::transmute(lut) };
        // Safety: by construction the lut is large enough for all tag values.
        // TODO check if gather_select_unchecked is faster than gather_or
        let ty =
            unsafe { Simd::gather_select_unchecked(lut, Mask::splat(true), idxs, Simd::splat(0)) };
        let ty = ty.to_array();
        let ty: [RedexTy; N] = unsafe { core::mem::transmute_copy(&ty) };
        (redexes, ty)
    }
    /// `ptr_to_fst` should point to `fst` with stage 1.
    #[inline]
    pub fn link_aux_ports(redexes: &mut Redexes, fst: &mut Ptr, snd: &mut Ptr, ptr_to_fst: Ptr) {
        uassert!(ptr_to_fst.tag() == PtrTag::LeftAux1 || ptr_to_fst.tag() == PtrTag::RightAux1);
        // All cases either swap (heterogenous cases) or or don't care that they swap (homogenous cases).
        // TODO perf: see if this is an anti-optimization and it is better to check the below and dispatch with 3 masks.
        core::mem::swap(fst, snd);
        // TODO how to remove this branching? Is it worthwhile to do all of them masked?
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
            // TODO remove branch or use a mask. Running `*a=b` masked should be very low-cost.
            if *a == Ptr::IN() {
                *a = b // redirect to the new principal port
            } else {
                Self::add_redex(&mut self.redex, b, *a);
                // TODO free port a's original location
            }
        }

        // Make new nodes and link their aux together so each has 1 out and 1 in.
        // All nodes start at stage 0 since handling left and right in separate stages is sufficient to avoid races here since no *port* has 2 incoming pointers, i.e., no node with 2 incoming pointers to same aux.
        // TODO does this prevent SIMD? What if uassert all the indices are non-equal?
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
        if *target == Ptr::IN() {
            // TODO this is the same code as in `interact_comm`
            // if target isn't a redirect
            *target = left;
        } else {
            // otherwise it's a redirect which must be followed again.
            Self::add_redex(&mut self.redex, left, *target);
            // TODO free original target port location
        }
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

#[used]
static ADD_REDEX: fn(&mut Redexes, left_ptr: [Ptr; 256], right_ptr: [Ptr; 256]) = {
    fn add_redex_batch(n: &mut Redexes, left_ptr: [Ptr; 256], right_ptr: [Ptr; 256]) {
        for (l, r) in core::iter::zip(left_ptr, right_ptr) {
            Net::add_redex(n, l, r);
        }
        core::hint::black_box(n);
    }
    add_redex_batch
};
#[used]
static ADD_REDEX_LUT_BATCH: fn(
    left_ptr: [Ptr; 64],
    right_ptr: [Ptr; 64],
) -> ([Redex; 64], [RedexTy; 64]) = Net::redex_lut_batch::<64>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_viz() {
        let mut net = Net::default();
        net.nodes.push(Node {
            left: Ptr::IN(),
            right: Ptr::new(PtrTag::LeftAux0, u29::new(1)),
        });
        net.nodes.push(Node {
            left: Ptr::IN(),
            right: Ptr::new(PtrTag::LeftAux0, u29::new(2)),
        });
        net.redex[RedexTy::Ann as usize].push(Redex(
            Ptr::new(PtrTag::Con, u29::new(1)),
            Ptr::new(PtrTag::Con, u29::new(2)),
        ));
        trace!(file "end.dot",; viz::mem_to_dot(&net));
    }

    fn _2layer_con_net() -> Net {
        let mut net = Net::default();
        let make_id = |net: &mut Net| {
            let slot = net.nodes.len();
            net.nodes.push(Node {
                left: Ptr::IN(),
                right: Ptr::new(PtrTag::LeftAux0, u29::new(slot.try_into().unwrap())),
            });
        };
        make_id(&mut net);
        make_id(&mut net);
        make_id(&mut net);
        make_id(&mut net);
        net.nodes.push(Node {
            left: Ptr::new(PtrTag::Con, u29::new(1)),
            right: Ptr::new(PtrTag::Con, u29::new(2)),
        });
        net.nodes.push(Node {
            left: Ptr::new(PtrTag::Con, u29::new(3)),
            right: Ptr::new(PtrTag::Con, u29::new(4)),
        });
        net.redex[RedexTy::Ann as usize].push(Redex(
            Ptr::new(PtrTag::Con, u29::new(5)),
            Ptr::new(PtrTag::Con, u29::new(6)),
        ));
        net.set_root(Ptr::new(PtrTag::Con, u29::new(1)));
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
                right: Ptr::new(PtrTag::LeftAux0, u29::new(slot.try_into().unwrap())),
            });
        };
        make_id(&mut net);
        make_id(&mut net);
        make_id(&mut net);
        make_id(&mut net);
        net.nodes.push(Node {
            left: Ptr::new(PtrTag::Con, u29::new(1)),
            right: Ptr::new(PtrTag::Con, u29::new(2)),
        });
        net.nodes.push(Node {
            left: Ptr::new(PtrTag::Con, u29::new(3)),
            right: Ptr::new(PtrTag::Con, u29::new(4)),
        });
        net.redex[RedexTy::Com as usize].push(Redex(
            Ptr::new(PtrTag::Con, u29::new(5)),
            Ptr::new(PtrTag::Dup, u29::new(6)),
        ));
        net.set_root(Ptr::new(PtrTag::Con, u29::new(1)));
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

    fn infinite_reduction_net() -> Net {
        let mut net = Net::default();
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
        net
    }

    #[test]
    fn speed_test() {
        let mut net = infinite_reduction_net();
        let mut i = 0;
        let mut redexes_avg = 0usize;
        let mut redexes_max = 0usize;
        let start = std::time::Instant::now();
        for _ in 0..10000000 {
            redexes_avg += net.redex.iter().flat_map(|x| &x.0).count();
            redexes_max = redexes_max.max(net.redex.iter().flat_map(|x| &x.0).count());
            if let Some(Redex(l, r)) = net.redex[RedexTy::Ann as usize].0.pop() {
                net.interact_ann(l, r);
                i += 1;
            }
            if let Some(Redex(l, r)) = net.redex[RedexTy::Com as usize].0.pop() {
                net.interact_com(l, r);
                i += 1;
            }
            if let Some(Redex(l, r)) = net.redex[RedexTy::FolL0 as usize].0.pop() {
                net.interact_follow(l, r);
                i += 1;
            }
            if let Some(Redex(l, r)) = net.redex[RedexTy::FolR0 as usize].0.pop() {
                net.interact_follow(l, r);
                i += 1;
            }
            if let Some(Redex(l, r)) = net.redex[RedexTy::FolL1 as usize].0.pop() {
                net.interact_follow(l, r);
                i += 1;
            }
            if let Some(Redex(l, r)) = net.redex[RedexTy::FolR1 as usize].0.pop() {
                net.interact_follow(l, r);
                i += 1;
            }
        }
        let end = std::time::Instant::now();
        eprintln!("Average redexes {}", redexes_avg / 10000000);
        eprintln!("Max redexes {}", redexes_max);
        eprintln!("Total time: {:?} for {i} interactions", end - start);
        eprintln!("MIPS: {}", i / (end.duration_since(start)).as_micros());
    }
}
