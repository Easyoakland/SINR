//! Data layout for reducible expressions ([`Redex`]es).

use crate::{node::Ptr, uassert, unsafe_vec::UnsafeVec};
use bilge::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Hash)]
pub struct Redex(pub Ptr, pub Ptr);
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
    bilge::TryFromBits,
)]
#[repr(u8)]
pub enum RedexTy {
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
pub struct Redexes {
    pub regular: [UnsafeVec<Redex>; RedexTy::LEN],
    pub erase: UnsafeVec<Ptr>,
}
