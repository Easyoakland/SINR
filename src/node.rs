//! Datatypes needed for storing nodes in the interaction net (inet).

use crate::{left_right::LeftRight, uassert};
use bilge::prelude::*;

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

pub type Slot = u61;

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
pub struct Ptr {
    pub tag: PtrTag,
    // A 32bit `Ptr` means we can address 2^29 nodes or 2^32 bytes since each node is u64 in size.
    // A 64bit `Ptr` means we can address 2^61 nodes or 2^67 bytes since each node is u128 in size, or make PtrTag 6 bits to get 2^64 bytes.
    pub slot: Slot,
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
pub struct Node {
    pub left: Ptr,
    pub right: Ptr,
}
impl Node {
    pub const EMP: fn() -> Node = || Node {
        left: Ptr::EMP(),
        right: Ptr::EMP(),
    };
}
