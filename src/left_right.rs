use bilge::prelude::*;
#[bilge::bitsize(1)]
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, bilge::BinaryBits, bilge::FromBits, Hash)]
pub enum LeftRight {
    #[default]
    Left,
    Right,
}
impl LeftRight {
    pub fn is_left(self) -> bool {
        matches!(self, LeftRight::Left)
    }
    pub fn is_right(self) -> bool {
        matches!(self, LeftRight::Right)
    }
}
impl core::fmt::Display for LeftRight {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LeftRight::Left => "L".fmt(f),
            LeftRight::Right => "R".fmt(f),
        }
    }
}
