//! Regular vec but checks (bounds checking, disjoint indexing) are not performed if he `unsafe` feature is active. This is necessary for SIMD and to improve performance.

#[inline]
fn unsafe_index<T>(val: &[T], index: usize) -> &T {
    #[cfg(not(feature = "unsafe"))]
    {
        core::ops::Index::index(val, index)
    }

    #[cfg(feature = "unsafe")]
    {
        unsafe {
            if val.len() < index {
                core::hint::unreachable_unchecked()
            }
        }
        unsafe { val.get_unchecked(index) }
    }
}
#[inline]
fn unsafe_index_mut<T>(val: &mut [T], index: usize) -> &mut T {
    #[cfg(not(feature = "unsafe"))]
    {
        core::ops::IndexMut::index_mut(val, index)
    }
    #[cfg(feature = "unsafe")]
    {
        unsafe {
            if val.len() < index {
                core::hint::unreachable_unchecked()
            }
        }
        unsafe { val.get_unchecked_mut(index) }
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct UnsafeVec<T>(pub Vec<T>);
impl<T> UnsafeVec<T> {
    #[inline]
    pub fn len(&self) -> usize {
        self.0.len()
    }
    #[inline]
    pub fn push(&mut self, value: T) {
        #[cfg(feature = "prealloc")]
        crate::macros::uassert!(self.0.len() < self.0.capacity());
        self.0.push(value);
    }
    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        self.0.pop()
    }

    #[inline]
    pub fn get_disjoint_mut<I, const N: usize>(&mut self, indices: [I; N]) -> [&mut I::Output; N]
    where
        I: core::slice::GetDisjointMutIndex + core::slice::SliceIndex<[T]>,
    {
        #[cfg(not(feature = "unsafe"))]
        let res = self.0.get_disjoint_mut(indices).unwrap();
        #[cfg(feature = "unsafe")]
        let res = unsafe { self.0.get_disjoint_unchecked_mut(indices) };
        res
    }
}
impl<T> core::ops::Index<usize> for UnsafeVec<T> {
    type Output = T;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        unsafe_index(&self.0, index)
    }
}

impl<T> core::ops::IndexMut<usize> for UnsafeVec<T> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        unsafe_index_mut(&mut self.0, index)
    }
}
