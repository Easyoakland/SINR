//! A low-latency barrier which avoids using read-modify-write atomic operations.

#![feature(sync_unsafe_cell)]

use core::sync::atomic::AtomicU8;
use crossbeam_utils::CachePadded;
use std::sync::Arc;

const STAGE_MASK: u8 = 0b1;
const DISABLED_MASK: u8 = 0b10;

#[derive(Debug, Default)]
struct Inner {
    waiting: Vec<CachePadded<AtomicU8>>,
}
/// A [`Waiter`] allows a thread to [`wait`] until [`wait`] has been called on all sibling [`Waiter`]s, to implement a low-latency barrier.
///
/// # Warning
/// - If the first [`Waiter`] returned by [`Waiter::new`] is dropped, future [`wait`]s by sibling [`Waiter`]s may panic or not complete.
/// - The [`Default`] impl makes a [`Waiter`] that is not part of a barrier and may panic on [`wait`].
///
/// [`wait`]: Self::wait()
#[derive(Debug, Default)]
pub struct Waiter {
    inner: Arc<Inner>,
    id: usize,
}
impl Waiter {
    /// Get `n` [`Waiter`]s which are all part of a single low-latency barrier.
    /// # Warning
    /// If the first [`Waiter`] returned by [`Waiter::new`] is dropped, future [`Waiter::wait`]s by sibling [`Waiter`]s may panic or not complete.
    pub fn new(n: usize) -> Vec<Self> {
        let shared = Arc::new(Inner {
            waiting: (0..n).map(|_| CachePadded::new(AtomicU8::new(0))).collect(),
        });
        (0..n)
            .map(|id| Self {
                inner: shared.clone(),
                id,
            })
            .collect()
    }

    /// Wait until all other sibling [`Waiter`]s are also waiting.
    pub fn wait(&self) {
        let my_slot = &self.inner.waiting[usize::try_from(self.id).unwrap()];
        // Get the previous value of the slot after the last synchronization.
        let current_val = my_slot.load(core::sync::atomic::Ordering::Relaxed);
        let new_val = current_val ^ STAGE_MASK; // addition mod 2

        // Slot 0 is the central thread coordinating everything.
        if self.id == 0 {
            // Wait for all other threads.
            for slot in &self.inner.waiting[1..] {
                loop {
                    let val = slot.load(core::sync::atomic::Ordering::Relaxed);
                    // If the other thread has either transitioned to the next stage or is disabled don't have to wait for it any more.
                    if (val & STAGE_MASK == new_val & STAGE_MASK) || val & DISABLED_MASK != 0 {
                        break;
                    };
                    // perf: Testing indicates that using [`std::thread::yield_now`] is slightly worse than [`core::hint::spin_loop`] in this branch.
                    core::hint::spin_loop();
                }
            }
            // Make sure all the previous loads are actually [`Acquire`] ordered.
            core::sync::atomic::fence(core::sync::atomic::Ordering::Acquire);

            // Now that every other thread is waiting indicate that this thread is also waiting with a [`Release`] so they [`Acquire`] every change this thread has picked up.
            my_slot.store(new_val, core::sync::atomic::Ordering::Release);
        } else {
            // [`Release`] is used here so thread 0 can acquire any changes before this barrier.
            my_slot.store(new_val, core::sync::atomic::Ordering::Release);
            // Spin until thread 0 indicates the barrier is done.
            loop {
                let val = self.inner.waiting[0].load(core::sync::atomic::Ordering::Relaxed);
                if val & STAGE_MASK == new_val & STAGE_MASK {
                    // If thread 0 indicated done, use [`Acquire`] ordering so all changes by all other threads participating in this barrier are now visible by synchronizing with the release of thread 0.
                    core::sync::atomic::fence(core::sync::atomic::Ordering::Acquire);
                    break;
                }
                // The 0th thread should not be dropped when other Waiters are still waiting.
                assert_eq!(
                    val & DISABLED_MASK,
                    0,
                    "First Waiter dropped before call to `wait` resulting in a deadlock"
                );
                // Unlike thread 0, this thread doesn't need to do anything for a while.
                // perf: Testing indicates that using a [`std::thread::yield_now`] works much better than [`core::hint::spin_loop`] for this branch.
                std::thread::yield_now();
            }
        }
    }

    /// Number of [`Waiter`] that are still active in this barrier.
    pub fn len(&self) -> usize {
        self.inner
            .waiting
            .iter()
            .filter(|x| x.load(core::sync::atomic::Ordering::Relaxed) & DISABLED_MASK == 0)
            .count()
    }
}
impl Drop for Waiter {
    fn drop(&mut self) {
        let Some(my_slot) = &self.inner.waiting.get(self.id) else {
            // Default impl has no waiters and vector is len 0.
            debug_assert_eq!(self.inner.waiting.len(), 0);
            return;
        };
        let val = my_slot.load(core::sync::atomic::Ordering::Relaxed);
        // Disable this waiter on drop.
        my_slot.store(val | DISABLED_MASK, core::sync::atomic::Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::{cell::SyncUnsafeCell, iter, sync::atomic::AtomicU32};

    #[test]
    #[ignore = "bench"]
    fn spinlock_latency() {
        static A: AtomicU32 = AtomicU32::new(0);
        const STORES: u32 = 10_000_000;
        let spawn_thread = move |mod_: u32, incr_on: u32| {
            let mut it = 0;
            std::thread::spawn(move || {
                loop {
                    let a = A.load(core::sync::atomic::Ordering::Relaxed);
                    if a > STORES {
                        break it;
                    } else if a % mod_ == incr_on {
                        A.store(a + 1, core::sync::atomic::Ordering::Relaxed);
                        // A.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
                    }
                    it += 1;
                    core::hint::spin_loop();
                }
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

    #[test]
    #[ignore = "bench"]
    fn std_barrier() {
        #[cfg(miri)]
        const SYNCS: u64 = 1_000;
        #[cfg(not(miri))]
        const SYNCS: u64 = 10_000;
        for n in 0..=((usize::from(std::thread::available_parallelism().unwrap())).max(3)) {
            let incrs = iter::repeat_with(|| SyncUnsafeCell::new(0u64))
                .take(n)
                .collect::<Vec<_>>();
            let barrier = std::sync::Barrier::new(n);
            let elapsed = {
                let incrs = &incrs;
                let barrier = &barrier;
                std::thread::scope(|s| {
                    let spawn_thread = move |mut tid: usize| {
                        s.spawn(move || {
                            loop {
                                let num = unsafe { &mut *incrs[tid].get() };
                                *num += 1;
                                if *num == SYNCS {
                                    break;
                                } else {
                                    tid = (tid + 1).rem_euclid(n);
                                    barrier.wait();
                                }
                            }
                        })
                    };
                    let start = std::time::Instant::now();
                    let join_handles = (0..n).map(|i| spawn_thread(i)).collect::<Vec<_>>();
                    join_handles
                        .into_iter()
                        .map(|x| x.join().unwrap())
                        .for_each(drop);
                    std::time::Instant::now().duration_since(start)
                })
            };
            eprintln!(
                "{n} incrs: {:?}, time: {:?}, sec_per_sync: {:?}",
                incrs
                    .into_iter()
                    .map(|mut x| *x.get_mut())
                    .collect::<Vec<_>>(),
                elapsed,
                elapsed.as_secs_f32() / (SYNCS as f32)
            );
        }
    }

    #[test]
    #[ignore = "bench"]
    fn hurdles_barrier() {
        #[cfg(not(miri))]
        const SYNCS: u64 = 1_000_000;
        #[cfg(miri)]
        const SYNCS: u64 = 1_000;
        for n in 0..=((usize::from(std::thread::available_parallelism().unwrap())).max(3)) {
            let incrs = iter::repeat_with(|| SyncUnsafeCell::new(0u64))
                .take(n)
                .collect::<Vec<_>>();
            let barrier = hurdles::Barrier::new(n);
            let elapsed = {
                let incrs = &incrs;
                let barrier = &barrier;
                std::thread::scope(|s| {
                    let spawn_thread = move |mut tid: usize, mut barrier: hurdles::Barrier| {
                        s.spawn(move || {
                            loop {
                                let num = unsafe { &mut *incrs[tid].get() };
                                *num += 1;
                                if *num == SYNCS {
                                    break;
                                } else {
                                    tid = (tid + 1).rem_euclid(n);
                                    barrier.wait();
                                }
                            }
                        })
                    };
                    let start = std::time::Instant::now();
                    let join_handles = (0..n)
                        .map(|i| spawn_thread(i, barrier.clone()))
                        .collect::<Vec<_>>();
                    join_handles
                        .into_iter()
                        .map(|x| x.join().unwrap())
                        .for_each(drop);
                    std::time::Instant::now().duration_since(start)
                })
            };
            eprintln!(
                "{n} incrs: {:?}, time: {:?}, sec_per_sync: {:?}",
                incrs
                    .into_iter()
                    .map(|mut x| *x.get_mut())
                    .collect::<Vec<_>>(),
                elapsed,
                elapsed.as_secs_f32() / (SYNCS as f32)
            );
        }
    }

    #[test]
    #[ignore = "bench"]
    fn waiter_barrier() {
        #[cfg(not(miri))]
        const SYNCS: u64 = 1_000_000;
        #[cfg(miri)]
        const SYNCS: u64 = 1_000;
        for n in 0..=((usize::from(std::thread::available_parallelism().unwrap())).max(3)) {
            let incrs = iter::repeat_with(|| SyncUnsafeCell::new(0u64))
                .take(n)
                .collect::<Vec<_>>();
            let mut barrier = Waiter::new(n);
            let elapsed = {
                let incrs = &incrs;
                std::thread::scope(|s| {
                    let spawn_thread = move |mut tid: usize, waiter: Waiter| {
                        s.spawn(move || {
                            loop {
                                let num = unsafe { &mut *incrs[tid].get() };
                                *num += 1;
                                if *num == SYNCS {
                                    break;
                                } else {
                                    tid = (tid + 1).rem_euclid(n);
                                    waiter.wait();
                                }
                            }
                        })
                    };
                    let start = std::time::Instant::now();
                    let join_handles = (0..n)
                        .map(|i| spawn_thread(i, barrier.pop().unwrap()))
                        .collect::<Vec<_>>();
                    join_handles
                        .into_iter()
                        .map(|x| x.join().unwrap())
                        .for_each(drop);
                    std::time::Instant::now().duration_since(start)
                })
            };
            eprintln!(
                "{n} incrs: {:?}, time: {:?}, sec_per_sync: {:?}",
                incrs
                    .into_iter()
                    .map(|mut x| *x.get_mut())
                    .collect::<Vec<_>>(),
                elapsed,
                elapsed.as_secs_f32() / (SYNCS as f32)
            );
        }
    }
}
