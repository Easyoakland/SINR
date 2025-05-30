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
//! # Future possibilities
//! ## Amb nodes
//! Since this design allows for synchronizing multiple pointers to a single port without atomics, implementing amb (a 2 principal port node) should be as simple as an FollowAmb0 and FollowAmb1 interactions.
//! ## Global ref nodes
//! Global nets can be supported simply by adding the interaction type as usual. To enable SIMD, the nets should be stored with offsets local to 0. Then, when instantiating the net, allocate a continuous block the size of the global subnet and add the start of the block to all the offsets of pointers in the global subnet.
//! # Goals
//! - [x] Basic interactions
//! - [x] Single-threaded scalar implementation
//! - [x] Memory deallocation and reclamation.
//! - [x] Minimize branching using lookup tables
//!     - This does not improve performance, and instead appears to decrease it. Remaining branching is very minimal.
//! - [x] SIMD
//!     - Attempting to implement some parts with SIMD appear to only serve to slow things down by increasing front-end load and performing unnecessary extra work to swap values inside registers. It's possible that the SIMD code was poor and could have been improved. See the `SIMD` branch for details.
//! - [x] Multiple threads
//! - [ ] Parse net from text
//!
//! [`RedexTy`]: redex::RedexTy

// # Safety
// In various places unsafe things are done without using the `unsafe` keyword and instead conditioning on `feature="unsafe"` and detecting the unsafe and panic-ing if not `(feature="unsafe")`. This will be changed after the design is more finalized.

#![feature(get_disjoint_mut_helpers)]
#![feature(unsafe_cell_access)]
#![allow(dead_code)] // TODO remove
#![allow(rustdoc::private_intra_doc_links)]

mod builder;
mod left_right;
mod macros;
mod net;
mod node;
mod redex;
mod unsafe_vec;
mod viz;

use macros::uassert;
use node::{Ptr, PtrTag, Slot};

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
    use crate::{
        macros::trace,
        net::{GlobalState, Net, ThreadState},
        node::{Node, SharedNode},
        redex::{Redex, RedexTy},
    };
    use core::sync::atomic::AtomicU32;
    use parking_lot::RwLock;

    #[test]
    fn test_viz() {
        let mut net = Net::default();
        net.nodes.push(SharedNode::new(Node {
            left: Ptr::IN(),
            right: Ptr::new(PtrTag::LeftAux0, Slot::new(1)),
        }));
        net.nodes.push(SharedNode::new(Node {
            left: Ptr::IN(),
            right: Ptr::new(PtrTag::LeftAux0, Slot::new(2)),
        }));
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
            net.nodes.push(SharedNode::new(Node {
                left: Ptr::IN(),
                right: Ptr::new(PtrTag::LeftAux0, Slot::new(slot.try_into().unwrap())),
            }));
        };
        make_id(&mut net);
        make_id(&mut net);
        make_id(&mut net);
        make_id(&mut net);
        net.nodes.push(SharedNode::new(Node {
            left: Ptr::new(PtrTag::Con, Slot::new(1)),
            right: Ptr::new(PtrTag::Con, Slot::new(2)),
        }));
        net.nodes.push(SharedNode::new(Node {
            left: Ptr::new(PtrTag::Con, Slot::new(3)),
            right: Ptr::new(PtrTag::Con, Slot::new(4)),
        }));
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
            net.nodes.push(SharedNode::new(Node {
                left: Ptr::IN(),
                right: Ptr::new(PtrTag::LeftAux0, Slot::new(slot.try_into().unwrap())),
            }));
        };
        make_id(&mut net);
        make_id(&mut net);
        make_id(&mut net);
        net.nodes.push(SharedNode::new(Node {
            left: Ptr::ERA_0(),
            right: Ptr::new(PtrTag::Con, Slot::new(1)),
        }));
        net.nodes.push(SharedNode::new(Node {
            left: Ptr::new(PtrTag::Con, Slot::new(2)),
            right: Ptr::new(PtrTag::Con, Slot::new(3)),
        }));
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
        net.interact_era(p);
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
        net.nodes[n1.value() as usize] = SharedNode::new(Node {
            left: Ptr::ERA_0(),
            right: Ptr::new(PtrTag::RightAux0, n2),
        });
        net.nodes[n2.value() as usize] = SharedNode::new(Node {
            left: Ptr::ERA_0(),
            right: Ptr::IN(),
        });
        net.redexes.regular[RedexTy::Com as usize]
            .push(Redex(Ptr::new(PtrTag::Dup, n1), Ptr::new(PtrTag::Con, n2)));
    }

    #[test]
    #[ignore = "bench"]
    fn speed_test() {
        let mut net = Net::default();
        const THREADS: usize = 14;
        let mut waiters = rmw_free_barrier::Waiter::new(THREADS);

        // Force page faults now so they don't happen while benchmarking.
        for _ in 0..10000000 {
            net.nodes.push(SharedNode::new(Node::default()));
        }
        for _ in 0..10000000 {
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

        // Insert original net for test
        for _ in 0..100000 {
            infinite_reduction_net(&mut net);
        }
        trace!(file "start.dot",;viz::mem_to_dot(&net));

        // Setup threads for reduction. Use the 0 waiter for `thread_state` and run it more times to match the `rmw_free_barrier` implementation's restriction.
        let mut thread_state = ThreadState::new(waiters.swap_remove(0));
        let mut thread_state2: [_; THREADS - 1] =
            core::array::from_fn(|_| ThreadState::new(waiters.pop().unwrap()));
        assert!(waiters.pop().is_none());

        let global_state = {
            let Net {
                nodes,
                redexes,
                free_list,
            } = net;
            GlobalState {
                nodes: RwLock::new(nodes),
                free_list: RwLock::new(free_list),
                redexes: RwLock::new(redexes),
            }
        };
        let mut nodes_max = 0u64;
        let mut redexes_max = 0u64;
        gdt_cpus::set_thread_priority(gdt_cpus::ThreadPriority::AboveNormal).unwrap();
        let mut ids = gdt_cpus::cpu_info().unwrap().logical_processor_ids();
        gdt_cpus::pin_thread_to_core(dbg!(ids.pop().unwrap())).unwrap();

        let mut start = std::time::Instant::now();

        let mut end = std::time::Instant::now();
        std::thread::scope(|s| {
            let t2 = thread_state2
                .get_disjoint_mut(core::array::from_fn::<_, { THREADS - 1 }, _>(|i| i))
                .unwrap();

            let t2 = t2.map(|thread_state2| {
                let global_state = &global_state;
                let id = ids.pop().unwrap();
                s.spawn(move || {
                    gdt_cpus::set_thread_priority(gdt_cpus::ThreadPriority::AboveNormal).unwrap();
                    gdt_cpus::pin_thread_to_core(id).unwrap();
                    for _ in 0..(1000000u32 - 5) {
                        // {
                        //     // perf: this kind of checks how lock performance affects multithreaded by repeatedly trying to read from the various global fields.
                        //     // in the happy path in `run_once` it only accessing local state and these contention with these shouldn't be a big difference.
                        //     core::hint::black_box(&global_state.free_list.try_read());
                        //     core::hint::black_box(&global_state.redexes.try_read());
                        //     core::hint::black_box(&global_state.nodes.try_read());
                        // };
                        thread_state2.run_once(&global_state);
                    }
                    core::mem::take(&mut thread_state2.waiter); // remove from barrier
                })
            });

            for _ in 0..5 {
                thread_state.run_once(&global_state);
            }
            start = std::time::Instant::now();

            for i in 0..1000000u32 {
                if i % 2u32.pow(14) == 0 {
                    dbg!(i);
                };
                thread_state.run_once(&global_state);

                nodes_max = nodes_max.max(
                    [
                        thread_state.local_net.nodes.len(),
                        global_state.nodes.read().len(),
                    ]
                    .into_iter()
                    .sum::<usize>()
                    .try_into()
                    .unwrap(),
                );
                redexes_max = redexes_max.max(
                    [
                        thread_state.local_net.active_redexes(),
                        net::active_redexes(&global_state.redexes.read()),
                    ]
                    .into_iter()
                    .sum::<u64>(),
                );
            }
            end = std::time::Instant::now();
            t2.map(|x| x.join().unwrap());
        });
        let net = {
            let GlobalState {
                nodes,
                free_list,
                redexes,
            } = global_state;
            Net {
                nodes: nodes.into_inner(),
                free_list: free_list.into_inner(),
                redexes: redexes.into_inner(),
            }
        };
        eprintln!("Max redexes: {}", redexes_max);
        eprintln!("Nodes max: {}", nodes_max);
        eprintln!(
            "Final free_list length: {}",
            [thread_state.local_net.free_list.len(), net.free_list.len()]
                .into_iter()
                .sum::<usize>()
        );
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
        let all_mips = [thread_state.interactions()]
            .into_iter()
            .chain(thread_state2.iter().map(|x| x.interactions()))
            .map(|x| x as f32)
            .sum::<f32>()
            / (end.duration_since(start)).as_micros() as f32;
        let non_follow_mips = [thread_state.non_follow_interactions()]
            .into_iter()
            .chain(thread_state2.iter().map(|x| x.non_follow_interactions()))
            .map(|x| x as f32)
            .sum::<f32>()
            / (end.duration_since(start)).as_micros() as f32;
        eprintln!(
            "---\n\
            All MIPS: {}\n\
            Non-follow MIPS: {}\n\
            Average MIPS: {}\n\
            Average Non-follow MIPS: {}",
            all_mips,
            non_follow_mips,
            all_mips / (THREADS as f32),
            non_follow_mips / (THREADS as f32)
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
    #[test]
    #[should_panic]
    #[cfg(not(feature = "unsafe"))]
    fn ub_detect_test() {
        let mut net = Net::default();
        infinite_reduction_net(&mut net);
        let mut a = net.nodes[0].get();
        let mut b = net.nodes[0].get();
        dbg!(&mut *a, &mut *b);
    }
    #[test]
    fn no_ub_detect_test() {
        let mut net = Net::default();
        infinite_reduction_net(&mut net);
        {
            let mut a = net.nodes[0].get();
            dbg!(&mut *a);
        }
        {
            let mut b = net.nodes[0].get();
            dbg!(&mut *b)
        };
    }
}
