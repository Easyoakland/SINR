<!-- cargo-rdme start -->

# SINR: Staged Interaction Net Runtime

Staged, Parallel, and SIMD capable Interpreter of Symmetric Interaction Combinators (SIC) + extensions.

# Layout
## Nodes
Nodes are represented as 2 tagged pointers and redexes as 2 tagged pointers. Nodes may have 0, 1, or 2 auxiliary ports. Although all currently implemented nodes are 2-ary others should work.
## Node tagged pointers
Tagged pointers contain a target and the type of the target they point to.
A Pointer is either "out" in which case its target is one of the [`PtrTag`] variants, `IN`, in which case it has no target, or `EMP` in which case it represents unallocated memory. The [`PtrTag`] represents the different port types, and the 4 auxiliary port types.

If you are familiar with inets you may wonder why 2-ary nodes have 4 separate auxiliary types instead of 2. The difference here is that auxiliary nodes have an associated stage, which dictates the synchronization of interactions. This makes it possible to handle annihilation interactions and re-use the annihilating nodes to redirect incoming pointers from the rest of the net in the case that two incoming pointers are to be linked. In this particular case one of the two auxiliary ports points to the other with a stage of 1 instead of 0, and therefore avoids racing when performing follows.

The reason that left and right follow interactions must be separated is to deallocate nodes only once both pointers in the node are no longer in use. This simplifies memory deallocation (TODO) by preventing any fragmentation (holes) less than a node in size. If this turns out to not be sufficiently useful, then left and right auxiliary follows can be performed simultaneously, only synchronizing between stage 0, 1, and principal interaction.

# Reduction
All redexes are stored into one of several redex buffers based upon the redex's interaction type, [`RedexTy`]. The regular commute and annihilate are two interactions. So are the 4 possible follow operations depending on the type of the auxiliary target. By separating the operations into these 2+4 types it becomes possible to perform all operations of the same type with minimal branching (for SIMD) and without atomic synchronization (for CPU SIMD and general perf improvements). All threads and SIMD lanes operate to reduce the same interaction type at the same time. When one of the threads runs out of reductions of that type (or some other signal such as number of reductions) all the threads synchronize their memory and start simultaneously reducing operations of a new type.
# Performance
See `Benchmarks.md` for performance measurements.
- On x86 single-threaded: Counting only commute and annihilate interactions appears to be ~145 million interactions per second (MIPS): 34.48 cycles per interaction.
- On x86 single-threaded: Counting commute, annihilate, and linking auxiliary ports appears to be ~330 MIPS: 15.15 cycles per interaction.
# Future possibilities
## Amb nodes
Since this design allows for synchronizing multiple pointers to a single port without atomics, implementing amb (a 2 principal port node) should be as simple as an FollowAmb0 and FollowAmb1 interactions.
## Global ref nodes
Global nets can be supported simply by adding the interaction type as usual. To enable SIMD, the nets should be stored with offsets local to 0. Then, when instantiating the net, allocate a continuous block the size of the global subnet and add the start of the block to all the offsets of pointers in the global subnet.
# Goals
- [x] Basic interactions
- [x] Single-threaded scalar implementation
- [ ] Memory deallocation and reclamation.
- [x] Minimize branching using lookup tables
    - This does not improve performance, and instead appears to decrease it. Remaining branching is very minimal.
- [x] SIMD
    - SIMD seems useless. Profiling indicates most of the program time is spent in `ptr::write`, `ptr::read`, and checking `len==0`, inside `Vec::push` and `Vec::pop`.
Consequently, SIMD is very unlikely to be useful since that part is not SIMD-able. Attempting to implement some parts with SIMD appear to only serve to slow things down by increasing front-end load and performing unnecessary extra work to swap values inside registers.
- [ ] Multiple threads

<!-- cargo-rdme end -->
