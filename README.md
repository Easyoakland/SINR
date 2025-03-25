<!-- cargo-rdme start -->

# SINR: Staged Interaction Net Runtime

Staged, Parallel, and SIMD capable Interpreter of Symmetric Interaction Combinators (SIC) + extensions.

# Layout
## Nodes
Nodes are represented as 2 tagged pointers and active pairs as 2 tagged pointers. Nodes may have 0, 1, or 2 auxillary ports. (Although all currently implemented nodes are 2ary others should be supported).
## Node tagged pointers
Tagged pointers contain a target and the type of the target they point to.
Pointers targets are one of the [`PtrTag`] variants. These are the different principal port types, and the 4 auxillary port types.

If you are familiar with inets you may wonder why 2ary nodes have 4 seperate auxillary types instead of 2. The difference here is that auxillary nodes have an associated stage, which dictates the synchronization of interactions. This makes it possible to handle annihilation interactions and re-use the annihilating nodes to redirect incoming pointers from the rest of the net in the case that two incoming pointers to be linked. In this particular case one of the two ports points to the other with a stage of 1 instead of 0, and therefore avoids racing when performing follows.

The reason that left and right follow interactions must be separated is to deallocate nodes only once both pointers in the node are no longer in use. This simplifies memory deallocation (TODO) by preventing any fragmentation (holes) less than a node in size. If this turns out to not be sufficiently useful, then left and right auxiliary follows can be performed simultaneously, only synchronizing between stage 0, 1, and principal interaction.

# Reduction
All redexes are stored into one of several redex buffers based upon the redex's interaction type, [`RedexTy`]. The regular commute and annihilate are two interactions. So are the 4 possible follow operations depending on the type of the auxillary target. By separating the operations into these 2+4 types it becomes possible to perform all operations of the same type with minimal branching (for SIMD) and without atomic synchronization (for CPU SIMD and general perf improvements). All threads and SIMD lanes operate reduce the same interaction type at the same time. When one of the threads runs out of reductions of that type (or some other signal such as number of reductions) all the threads synchronize their memory and start simultaneously reducing operations of a new type.
# Future possibilities
## Amb
Since this design allows for synchronizing multiple pointers to a single port without atomics, implementing amb (a 2 principal port node) should be as simple as an FollowAmb0 and FollowAmb1 interactions.
# TODO
- [x] Single-threaded scalar implementation with branching
- [ ] Memory deallocation and reclamation.
- [ ] Minimize branching using lookup tables
- [ ] SIMD
- [ ] Multiple threads
## Ref nodes
Global nets can be supported simply by adding the interaction type as usual. To enable SIMD, the nets should be stored with offsets local to 0. Then, when instantiating the net, allocate a continuous block the size of the global subnet and add the start of the block to all the offsets of pointers in the global subnet.

<!-- cargo-rdme end -->
