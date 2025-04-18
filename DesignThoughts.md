# Tagging and queues
Need to be able to differentiate a pointer which points towards the following possibilities:
Era
Con
Dup
Redirect
In aux
Out towards left aux/Out towards right aux

An aux might become a redirect so the way to differentiate these two must be stored where the pointer points instead of tagging the pointer. May as well store that where the data for each node goes.

Whether the pointer is towards the left or right aux must either be in the pointer tag since it isn't shared between pointers. If the In Aux included the information it would require pointers to point to the left vs right which would be the same bit that could have been used to differentiate to begin with.

Two redirects must be able to fit inside 1 node for the same reason as the above.

The alternative is a substitution map indexed by the auxiliary pointer, but I don't think this is needed so I 'm going to try the above.


# SIMD
- Either place all redexes in the same queue and sort them scalar into queues before running each queue in SIMD, or sort active pairs when placing them.
    - The second might prevent SIMD on interactions themselves because they have to branch which stack to place in, so the pushes aren't SIMDable
        - What about an unsorted buffer that is sorted at the end? Maybe that will enable doing everything SIMD, then scalar sorting the unsorted stack into the right redex_ty/bucket.
        - The scalar code will run an exact amount of loop iterations, which is nicer than the other option of running an indeterminate amount of time until a local queue has enough to perform a SIMD.
        - There exists methods of performing filtering and compacting with SIMD such as [this](https://quickwit.io/blog/simd-range) method which reports almost perfect speedup.
    - The first is not SIMDable when loading, but after a queue gets enough interactions the interaction itself should be SIMD.
    - The first will be hard to work with the idea of 3 stages for pri-pri, laux, raux which I hope will avoid atomics on interactions themselves.

# Erasure
Originally I thought that erasure could be implemented with a regular binary combinator node with its own label that happens to always be created with its two auxiliary ports linked. That doesn't work when doing memory reclamation, though, because when the two combinators erase each other they create vicious circles that are not easily freed. Vicious circles may consume memory forever because memory slots can contain the links of that vicious circle. In other words, because redirections exist in memory a vicious circle with no interaction net nodes can still contain redirecting pseudo-nodes.

# Synchronization
There should be a vec/slice/array with slots equal to number of stages. When a thread wants to transition it increments the next stage's slot. If a thread wants to skip a stage it increments up to the stage it wants to work at. For each incremented slot it checks to see if the increment equals the number of threads. If yes, set it back to zero. All threads wait for a slot they incremented to be set back to 0 to indicate that stage can start. As an example with 2 threads and a thread wants to skip stage 2 and 3 and the active stage is 1 (0 indexed), then the array will look like [0,0,0,0,0]. Then when the thread wants to transition it atomically updates each slot in sequence like: fetch_add(2), fetch_add(3), fetch_add(4) so now the array is [0,0,1,1,1]. Now the other thread (say it doesn't want to skip) will increment [0,0,2,1,1], notice that it was last to be ready for stage 2, set the count to 0: [0,0,0,1,1] complete stage 2 and repeat. Then when stage 4 is reached [0,0,0,0,2]=>[0,0,0,0,0]. now both threads start completing stage 4. (Note that the thread which sets a slot to 0 can't skip back to that stage because it might set the slot to 1 before the others see the update to 0, and then you get deadlock) The advantage here is that the latency of thread communication should now be much less visible if a thread does these fetch adds one-after-another. Further, the skipped stages can be fetch_add_acquire instead of acq/rel since the skipping thread made no changes and doesn't need to publish any. (acq/rel when not skipping because this is the mechanism to ensure threads synchronize changes between stages). A thread might want to skip a stage if the number of redexes it has for that stage are little compared to other stages it has work for. Likely to happen a lot with the R1 and L1 stages.

Also note that global stage sync is only "global" up to whatever domain has potentially shared nodes. The graph could be partitioned into separate pieces which are known not to be connected. This could also be done with trees: If you exclude the leaves with redirects/vars/indirects you can compute the whole subtree in parallel only synchronizing stage transitions with any thread/SIMD lane which is also operating on that tree. Though this might be re-creating hardware mediated atomic operations.

## Atomic swap
You could avoid stage transitions if you use atomic exchanges instead. If stage transitions turn out to be very costly, might be worth performing all follow interactions with atomic swaps and then regular principal interactions in their own non-atomic stage. I think that's just reinventing the atomic swap algorithm again, though.
