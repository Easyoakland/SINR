= Tagging and queues
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

= Erasure
Originally I thought that erasure could be implemented with a regular binary combinator node with its own label that happens to always be created with its two auxiliary ports linked. That doesn't work when doing memory reclamation, though, because the vicious circles that are created when the two combinator erasers result in not realizing that certain memory slots can be freed which contain the links of that vicious circle. In other words, because redirections exist in memory a vicious circle with no interaction net nodes can still contain redirecting pseudo-nodes.