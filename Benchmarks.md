Benchmarks all use `--features unsafe,prealloc` and `-C target-cpu=native` unless otherwise specified.

Device 1: AMD Ryzen 5 7640U
Device 2: Intel Core i9-10900K

Device 1 for speed_test of infinite net single-threaded, scalar, no memory re-use.
```
Average redexes 277781
Max redexes 555560
Total time: 245.3175ms for 32222221 interactions
MIPS: 131
```

Counting only non-follow interactions (Com and Ann)
```
Total time: 217.6439ms for 12222223 interactions
MIPS: 56
```

Device 2 for speed_test of infinite net single-threaded, scalar, no memory re-use.
```
Average redexes 277781
Max redexes 555560
Total time: 153.155ms for 32222221 interactions
MIPS: 210
```

Counting only non-follow interactions (Com and Ann)
```
Total time: 144.6203ms for 12222223 interactions
MIPS: 84
```

Not tracking redex stats
```
Total time: 139.5743ms for 32222221 interactions
MIPS: 230
```

COMMIT: 697ca99e978daa0d05d9ddcd08673464487c6d7f
Device 2 with page fault warmup, while let, speed_test of infinite net single-threaded, SIMD64 L0 and R0 follows, no memory re-use

all interactions:
```
Average redexes 665
Max redexes 666
Nodes max 532798225
Total time: 3.0037512s for 616398079 interactions
MIPS: 205.20944
```

non-follow interactions:
```
Average redexes 665
Max redexes 666
Nodes max 532798225
Total time: 3.0044955s for 266398890 interactions
MIPS: 88.66678
```

Device 2 with page fault warmup, while let, speed_test of infinite net single-threaded, SIMD32 L0 and R0 follows, no memory re-use

all interactions:
```
Average redexes 665
Max redexes 666
Nodes max 532798225
Total time: 3.1203954s for 621598335 interactions
MIPS: 199.20502
```

non-follow interactions:
```
Average redexes 665
Max redexes 666
Nodes max 532798225
Total time: 3.1884945s for 266398890 interactions
MIPS: 83.55007
```

Device 2 with page fault warmup, while let, speed_test of infinite net single-threaded, SIMD16 L0 and R0 follows, no memory re-use

all interactions:
```
Average redexes 665
Max redexes 666
Nodes max 532798225
Total time: 2.8666402s for 621598335 interactions
MIPS: 216.83865
```

non-follow interactions:
```
Average redexes 665
Max redexes 666
Nodes max 532798225
Total time: 2.8915461s for 266398890 interactions
MIPS: 92.130264
```

Device 2 with page fault warmup, while let, speed_test of infinite net single-threaded, SIMD8 L0 and R0 follows, no memory re-use

all interactions:
```
Average redexes 665
Max redexes 666
Nodes max 532798225
Total time: 2.816974s for 621598335 interactions
MIPS: 220.66173
```

non-follow interactions:
```
Average redexes 665
Max redexes 666
Nodes max 532798225
Total time: 2.8140446s for 266398890 interactions
MIPS: 94.66764
```

Device 2 with page fault warmup, while let, speed_test of infinite net single-threaded, SIMD4 L0 and R0 follows, no memory re-use

all interactions:
```
Average redexes 665
Max redexes 666
Nodes max 532798225
Total time: 3.5892714s for 621598335 interactions
MIPS: 173.18234
```

non-follow interactions:
```
Average redexes 665
Max redexes 666
Nodes max 532798225
Total time: 3.5544616s for 266398890 interactions
MIPS: 74.94776
```

Device 2 with page fault warmup, while let, speed_test of infinite net single-threaded, SIMD2 L0 and R0 follows, no memory re-use

all interactions:
```
Average redexes 665
Max redexes 666
Nodes max 532798225
Total time: 2.6684194s for 621598335 interactions
MIPS: 232.9463
```

non-follow interactions:
```
Average redexes 665
Max redexes 666
Nodes max 532798225
Total time: 2.6716652s for 266398890 interactions
MIPS: 99.712685
```

Device 2 with page fault warmup, while let, speed_test of infinite net single-threaded, SIMD1 L0 and R0 follows, no memory re-use

all interactions:
```
Average redexes 665
Max redexes 666
Nodes max 532798225
Total time: 2.3841635s for 621598335 interactions
MIPS: 260.71973
```

non-follow interactions:
```
Average redexes 665
Max redexes 666
Nodes max 532798225
Total time: 2.299392s for 266398890 interactions
MIPS: 115.85623
```

Device 2 with page fault warmup, while let, speed_test of infinite net single-threaded, scalar L0 and R0 follows, no memory re-use

all interactions:
```
Average redexes 665
Max redexes 666
Nodes max 532798225
Total time: 2.2518212s for 621598335 interactions
MIPS: 276.0425
```

non-follow interactions:
```
Average redexes 665
Max redexes 666
Nodes max 532798225
Total time: 2.1299137s for 266398890 interactions
MIPS: 125.07501
```

Device 2 with page fault warmup, while let, speed_test of infinite net single-threaded, scalar L0 and R0 follows, no memory re-use, lto = "fat", codegen-units=1

all interactions:
```
Average redexes 665
Max redexes 666
Nodes max 532798225
Total time: 2.0610257s for 621598335 interactions
MIPS: 301.5967
```

non-follow interactions:
```
Average redexes 665
Max redexes 666
Nodes max 532798225
Total time: 2.0261774s for 266398890 interactions
MIPS: 131.47859
```

Commit: 708304d5834521f76f732bb172673964cdce6efb

Device 2 with page fault warmup, while let, speed_test of infinite net single-threaded, scalar L0 and R0 follows, no memory re-use, lto = "fat", codegen-units=1, inline(always) interact_com

all interactions:
```
Max redexes 666
Nodes max 532798225
Total time: 1.8899441s for 621598335 interactions
MIPS: 328.89777
```

non-follow interactions:
```
Max redexes 666
Nodes max 532798225
Total time: 1.836576s for 266398890 interactions
MIPS: 145.05193
```

Device 1 with page fault warmup, while let, speed_test of infinite net single-threaded, scalar L0 and R0 follows, no memory re-use, lto = "fat", codegen-units=1, inline(always) interact_com

all interactions:
```
Max redexes 666
Nodes max 532798225
Total time: 2.2039997s for 621598335 interactions
MIPS: 282.03204
```

non-follow interactions:
```
Max redexes 666
Nodes max 532798225
Total time: 2.168278s for 266398890 interactions
MIPS: 122.86196
```

Commit: XXX

Device 1 adding memory re-use to `speed_test` of infinite net
```
Max redexes 333
Nodes max 889
Total time: 2.3478049s
---
total: 310799667
commute: 44400000
annihilate: 0
erase: 88800000
follow: 177599667
---
All MIPS: 132.37889
Non-follow MIPS: 56.733868
```

Device 2 adding memory re-use to `speed_test` of infinite net
```
Max redexes 333
Nodes max 889
Total time: 1.0197678s
---
total: 310799667
commute: 44400000
annihilate: 0
erase: 88800000
follow: 177599667
---
All MIPS: 304.77518
Non-follow MIPS: 130.61807
```
