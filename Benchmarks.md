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

COMMIT: XXX
Device 2 with page fault warmup, while let, speed_test of infinite net single-threaded, SIMD64 L0 and R0 follows, no memory re-use
all interactions:
Average redexes 665
Max redexes 666
Nodes max 532798225
Total time: 3.0037512s for 616398079 interactions
MIPS: 205.20944

non-follow interactions:
Average redexes 665
Max redexes 666
Nodes max 532798225
Total time: 3.0044955s for 266398890 interactions
MIPS: 88.66678

Device 2 with page fault warmup, while let, speed_test of infinite net single-threaded, SIMD32 L0 and R0 follows, no memory re-use
all interactions:
Average redexes 665
Max redexes 666
Nodes max 532798225
Total time: 3.1203954s for 621598335 interactions
MIPS: 199.20502

non-follow interactions:
Average redexes 665
Max redexes 666
Nodes max 532798225
Total time: 3.1884945s for 266398890 interactions
MIPS: 83.55007

Device 2 with page fault warmup, while let, speed_test of infinite net single-threaded, SIMD16 L0 and R0 follows, no memory re-use
all interactions:
Average redexes 665
Max redexes 666
Nodes max 532798225
Total time: 2.8666402s for 621598335 interactions
MIPS: 216.83865

non-follow interactions:
Average redexes 665
Max redexes 666
Nodes max 532798225
Total time: 2.8915461s for 266398890 interactions
MIPS: 92.130264

Device 2 with page fault warmup, while let, speed_test of infinite net single-threaded, SIMD8 L0 and R0 follows, no memory re-use

all interactions:
Average redexes 665
Max redexes 666
Nodes max 532798225
Total time: 2.816974s for 621598335 interactions
MIPS: 220.66173

non-follow interactions:
Average redexes 665
Max redexes 666
Nodes max 532798225
Total time: 2.8140446s for 266398890 interactions
MIPS: 94.66764

Device 2 with page fault warmup, while let, speed_test of infinite net single-threaded, SIMD4 L0 and R0 follows, no memory re-use
all interactions:
Average redexes 665
Max redexes 666
Nodes max 532798225
Total time: 3.5892714s for 621598335 interactions
MIPS: 173.18234

non-follow interactions:
Average redexes 665
Max redexes 666
Nodes max 532798225
Total time: 3.5544616s for 266398890 interactions
MIPS: 74.94776

Device 2 with page fault warmup, while let, speed_test of infinite net single-threaded, SIMD2 L0 and R0 follows, no memory re-use
all interactions:
Average redexes 665
Max redexes 666
Nodes max 532798225
Total time: 2.6684194s for 621598335 interactions
MIPS: 232.9463

non-follow interactions:
Average redexes 665
Max redexes 666
Nodes max 532798225
Total time: 2.6716652s for 266398890 interactions
MIPS: 99.712685

Device 2 with page fault warmup, while let, speed_test of infinite net single-threaded, SIMD1 L0 and R0 follows, no memory re-use
all interactions:
Average redexes 665
Max redexes 666
Nodes max 532798225
Total time: 2.3841635s for 621598335 interactions
MIPS: 260.71973

non-follow interactions:
Average redexes 665
Max redexes 666
Nodes max 532798225
Total time: 2.299392s for 266398890 interactions
MIPS: 115.85623

Device 2 with page fault warmup, while let, speed_test of infinite net single-threaded,scalar L0 and R0 follows, no memory re-use
all interactions:
Average redexes 665
Max redexes 666
Nodes max 532798225
Total time: 2.2518212s for 621598335 interactions
MIPS: 276.0425

non-follow interactions:
Average redexes 665
Max redexes 666
Nodes max 532798225
Total time: 2.1299137s for 266398890 interactions
MIPS: 125.07501
