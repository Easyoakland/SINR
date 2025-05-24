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

Commit: 697ca99e978daa0d05d9ddcd08673464487c6d7f

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

~~Device 1 with page fault warmup, while let, speed_test of infinite net single-threaded, scalar L0 and R0 follows, no memory re-use, lto = "fat", codegen-units=1, inline(always) interact_com~~

Can't replicate this result. Might have confused the device used.

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

Commit: b50197c6cf801e360b1bc1fc87bd5a66059db34d

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

Device 1 re-run
```
Max redexes 333
Nodes max 889
Total time: 1.896017s
---
total: 310799667
commute: 44400000
annihilate: 0
erase: 88800000
follow: 177599667
---
All MIPS: 163.92241
Non-follow MIPS: 70.25253
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


Commit: bc17e17d4aca29144d36c6024cb4f0d5f6b5597f

Delta: Use 64 bit pointers, limit commute interactions to 512 per loop, use [`ThreadState`]

Device 1:
```
Max redexes: 101024
Nodes max: 203073
Final free_list length: 2048
Total time: 5.6967864s
---
total: 1433598464
commute: 204800000
annihilate: 0
erase: 409600000
follow: 819198464
---
All MIPS: 251.65039
Non-follow MIPS: 107.85029
```

Device 2:
```
Max redexes: 101024
Nodes max: 203073
Final free_list length: 2048
Total time: 5.1821172s
---
total: 1433598464
commute: 204800000
annihilate: 0
erase: 409600000
follow: 819198464
---
All MIPS: 276.6434
Non-follow MIPS: 118.561584
```

Commit: 0f3a6e45feb63980ea65f601432618e19367a61c

Delta: Decrease comm interaction limit to 128 based on benchmark.

Device 1:
```
Max redexes: 100256
Nodes max: 200769
Final free_list length: 512
Total time: 1.3840124s
---
total: 358399616
commute: 51200000
annihilate: 0
erase: 102400000
follow: 204799616
---
All MIPS: 258.957
Non-follow MIPS: 110.9817
```

Device 2:
```
Max redexes: 100256
Nodes max: 200769
Final free_list length: 512
Total time: 1.1411338s
---
total: 358399616
commute: 51200000
annihilate: 0
erase: 102400000
follow: 204799616
---
All MIPS: 314.0735
Non-follow MIPS: 134.60307
```

Commit: XXX

Device 1 THREADS=10:
Max redexes: 91040
Nodes max: 207682
Final free_list length: 512
Total time: 13.0609293s
---
total: 896003840
commute: 128000640
annihilate: 0
erase: 256001024
follow: 512002176
---
All MIPS: 686.0123
Non-follow MIPS: 294.0053
Average MIPS: 68.601234
Average Non-follow MIPS: 29.400532

Device 1 THREADS=8:
Max redexes: 93088
Nodes max: 206146
Final free_list length: 512
Total time: 11.6986008s
---
total: 896003840
commute: 128000640
annihilate: 0
erase: 256001024
follow: 512002176
---
All MIPS: 612.72015
Non-follow MIPS: 262.5944
Average MIPS: 76.59002
Average Non-follow MIPS: 32.8243

Device 1 THREADS=11:
Max redexes: 90016
Nodes max: 208450
Final free_list length: 512
Total time: 14.2454032s
---
total: 896003840
commute: 128000640
annihilate: 0
erase: 256001024
follow: 512002176
---
All MIPS: 691.869
Non-follow MIPS: 296.51535
Average MIPS: 62.897182
Average Non-follow MIPS: 26.95594

Below is two cherry-picked examples. One which is slower than usual one which is faster than usual.
Device 2 slow sample THREADS=14:
Max redexes: 86944
Nodes max: 210754
Final free_list length: 512
Total time: 15.9709387s
---
total: 896003840
commute: 128000640
annihilate: 0
erase: 256001024
follow: 512002176
---
All MIPS: 785.4227
Non-follow MIPS: 336.6098
Average MIPS: 56.101624
Average Non-follow MIPS: 24.043558

Device 2 fast sample THREADS=14:
Max redexes: 86944
Nodes max: 210754
Final free_list length: 512
Total time: 10.2035716s
---
total: 896003840
commute: 128000640
annihilate: 0
erase: 256001024
follow: 512002176
---
All MIPS: 1229.3674
Non-follow MIPS: 526.8719
Average MIPS: 87.81196
Average Non-follow MIPS: 37.633705

Device 2 THREADS=1:
Max redexes: 100256
Nodes max: 200770
Final free_list length: 512
Total time: 2.9797723s
---
total: 896003840
commute: 128000640
annihilate: 0
erase: 256001024
follow: 512002176
---
All MIPS: 300.69543
Non-follow MIPS: 128.86948
Average MIPS: 300.69543
Average Non-follow MIPS: 128.86948

Device 2 THREADS=2 (pinned to different logical core on same physical core):
Max redexes: 99232
Nodes max: 201538
Final free_list length: 512
Total time: 6.2938487s
---
total: 896003840
commute: 128000640
annihilate: 0
erase: 256001024
follow: 512002176
---
All MIPS: 284.7223
Non-follow MIPS: 122.023834
Average MIPS: 142.36115
Average Non-follow MIPS: 61.011917S

Device 2 THREADS=2 (pinned to different physical core):
Max redexes: 99232
Nodes max: 201538
Final free_list length: 512
Total time: 5.03342s
---
total: 896003840
commute: 128000640
annihilate: 0
erase: 256001024
follow: 512002176
---
All MIPS: 356.0201
Non-follow MIPS: 152.58005
Average MIPS: 178.01006
Average Non-follow MIPS: 76.29002
