# Results per shape (judge runs of our leaderboard rows)

Ranks read from the public leaderboard API at 2026-10-29 20:54 UTC. Speedup = SGLang baseline time / our time, as the judge reports it per shape (CUDA-graph replays, cold L2). The board score is the judge's weighted geomean over these shapes. Positions are provisional until the organizers' final reruns.

## muse_decode_layer

Rank **40 of 989**, speedup **2.2058x** over SGLang (95% CI 2.1727-2.2389), judge job `a61afb9687f14eff`. File: `kernels/muse_decode_layer.py` (our work on vjeux's base (Apache-2.0), with code following Matsvei (kernarc 'matvey')).

| shape | SGLang (us) | ours (us) | speedup |
|---|---:|---:|---:|
| batch=1, ctx=80000, dist=soft, layer=sliding, load=decode 1, page=1, repeats=60, weight=46 | 173.09 | 69.33 | 2.497x |
| batch=1, ctx=80000, dist=soft, layer=full, load=decode 1, page=1, repeats=60, weight=16 | 181.94 | 74.43 | 2.445x |
| batch=16, ctx=80000, dist=soft, layer=sliding, load=decode 16, page=1, spread=agentic, weight=46 | 266.50 | 129.35 | 2.060x |
| batch=16, ctx=80000, dist=soft, layer=full, load=decode 16, page=1, spread=agentic, weight=16 | 432.26 | 203.96 | 2.119x |
| batch=32, ctx=80000, dist=soft, layer=sliding, load=decode 32, page=1, spread=agentic, weight=46 | 261.04 | 149.13 | 1.750x |
| batch=32, ctx=80000, dist=soft, layer=full, load=decode 32, page=1, spread=agentic, weight=16 | 616.21 | 304.34 | 2.025x |
| batch=1, ctx=80000, dist=soft, layer=sliding, load=verify 1, page=1, q_len=8, weight=46 | 268.99 | 110.12 | 2.443x |
| batch=1, ctx=80000, dist=soft, layer=full, load=verify 1, page=1, q_len=8, weight=16 | 281.22 | 117.16 | 2.400x |
| batch=16, ctx=80000, dist=soft, layer=sliding, load=verify 16, page=1, q_len=8, spread=agentic, weight=46 | 527.47 | 244.17 | 2.160x |
| batch=16, ctx=80000, dist=soft, layer=full, load=verify 16, page=1, q_len=8, spread=agentic, weight=16 | 693.57 | 327.81 | 2.116x |
| batch=32, ctx=80000, dist=soft, layer=sliding, load=verify 32, page=1, q_len=8, spread=agentic, weight=46 | 704.72 | 303.85 | 2.319x |
| batch=32, ctx=80000, dist=soft, layer=full, load=verify 32, page=1, q_len=8, spread=agentic, weight=16 | 1040.11 | 464.10 | 2.241x |

## muse_moe_w4a16

Rank **41 of 1122**, speedup **2.4015x** over SGLang (95% CI 2.3655-2.4376), judge job `e1bcad5a55cd4a29`. File: `kernels/muse_moe_w4a16.py` (our work on vjeux's base (Apache-2.0), with code following Matsvei (kernarc 'matvey')).

| shape | SGLang (us) | ours (us) | speedup |
|---|---:|---:|---:|
| bs=1, load=decode 1, sharding=ep8, weight=60 | 77.90 | 24.64 | 3.162x |
| bs=16, load=decode 16, sharding=ep8, weight=60 | 188.88 | 92.27 | 2.047x |
| bs=32, load=decode 32, sharding=ep8, weight=60 | 206.99 | 123.96 | 1.670x |
| bs=8, load=verify 1, sharding=ep8, weight=60 | 142.38 | 63.61 | 2.238x |
| bs=128, load=verify 16, sharding=ep8, weight=60 | 468.96 | 175.24 | 2.676x |
| bs=256, load=verify 32, sharding=ep8, weight=60 | 615.50 | 207.76 | 2.963x |

## muse_bf16_proj

Rank **75 of 1401**, speedup **1.5502x** over SGLang (95% CI 1.5269-1.5734), judge job `f07235a6ca8a46e2`. File: `kernels/muse_bf16_proj.py` (our work on vjeux's base (Apache-2.0) with Nathen Fernandes' routes (MIT) and Matsvei's code).

| shape | SGLang (us) | ours (us) | speedup |
|---|---:|---:|---:|
| layer=qkv_gate, load=decode 1, m=1, weight=62 | 13.82 | 8.83 | 1.566x |
| layer=o, load=decode 1, m=1, weight=62 | 7.04 | 5.24 | 1.345x |
| layer=pooler, load=decode 1, m=1, weight=62 | 21.25 | 13.36 | 1.591x |
| layer=unpooler, load=decode 1, m=1, weight=62 | 7.01 | 3.58 | 1.956x |
| layer=router, load=decode 1, m=1, weight=62 | 10.50 | 3.69 | 2.842x |
| layer=qkv_gate, load=decode 16, m=16, weight=62 | 13.82 | 9.65 | 1.432x |
| layer=o, load=decode 16, m=16, weight=62 | 6.72 | 5.45 | 1.234x |
| layer=pooler, load=decode 16, m=16, weight=62 | 20.94 | 13.54 | 1.547x |
| layer=unpooler, load=decode 16, m=16, weight=62 | 18.56 | 13.64 | 1.361x |
| layer=router, load=decode 16, m=16, weight=62 | 14.82 | 4.50 | 3.289x |
| layer=qkv_gate, load=decode 32, m=32, weight=62 | 14.24 | 10.78 | 1.321x |
| layer=o, load=decode 32, m=32, weight=62 | 7.07 | 6.09 | 1.162x |
| layer=pooler, load=decode 32, m=32, weight=62 | 21.82 | 14.87 | 1.467x |
| layer=unpooler, load=decode 32, m=32, weight=62 | 18.62 | 14.46 | 1.288x |
| layer=router, load=decode 32, m=32, weight=62 | 15.14 | 5.24 | 2.891x |
| layer=qkv_gate, load=verify 1, m=8, weight=62 | 14.18 | 9.78 | 1.449x |
| layer=o, load=verify 1, m=8, weight=62 | 6.94 | 5.52 | 1.259x |
| layer=pooler, load=verify 1, m=8, weight=62 | 21.20 | 13.55 | 1.565x |
| layer=unpooler, load=verify 1, m=8, weight=62 | 6.96 | 4.11 | 1.693x |
| layer=router, load=verify 1, m=8, weight=62 | 26.54 | 4.38 | 6.065x |
| layer=qkv_gate, load=verify 16, m=128, weight=62 | 16.86 | 13.79 | 1.223x |
| layer=o, load=verify 16, m=128, weight=62 | 7.36 | 6.96 | 1.057x |
| layer=pooler, load=verify 16, m=128, weight=62 | 25.65 | 19.94 | 1.286x |
| layer=unpooler, load=verify 16, m=128, weight=62 | 20.00 | 19.07 | 1.048x |
| layer=router, load=verify 16, m=128, weight=62 | 16.16 | 5.77 | 2.802x |
| layer=qkv_gate, load=verify 32, m=256, weight=62 | 20.02 | 17.27 | 1.159x |
| layer=o, load=verify 32, m=256, weight=62 | 9.50 | 9.26 | 1.026x |
| layer=pooler, load=verify 32, m=256, weight=62 | 31.58 | 24.36 | 1.297x |
| layer=unpooler, load=verify 32, m=256, weight=62 | 24.26 | 23.20 | 1.046x |
| layer=router, load=verify 32, m=256, weight=62 | 17.76 | 7.35 | 2.418x |

## muse_prefill_layer

Rank **34 of 648**, speedup **1.8167x** over SGLang (95% CI 1.7895-1.8440), judge job `7f87198d13a34650`. File: `kernels/muse_prefill_layer.py` (vjeux's public file, submitted unchanged (attribution header added)).

| shape | SGLang (us) | ours (us) | speedup |
|---|---:|---:|---:|
| batch=1, ctx=80000, dist=soft, layer=sliding, load=turn 1, page=1, q_len=753, repeats=6, weight=46 | 1134.78 | 585.11 | 1.939x |
| batch=1, ctx=80000, dist=soft, layer=full, load=turn 1, page=1, q_len=753, repeats=6, weight=16 | 1488.30 | 913.02 | 1.630x |
| batch=4, ctx=80000, dist=soft, layer=sliding, load=turn 4, page=1, q_len=753, repeats=6, weight=46 | 3379.42 | 1785.24 | 1.893x |
| batch=4, ctx=80000, dist=soft, layer=full, load=turn 4, page=1, q_len=753, repeats=6, weight=16 | 4934.94 | 3134.27 | 1.575x |
| batch=1, ctx=0, dist=soft, layer=full, load=first turn, page=1, q_len=16384, repeats=6, weight=16 | 17688.19 | 9426.50 | 1.876x |
| batch=1, ctx=32768, dist=soft, layer=sliding, load=first turn, page=1, q_len=16384, repeats=6, weight=138 | 17043.28 | 8879.56 | 1.919x |
| batch=1, ctx=32768, dist=soft, layer=full, load=first turn, page=1, q_len=16384, repeats=6, weight=16 | 21117.58 | 12402.68 | 1.703x |
| batch=1, ctx=65536, dist=soft, layer=full, load=first turn, page=1, q_len=16384, repeats=6, weight=16 | 24177.55 | 15194.82 | 1.591x |

## muse_attention_decode

Rank **46 of 1077**, speedup **2.6530x** over SGLang (95% CI 2.6132-2.6928), judge job `1c8d81a0dbd94ee0`. File: `kernels/muse_attention_decode.py` (vjeux's public file, submitted unchanged (attribution header added)).

| shape | SGLang (us) | ours (us) | speedup |
|---|---:|---:|---:|
| batch=1, ctx=80000, dist=soft, heads=tp8, layer=sliding, load=decode 1, page=1, weight=46 | 20.75 | 5.19 | 4.002x |
| batch=1, ctx=80000, dist=coherent, heads=tp8, layer=full, load=decode 1, page=1, weight=16 | 31.65 | 10.95 | 2.890x |
| batch=16, ctx=80000, dist=soft, heads=tp8, layer=sliding, load=decode 16, page=1, spread=agentic, weight=46 | 27.15 | 7.91 | 3.434x |
| batch=16, ctx=80000, dist=coherent, heads=tp8, layer=full, load=decode 16, page=1, spread=agentic, weight=16 | 202.02 | 83.65 | 2.415x |
| batch=32, ctx=80000, dist=soft, heads=tp8, layer=sliding, load=decode 32, page=1, spread=agentic, weight=46 | 31.81 | 9.70 | 3.279x |
| batch=32, ctx=80000, dist=coherent, heads=tp8, layer=full, load=decode 32, page=1, spread=agentic, weight=16 | 387.33 | 159.93 | 2.422x |
| batch=1, ctx=80000, dist=soft, heads=tp8, layer=sliding, load=verify 1, page=1, q_len=8, weight=46 | 23.23 | 6.77 | 3.430x |
| batch=1, ctx=80000, dist=coherent, heads=tp8, layer=full, load=verify 1, page=1, q_len=8, weight=16 | 36.58 | 14.25 | 2.567x |
| batch=16, ctx=80000, dist=soft, heads=tp8, layer=sliding, load=verify 16, page=1, q_len=8, spread=agentic, weight=46 | 29.90 | 11.77 | 2.540x |
| batch=16, ctx=80000, dist=coherent, heads=tp8, layer=full, load=verify 16, page=1, q_len=8, spread=agentic, weight=16 | 206.98 | 99.98 | 2.070x |
| batch=32, ctx=80000, dist=soft, heads=tp8, layer=sliding, load=verify 32, page=1, q_len=8, spread=agentic, weight=46 | 34.94 | 14.82 | 2.359x |
| batch=32, ctx=80000, dist=coherent, heads=tp8, layer=full, load=verify 32, page=1, q_len=8, spread=agentic, weight=16 | 392.46 | 182.13 | 2.155x |
