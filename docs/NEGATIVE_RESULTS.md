# Negative results: GPU kernel optimizations that did not work (Open Kernel Challenge, H200)

This file lists optimizations that team alygg77 tried and closed with evidence while working on the Muse Spark Open Kernel Challenge (OKC, https://openkernelchallenge.dev) on the NVIDIA H200 SXM GPU. The model is Muse Spark Open, a 256-expert mixture-of-experts (MoE) model with NVFP4 expert weights, BF16 dense weights and an FP8 KV cache. The boards are `muse_decode_layer`, `muse_moe_w4a16`, `muse_bf16_proj`, `muse_prefill_layer` and `muse_attention_decode`. Each entry says what we tried, the measured result with numbers, and why it lost. Topics include ring buffers and pipeline depth, MoE tail fixes, GPU-scope fences, the 16-token decode MoE (d16), 32-token job tiles (WGS32), qkv stream-K and split-K, balancing of full verify attention, the V transpose, prefill overlap, boundary kernels and stage counts, FP32 router variants, and gaps between local and judge results. The companion file `H200_MEASURED_FACTS.md` lists the measured facts behind these results.

## How to read this file

- Every entry is a measurement. A closed lever is closed only inside its scope. Where we measured on an earlier kernel base, the entry says so.
- "Faster by X" and "slower by X" are written out on purpose, because our tools used different sign conventions.
- Load names: `d1`, `d16`, `d32` are decode loads with 1, 16 and 32 tokens. `v1`, `v16`, `v32` are verify loads with 1, 16 and 32 requests of 8 tokens each. The suffix `s` means sliding-window attention (2,048-token window). The suffix `f` means full attention (about 80,000 tokens of context). `t1` and `t4` are prefill turn 1 (753 tokens) and turn 4 (3012 tokens).
- Terms: sk2 = our name for the swapped-operand dense GEMV kernel family used for 32 tokens or fewer. G1 and G2 = the first and second expert GEMM of the Marlin MoE kernel pair. SM = streaming multiprocessor. CTA = thread block. L2 = the GPU-wide cache. TMA = tensor memory accelerator (bulk copy). wgmma / HGMMA = warpgroup matrix multiply instruction. GEMV and GEMM = matrix-vector and matrix-matrix product; `m` is the number of activation rows. PDL = programmatic dependent launch. DSMEM = shared memory read across the CTAs of one cluster. NCU = NVIDIA Nsight Compute. SASS = compiled GPU code. KPERM = K-permutation, a lossless re-layout of NVFP4 weights that shares scale decoding (see the facts file).
- WGS = the label we use for the persistent MoE expert kernel build that we derived from vjeux's code (https://github.com/vjeux/okc-2026, Apache-2.0). It works in "units" of 16 weight stages. "Phase A" is the first half of a unit and "phase B" the second half.
- A/A = a test that compares two identical arms. It measures the noise floor. CI = confidence interval of a paired difference. Pad = a random 0 to 32 MiB allocation before an arm's `prepare()` call, to imitate the judge's memory layout.
- Judge clusters are named `sfo3` and `sfo3c` in the judge output.

---

## muse_decode_layer

### 1. Starting the pooler and router stream before attention
- **Tried:** Run the 58.7 MB pooler and router weight stream on a side stream while attention runs, because that stream depends on attention only through a 1024-wide term.
- **Result:** `d32s` was slower by 2.1 to 7.7 us.
- **Why it lost:** Only about 6 us of overlap exists before attention ends. The second root branch of a graph-start fork starts 21 to 33 us late. Side streams take SM slots from attention and from the `o` projection. Verify attention (`attn_kernel<8,false,3>`, 128 CTAs of 94 KB) slows by 50% to 80% next to any co-resident GEMM: `v32s` goes from 14.9 us to 22 to 27 us and `v16s` from 11.6 us to 16 to 17 us.

### 2. Side-stream splits of router and experts (earlier kernel base)
- **Tried:** Split the router and pooler across two streams, and split the Marlin expert pair into two half-grids on two streams.
- **Result:** The router and pooler split was slower by 2.132 us (`v16s`), 3.223 us (`v32s`) and 3.177 us (`v32f`). The two-stream Marlin halves changed the partitioning, broke bit identity, and were slower by up to 35 us (`d16f` 35.3 us; `d32f` 11.9 and 21.8 us).
- **Why it lost:** Each junction that joins a side stream back into a PDL chain cost at least 2 us per layer. A side-stream design must hide more than about 3 us of measured serial gap to win. Concurrency inside one kernel is not covered by this result.

### 3. Skipping glue launches, more pre-wait stages, and early PDL triggers
- **Tried:** Remove glue launches at T of 8 or less, issue more weight stages before the PDL wait, and move the PDL trigger of the projection kernels after the weight issue.
- **Result:** Skipping one glue launch saves only 1.2 to 2.0 us (4.0 to 4.8 us for all three at `d1s` and `v1s`). Two pre-wait stages are optimal; four or six stages slow attention by 1.7 us and the boundary kernel `b1` by 0.9 us. Early PDL triggers gained at most 0.736 us in real-kernel pairs.
- **Why it lost:** At T of 8 or less the layer is one serial PDL path with no idle gaps. The next GEMV already streams two pre-wait stages during the glue kernel. A matched synthetic stand-in with identical bytes and fake math had predicted 4.67 to 5.46 us per sliding shape, about 6.6 times the real gain.

### 4. L2 prefetch of the next kernel's weights (earlier kernel base)
- **Tried:** Prefetch the next kernel's weights into L2 from inside sliding attention, prefetch the unpooler weights during MoE, and add early L2 hints in the Triton sliding-attention prologue.
- **Result:** Prefetch from sliding attention was slower by 2.6 to 5.1 us at `d1`, `d16` and `d32` (two seeds). The unpooler prefetch during MoE lost because the lines were evicted. L2 hints for page-table rows gave 0 us. Hints for K/V rows made `v32s` slower by 2.1 and 3.0 us, and both together by 3.1 and 3.9 us.
- **Why it lost:** The unpooler lines were evicted. We did not isolate why prefetch from sliding attention was slower. Prefetch inside one persistent kernel that owns the L2 traffic was not measured.

### 5. Launch-table knob search (earlier kernel base)
- **Tried:** A 20-point evolutionary search over the launch-table knobs of `v16s`.
- **Result:** The best point gained 0.15 us, which is inside A/A noise. Identical arms differed by 1 to 1.5 us.
- **Why it lost:** A gain below about 1.5 us needs a repeat to leave the A/A band.

### 6. cuBLASLt algorithm search for the dense verify GEMMs
- **Tried:** A full sweep of tiles, stages, clusters, swizzles, split-K 2 to 8 and batch layouts (1,789 to 2,309 valid configurations per shape) at m128 and m256.
- **Result:** About 2 us per layer of total headroom. The best five picks made `v32s` faster by 1.97 and 2.16 us and `v32f` faster by 2.8 and 3.4 us (judge seed and second seed). Standalone savings shrank by about 35% inside the layer. No split-K appeared in the top 60.
- **Why it lost:** The m256 GEMMs are tensor-bound (the pooler-router GEMM runs 19.3 GFLOP at 60% to 62% of 989 TFLOP/s). Stream-K and split-K dense kernels were tested separately (entry 8 below and `muse_bf16_proj` entry 1).

### 7. Removing split-K and its combine from the dense chain (earlier kernel base)
- **Tried:** Premise check for removing split-K in the `d16` and `d32` dense chain.
- **Result:** The router chain ended 7.0 to 11.1 us before the expert kernel. The exposed combine time was 0.00 us on 6 of 6 rows. The only exposed split-K was the qkv reduce at 1.25 to 1.47 us. The ceiling of the idea is about 0.12% of the board at `d16`.
- **Why it lost:** Split-K was almost never on the critical path.

### 8. Stream-K or more CTAs for the decode qkv GEMV
- **Tried:** Stream-K and SM fill for the sk2 qkv GEMV at T of 16 and 32, on the premise that 36 SMs sit idle.
- **Result:** The same 1,536 CTA-stages end after 10.8 to 11.4 us on 96 CTAs and after 10.7 to 10.8 us on 128 CTAs (m32). The ceiling for stream-K is at most 0.6 us. The measured fix-up tail is 1.5 to 4 us.
- **Why it lost:** The kernel saturates its aggregate CTA-stage rate near 96 CTAs, so the idle-SM premise does not hold. The result does not change with x loads, the escape warp, stream layout or ring depth.

### 9. Rewrites of the `d1` BF16 codec GEMV (earlier kernel base)
- **Tried:** A lane-major weight repack, and a producer-warp decode stage (4 variants).
- **Result:** The repack was slower by 0.78 to 1.02 us. The producer-warp decode was slower by 6.28 and 6.50 us (two seeds).
- **Why it lost:** We did not isolate the cause. Fewer bytes (a new codec format) were not measured here.

### 10. Nathen Fernandes's T = 128 QKV block added to the decode layer
- **Tried:** The T = 128 QKV wgmma block by Nathen Fernandes (https://github.com/NathenFernandes/muse-bf16-proj, MIT) appended at T = 128 only.
- **Result:** The board was slower by 0.091% (judge seed) and 0.160% (second seed), pooled over 4 pads, 2 processes and 2 seeds. `v16` was faster by 0.30 to 1.51 us. `d32f` was slower by 1.82 and 1.87 us, `v32` slower by 0.23 to 0.99 us, and `d16` slower by 0.19 to 0.72 us. On an older base the same block at T = 256 lost 5.5 to 7.6 us.
- **Why it lost:** The block runs no GPU code at T of 32 or 16, so the losses are most likely a memory-layout effect of the larger module. We did not measure that cause.

### 11. Growing `prepare()` memory
- **Tried:** Add 3 MB to one scratch buffer that `prepare()` allocates, and add an extra build with a prepare-time extension load.
- **Result:** `d1` and `v1` were slower by 0.5 to 1.3 us with SASS identical to the base. Over 4 pads the extra scratch cost `v32s` 1.12 us. On the MoE board, `v32` was slower by 2.43 us on unchanged kernels. Reusing unused rows of an existing buffer removed the loss.
- **Why it lost:** Allocation layout, TLB or L2 set placement may be the cause. We did not measure the mechanism. A regression on SASS-identical shapes is a layout suspect.

### 12. FP32 router variants at T of 8 or less (precision rule)
- **Tried:** Several ways to give the router an FP32 activation as the SGLang reference does (rule found by Matsvei, kernarc user `matvey`, https://agents.kernarc.com).
- **Result:** A separate FP32 GEMV after the pooler was slower by 5.5 us (`d1`) and 6.5 us (`v1`), a 2.27% board loss. A separate pooler launch plus cuBLAS `F.linear` was slower than the high-plus-low-row design by 5.5 to 6.0 us (`d1`) and 19.8 to 20.4 us (`v1`), about 5.0% of the board. A router in the fused pooler tail was slower by 0.23% to 0.24% of the board. The interleaved FP32 stage reached parity.
- **Why it lost:** The FP32 stages move twice the bytes of the BF16 high-plus-low stages. About 0.6 us of the 1.5 us cost at `v1` is FP32 FMAs and x bytes; about 0.7 us is unexplained edit overhead. See `H200_MEASURED_FACTS.md`, section 5.

---

## muse_moe_w4a16

### 1. Deeper rings and more weight slots
- **Tried:** A fourth weight slot in the WGS pipeline, other x-ring geometries, non-warp-uniform stage offsets, and (on earlier kernels) deeper rings and L2 prefetch of the next bytes.
- **Result:** A fourth slot changed every WGS load by between 1.0 us slower and 0.9 us faster. X-ring geometry changes only lost. Non-warp-uniform stage offsets spill at the 96-register cap. On earlier kernels, codec GEMVs stayed at 2.2 to 3.2 TB/s and NVFP4 experts at 2.3 to 3.9 TB/s with 108 KiB in flight per SM.
- **Why it lost:** The weight pipeline saturates at 3 slots. The stream is boundary-bound (unit ends), not depth-bound.

### 2. Tail fixes: dynamic claims, dispersed ends, static-split retunes
- **Tried:** (a) A dynamic phase-B tail with whole-unit claims. (b) Dispersing the unit ends. (c) Moving the read-back into one helper warp. (d) A refit per-n-tile cost table for the static split.
- **Result:** (a) cannot win: one phase-B unit costs a CTA 18.5 us (`d32`, `v16`) or 19.4 us (`v32`), each CTA holds only 1.7 to 3.8 units, and the end spread to remove is 2.7 to 3.7 us or 5.9 to 8.2 us. The unit is 2.5 to 7 times the spread, so the last finisher can only move later. Sub-unit claims need stream-K fix-ups (at most 28 free slots, about 4 us of work per CTA) whose tails (1.5 to 4 us) eat the gain. (b) gained only about 1 us at `d32`. (c) lost. (d) halved the phase-A duration spread on held-out seeds but was slower by 0.8 to 4.6 us on the `d32` judge cases.
- **Why it lost:** The slowest CTA just moves to another range. What the kernel waits on is the phase-B end spread (maximum minus mean): 3.5 us at `d32`, 3.9 to 4.6 us at batch sizes 64 to 192, 6.9 us at `v32`. Every 16-stage unit end stalls a CTA for 1 to 7 us, and the stream is boundary-bound.

### 3. Fences: moving them, and removing the split-unit read-back fence
- **Tried:** Move releases, counts and read-backs to a helper or producer warp. Drop the `fence.acq_rel.gpu` before the split-unit read-back (a build we call NOFENCE).
- **Result:** One fence per phase-A issue in the x producer slowed the kernel by 6.8 to 19.5 us (about 0.13 us per fence). NOFENCE made `v16` faster by 0.86 to 0.96 us and `v32` faster by 1.19 to 1.78 us in 8 of 8 rows each, and `d32` faster by 0.94 and 0.04 us. The board gain was +0.234% (A/A -0.04%), and no row beat A/A noise (0.7 to 3.1 us). On the decode layer the same edit gave no gain (`d32f` slower by 1.35 us in 4 of 4 pads). The edit is memory-safe under the PTX model and passed 10,000 reference-checked replays at each of `d32`, `v16` and `v32` with no failures.
- **Why it lost:** Fence count is the cost, not the warp that runs the fence. Moving the work cannot win unless the number of fences per CTA drops. The one fence removed gains less than the judge noise.

### 4. A 32-token WGS job tile (WGS32)
- **Tried:** A job tile for 17 to 32 tokens, which gained 1.3 to 3.8 us on another public H200 base.
- **Result:** Our `d32` kernels were slower by 2.2 to 3.0 us. The phase-B end moved later by 1.3 to 3.9 us. Route, gather and tail were unchanged.
- **Why it lost:** The instruction count per weight word was equal, but the compiler's stage-loop schedule differed. Stream rate is sensitive to schedule, not only to instruction count. A public gain must be re-timed on our base.

### 5. The 16-token MoE (d16) second weight read
- **Tried:** Remove the second weight read of the split 16-token expert, which reads its weights twice (7.7% to 11% of the bytes).
- **Result:** Other CTAs sped up by only 1%. Byte floors are about 90 us with the second read and about 84 us without it.
- **Why it lost:** The `d16` mid path is co-bound: phase A streams at the size-matched ceiling (4.10 to 4.16 TB/s) while each CTA runs 1.12 us per 36 KB unit, and phase B is consumer-bound (NCU issue utilization 51.9%).

### 6. Expert byte compression
- **Tried:** Lossless coding of NVFP4 expert weights.
- **Result:** The E2M1 codes use exactly 4.000 bits and the E4M3 scales 3.71 bits. The best practical code saves about 5% of the bytes.
- **Why it lost:** The weights behave like independent random draws. Kernels that beat our byte floor are faster for other reasons.

### 7. Marlin and fused-lean replacements (earlier kernel base)
- **Tried:** Three drop-in replacements for the Marlin G1/G2 pair at 16 and 32 tokens, SIMT FP32 bodies for 1-token and 2-token route groups, a readiness-aware G2-first queue, and more Marlin CTAs per SM.
- **Result:** The replacements lost at 16 and 32 tokens: the fused-lean path was slower by 10 to 29 us, and one variant reached only 1.1 to 1.4 TB/s. The phi pair gained only at 16-token sliding (4.8 and 3.1 us faster). The SIMT bodies were slower by 52.8 to 84.1 us (1-token groups) and 80.2 to 149.2 us (up to 2 tokens). A zero-decode probe showed a ceiling of +0.53% of the board. The G2-first queue was slower by 0.86 to 1.65 us at `v16s` and `v32s`. Each extra 132 Marlin CTAs cost about 1.0 us at `d32` (up to 4.05 us for the largest step).
- **Why it lost:** The per-weight NVFP4 decode, not the MMA padding, is the cost of tail items. Per-expert ready counters had a measured ceiling near zero at `d1` and `v1` (publish plus acquire about 0.9 us).

### 8. Local gains that did not show on the judge
- **Tried:** A MoE build with extra changes beyond KPERM3, whose local `d16` and `v1` gains were +0.97% and +1.33% against the older file.
- **Result:** On the judge, `d16` read -0.7% and `v1` read +0.2% (noise). The gain showed only on the loads that run the WGS build (`d32`, `v16`, `v32`).
- **Why it lost:** We did not isolate the cause. The judge (2 runs per arm, one cluster) showed gains only on the loads that run the WGS build.

---

## muse_bf16_proj

### 1. Split-K and stream-K for qkv at m256
- **Tried:** A 128x256 split-K-7 kernel on 126 CTAs for qkv at m256.
- **Result:** The K loop ran at 94% of the clock-corrected peak, but the kernel was slower by 3.3 to 4.6 us.
- **Why it lost:** Per-SM write bandwidth to L2 is only about 40 to 50 GB/s. Writing a 128 KB FP32 partial per SM takes 2.6 to 4.1 us and reading the peers' 110 KB back takes 1.3 to 2.8 us. So split-K at m of 128 or more pays at least tile area times 4 B divided by about 45 GB/s per SM. DSMEM reads run at only about 30 GB/s per SM.

### 2. In-register lossless weight decode at m of 128 or more
- **Tried:** Decode the 12-bit lossless BF16 code in registers inside the GEMM.
- **Result:** The unpooler and `o` at m128 were slower by 2.4 to 4.1 us. The pooler at m128 was slower by about 12 us (32.6 us against 20.0 to 21.4 us).
- **Why it lost:** Integer ALU work does not overlap an in-flight HGMMA on the same SM (ALU only 286 clocks plus MMA only 256 clocks became 489 clocks together). The codec pays only where the matrix-multiply work per weight byte is small (m of 32 or less). Even with free decode the byte saving is at most +2.4%, and about +0.5% to +1% in practice.

### 3. The uncompressed cuBLASLt route search at m256
- **Tried:** QKV and `o` routes at m256 (verify 32) through uncompressed cuBLASLt configurations.
- **Result:** QKV was slower by 1.12 to 5.03 us and `o` by 0.11 to 0.72 us (8 captures per arm). Three more QKV configurations failed the algorithm check.
- **Why it lost:** We did not isolate the cause. Related measurements: the base qkv m256 kernel moves 66 B per clock per SM from L2 with the tensor pipe 53% busy, and the `o` m256 K loop is already 87% tensor-busy.

### 4. Small-m routes from Nathen Fernandes that lost
- **Tried:** Three of Nathen Fernandes's small-m routes (https://github.com/NathenFernandes/muse-bf16-proj, MIT) against our own paths.
- **Result:** In 8 of 8 local runs (4 pads times 2 seeds), his unpooler m8 route was slower by 0.175 us, his router m1 route by 0.163 us, and his `o` m16 route by 0.098 us. His `o` m8, pooler m8 and pooler m16 routes gained, and we kept them (see `H200_MEASURED_FACTS.md`, section 8).
- **Why it lost:** We measured these routes against our own unpooler m8 path (direct sk2 escape) and router m1 path (`rgemv` or a lossless `iv_router24`), and they were slower. The full graft of all his routes gained only +0.36% of the board, against +0.60% for the three that won.

### 5. Forcing the router m1 path
- **Tried:** Force the router at m1 to `rgemv` and add a per-thread cuBLASLt context bind.
- **Result:** The local change was +0.054% (CI -0.653% to +0.760%, 4 pads). The judge scored 1.5334 against 1.5427, 1.5489 and 1.5387 for the previous file.
- **Why it lost:** The previous file already chose `rgemv` or the lossless `iv_router24` path at prepare time, both 2.5 us on the judge. The context bind remains a robustness fix for fresh threads and not a speed change. Matsvei (kernarc user `matvey`) wrote the bind helper.

---

## muse_prefill_layer

### 1. Hiding memory-bound glue under the GEMMs
- **Tried:** Run a persistent boundary kernel (residual and RMSNorm glue) on `k` SMs beside the pooler GEMM, and other overlap layouts.
- **Result:** The pooler GEMM plus boundary on `k` SMs took 1,570 us against 1,650 us serial (4.8% less time), below the 8% we needed to justify the build. The union made the GEMM 8% to 15% slower. Memory traffic from other work slows these GEMMs a further 12% to 18%. Hiding the 1.07 GB boundary would take at least 12 GEMM SMs. The best case saves about 6% on the pair as built and about 10% with a free consumer.
- **Why it lost:** The GEMMs are power-capped (the pooler ran at 1.53 GHz and read 1.16 TB/s). The boundary kernel reaches only 58 GB/s per SM with its full math (118 GB/s with loads only, 157 GB/s for the bulk-copy ring alone). Gains must come from inside the glue kernels, with fewer bytes or faster kernels.

### 2. Persistent boundary kernels
- **Tried:** A persistent 12-slot bulk-copy boundary kernel.
- **Result:** It used 4% less time in the first-boundary recompute mode (about +0.04% of the board). It used 3% to 32% more time in the other modes and spilled registers. The old kernels already stream 3.4 to 4.1 TB/s at T = 16,384 (head-to-head, same process). An earlier 260 us against 285 to 294 us gap came from different processes.
- **Why it lost:** The boundary kernels are already near the memory ceiling. Glue gains must remove bytes.

### 3. Stage counts, narrower wgmma and extra copies in the expert kernels
- **Tried:** Narrower wgmma instantiations, deeper waits, composite tiles, and extra bulk copies per stage.
- **Result:** Narrower instantiations and deeper waits hit `ptxas` warnings C7511 and C7512 (serialized wgmma). Composite tiles cost about 10 ns per extra wgmma per k16 step. Extra bulk copies per stage slowed every N32 item, even when fewer bytes moved. N32 items keep a floor of about 9.2 us whether or not we remove the decode or the MMA.
- **Why it lost:** The T = 753 expert kernel is consumer-instruction bound (removing loads gains 3%; removing 12 instructions per k-block gains 5.8%). The T = 3012 kernel keeps its tensor pipe 82% to 85% active at 1.70 GHz, with DRAM use at only 22% to 29% of the ceiling.

### 4. KPERM at the largest token counts
- **Tried:** Apply the KPERM scale sharing to the T = 16,384 expert kernel.
- **Result:** The board was slower by 0.4% to 0.5%.
- **Why it lost:** We did not isolate the cause. We keep KPERM below 2,048 tokens (with a fine-plan build up to 4,096).

### 5. Merging the turn-1 attention splits through global partials
- **Tried:** A last-arriving-CTA merge through global partials plus `__threadfence`.
- **Result:** Slower than the combine kernel by 2 to 4 us. The 2-CTA cluster merge through DSMEM instead saved about 2 us per call.
- **Why it lost:** We did not isolate the cause.

### 6. Attention exponent tricks at q-length 16,384
- **Tried:** A premise check for faster exponentials in prefill attention.
- **Result:** Attention is only 12.3% of the first-turn load. A kernel with no exp at all still takes 87% to 91% of the time. Even a MUFU floor everywhere would give +1.2% of the board.
- **Why there is little to gain:** The kernel is co-bound by MUFU `ex2` (72% to 80% busy) and by issue (349 instructions per warp-tile).

### 7. Local gain against judge result
- **Tried:** A stack of KPERM experts below 2,048 tokens plus the cluster merge; later a fine-plan build up to 4,096 tokens.
- **Result:** The judge gain was +0.31% against a same-hour base (2 runs per arm; the pairs disagreed in sign), against a local estimate of +0.35%. The later stack (counted local gain +0.41%) scored 1.8040 and did not beat the earlier row of 1.8167. Same-file base re-checks read 1.7958 and 1.8002.
- **Why it lost:** The earlier row was a fast run. On this board, a gain below about 1% cannot raise the row unless the run lands fast.

---

## muse_attention_decode

### 1. Capping the finalize rows at 4 (two implementations)
- **Tried:** Cap the coop finalize at 4 rows, then a second occupancy-safe version, then the same with an added STSM (matrix store instruction) in flat verify.
- **Result:** The first version made `v16s` faster by 0.19 and 0.28 us (4 pads, two seeds) but `v1s` slower by 0.04 and 0.08 us and `v32s` slower by 0.09 and 0.15 us in 9 of 10 rows. The board was -0.045% (CI -0.174% to +0.085%). The second version and the STSM stack fell below the A/A control on the weighted result over all 12 shapes.
- **Why it lost:** Gains at one shape were paid back at the other shapes. Other finalize designs are not covered.

### 2. Balancing full verify attention (`v16f`, `v32f`)
- **Tried:** A balanced dynamic pair split, weighted splits, a polynomial `exp2` (degree 5, relative error 2.1e-7), reordered exponential work, and a one-CTA FA3 (FlashAttention-3) style intra-warpgroup pipeline.
- **Result:** The dynamic split made both CTAs of an SM pair end together (from 152 and 172 us to 172.4 and 173.5 us) but left the SM finish time unchanged. An open-loop consumer costs about 5 us of code generation by itself. The one-CTA pipeline took 116.4 us against 100.5 us. The other arms did not pass our gain gate.
- **Why it lost:** Full verify attention is bound by per-SM throughput under consumer load. One CTA per SM is only 16% to 18% slower than two. The consumer loop spends 40.5% on exponentials, 17.5% on QK plus the V transpose, 12.7% on the PV warpgroup barrier and 12.5% on data waits. The only real balance slack is about 4 us of spread across SMs. Time a feature-off build of every control-flow change.

### 3. Moving the V transpose to the producer warps
- **Tried:** Do the V transpose of flat q-length-8 verify in the producer warps.
- **Result:** `v16f` was slower by 0.55 to 0.87 us (CI below zero in 4 of 4 captures). `v32f` was inside A/A. A probe that skipped the producer's own-copy wait gave 99.81 us against 100.06 us.
- **Why it lost:** The consumer transpose was hidden behind the QK wgmma latency. Removing it only exposes that latency. One tile per warpgroup needs 72 accumulator, 32 score and 8 P registers in a 104-register budget, so we could not build it.

### 4. Balancing the waves of one-request full attention (`d1f`, `v1f`)
- **Tried:** 11 real-kernel arms: a 4-stage ring, a wide finalize, wave-weighted splits, and clusters.
- **Result:** All 11 arms lost. Balancing the waves without cutting the merge volume made the run slower by 0.75 us and raised the merge median from 0.96 to 1.5 us.
- **Why it lost:** One-request full attention is fixed-latency bound: 2 dependent DRAM round trips before the stream and 3 L2 round trips after the last merge (2.5 us at `d1f`, 2.8 us at `v1f`). A wave imbalance of about 2 us staggers 8 MB of FP32 `red.v4` merge atomics, and same-address `red.v4.f32` serializes at about 17 ns per operation.

### 5. Merging 128 to 318 splits in one CTA or one cluster (earlier kernel base)
- **Tried:** A single-CTA last-arrival merge, a cluster DSMEM merge (2.6 to 3.5 us push plus about 4 us wait), and a final `d1` design with 240 split CTAs in 30 eight-CTA clusters.
- **Result:** The single-CTA merge was slower by about 60 us (`d1` full) and 96 us (`v1` full). The cluster merge of the final design was slower by 1.181 and 1.118 us (judge seed and second seed).
- **Why it lost:** The merge was serial in one CTA or one cluster. The scope is decode `d1` and `v1` full attention with 128 to 318 splits. A merge of only two splits gained on prefill turn 1 (see `muse_prefill_layer` entry 5). Parallel multi-CTA tree merges were not measured.

### 6. Page sorting and length-sorted queues for full attention (earlier kernel base)
- **Tried:** Global per-request page sorting, and one persistent length-sorted CTA queue over the unchanged `_cuda_decode` splits.
- **Result:** Page sorting was slower by 161.9 to 275.7 us per full load (`d16f`, `d32f`, `v16f`, `v32f`, judge seed). The queue was slower by 3.5 to 6.6 us with 256 CTAs and by 38.9 to 70.0 us with 132 CTAs (`d16f`, `d32f`, two seeds, 8 of 8 rows bit-identical).
- **Why it lost:** These changes did not raise the stream rate, and fewer resident CTAs cost far more than any balance gain. Fewer bytes and overlap were not tested.

### 7. Folding verify Q preparation into the split attention core (earlier kernel base)
- **Tried:** Fold the 2.9 to 3.2 us `prep_queries` step into the split-K attention core at `v16f` and `v32f` (3 variants).
- **Result:** The net gain was only about 1.6 us. `v32f` gained 1.69, 1.84 and 2.15 us (judge seed) and 1.53, 2.64 and 0.68 us (second seed). `v16f` stayed inside A/A. The ceiling is about 0.09% of the board.
- **Why it lost:** Every split recomputed the 128-row Q tile (16 to 33 times redundant, 0.7 to 2.5 us more core time), and the K/V store and release added 1.3 to 1.5 us.

---

## Gaps between local results and judge results

- **Stand-in versus real kernel.** A synthetic stand-in (identical bytes, fake math) predicted about 6.6 times the real gain of the early PDL trigger change. Size bets from real-kernel runs, not from stand-ins.
- **Cross-cluster comparisons.** A `muse_bf16_proj` candidate on `sfo3c` scored 1.5489 against a base of 1.4866 on `sfo3` (+4.19%). The base value was the lowest of four runs, and the real gain was about +2.4%. On `muse_moe_w4a16` the cross-cluster gain was +2.46% against +1.25% inside one cluster. Read board deltas only inside one cluster.
- **Single-pair decisions.** A check of an unchanged `muse_bf16_proj` file read 1.5573 on a fast base run, and the submit of the same file scored 1.5420. A decision needs at least 2 judge runs per arm or a per-shape read where the changed shapes move and the unchanged ones do not.
- **The FP32 router costs twice as much on the judge.** At `v1` the judge priced the fused FP32 router at 3.8% (`v1s`) and 3.8% (`v1f`), against about 1.8% in local runs.
- **A local +0.04% became -1.7% on the judge.** A hi-plus-low-row FP32 router build was +0.04% locally and 1.7% slower on one judge pair (2.1780 against 2.2148, both `sfo3`, one run each). Board-level noise on unchanged files reached 1.6% to 2.2% in our runs.
- **Decode bundle.** A bundle with a local gain of +0.40% (CI +0.31% to +0.55%) read +0.91% on one same-cluster judge check (2.2094 against 2.1895, both `sfo3c`) and scored 2.1942 on `sfo3`. The `v32` expert region moved by +23.3 and -16.8 us between the two runs.
- **Unchanged files.** Four submits of one unchanged `muse_bf16_proj` file spread over 1.6% (1.5259 to 1.5502). Two judge submissions of decode files with the same kernels scored 0.5% apart.
- **An outside win that did not transfer.** A kernarc result (agents.kernarc.com, from Matsvei, user `matvey`) showed +2.4% to +3.0% at M = 8 for more Marlin CTAs per SM against stock Marlin. It did not transfer to a file that already sets the CTAs per SM for each load.
- **Memory layout.** A single-layout gain or loss smaller than about 1.5 us at `v32` (0.7 us at `d1`, `v1`, `d32`) moved with the pad. Measure with at least 4 pads.
