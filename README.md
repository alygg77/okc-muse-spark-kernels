# Muse Spark Open Kernel Challenge (OKC): H200 CUDA kernels by team alygg77

Open-source kernels, judge results, measured H200 facts and negative results from team **alygg77** in the
[Muse Spark Open Kernel Challenge](https://openkernelchallenge.dev) (OKC). The challenge asks for faster
GPU kernels for Meta's **Muse Spark Open** model as SGLang serves it under TP8/EP8 on **NVIDIA H200 SXM**.
Muse Spark Open is a mixture-of-experts (MoE) model with 256 experts, NVFP4 expert weights, BF16
projections and an FP8 KV cache.

All five kernel boards are covered: `muse_decode_layer`, `muse_moe_w4a16`, `muse_bf16_proj`,
`muse_prefill_layer` and `muse_attention_decode`.

**Keywords:** CUDA, H200, Hopper, sm_90a, WGMMA, CUDA graphs, SGLang, mixture of experts, MoE, NVFP4,
W4A16, BF16 GEMV/GEMM, FP8 KV cache, paged attention, decode attention, speculative-decoding verify,
grouped-query attention (GQA), Muse Spark Open, Open Kernel Challenge, OKC, kernel optimization,
GPU performance.

## Results

These are the leaderboard rows read from the public API on **2026-10-29 at 20:54 UTC**. Speedup is the
judge's weighted geomean of SGLang's time divided by ours, measured with CUDA-graph replays from a cold
L2 cache. **All positions are provisional.** The organizers rerun the top entries after the kernel boards
close (Nov 4, 2026). Final positions come from the medians of those reruns.

| Board | Speedup vs SGLang | 95% CI | Rank | Kernel file |
|---|---:|---|---:|---|
| Overall kernel leaderboard | - | - | **40 of 1612** | - |
| `muse_decode_layer` (whole decoder layer) | **2.2058x** | 2.173-2.239 | 40 of 989 | [kernels/muse_decode_layer.py](kernels/muse_decode_layer.py) |
| `muse_moe_w4a16` (routed NVFP4 experts) | **2.4015x** | 2.366-2.438 | 41 of 1122 | [kernels/muse_moe_w4a16.py](kernels/muse_moe_w4a16.py) |
| `muse_bf16_proj` (BF16 projections + FP32 router) | **1.5502x** | 1.527-1.573 | 75 of 1401 | [kernels/muse_bf16_proj.py](kernels/muse_bf16_proj.py) |
| `muse_prefill_layer` (prefill layer) | **1.8167x** | 1.789-1.844 | 34 of 648 | [kernels/muse_prefill_layer.py](kernels/muse_prefill_layer.py) |
| `muse_attention_decode` (gated attention decode) | **2.6530x** | 2.613-2.693 | 46 of 1077 | [kernels/muse_attention_decode.py](kernels/muse_attention_decode.py) |

Per-shape tables (SGLang time, our time and speedup for every scored shape) are in [RESULTS.md](RESULTS.md).
The same data is in machine-readable form in [results.json](results.json), with each file's SHA-256 and judge job id.

The judge is noisy. One unchanged `muse_bf16_proj` file scored 1.526x, 1.541x, 1.544x and 1.550x in four
runs on one day, and the two judge clusters can differ by 0.4-1%. Treat differences under ~2% between
single runs as noise.

## Where each file comes from (credit)

All five files start from **vjeux's public OKC kernels** ([github.com/vjeux/okc-2026](https://github.com/vjeux/okc-2026),
Apache-2.0, commit f35a41e). Most of the speed on every board comes from that base. See [NOTICE](NOTICE).

- `muse_prefill_layer.py` and `muse_attention_decode.py` are **vjeux's files, unchanged**, with an attribution
  header added. Our own variants of these two stayed within judge noise of them.
- `muse_decode_layer.py`: our changes on vjeux's base include:
  - the exact FP32 router matvec (FP32 activation x FP32 weight, FP32 accumulation, the precision the
    rules require at T <= 8) fused into the pooler kernel, pinned so every prepare uses it;
  - the KPERM3 expert re-layout below for the expert stream at 17 or more tokens;
  - deepcopy-safe cuBLASLt holders;
  - a per-thread CUDA-context bind.
- `muse_moe_w4a16.py`: **KPERM3**, a lossless re-layout of the NVFP4 expert weights. The codes are permuted
  along K, so one scale pair serves four 16-wide K steps and the inner loop stops spilling registers. The
  codes, the scales and every dequantized value stay the same.
  - On the judge, the loads that use this path got faster: d32 +2.6%, v16 +3.9%, v32 +3.1% (same cluster).
  - The board score rose about +1.25%.
- `muse_bf16_proj.py` keeps vjeux's lossless 12-bit BF16 weight format. On top of it:
  - custom WGMMA O-projection kernels with FP32 accumulation;
  - pooler tuning;
  - **Nathen Fernandes'** small-M routes ([github.com/NathenFernandes/muse-bf16-proj](https://github.com/NathenFernandes/muse-bf16-proj),
    MIT; full text in the file header and in `licenses/`);
  - a `__deepcopy__` fix, so each copied state owns its weights. **Matsvei** (kernarc user `matvey`) found the
    pattern. The fix is speed-neutral (+0.02% +-0.06%).
- The per-thread CUDA-context bind and the cuBLASLt plan pins follow code posted by **Matsvei** on
  [agents.kernarc.com](https://agents.kernarc.com).

Each file's docstring describes its pipeline in detail.

## What is in this repo

| Path | What |
|---|---|
| `kernels/` | The exact files behind our five leaderboard rows (one self-contained Python file per board, CUDA inline) |
| `RESULTS.md`, `results.json` | Judge results per board and per shape |
| `docs/H200_MEASURED_FACTS.md` | Measured H200 and judge facts: bandwidth ceilings by size, L2 write rate, what bounds each kernel |
| `docs/NEGATIVE_RESULTS.md` | Ideas we built, measured and dropped, with numbers and the reason each one lost |
| `timing_kit/` | The paired cold-L2 CUDA-graph A/B/A timing harness used for every local claim |
| `llms.txt` | A short machine-readable summary for AI agents |
| `NOTICE`, `LICENSE`, `licenses/` | Credits and licenses |

## How to run a kernel file

Each file is a complete OKC submission. It defines `prepare(weights)` and `kernel_fn(data, state)`, and it
runs in the challenge's judge image (`lmsysorg/sglang:dev-muse-spark-open`) on one H200:

```bash
curl -fsSL https://openkernelchallenge.dev/install.sh | sh   # the official okc CLI
okc init muse_moe_w4a16                                       # problem files + local tests
okc check muse_moe_w4a16 kernels/muse_moe_w4a16.py            # unscored judge run
```

The rules: weights may be repacked or compressed only losslessly, activations stay BF16 where SGLang keeps
them BF16, the FP8 KV cache stays FP8, and every kernel must be CUDA-graph capturable. See
[openkernelchallenge.dev/llms.txt](https://openkernelchallenge.dev/llms.txt).

## License

Apache License 2.0 ([LICENSE](LICENSE)). The challenge terms license every submission under Apache-2.0.
Nathen Fernandes' code inside `kernels/muse_bf16_proj.py` stays under the MIT License, whose text is kept.
