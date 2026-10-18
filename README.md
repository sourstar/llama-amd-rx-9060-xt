# llama-embed — Qwen3-Embedding-8B on AMD Radeon RX 9060 XT

The AMD counterpart of [`llama-intel-arc-b70`](https://github.com/sourstar/llama-intel-arc-b70):
the same harness, the same fixture pools and the same `history.csv` columns,
so a row here and a row there differ in the card and the stack, nothing else.
Started 2026-10-18 on Sofia's PC in Perugia (the dev peer of louise-rag).

## The machine

| | |
|---|---|
| card | AMD Radeon RX 9060 XT, 16 GB GDDR6 (16,304 MiB), gfx1200 (RDNA 4), 32 CUs, PCIe 5.0 x16 at `0000:03:00.0` |
| driver | AMD Software Adrenalin 26.8.1 (`26.10.41.01-260811a`) on Windows 11 Pro |
| CPU / RAM | Ryzen 7 9700X (8 cores, AVX-512), 61 GB |
| container host | Docker Desktop 29.8 (WSL2 kernel 6.18.33.2), the card reached as `/dev/dxg` |
| storage | everything on D: (C: is full) |

## The engine

LocalAI **v4.11.0** (`58830f7`) with louise-rag's patch series on its llama.cpp
backend (P1 load options, P2 token-id embedding; series digest
`724329e027f2`), built for gfx1200 by louise-rag's
`deploy/localai/llama-cpp/build.sh hipblas` and installed as
`louise-rocm-llama-cpp`. ROCm 7.2.1; librocdxg 1.2.2 is what lets ROCm reach
the card through `/dev/dxg` (`louise-localai:rocm`,
louise-rag `deploy/localai-desktop`).

The B70's numbers come from a standalone llama-server (SYCL b10630). This card
is benchmarked through LocalAI because that is what serves Louise here; the
LocalAI hop costs a gRPC call per request, small against an 8B prefill, and
a standalone ROCm llama-server row is a separate measurement if wanted.

## Configuration under test — the B70's

`localai/bench-qwen3-embedding-8b.yaml`: Qwen3-Embedding-8B Q4_K_M (sha256
`3fcd3feb…`, identical to the node's file), `context_size 8192`,
`parallel:1`, `batch = ubatch = 8192`, flash attention on, pooling last,
`gpu_layers 999`, and `ROCBLAS_DEFAULT_ATOMICS_MODE=0` (louise-rag CLAUDE.md
invariant 12 — AMD's bitwise-reproducibility mode).

`-np 1` is load-bearing on both cards: overlapping requests on a multi-slot
server corrupt embeddings (louise-rag memory, measured 2026-08-28). Client
concurrency above 1 only queues at the server.

## Run

```
sh scripts/run_bench.sh ["note"]
```

It copies the model YAML into LocalAI's models directory, restarts LocalAI,
runs `bench.py` on the bench pool (3 passes, warm-up of one full pass) at
client concurrency 1, 2 and 4, appends one row each to
`baselines/history.csv`, then runs the correctness gate (`capture.py`
twice in shuffled order + `compare.py --determinism`) and removes the YAML.
**One job on the card at a time.**

## The pools

Copied byte for byte from llama-intel-arc-b70 at `a2dae24`:

| file | sha256 | use |
|---|---|---|
| `fixtures/bench.json` | `7c1bf2d0…a70680e1e` | throughput: 24 real gutenberg chunks |
| `fixtures/chunks.json` | `77734410…7a2ca0df` | the gate: 16 boundary-straddling lengths, 8 to 7,900 tokens |

## Changes against the B70 harness

- `scripts/bench.py`, `scripts/capture.py`: the health path and the model
  name come from `HEALTH_PATH` (default `/health`) and `EMBED_MODEL`
  (default `embed`) — LocalAI answers `/readyz` and routes by model. Nothing
  else differs.
- `scripts/run_bench.sh` replaces the B70's container-starting scripts:
  LocalAI loads the model; nothing starts a llama-server here.

See `KNOWN-ISSUES.md` for what this card and this host do differently.
