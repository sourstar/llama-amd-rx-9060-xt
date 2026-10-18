# Known issues — RX 9060 XT under Docker Desktop (all measured 2026-10-18)

## Host and driver

**H1. ROCm works in Docker Desktop on RDNA 4 — with librocdxg.** There is no
`/dev/kfd` or `/dev/dri` under WSL2; Docker Desktop's VM does expose
`/dev/dxg` and `/usr/lib/wsl/lib`. AMD's librocdxg (1.2.x pairs with ROCm
7.2.x) bridges ROCm to the Windows driver: install `rocdxg-roct_1.2.2_amd64.deb`
in the image, mount `/usr/lib/wsl/lib/libdxcore.so` to `/usr/lib/libdxcore.so`,
set `HSA_ENABLE_DXG_DETECTION=1`. `rocminfo` then lists gfx1200. An Ubuntu
distro with its own Docker Engine is NOT needed.

**H2. Vulkan in a container finds only `llvmpipe`.** Ubuntu's Mesa ships no
`dzn` (Vulkan-on-D3D12) driver and RADV needs `/dev/dri`. The native Windows
Vulkan llama.cpp works on the card but has no bitwise mode and measured
14/16 run to run at `-np 4`.

**H3. Task Manager shows 0 %.** Its default GPU graphs are 3D/Copy/Video;
WSL compute lands on **Compute 0**. Read `\GPU Engine(*engtype_Compute*)\Utilization Percentage`.

**H4. GPU monitoring goes through the dxg build of amd-smi**
(`rocdxg-amd-smi-lib`, `/opt/rocm-wsl/bin/amd-smi`, source `/opt/rocm-wsl/.env.sh`).
`rocm-smi` reads `/sys/class/drm` and finds nothing. amd-smi's `process`
reports THREAD ids (map with `/proc/<tid>/status` `Tgid`) and lists every
process holding a GPU context, one running wholly on the CPU too; per-process
VRAM is `N/A`. Card-wide `metric --mem-usage` is real.

## Engine

**E1. `gpu_layers: 0` on a GPU build is not a CPU run.** llama.cpp still
offloads large batched matmuls to the card. A CPU twin needs
`no_op_offload:true` (and `HIP_VISIBLE_DEVICES=-1` to show HIP no device).

**E2. Which path does rocBLAS see?** Traced with `ROCBLAS_LAYER=1`: F16
models (bge-m3) run their matmuls through `rocblas_gemm_ex` (f16 in, f32
out); Q4_K models (Qwen3-Embedding) never call rocBLAS — llama.cpp's own
quantised kernels. `ROCBLAS_DEFAULT_ATOMICS_MODE=0` therefore matters for the
F16 models; it is set for all.

**E3. Bit-exact run to run** (byte compare of f32): bge-m3 F16 15/15 chunks
over 120 vectors (2 fresh loads x 4 orders), atomics allowed and disallowed
alike and with the same bytes; Qwen3-Embedding 4B/8B 16/16; granite 16/16;
llama.cpp CPU (same build) 16/16.

**E4. GPU vs CPU is a precision gap, not determinism.** Same build and
options, Qwen3-4B: worst cosine 0.9964, uniform across lengths — but that
"CPU" side lacked `no_op_offload` (E1), so the figure is to be re-measured.

**E5. Flash attention off with ubatch 8192 stalls the load** (>12 min, most
likely the attention buffer overrunning 16 GB). Keep `fa on`.

**E6. VRAM.** The deployed native-Vulkan config once copied the node's
`-c 65536` and paged the 16 GB card (every slot froze ~9 s per ~30 s):
`-c` = slots x ctx, sized to THIS card.
