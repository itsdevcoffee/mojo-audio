# Dual DGX Spark NCCL Bandwidth Investigation

**Date:** April 11, 2026
**Status:** Root cause identified, partial workaround found, forum post filed
**Prior context:** [04-09-2026-dual-spark-cluster-setup.md](./04-09-2026-dual-spark-cluster-setup.md)

## TLDR

Two-session investigation into the 3 GB/s NCCL bandwidth cap on the dual DGX Spark FE cluster. Firmware, MTU, and common causes from forum research were ruled out. Root cause identified: the GB10 GPU's PCIe link is reported as Gen1 x1 by sysfs, and NCCL's topology cost model uses that value as its ring bandwidth ceiling. A `NCCL_TOPO_FILE` override changes NCCL's plan from `totalBw 3.0` to `totalBw 48.0`, but a secondary cap (likely mlx5 write combining disabled on ARM64 Grace) holds actual throughput at about 1.5 GB/s per NIC. Filed on the NVIDIA DGX Spark forum: [forum thread 366266](https://forums.developer.nvidia.com/t/nccl-bandwidth-capped-at-3-gb-s-gpu-pcie-topology-reports-gen1-x1-on-dgx-spark-fe/366266).

## Current cluster state

Same hardware as the 04-09 doc, no physical changes. Software state as of 04-11:

| | Spark 1 (visage) | Spark 2 (spark-281c) |
|---|---|---|
| SoC FW | `0x0200941a` | `0x0200941a` |
| EC FW | `0x02004e18` | `0x02004e18` |
| Kernel | `6.17.0-1014-nvidia` | `6.17.0-1014-nvidia` |
| NVIDIA driver | `580.142` | `580.126.09` |
| CUDA | 13.0 | 13.0 |
| CX7 FW | `28.45.4028` | `28.45.4028` |
| MTU (all 4 interfaces) | 9000 | 9000 |
| NCCL built from source | 2.28.9 and 2.29.7 | 2.28.9 and 2.29.7 |

The driver version mismatch is unintended but the investigation showed results are identical on both, so it's actually a useful data point (rules out driver as a variable). Worth aligning on the next convenient reboot.

## The root cause

NCCL's topology detector reads `/sys/bus/pci/devices/000f:01:00.0/max_link_speed` for the GPU and gets back `2.5 GT/s PCIe`:

```
$ cat /sys/bus/pci/devices/000f:01:00.0/{current,max}_link_{speed,width}
2.5 GT/s PCIe
2.5 GT/s PCIe
1
16
$ nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current,pcie.link.gen.max,pcie.link.width.max --format=csv
1, 1, 1, 16
```

Note `max_link_gen=1` — not just current. This is either a sysfs reporting bug or reflects a real PCIe Gen1 x16 "stub" used for config space while the actual data path is NVLink-C2C. Either way, NCCL's cost model reads it, computes ~3 GB/s as the GPU edge bandwidth, and caps all collective operations there. The match is exact:

```
=== System : maxBw 12.0 totalBw 3.0 ===
Pattern 4, crossNic 0, nChannels 8, bw 3.000000/3.000000, type LOC/P2C
```

Benchmark: `3.02 GB/s busbw` on `all_gather_perf -b 1G -e 4G -f 2`.

Other DGX Spark users report 22-24 GB/s on similar hardware, so the sysfs value is almost certainly wrong (or there's a different code path other users hit that bypasses it).

## The workaround: NCCL_TOPO_FILE override

**Reusable recipe.** This is a partial fix but it's real. It changes NCCL's cost model, which matters for anyone debugging further or wanting to report accurate numbers.

Step 1: let NCCL dump its auto-detected topology once
```bash
# Add to your mpirun env:
-x NCCL_TOPO_DUMP_FILE=/tmp/nccl_topo.xml
```

Step 2: fix the GPU's link_speed and collapse the NCCL-inserted newlines (NCCL's own dump format has literal newlines mid-tag that its own parser can't read back):
```bash
sed ':a;N;$!ba;s/"\nlink_/" link_/g' /tmp/nccl_topo.xml \
  | sed 's|link_speed="2.5 GT/s PCIe" link_width="16"|link_speed="32.0 GT/s PCIe" link_width="16"|' \
  > /tmp/nccl_topo_fixed.xml
```

Step 3: copy to all nodes and use it:
```bash
scp /tmp/nccl_topo_fixed.xml maximus@192.168.100.11:/tmp/nccl_topo_fixed.xml

# In mpirun env:
-x NCCL_TOPO_FILE=/tmp/nccl_topo_fixed.xml
```

Effect on NCCL's plan:
```
Before: === System : maxBw 12.0 totalBw 3.0 ===
        Pattern 4, nChannels 8, bw 3.000000/3.000000

After:  === System : maxBw 12.0 totalBw 48.0 ===
        Pattern 4, nChannels 1, bw 12.000000/12.000000
        Pattern 3, nChannels 1, bw 24.000000/12.000000
```

Note that after the override, NCCL reduces `nChannels` because it believes fewer channels are needed to saturate the higher per-channel bandwidth. Actual throughput does not improve because of the secondary cap (see below).

## The secondary cap (unresolved)

After the topology fix, real bandwidth is still limited to roughly 1.5 GB/s per NIC regardless of channel count. Forcing `NCCL_MIN_NCHANNELS=16` doesn't help. This points to a layer below NCCL. Best guess: **mlx5 write combining disabled**.

All 4 mlx5_core devices report in dmesg:
```
mlx5_core_test_wc: Write combining is not supported
```

Grace CPUs have a known-unreliable WC implementation. The kernel's mlx5 driver boot-time WC test uses standard 8-byte stores which fail on Grace about 30% of the time. When it fails, BlueFlame (the fast MMIO path for RDMA doorbells) is disabled driver-wide. Every `ibv_post_send()` then needs an extra DMA round-trip to fetch the WQE, which is catastrophic for NCCL's high-frequency proxy thread posting pattern.

A NEON-based mlx5 WC test patch was submitted to LKML in Sep 2025 targeting net-next (kernel 6.13+). Unclear if it's in NVIDIA's `6.17.0-1014-nvidia` kernel — dmesg still shows the WC failure, suggesting not, or that it's included but still failing on this hardware.

## Ruled out

- **SoC firmware regression** — both Sparks on current `0x0200941a`, verified via `fwupdmgr get-devices`. Not the 10500→10600 issue others hit.
- **MTU mismatch** — 9000 confirmed on both uppercase and lowercase interfaces on both Sparks via `ip link show`.
- **GID index** — initial test used GID 3 (worked at first, GID table later regenerated and GID 3 became empty on `rocep1s0f1`). GID 1 is the stable link-local RoCEv2 entry.
- **NCCL version** — 2.28.9 and 2.29.7 produce byte-identical results.
- **Driver version** — different on each unit (580.142 vs 580.126.09), identical results.
- **Cable / port / physical** — `ib_write_bw` achieves 109 Gbps on the same link, confirming the hardware path.
- **NCCL channel count** — forcing 16 channels via `NCCL_MIN_NCHANNELS` doesn't change actual throughput.

## Lowercase interface needs IP configuration

`enp1s0f1np1` is UP and has a link-local MAC but no IPv4/IPv6 configured. Its RoCE device `rocep1s0f1` has no IPv4-mapped GID 3 entry. For full dual-NIC operation:

```bash
# On Spark 1 (requires sudo):
sudo ip addr add 192.168.101.10/24 dev enp1s0f1np1

# On Spark 2:
sudo ip addr add 192.168.101.11/24 dev enp1s0f1np1
```

Persist in netplan afterwards. This was not done during the investigation because the sudo requirement and the fact that the topology cap was the primary concern. Worth doing next session — it's prerequisite to any dual-NIC NCCL work.

## Files and artifacts

On Spark 1:
- `/tmp/nccl_test.sh` — the final test script with the topology override (points at `/tmp/nccl_topo_fixed.xml`)
- `/tmp/nccl_baseline.sh` — clean baseline script, no env tuning
- `/tmp/nccl_topo.xml` — raw NCCL-dumped topology (shows the Gen1 x1 bug)
- `/tmp/nccl_topo_fixed.xml` — the corrected topology file
- `/tmp/nccl_baseline_out.txt` — NCCL 2.29.7 baseline run, 3.02 GB/s (the main evidence)
- `/tmp/nccl_output_2297.txt` — NCCL 2.29.7 with topology fix + forced channels, 1.42 GB/s
- Various `/tmp/nccl_output[3-10].txt` from iteration runs

On Spark 2: same files under `/tmp/` for the topology file only.

On Fedora workstation (`/tmp/`): copies of `nccl_topo.xml` and several output files from the session.

## Forum post

Filed on the NVIDIA DGX Spark / GB10 forum: **[forum thread 366266](https://forums.developer.nvidia.com/t/nccl-bandwidth-capped-at-3-gb-s-gpu-pcie-topology-reports-gen1-x1-on-dgx-spark-fe/366266)**

Tags: `pcie`, `performance`, `kernel`, `debugging-and-troubleshooting`, `nics`, `rdma`

People to watch for responses:
- **eugr** — most responsive community member on CX7 bandwidth issues
- **Balaxxe** — identified the SoC firmware regression in earlier threads
- **aniculescu**, **NVES** — NVIDIA staff

## Practical implications for dev work

The cluster is **usable** for distributed inference right now, with caveats:

- **256 GB unified memory pool across both Sparks** is real and accessible
- **Pipeline parallelism (PP) is the right strategy** — only activations cross the link once per forward pass (~1 MB per token), so the 3 GB/s limit is effectively invisible
- **Tensor parallelism (TP) is painful** — all-reduces after every attention + MLP layer compound the comms bottleneck. Usable for large-batch throughput but bad for interactive decode.
- **Data parallelism (training) works** if batches are large enough to amortize gradient all-reduce

Recommended config for large-model inference:
```bash
# vLLM
vllm serve MODEL --pipeline-parallel-size 2 --tensor-parallel-size 1

# TensorRT-LLM
# pp_size=2, tp_size=1 in build config
```

Models that fit in 256 GB that wouldn't fit on one Spark:
- Llama-3.1-405B Q4 (~230 GB)
- DeepSeek-V3 Q4 (~350 GB — won't fit, but V3-Lite will)
- Qwen2.5-72B BF16 (~145 GB)
- Any ~200B class model in Q4-Q6

## Open questions

1. Is the `max_link_gen=1` sysfs reporting a bug, or does it reflect an actual PCIe Gen1 stub that coexists with the NVLink-C2C data path? (Waiting on forum response)
2. Is Tariq Toukan / Patrisious Haddad's NEON-based mlx5 WC test patch in NVIDIA's 6.17.0-1014 kernel, or do we need 6.18+?
3. For users getting 22-24 GB/s on similar hardware: different kernel, different NCCL build, or something in the RDMA userspace we're not setting?
4. Does the NCCL_TOPO_FILE override actually improve any real workload (e.g., vLLM with PP), or is the secondary cap the only thing that matters in practice? (Worth testing next session)

## Next steps

1. Wait 24-48h for forum responses before sinking more time into this
2. If responses come with concrete fixes, try them
3. In parallel: start workload testing with PP=2 on the current 3 GB/s bandwidth and document real-world inference perf numbers on ~70B-class models
4. Configure IPs on the lowercase interface (`enp1s0f1np1`) for future dual-NIC work
5. Align Spark 2 driver to `580.142` on the next convenient reboot
