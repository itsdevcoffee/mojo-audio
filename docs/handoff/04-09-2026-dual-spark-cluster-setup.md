# Dual DGX Spark Cluster Setup — Handoff

**Date:** April 9, 2026
**Status:** Link working, NCCL functional but bandwidth suboptimal

## Hardware

| | Spark 1 (visage) | Spark 2 (maximus) |
|---|---|---|
| Hostname | visage-spark | spark-281c |
| User | visage | maximus |
| IP (CX7 link) | 192.168.100.10 | 192.168.100.11 |
| IP (lowercase iface) | 192.168.101.10 | 192.168.101.11 |
| WiFi IP | 10.0.0.46 | (not noted) |
| OTA Version | 7.5.0 | 7.5.0 |
| Kernel | 6.17.0-1014-nvidia | 6.17.0-1014-nvidia |
| Driver | 580.142 | 580.142 |
| CUDA | 13.0 | 13.0 |
| CX7 Firmware | 28.45.4028 | 28.45.4028 |
| Serial | 1983925015499 | 1983925012279 |

**Cable:** Single QSFP cable between the two units, plugged into the port that maps to `enP2p1s0f1np1` (f1) on each Spark. The other port (f0) is unused.

## Network Configuration

### Interface naming
Each physical QSFP port exposes **two** logical interfaces due to the GB10's multi-host PCIe mode:
- `enP2p1s0f1np1` (uppercase) — PCI bus 0002:01
- `enp1s0f1np1` (lowercase) — PCI bus 0000:01

Both map to the same physical port but represent separate 100G PCIe paths. The ConnectX-7 aggregates them for 200G total.

Corresponding RDMA devices:
- `roceP2p1s0f1` → `enP2p1s0f1np1`
- `rocep1s0f1` → `enp1s0f1np1`

### Netplan (persists through reboot)
Spark 1: `/etc/netplan/40-cx7.yaml`
```yaml
network:
  version: 2
  ethernets:
    enP2p1s0f0np0:
      link-local: [ ipv4 ]
    enP2p1s0f1np1:
      addresses:
        - 192.168.100.10/24
```

Spark 2: same file but with `192.168.100.11/24`.

**Note:** The original NVIDIA playbook netplan uses lowercase interface names (`enp1s0f1np1`) which don't match the Founder's Edition naming. We had to manually fix this.

### MTU
Set to 9000 on Spark 1 (both interfaces). **Not confirmed on Spark 2** — may still be 1500. This is ephemeral (not in netplan) and would reset on reboot. To persist, add `mtu: 9000` to the netplan config.

### SSH
- Spark 1 → Spark 2: configured via `~/.ssh/config` (key: `~/.ssh/spark_cluster`, user: maximus)
- Spark 2 → Spark 1: configured via `~/.ssh/config` (key: `~/.ssh/spark_cluster`, user: visage)
- Passwordless SSH works both directions
- Fedora workstation can also SSH to Spark 1 (`ssh visage@visage-spark.local`)

## NCCL Setup

### Installed on both Sparks
- NCCL v2.28.9-1 built from source with `NVCC_GENCODE="-gencode=arch=compute_121,code=sm_121"`
- Location: `~/nccl/` (build output in `~/nccl/build/`)
- nccl-tests: `~/nccl-tests/` (binaries in `~/nccl-tests/build/`)
- OpenMPI: `libopenmpi-dev` installed via apt

### Environment variables needed before running tests
```bash
export CUDA_HOME="/usr/local/cuda"
export MPI_HOME="/usr/lib/aarch64-linux-gnu/openmpi"
export NCCL_HOME="$HOME/nccl/build/"
export LD_LIBRARY_PATH="$NCCL_HOME/lib:$CUDA_HOME/lib64/:$MPI_HOME/lib:$LD_LIBRARY_PATH"
```

### Working mpirun command
```bash
mpirun -np 2 -H 192.168.100.10:1,192.168.100.11:1 \
  --mca plm_rsh_agent "ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no" \
  --mca btl_tcp_if_include enP2p1s0f1np1 \
  --mca oob_tcp_if_include enP2p1s0f1np1 \
  -x NCCL_SOCKET_IFNAME=enP2p1s0f1np1 \
  bash -c 'export LD_LIBRARY_PATH=$HOME/nccl/build/lib:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu/openmpi/lib:$LD_LIBRARY_PATH && $HOME/nccl-tests/build/all_gather_perf -b 1G -e 4G -f 2'
```

**Important:** The `bash -c '$HOME/...'` pattern is required because usernames differ (visage vs maximus), so `$HOME` must expand on each node locally.

## Benchmark Results

### Raw RDMA (ib_write_bw)
- **109 Gbps** — single logical interface, single QP
- This confirms the physical link and RDMA stack are healthy

### NCCL all_gather_perf
- **3 GB/s bus bandwidth** (~24 Gbps) — consistent across all buffer sizes (64MB to 16GB)
- **6 GB/s algorithm bandwidth**
- Uses NET/IB transport (RDMA, not TCP sockets)
- 16 channels, alternating between IB/0 (rocep1s0f1) and IB/1 (roceP2p1s0f1)
- GDR 0 (GPU Direct RDMA disabled — expected, GB10 has no dedicated GPU memory)
- Zero validation errors

### Expected performance
Forum reports from other DGX Spark users: **16-23 GB/s bus bandwidth**. Our 3 GB/s is ~5-7x below expected.

## What We Tried (Bandwidth Tuning)

| Attempt | Result |
|---|---|
| Default config | 3.0 GB/s |
| MTU 9000 on Spark 1 | 3.0 GB/s (unchanged) |
| NCCL_IB_HCA=roceP2p1s0f1 (single device) | 1.4 GB/s (worse — confirms both devices needed) |
| NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1 (explicit both) | 3.0 GB/s |
| NCCL_BUFFSIZE=8388608 | 3.0 GB/s |
| NCCL_IB_QPS_PER_CONNECTION=4 | 3.0 GB/s |
| NCCL_IGNORE_CPU_AFFINITY=1 | 3.0 GB/s |
| Adding IPs to lowercase interface (192.168.101.x) | 3.0 GB/s |
| Buffer sizes from 64MB to 16GB | 3.0 GB/s (no scaling) |

## Known Issues / Leads

1. **"Write combining is not supported"** — All 4 mlx5_core devices report this in dmesg. Write combining optimizes MMIO writes for RDMA and its absence could significantly impact NCCL ring buffer performance on ARM/Grace. This may be the root cause.

2. **SoC firmware known issue** — Forum threads reference FE-specific CX7 bandwidth regressions tied to SoC firmware updates (10500→10600). A March 2026 patch partially addressed this. Our firmware is current (28.45.4028) but the fix may be incomplete.

3. **nvidia-peermem won't load** — `modprobe nvidia-peermem` returns "Invalid argument". This is expected on GB10 (no dedicated GPU memory = no GDR), but was confirmed not to affect the 16+ GB/s results others get.

4. **GID table changed warning** — Appeared once during NCCL testing: `NET/IB : rocep1s0f1:1 GID table changed`. May indicate instability in RoCE GID resolution.

## Recommended Next Steps

1. **Post on NVIDIA forum** with the ib_write_bw (109 Gbps) vs NCCL (3 GB/s) comparison — this is a clear, reproducible data point. Target: [DGX Spark / GB10 forum](https://forums.developer.nvidia.com/c/accelerated-computing/dgx-spark-gb10/719)

2. **Try perftest between both logical interfaces** — run `ib_write_bw` on `rocep1s0f1` (the lowercase one) to verify both 100G paths work independently

3. **Persist MTU 9000 in netplan** on both Sparks and both interfaces

4. **Try the precompiled NCCL binaries** from [assix/dgx-spark-nccl-blackwell](https://github.com/assix/dgx-spark-nccl-blackwell) to rule out a build issue

5. **Check Spark 2 standalone** — run the same benchmark from Spark 2 as the MPI initiator to rule out a Spark-1-specific issue

6. **Don't let bandwidth block practical use** — 3 GB/s is still usable for distributed inference (vLLM/TensorRT-LLM). The 256 GB unified memory pool is the bigger win for large models.

## Services Running on Spark 1

Key services (all systemd, auto-restart on reboot):
- ollama, pm2-visage, redis-server, tailscaled, syncthing@visage, docker, gnome-remote-desktop, nvidia-persistenced

## GPU Throttle Warning

There's a known GPU throttling issue after the kernel 6.17.0-1014 / driver 580.142 update where GPU clocks drop from ~2400MHz to 750MHz due to USB Power Delivery negotiation bugs. Fix: unplug the power brick from wall for 30+ seconds. Tool: [spark-gpu-throttle-check](https://github.com) on GitHub.

## Reference Links

- [Spark Stacking Guide (official)](https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html)
- [Connect Two Sparks Playbook](https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/connect-two-sparks)
- [NCCL Playbook](https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/nccl)
- [Optimized NCCL binaries for Spark](https://github.com/assix/dgx-spark-nccl-blackwell)
- [CX7 NIC Discussion (forum)](https://forums.developer.nvidia.com/t/connectx-7-nic-in-dgx-spark/350417?page=4)
- [GPU Direct RDMA on Spark (forum)](https://forums.developer.nvidia.com/t/enabling-gpu-direct-rdma-for-dgx-spark-clustering/352051)
- [CX7 Connectivity Persistence Issue](https://forums.developer.nvidia.com/t/dual-spark-cx7-connectivity-doesnt-persist/366023)
