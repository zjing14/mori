## SDMA Transport — Architecture Overview

SDMA (**System DMA**) is one of three transport backends in Mori, alongside RDMA and P2P. It uses the dedicated DMA copy engines built into AMD GPUs rather than GPU shader threads (P2P) or network NICs (RDMA).

| Transport | Enum Value | Use Case |
|---|---|---|
| **RDMA** | 0 | Inter-node, GPU-initiated RDMA via NIC (MLX5/BNXT/Ionic providers) |
| **P2P** | 1 | Intra-node, direct GPU memory copies via PCIe/xGMI peer access |
| **SDMA** | 2 | Intra-node, DMA engine-initiated copies via AMD SDMA hardware queues |

### How SDMA Works — The Full Stack

#### 1. Hardware Layer: AMD SDMA Engines

Each AMD GPU has multiple SDMA engines — fixed-function DMA copy units that can perform memory-to-memory copies independently of the GPU's compute units (CUs). On MI300X there are 8 such engines. The `anvil.hpp` file contains an OAM (On-Accelerator Module) map that selects which SDMA engine to use for a given src→dst GPU pair.

#### 2. Anvil Library (`anvil.hpp` / `anvil_device.hpp`)

Anvil is the host+device abstraction layer for SDMA queues:

- **`SdmaQueue`** (host side) — wraps the KFD (Kernel Fusion Driver) queue creation via `HsaQueueResource`. Each queue is associated with a specific SDMA engine and allocated with a 256KB ring buffer.

- **`SdmaQueueDeviceHandle`** (device side) — the GPU-accessible handle for a queue, containing:
  - `queueBuf` — the ring buffer (array of `uint32_t` DWORDs)
  - `rptr` — hardware read pointer (read by GPU to check consumption)
  - `wptr` — hardware write pointer (written to submit work)
  - `doorbell` — MMIO doorbell address (written to notify hardware)
  - `cachedWptr` / `committedWptr` — software tracking for lock-free multi-producer ring buffer management

The key device-side operations on `SdmaQueueDeviceHandle`:

| Method | What it does |
|---|---|
| `ReserveQueueSpace(size)` | Atomically reserves `size` bytes in the ring buffer using CAS on `cachedWptr`. Handles wraparound by padding with NOPs. |
| `placePacket<T>(packet, wptr)` | Writes a packet's DWORDs into the ring buffer at the current write position. |
| `submitPacket(base, pendingWptr)` | Waits for ordering (via `committedWptr`), writes `wptr`, issues barriers, writes `doorbell`. |

There is also a `SdmaQueueSingleProducerDeviceHandle` variant that skips atomics for the single-producer case.

#### 3. SDMA Packet Types (`sdma_pkt_struct.h`)

SDMA commands are submitted as hardware-defined packet structures:

- **`SDMA_PKT_COPY_LINEAR`** (7 DWORDs) — Linear memory copy. Contains: `{op=COPY, sub_op=LINEAR, count, src_addr_lo/hi, dst_addr_lo/hi}`. The count field is `copy_size - 1`.

- **`SDMA_PKT_ATOMIC`** — Atomic operation. Used with `SDMA_ATOMIC_ADD64` to atomically increment a signal counter at a given address.

- **`SDMA_PKT_FENCE`** — Fence/barrier packet that writes a value to a memory address.

Packet creation helpers:
- `CreateCopyPacket(src, dst, size)` → `SDMA_PKT_COPY_LINEAR`
- `CreateAtomicIncPacket(signal)` → `SDMA_PKT_ATOMIC` with `ADD64` of 1
- `CreateFencePacket(addr, data)` → `SDMA_PKT_FENCE`

#### 4. Core SDMA Device Primitives (`core/transport/sdma/device_primitives.hpp`)

This is the core transport layer that bridges the shmem API to Anvil:

- **`SdmaPutThread`** — Selects a queue by `qpId`, reserves space for a COPY_LINEAR + ATOMIC signal packet pair, places both, submits to hardware, and increments the expected signal counter.

- **`SdmaPutWarp`** — Distributes the copy across **all `numQueues` SDMA queues** in parallel (one per warp lane). Each lane handles `copy_size / numQueues` bytes (last lane gets remainder). This achieves multi-queue parallelism — the copy is split across up to 8 SDMA engines simultaneously.

- **`SdmaQuietThread`** — Iterates over all queues, spin-waiting on each signal counter via `waitForSignal(signal, expected)` using `__hip_atomic_load`.

- **`SdmaQuietWarp`** — Same but parallelized: lane `i` waits on queue `i`'s signal.

#### 5. Shmem SDMA Kernel Specializations (`shmem_sdma_kernels.hpp`)

Template specializations of the transport-agnostic shmem kernel API for `TransportType::SDMA`:

- **`ShmemPutMemNbi{Thread,Warp}Kernel<SDMA>`** — Extracts pointers and per-PE SDMA resources from `SymmMemObjPtr`, then calls the corresponding `core::SdmaPut{Thread,Warp}`.

- **`ShmemPutSizeImmNbi{Thread,Warp}Kernel<SDMA>`** — Empty stubs (SDMA does not support inline immediate puts).

- **`ShmemAtomicSizeNonFetch{Thread,Warp}Kernel<SDMA>`** — Falls back to P2P for atomic operations, since SDMA engines do not support arbitrary atomics on remote memory.

- **`ShmemQuiet{Thread,Warp}Kernel<SDMA>`** — Completion waiting via `core::SdmaQuiet{Thread,Warp}`, using per-PE signal counters from the `SymmMemObjPtr`.

#### 6. SymmMemObj SDMA Fields (`symmetric_memory.hpp`)

Each symmetric memory object tracks SDMA-specific state:

```cpp
anvil::SdmaQueueDeviceHandle** deviceHandles_d;  // [numPEs * sdmaNumQueue] GPU-resident handles
HSAuint64* signalPtrs;                            // [numPEs * sdmaNumQueue] completion signals
uint32_t sdmaNumQueue = 8;                        // default 8 SDMA queues per peer
HSAuint64* expectSignalsPtr;                      // [numPEs * sdmaNumQueue] expected signal values
```

These are indexed as `pe * sdmaNumQueue + queueId`, giving each PE its own independent set of SDMA queues.

#### 7. Transport Dispatch (`shmem_device_api.hpp`)

The runtime dispatches to SDMA based on a per-PE `transportTypes[]` array in `GpuStates`:

```cpp
#define DISPATCH_TRANSPORT_TYPE(func, pe, ...)
  TransportType transportType = globalGpuStates->transportTypes[pe];
  if (transportType == TransportType::RDMA)  { func<RDMA>(...); }
  else if (transportType == TransportType::P2P)  { func<P2P>(...); }
  else if (transportType == TransportType::SDMA) { func<SDMA>(...); }
```

### SDMA vs P2P vs RDMA

| Aspect | P2P | SDMA | RDMA |
|---|---|---|---|
| **Copy mechanism** | GPU threads do `memcpy` via load/store | GPU submits packet to DMA engine; engine does copy | GPU posts WQE to NIC; NIC does RDMA write |
| **Who does the work** | Compute Units (CUs) | SDMA fixed-function engines | NIC hardware |
| **Occupies CUs?** | Yes | No (CUs only submit, don't copy) | No (CUs only post, don't transfer) |
| **Inline puts** | Yes (atomic stores) | No (stubs are empty) | Yes (NIC inline write) |
| **Atomics** | CAS loops on GPU | Falls back to P2P | NIC-based RDMA atomics |
| **Quiet/fence** | No-op (writes are immediately visible) | Spin-wait on signal counters | Poll CQ for completions |
| **Multi-queue** | N/A | Yes, up to 8 parallel DMA engines | Yes, multiple QPs |
| **Scope** | Intra-node only | Intra-node only | Inter-node (and intra-node) |

### Data Flow Summary

```
User kernel calls:  ShmemPutMemNbiThread(dest, offset, source, offset, bytes, pe)
        ↓ DISPATCH_TRANSPORT_TYPE (checks transportTypes[pe] == SDMA)
        ↓
ShmemPutMemNbiThreadKernel<SDMA>(dest, offset, source, offset, bytes, pe, qpId)
        ↓ Extract pointers: srcPtr, dstPtr, deviceHandles, signals, expectedSignals
        ↓
core::SdmaPutThread(srcPtr, dstPtr, bytes, deviceHandles, signals, expectedSignals, numQueue, qpId)
        ↓
anvil::SdmaQueueDeviceHandle::ReserveQueueSpace(sizeof(COPY_LINEAR))
anvil::CreateCopyPacket(src, dst, size) → SDMA_PKT_COPY_LINEAR
handle.placePacket(copy_packet, wptr)
        ↓
handle.ReserveQueueSpace(sizeof(ATOMIC))
anvil::CreateAtomicIncPacket(signal) → SDMA_PKT_ATOMIC
handle.placePacket(atomic_packet, wptr)
        ↓
handle.submitPacket(startBase, pendingWptr)
  → writes wptr register
  → memory barriers (__builtin_amdgcn_s_waitcnt, __atomic_signal_fence)
  → rings doorbell (MMIO write)
        ↓
SDMA engine reads packets from ring buffer, performs DMA copy, atomically increments signal
        ↓
User kernel calls:  ShmemQuietThread(pe, dest)
        ↓
core::SdmaQuietThread(signals, expectedSignals, numQueue)
        ↓ for each queue: waitForSignal(signal, expected) — spin on atomic load
```

The key advantage of SDMA: it offloads memory copies from GPU compute units to dedicated fixed-function DMA engines. The GPU threads only do lightweight work — constructing packets in a ring buffer and ringing a doorbell — while the actual data movement happens asynchronously on the SDMA hardware. This frees up CUs for computation, making SDMA attractive for overlapping communication with computation on intra-node transfers.

### AMD SDMA vs NVIDIA Copy Engine — Comparison

AMD SDMA engines and NVIDIA Copy Engines (CEs) serve the same fundamental purpose — dedicated fixed-function DMA units that move data without occupying GPU compute resources — but differ significantly in architecture, programmability, and how they are exposed to users.

#### Hardware

| Aspect | AMD SDMA | NVIDIA Copy Engine |
|---|---|---|
| **Name** | System DMA (SDMA) engine | Copy Engine (CE) |
| **Count (datacenter)** | MI300X: 8 SDMA engines (shared across the 8 XCDs) | H100: 3 CEs; A100: 3 CEs; older GPUs: 1–2 CEs (`asyncEngineCount` in device properties) |
| **Engine independence** | Each engine has its own command queue; fully independent | Each CE is independent; can run concurrently with compute and with each other |
| **Transfer types** | Linear copy, fill, atomic, fence, timestamp, trap, etc. | Host↔Device, Device↔Device (peer), plus specialized async copy instructions |
| **Bandwidth** | Shares the same memory fabric (xGMI/Infinity Fabric) as P2P | Shares PCIe/NVLink bandwidth depending on transfer direction |

#### Programming Model

| Aspect | AMD SDMA (Mori/Anvil) | NVIDIA Copy Engine |
|---|---|---|
| **Host-side submission** | Via KFD (HSA Kernel Fusion Driver): `hsaKmtCreateQueue` with `HSA_QUEUE_SDMA` type | Via CUDA runtime: `cudaMemcpyAsync` on a CUDA stream; driver maps to available CE |
| **Device-side (GPU-initiated) submission** | Directly supported — GPU threads write SDMA packets into a ring buffer and ring a doorbell (this is what Mori does) | Historically not supported. New in CUDA 12.x / NCCL 2.26+: "CE collectives" allow device-initiated CE submission for NCCL operations, but the API is not publicly exposed for general use |
| **Queue management** | Explicit: user creates per-engine queues, selects engine ID, manages ring buffer write/read pointers | Implicit: the CUDA driver manages CE allocation; users submit to CUDA streams and the driver routes to CEs transparently |
| **Packet format** | User constructs hardware packets directly (`SDMA_PKT_COPY_LINEAR`, `SDMA_PKT_ATOMIC`, `SDMA_PKT_FENCE`, etc.) — visible and documented in open-source AMD headers | Opaque: NVIDIA driver constructs internal command packets; the hardware packet format is not publicly documented |
| **Multi-queue parallelism** | Explicit: Mori's `SdmaPutWarp` splits a copy across all 8 SDMA queues, one per warp lane | Implicit: multiple CUDA streams can use different CEs concurrently, but a single `cudaMemcpyAsync` call uses one CE |
| **Completion signaling** | User-managed: atomic increment on a signal address in GPU memory; software spin-wait via `waitForSignal` | CUDA events (`cudaEventRecord` / `cudaEventSynchronize`); internally uses CE semaphore/fence mechanisms |

#### Architectural Differences

| Aspect | AMD SDMA | NVIDIA Copy Engine |
|---|---|---|
| **Openness** | Fully open: packet formats in public headers (`sdma_pkt_struct.h`), KFD interface in open-source kernel driver, GPU-side ring buffer protocol documented | Closed: CE internals are proprietary; users interact through CUDA runtime/driver abstractions |
| **GPU-initiated control** | First-class: designed for GPU kernels to directly construct and submit packets — the core of Mori's SDMA transport | Emerging: NVIDIA added device-initiated CE support in recent NCCL versions for collective operations, but it remains limited in scope and not a general-purpose API |
| **Granularity of control** | Fine-grained: choose specific engine, manage queue depth, combine copy + atomic + fence in a single submission batch | Coarse-grained: CUDA driver decides which CE to use; user controls concurrency only at the stream level |

#### Summary of Key Differences

1. **Openness**: AMD SDMA is fully open and hackable — Mori/Anvil directly constructs hardware packets and manages ring buffers. NVIDIA CEs are abstracted behind the closed-source CUDA driver.

2. **GPU-initiated submission**: AMD SDMA was designed from the start for GPU kernels to submit work to DMA engines (via KFD queues accessible from device code). NVIDIA only recently added limited device-initiated CE support in NCCL, and it is not a general-purpose API.

3. **Engine count**: MI300X has 8 SDMA engines vs H100's 3 CEs. This gives AMD more parallelism for DMA operations, which Mori exploits via `SdmaPutWarp` (splitting a single copy across all 8 engines).

4. **Queue management**: AMD requires explicit queue setup and management (create queue per engine, manage ring buffer pointers, handle wraparound). NVIDIA handles this transparently in the driver.

6. **Completion model**: AMD SDMA uses explicit atomic signals in GPU memory with software spin-waits. NVIDIA uses CUDA events backed by internal CE semaphores — a higher-level abstraction.

7. **Scope overlap with in-SM async copy**: Both vendors also have in-CU/SM async copy mechanisms (`async_copy` on AMD, `cp.async`/TMA on NVIDIA) that are architecturally separate from the DMA engines. These operate at the shared memory (LDS/SMEM) level and serve different use cases (prefetching tiles for matrix operations) than the system-level DMA engines.
