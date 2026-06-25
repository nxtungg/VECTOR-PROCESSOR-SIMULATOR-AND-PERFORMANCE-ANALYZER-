# Công Thức & Tham Số Tính Toán

## 1. Cơ Bản

### Định Nghĩa Biến
```
VL          = Vector Length (số phần tử tối đa)
L           = Latency (độ trễ tính bằng cycle)
BW          = Memory Bandwidth (phần tử/cycle)
N_lanes     = Số lanes xử lý song song
Freq        = Clock frequency (MHz)
N_elem      = Số phần tử thực tế trong vector
```

### Từ Config
```json
{
  "vector_length": 16,          // VL
  "num_lanes": 4,               // N_lanes
  "load_latency": 3,            // L_load
  "store_latency": 3,           // L_store
  "add_latency": 2,             // L_add
  "mul_latency": 4,             // L_mul
  "div_latency": 8,             // L_div
  "reduction_latency": 3,       // L_red
  "memory_bandwidth": 4,        // BW
  "clock_frequency_mhz": 1000,  // Freq
  "num_alu_units": 1,           // Số ALU
  "num_memory_units": 1         // Số memory unit
}
```

---

## 2. Scalar Simulator (Baseline)

### VLOAD / VSTORE
```
cycles_load = L_load + ceil(N_elem / BW)
cycles_store = L_store + ceil(N_elem / BW)
```

**Ví dụ:**
```
VLOAD V1, A          (N_elem=8, BW=4, L_load=3)
cycles = 3 + ceil(8/4) = 3 + 2 = 5 cycles
```

### ALU Operations (VADD, VSUB, VMUL, VDIV)
```
cycles = L_op × N_elem
```

**Ví dụ:**
```
VADD V3, V1, V2      (N_elem=8, L_add=2)
cycles = 2 × 8 = 16 cycles
(sequential: elem 0, 1, 2, ..., 7)
```

### Reduction (VREDUCE_SUM, VREDUCE_MAX, VREDUCE_MIN)
```
cycles = L_red × (N_elem - 1)
```

**Ví dụ:**
```
VREDUCE_SUM V1, V2   (N_elem=8, L_red=3)
cycles = 3 × (8 - 1) = 21 cycles
(s0=a0, s1=s0+a1, s2=s1+a2, ..., s7=s6+a7)
```

### VDOT (Dot Product)
```
cycles = L_mul × (N_elem - 1) + L_add × (N_elem - 1)
       ≈ (L_mul + L_add) × N_elem
```

**Ví dụ:**
```
VDOT V1, V2, V3      (N_elem=8, L_mul=4, L_add=2)
cycles ≈ (4+2) × 8 = 48 cycles
```

### Gather / Scatter
```
cycles_gather = L_load + ceil(N_elem / BW) + overhead_random_access
cycles_scatter = L_store + ceil(N_elem / BW) + overhead_random_access
```

---

## 3. Vector Non-Pipeline Simulator

### Tính toán Duration

#### 3.1 ALU Operations (VADD, VSUB, VMUL, VDIV, VMAX, VMIN)
```
iterations = ceil(VL / N_lanes)
duration = L_op + iterations - 1

= L_op + ceil(VL / N_lanes) - 1
```

**Ví dụ:**
```
VADD V3, V1, V2   (VL=8, N_lanes=4, L_add=2)
iterations = ceil(8/4) = 2
duration = 2 + 2 - 1 = 3 cycles

Timeline:
  Cycle 0-1: Process lanes 0-3 (latency=2)
  Cycle 2-3: Process lanes 4-7 (iterations=2)
  Total = 3 cycles
```

#### 3.2 Memory Operations (VLOAD, VSTORE)
```
iterations = ceil(VL / BW)
duration = L_mem + iterations - 1

= L_mem + ceil(VL / BW) - 1
```

**Ví dụ:**
```
VLOAD V1, A   (VL=8, BW=4, L_load=3)
iterations = ceil(8/4) = 2
duration = 3 + 2 - 1 = 4 cycles

Timeline:
  Cycle 0-2: Start load, latency 3
  Cycle 3: Load next batch (4 elements)
  Total = 4 cycles
```

#### 3.3 Reduction Operations (VREDUCE_SUM, VREDUCE_MAX, VREDUCE_MIN)
```
duration = L_red × ceil(log2(VL))
```

**Ví dụ:**
```
VREDUCE_SUM V1, V2   (VL=8, L_red=3)
duration = 3 × ceil(log2(8)) = 3 × 3 = 9 cycles

Tree reduction:
  Level 0: [a0, a1, a2, a3, a4, a5, a6, a7]
           └─┬─┘ └─┬─┘ └─┬─┘ └─┬─┘
  Level 1: [a0+a1, a2+a3, a4+a5, a6+a7]  (lat=3)
           └───┬──┘     └───┬──┘
  Level 2: [(a0+a1)+(a2+a3), (a4+a5)+(a6+a7)]  (lat=3)
           └─────────┬──────┘
  Level 3: Final result  (lat=3)
  
  Total = 3 levels × 3 = 9 cycles
```

#### 3.4 VDOT (Dot Product)
```
duration_mul = L_mul + ceil(VL / N_lanes) - 1
duration_add = L_add + iterations_reduction - 1
duration_total = max(duration_mul, duration_add for reduction)

≈ L_mul + ceil(VL / N_lanes) - 1 + L_add + ceil(log2(VL)) - 1
```

### 3.5 Gather/Scatter
```
duration_gather = L_load + ceil(VL / BW) - 1 + bank_conflict_penalty
duration_scatter = L_store + ceil(VL / BW) - 1 + bank_conflict_penalty
```

### Tổng Cycles (Non-Pipeline)
```
total_cycles = Σ(duration_i)  for i = 0 to num_instructions - 1
```

Vì **không overlap lệnh**, các duration cộng lại.

**Ví dụ:**
```
Program:
  VLOAD V1, A     → duration = 4
  VLOAD V2, B     → duration = 4
  VADD V3, V1, V2 → duration = 3
  VSTORE C, V3    → duration = 4

Total cycles = 4 + 4 + 3 + 4 = 15 cycles
```

---

## 4. Vector Pipeline Simulator

### Hazard Detection

#### 4.1 RAW (Read-After-Write) Hazard
```
earliest_raw = 1
for each src_reg in instruction.src:
  if src_reg written by previous_instr j:
    earliest_raw = max(earliest_raw, end_cycle[j] + 1)
```

**Ví dụ:**
```
I0: VADD V1, V2, V3    [end = 3]
I1: VSUB V4, V1, V5    [wait for V1]

RAW delay: V1 ready at cycle 3 + 1 = 4
earliest_raw[I1] = 4
```

#### 4.2 Structural Hazard (Unit Conflict)
```
earliest_struct = 1
for each required_functional_unit in instruction:
  if unit_free_at[unit] > current_cycle:
    earliest_struct = max(earliest_struct, unit_free_at[unit] + 1)
```

**Ví dụ:**
```
I0: VADD V1, V2, V3    [uses ALU, ends at 3]
I1: VMUL V4, V5, V6    [uses ALU]

Structural conflict: ALU busy until cycle 3 + 1 = 4
earliest_struct[I1] = 4
```

### Actual Issue Cycle
```
actual_issue[i] = max(earliest_raw[i], earliest_struct[i], 1)
actual_end[i] = actual_issue[i] + duration[i]
```

### Total Cycles (Pipeline)
```
total_cycles = max(end_cycle[i]) for all i
```

**Ví dụ:**
```
Config: 2 load units

I0: VLOAD V1, A    [issue=0, end=3]
I1: VLOAD V2, B    [issue=0, end=3]  ← overlap! (diff units)
I2: VADD V3, V1, V2 [issue=4, end=6]  ← wait for V2
I3: VSTORE C, V3   [issue=7, end=10]

Total cycles = 10 (vs 15 non-pipeline)
Speedup = 15/10 = 1.5x
```

---

## 5. Chaining / Data Forwarding

### Normal Pipeline (No Chaining)
```
I0: VLOAD V1      [0-3]
I1: VADD V2, V1   [4-6]  ← phải chờ V1 ready ở cycle 4

earliest_raw[I1] = 3 + 1 = 4
```

### With Chaining
```
I0: VLOAD V1      [0-3]
I1: VADD V2, V1   [3-5]  ← forward từ memory unit

forward_latency ≈ 1 cycle (từ last pipeline stage)
earliest_raw_chain[I1] = 3 - forward_latency + 1 = 3
actual_issue[I1] = 3
```

### Chaining Gain
```
cycles_gain = normal_cycles - chain_cycles
           ≈ forward_latency × number_of_dependent_chains
```

---

## 6. Reduction (Parallel Tree)

### Scalar (Sequential)
```
cycles = L_red × (N_elem - 1)
```

**Ví dụ:** N_elem=8, L_red=3
```
cycles = 3 × 7 = 21
```

### Vector (Parallel Tree)
```
cycles = L_red × ceil(log2(N_elem))
       = L_red × num_levels
```

**Ví dụ:** N_elem=8, L_red=3
```
cycles = 3 × 3 = 9

vs Scalar: 21
Speedup = 21/9 = 2.33x
```

### Speedup Reduction
```
speedup_reduction = (L_red × (N_elem - 1)) / (L_red × ceil(log2(N_elem)))
                  = (N_elem - 1) / ceil(log2(N_elem))

For N_elem = 8:
speedup = 7 / 3 ≈ 2.33x

For N_elem = 1024:
speedup = 1023 / 10 ≈ 102x
```

---

## 7. Gather/Scatter

### Memory Access Pattern
```
VGATHER V1, base_array, indices
→ for each lane l in parallel:
    V1[l] = base_array[indices[l]]

cycles = L_load + ceil(N_elem / BW) + bank_conflict_delay
       ≈ L_load + ceil(N_elem / BW)  (best case)
       ≈ L_load + N_elem × average_conflict_factor  (worst case)
```

### Bank Conflict Penalty
```
bank_conflict_delay = (number_of_conflicts - 1) × L_load
```

---

## 8. Masking / Predicate Execution

### Masked Operation
```
for each lane l:
  if mask[l] == 1:  (active)
    result[l] = operation(src1[l], src2[l])
  else:  (inactive)
    result[l] = result[l]  (merge mode) OR 0 (zero mode)

cycles: SAME (không tăng)
lane_utilization: active_lanes / total_lanes
```

### Lane Utilization Impact
```
throughput = N_lanes × active_ratio
           = N_lanes × (num_active_lanes / N_lanes)
           = num_active_lanes

Example: N_lanes=4, mask=[1,0,1,0]
  active = 2
  effective_throughput = 2 elem/cycle
  utilization = 2/4 = 50%
```

---

## 9. Performance Metrics

### Speedup
```
Speedup = scalar_cycles / vector_cycles

Example: scalar=100, vector=20
Speedup = 100/20 = 5x
```

### Throughput
```
Throughput_elem_per_cycle = N_elem / vector_cycles

Example: N_elem=8, cycles=4
Throughput = 8/4 = 2 elem/cycle
```

### Lane Utilization
```
Lane_Utilization = useful_lane_work / (N_lanes × total_cycles)
                 = total_element_operations / (N_lanes × total_cycles)

Example: 
  Workload: VADD 4 lanes, 8 elements
  cycles = 3 (for 2 iterations)
  elem_ops = 8
  Lane_util = 8 / (4 × 3) = 8/12 = 67%
```

### Execution Time
```
Execution_Time_ns = cycles / (Freq_MHz × 1e6)
                  = cycles × (1000 / Freq_MHz) / 1e6
                  = cycles / Freq_ns

Example: cycles=100, Freq=1000 MHz
  Exec_time = 100 / (1000 × 1e6) = 100 ns
```

---

## 10. Combined Metrics

### CPI (Cycles Per Instruction)
```
CPI = total_cycles / num_instructions

Example: 15 cycles, 4 instructions
CPI = 15/4 = 3.75 cycles/instr
```

### IPC (Instructions Per Cycle) - Pipeline
```
IPC = num_instructions / total_cycles
    = 1 / CPI

Example: 4 instr, 10 cycles
IPC = 4/10 = 0.4 instr/cycle
```

### Arithmetic Intensity
```
Arithmetic_Intensity = floating_point_operations / memory_bytes
```

---

## 11. Quick Reference Table

| Operation | Formula | Time Complexity |
|-----------|---------|-----------------|
| **VLOAD** | L + ceil(VL/BW) | O(VL/BW) |
| **VSTORE** | L + ceil(VL/BW) | O(VL/BW) |
| **VADD (scalar)** | L × N_elem | O(N_elem) |
| **VADD (vector)** | L + ceil(VL/lanes) - 1 | O(VL/lanes) |
| **VADD (pipeline)** | hazard + L + ... | O(hazard+latency) |
| **VREDUCE (scalar)** | L × (N_elem-1) | O(N_elem) |
| **VREDUCE (vector)** | L × ceil(log2(N_elem)) | O(log2(N_elem)) |
| **VGATHER** | L + ceil(VL/BW) + conflict | O(VL/BW) |
| **VDOT (scalar)** | (L_mul+L_add) × N_elem | O(N_elem) |
| **VDOT (vector)** | L_mul + ceil(VL/lanes) + L_red × log2(VL) | O(VL/lanes + log2 VL) |

---

## 12. Memory Hierarchy Latencies (Typical)

```
L1 Cache:      2-4 cycles
L2 Cache:      10-20 cycles
L3 Cache:      40-75 cycles
Main Memory:   200-300 cycles
```

---

## 13. Common Hardware Configurations

### Config 1: Small (Embedded)
```
vector_length: 4
num_lanes: 2
load_latency: 2
add_latency: 1
memory_bandwidth: 1
```

### Config 2: Medium (Mobile GPU)
```
vector_length: 16
num_lanes: 4
load_latency: 3
add_latency: 2
memory_bandwidth: 4
```

### Config 3: Large (High-End GPU)
```
vector_length: 64
num_lanes: 16
load_latency: 5
add_latency: 2
memory_bandwidth: 8
```

### Config 4: Supercomputer (Vector CPU)
```
vector_length: 256
num_lanes: 64
load_latency: 7
add_latency: 3
memory_bandwidth: 16
```

---

## 14. Dominance & Optimization Trade-offs

### Memory Bound (I/O dominated)
```
cycles ≈ sum(memory_operations × (latency + ceil(VL/BW)))

Optimization: Increase bandwidth, reduce latency, hide latency with prefetch
```

### Compute Bound (ALU dominated)
```
cycles ≈ sum(alu_operations × latency)

Optimization: Increase num_lanes, use chaining, ILP
```

### Balanced (Memory + Compute)
```
Arithmetic Intensity = ops / memory_bytes

If AI < Roofline → memory bound
If AI > Roofline → compute bound
```

---

## 15. Design Space Exploration

### Vector Length Impact
```
As VL increases:
  • Throughput ↑ (more elem processed)
  • Lanes utilized ↑ (better parallelism)
  • Register pressure ↑ (spilling risk)
  • Code size ↓ (fewer iterations)
```

### Number of Lanes Impact
```
As N_lanes increases:
  • Duration ↓ (faster per iteration)
  • Lane_utilization ↓ (harder to keep full)
  • Power ↑ (more hardware)
  • Cost ↑
```

### Memory Bandwidth Impact
```
As BW increases:
  • Memory cycles ↓ (faster data fetch)
  • Pipeline utilization ↑
  • Power ↑ (more buses)
  • Cost ↑
```

---

## Summary Cheatsheet

```
Quick Lookup:

Scalar cycles = latency × num_elements
Vector cycles = latency + ceil(VL/lanes) - 1
Pipeline = vector + hazard_delay
Reduction = latency × ceil(log2(N))

Speedup = scalar / vector
Throughput = elements / cycles
Utilization = work / (lanes × cycles)
Exec_time = cycles / frequency
```

