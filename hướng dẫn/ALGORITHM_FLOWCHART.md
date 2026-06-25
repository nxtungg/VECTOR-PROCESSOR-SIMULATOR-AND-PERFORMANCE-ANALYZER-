# Sơ Đồ Thuật Toán và Luồng Chạy

## 1. Luồng Chạy Chủ (Main Flow)

```
┌─────────────────────────────────────────┐
│      main.py → run_program()            │
├─────────────────────────────────────────┤
│  1. Load config từ config.json           │
│  2. Parse file program.txt               │
│     ├→ Extract DATA arrays              │
│     ├→ Extract SCALAR values            │
│     └→ Extract instructions             │
│  3. Tạo 3 simulator instances           │
│  4. Chạy tuần tự:                       │
│     ├→ Scalar simulator (baseline)      │
│     ├→ Vector simulator (non-pipeline)  │
│     └→ Pipeline simulator               │
│  5. So sánh kết quả & xuất report       │
└─────────────────────────────────────────┘
```

---

## 2. Parser Flow (parser.py)

```
┌──────────────────────────────────────────────┐
│        parse_file(filepath)                  │
├──────────────────────────────────────────────┤
│                                              │
│  for each line in file:                     │
│    ├─ Remove comment (#...)                │
│    ├─ Skip empty line                      │
│    │                                        │
│    ├─ If "DATA name = [...]"               │
│    │   └─ Parse list → store in data{}    │
│    │                                        │
│    ├─ If "SCALAR name = value"             │
│    │   └─ Parse value → store in scalars{}│
│    │                                        │
│    └─ If opcode (VLOAD, VADD, ...)         │
│        ├─ Validate opcode in valid_opcodes│
│        ├─ Split operands                  │
│        ├─ Check operand count             │
│        └─ Create Instruction object       │
│                                              │
│  Return: (data_dict, scalars_dict, inst_list)
│                                              │
└──────────────────────────────────────────────┘
```

---

## 3. Scalar Simulator Flow (scalar_simulator.py)

```
┌──────────────────────────────────────────┐
│   estimate_cycles(instructions, data)    │
├──────────────────────────────────────────┤
│                                          │
│  total_cycles = 0                        │
│  num_elements = len(data[0])             │
│                                          │
│  for each instruction:                   │
│                                          │
│    ┌──── if VLOAD ────┐                 │
│    │  cycles += load_latency            │
│    │  cycles += ceil(len / bandwidth)   │
│    └──────────────────┘                 │
│                                          │
│    ┌──── if VSTORE ───┐                 │
│    │  cycles += store_latency           │
│    │  cycles += ceil(len / bandwidth)   │
│    └──────────────────┘                 │
│                                          │
│    ┌──── if VADD, VSUB ────┐            │
│    │  cycles += add_latency             │
│    │  cycles *= num_elements            │
│    │  (tuần tự, phần tử 0, rồi 1, ...)│
│    └────────────────────┘               │
│                                          │
│    ┌──── if VREDUCE_SUM ────┐           │
│    │  cycles += reduction_latency       │
│    │  cycles *= log(num_elements)       │
│    │  (tuần tự accumulate)              │
│    └────────────────────┘               │
│                                          │
│  return total_cycles                    │
│                                          │
└──────────────────────────────────────────┘

Ví dụ: A[4] + B[4] + VSTORE
  VLOAD: 3 cycles (load_latency)
  VLOAD: 3 cycles
  VADD:  2 × 4 = 8 cycles (add_latency × num_elem)
  VSTORE: 3 cycles
  ─────────────────
  Total: 17 cycles
```

---

## 4. Vector Non-Pipeline Simulator Flow (vector_simulator.py)

```
┌────────────────────────────────────────────────┐
│   run(instructions, vector_length, num_lanes)  │
├────────────────────────────────────────────────┤
│                                                │
│  timeline = []                                 │
│  current_cycle = 0                             │
│  register_file = {}                            │
│                                                │
│  for each instruction i:                       │
│                                                │
│    ┌─────── ALU Operation (VADD, VMUL) ───────┐
│    │ iterations = ceil(VL / num_lanes)        │
│    │ duration = latency + iterations - 1      │
│    │                                           │
│    │ Example: VADD V1, V2, V3                 │
│    │   VL=8, lanes=4, add_latency=2           │
│    │   iterations = ceil(8/4) = 2             │
│    │   duration = 2 + 2 - 1 = 3 cycles        │
│    └──────────────────────────────────────────┘
│                                                │
│    ┌─────── Memory (VLOAD, VSTORE) ───────────┐
│    │ iterations = ceil(VL / memory_bandwidth) │
│    │ duration = latency + iterations - 1      │
│    │                                           │
│    │ Example: VLOAD V1, A                     │
│    │   VL=8, bandwidth=4, load_latency=3      │
│    │   iterations = ceil(8/4) = 2             │
│    │   duration = 3 + 2 - 1 = 4 cycles        │
│    └──────────────────────────────────────────┘
│                                                │
│    ┌─────── Reduction (VREDUCE_SUM) ──────────┐
│    │ duration = latency × ceil(log2(VL))      │
│    │                                           │
│    │ Example: VREDUCE_SUM V1, V2              │
│    │   VL=8, red_latency=3                    │
│    │   duration = 3 × ceil(log2(8)) = 9       │
│    └──────────────────────────────────────────┘
│                                                │
│    end_cycle = current_cycle + duration       │
│    timeline.append((i, start, end_cycle))     │
│    current_cycle = end_cycle                  │
│                                                │
│  return timeline, current_cycle                │
│                                                │
└────────────────────────────────────────────────┘

KEY POINT: Không có overlapping lệnh!
  Lệnh i+1 bắt đầu sau lệnh i xong.
```

---

## 5. Vector Pipeline Simulator Flow (pipeline_simulator.py)

```
┌───────────────────────────────────────────────────────┐
│   run(instructions, vector_length, num_lanes)         │
├───────────────────────────────────────────────────────┤
│                                                       │
│  timeline = []                                        │
│  current_cycle = 0                                    │
│  register_file = {}  # dst → ready_cycle             │
│  unit_free_at = {}   # unit → available_cycle        │
│                                                       │
│  for each instruction i:                             │
│                                                       │
│    ┌─ Step 1: Compute earliest_issue due to RAW ─┐   │
│    │ RAW (Read-After-Write) hazard:               │   │
│    │   if instruction i reads register X:         │   │
│    │     wait for any previous write to X        │   │
│    │                                              │   │
│    │ earliest_issue = 1                           │   │
│    │ for each src_reg in src:                     │   │
│    │   if src_reg in register_file:               │   │
│    │     earliest_issue = max(earliest_issue,    │   │
│    │         register_file[src_reg] + 1)          │   │
│    └──────────────────────────────────────────────┘   │
│                                                       │
│    ┌─ Step 2: Compute earliest due to Structural ─┐   │
│    │ Structural hazard:                           │   │
│    │   if unit is busy (cạnh tranh tài nguyên)   │   │
│    │                                              │   │
│    │ unit = get_unit(instruction.opcode)          │   │
│    │ earliest_struct = 1                          │   │
│    │ if unit in unit_free_at:                     │   │
│    │   earliest_struct = unit_free_at[unit] + 1  │   │
│    └──────────────────────────────────────────────┘   │
│                                                       │
│    ┌─ Step 3: Actual issue = max both ────────────┐   │
│    │ actual_issue = max(earliest_issue,           │   │
│    │                    earliest_struct)          │   │
│    └──────────────────────────────────────────────┘   │
│                                                       │
│    ┌─ Step 4: Calculate duration ─────────────────┐   │
│    │ duration = calculate_duration(instruction)   │   │
│    │ (same as vector_simulator)                  │   │
│    └──────────────────────────────────────────────┘   │
│                                                       │
│    ┌─ Step 5: Record timeline & update state ───┐    │
│    │ end_cycle = actual_issue + duration         │    │
│    │ timeline.append((i, actual_issue, end))     │    │
│    │                                              │    │
│    │ register_file[dst] = end_cycle               │    │
│    │ unit_free_at[unit] = end_cycle               │    │
│    └──────────────────────────────────────────────┘   │
│                                                       │
│  return timeline, end_cycle                          │
│                                                       │
└───────────────────────────────────────────────────────┘

KEY POINT: Overlap được nếu không có hazard!

Ví dụ: 
  I0: VLOAD V1, A      [cycle 0-3]
  I1: VLOAD V2, B      [cycle 4-7]  (wait load unit)
  I2: VADD V3, V1, V2  [cycle 8-10] (wait V2 ready)
  
  vs (nếu 2 load unit):
  I0: VLOAD V1, A      [cycle 0-3]
  I1: VLOAD V2, B      [cycle 0-3]  (overlap!)
  I2: VADD V3, V1, V2  [cycle 4-6]  (overlap!)
```

---

## 6. Hazard Detection Chi Tiết

### 6.1 RAW (Read-After-Write) Hazard
```
Lệnh i:  VADD V1, V2, V3   (V1 is destination)
Lệnh i+1: VSUB V4, V1, V5  (V1 is source)

Problem: V1 chưa sẵn khi i+1 execute
Solution: i+1 phải chờ tới cycle (i.end_cycle + 1)
```

### 6.2 WAR (Write-After-Read) Hazard
```
Lệnh i:  VADD V1, V2, V3   (V2 is source)
Lệnh i+1: VMUL V2, V4, V5  (V2 is destination)

Problem: i+1 ghi V2 trước i dùng xong
Solution: i+1 phải chờ cho i execute xong
(Modern out-of-order xử lý rename register, bypass hazard này)
```

### 6.3 Structural Hazard
```
Lệnh i:  VADD V1, V2, V3   (use ALU)  [cycle 0-2]
Lệnh i+1: VMUL V4, V5, V6  (use ALU)  [cycle 3-6]
Lệnh i+2: VSUB V7, V8, V9  (use ALU)  

Problem: Chỉ có 1 ALU, chỉ 1 lệnh xài được cùng lúc
Solution: i+1 chờ i xong, i+2 chờ i+1 xong
(num_alu_units = 3 → tất cả overlap)
```

---

## 7. Chaining / Data Forwarding

```
Normal Pipeline (no chaining):
  I0: VLOAD V1, A    [0-3]
  I1: VADD V2, V1, ...  [4-6]  (chờ V1 ready ở cycle 3+1=4)

With Chaining:
  I0: VLOAD V1, A    [0-3]
  I1: VADD V2, V1, ...  [3-5]  (forward từ memory, không chờ hết)
  
  Công thức: 
  earliest_with_chain = max(earliest_raw - forward_latency, 1)
```

---

## 8. Reduction (Tree-based Parallel)

```
VREDUCE_SUM([a0, a1, a2, a3, a4, a5, a6, a7])

Scalar (Sequential):
  s0 = a0
  s1 = s0 + a1
  s2 = s1 + a2
  ...
  s7 = s6 + a7
  Cycles = 7 × latency

Vector (Parallel Tree):
  
  Level 0:  a0  a1  a2  a3  a4  a5  a6  a7
             │   │   │   │   │   │   │   │
             └─┬─┘   └─┬─┘   └─┬─┘   └─┬─┘
  Level 1: (a0+a1) (a2+a3) (a4+a5) (a6+a7)  [latency cycles]
             │       │       │       │
             └───┬───┘       └───┬───┘
  Level 2: (a0+..+a3)    (a4+..+a7)        [latency cycles]
             │               │
             └───────┬───────┘
  Level 3: (a0+..+a7)                      [latency cycles]
  
  Cycles = ceil(log2(8)) × latency = 3 × latency
  
  Công thức: cycles = ceil(log2(vector_length)) × reduction_latency
```

---

## 9. Gather/Scatter Pattern

```
VGATHER V1, base_array, indices:

Input:  base_array = [a0, a1, a2, a3, a4, a5, a6, a7]
        indices = [1, 3, 0, 5, 2, 4, 7, 6]

Execution (parallel per lane):
  Lane 0: V1[0] = base_array[1]   = a1
  Lane 1: V1[1] = base_array[3]   = a3
  Lane 2: V1[2] = base_array[0]   = a0
  ...
  
Cycles:
  - Compute effective address: 1 cycle (startup)
  - Memory access (possibly random): load_latency + ceil(VL/bw)
  
Total: startup + load_latency + ceil(VL / memory_bandwidth)

Risk: Bank conflict nếu 2+ lane access cùng address
      → serialization, cycles tăng
```

---

## 10. Masking (Predicate Execution)

```
VADD_MASKED V3, V1, V2, V9:

Trước:  V1 = [1, 2, 3, 4]
        V2 = [10, 20, 30, 40]
        V9 = [1, 0, 1, 0]  (mask: 1=active, 0=inactive)

Sau:    V3 = [1+10, ?, 3+30, ?]
             = [11, (unchanged), 33, (unchanged)]
        
        Hoặc:
        V3 = [11, 0, 33, 0]  (nếu mask_mode = "zero")

Cycles: Vẫn = add_latency + iterations - 1
        (không tăng, nhưng inactive lane skip)

Lane utilization giảm = (num_active_lanes / num_lanes)
```

---

## 11. Timeline Output Example

```
Program:
  DATA A = [1,2,3,4]
  DATA B = [5,6,7,8]
  VLOAD V1, A
  VLOAD V2, B
  VADD V3, V1, V2
  VSTORE C, V3

Vector Simulator Output:
┌────────┬────────────┬──────────┬────────────┐
│ Instr# │ Start Cyc  │ End Cyc  │ Duration   │
├────────┼────────────┼──────────┼────────────┤
│ 0: VL1 │     0      │    3     │ 3 cycles   │
│ 1: VL2 │     4      │    7     │ 4 cycles   │
│ 2: VADD│     8      │   10     │ 3 cycles   │
│ 3: VSTO│    11      │   14     │ 4 cycles   │
└────────┴────────────┴──────────┴────────────┘

Total cycles (non-pipeline): 14

Pipeline Simulator Output (same config, but 2 load units):
┌────────┬────────────┬──────────┬────────────┐
│ Instr# │ Start Cyc  │ End Cyc  │ Duration   │
├────────┼────────────┼──────────┼────────────┤
│ 0: VL1 │     0      │    3     │ 3 cycles   │
│ 1: VL2 │     0      │    3     │ 3 cycles   │ (overlap!)
│ 2: VADD│     4      │    6     │ 3 cycles   │ (wait V2)
│ 3: VSTO│     7      │   10     │ 4 cycles   │
└────────┴────────────┴──────────┴────────────┘

Total cycles (pipeline): 10
Speedup = 14/10 = 1.4x
```

---

## 12. Performance Analysis Flow

```
┌────────────────────────────────┐
│ PerformanceAnalyzer.summarize()│
├────────────────────────────────┤
│                                │
│ Input:                         │
│   - mode: "scalar"/"vector"/.. │
│   - scalar_cycles              │
│   - vector_cycles              │
│   - num_elements               │
│   - vector_length              │
│   - clock_frequency_mhz        │
│                                │
│ Calculate:                     │
│                                │
│ 1. Speedup                     │
│    = scalar_cycles / vector_   │
│      cycles                    │
│                                │
│ 2. Throughput                  │
│    = num_elements /            │
│      vector_cycles             │
│                                │
│ 3. Lane Utilization            │
│    = useful_lane_work /        │
│      (num_lanes × cycles)      │
│                                │
│ 4. Execution Time              │
│    = vector_cycles / (freq/1e6)│
│                                │
│ Return: dict with metrics      │
│                                │
└────────────────────────────────┘
```

---

## 13. Comparison Matrix: Scalar vs Vector vs Pipeline

```
┌─────────────┬────────────┬───────────────┬──────────────┐
│ Aspect      │ Scalar     │ Vector        │ Pipeline     │
├─────────────┼────────────┼───────────────┼──────────────┤
│ Parallel    │ None       │ Yes (lanes)   │ Yes (lanes+  │
│ Execution   │            │               │  instruction)│
├─────────────┼────────────┼───────────────┼──────────────┤
│ Hazard      │ N/A        │ None          │ RAW, WAR,    │
│ Handling    │            │ (sequential)  │ Structural   │
├─────────────┼────────────┼───────────────┼──────────────┤
│ Cycles      │ sum(lat ×  │ max(lat) ×    │ Hazard       │
│ Formula     │ elements)  │ ceil(VL/lanes)│ delayed      │
├─────────────┼────────────┼───────────────┼──────────────┤
│ Speedup     │ 1x         │ 2-4x          │ 2-8x+        │
│ Range       │ (baseline) │ (typical)     │ (with chain) │
├─────────────┼────────────┼───────────────┼──────────────┤
│ Best for    │ Simple,    │ Data-         │ Complex      │
│             │ sequential │ parallel      │ deps, ILP    │
│             │ code       │ workloads     │ workloads    │
└─────────────┴────────────┴───────────────┴──────────────┘
```

---

## 14. Quick Algorithm Lookup Table

| Operation | Cycles | Notes |
|-----------|--------|-------|
| VLOAD | latency + ceil(VL/BW) | BW = bandwidth |
| VSTORE | latency + ceil(VL/BW) | Same as VLOAD |
| VADD | latency + ceil(VL/lanes) - 1 | Pipelining within |
| VMUL | mul_latency + ceil(VL/lanes) - 1 | Longer latency |
| VREDUCE | red_latency × ceil(log2(VL)) | Tree reduction |
| VGATHER | latency + ceil(VL/BW) | Random access risk |
| VSCATTER | latency + ceil(VL/BW) | Write after gather |
| VAND/OR/XOR | latency + ceil(VL/lanes) - 1 | Fast logic ops |

