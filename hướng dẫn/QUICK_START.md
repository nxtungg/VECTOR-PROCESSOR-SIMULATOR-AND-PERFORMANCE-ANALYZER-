# Quick Start: Các File Quan Trọng & Thuật Toán

## 🎯 Thứ Tự Đọc Tối Ưu (Theo Mức Độ Phức Tạp)

### **Tier 1: Cơ bản (30 phút)**
Đọc những file này để hiểu flow chính:

1. **[config.py](config.py)** - Cấu hình phần cứng
   - Dataclass với fields: vector_length, num_lanes, latencies, ...
   - Nơi tất cả các tham số phần cứng được định nghĩa

2. **[instruction.py](instruction.py)** - Định nghĩa lệnh
   - `Instruction` dataclass: opcode, dst, src, mask
   - `InstructionType` enum: MEMORY_LOAD, ARITHMETIC_BINARY, ...
   - 30 dòng đầu là cốt lõi

3. **[parser.py](parser.py)** (dòng 1-100)
   - `ProgramParser` class
   - Hàm `parse_file()` - chuyển file `.txt` → instruction list

4. **[main.py](main.py)** (dòng 35-133)
   - Hàm `run_program()` - entry point chính
   - Khái quát 3 simulator: scalar → vector → pipeline

**Bài tập:** Chạy một ví dụ đơn giản
```bash
python main.py --program examples/vector_add.txt --mode all
```

---

### **Tier 2: Simulators (1-2 giờ)**
Các simulator chính lôgic:

1. **[scalar_simulator.py](scalar_simulator.py)**
   - Baseline đơn giản nhất
   - Công thức: `cycles = sum(latency × num_elements)`
   - Không parallelism

2. **[vector_simulator.py](vector_simulator.py)** ⭐ **QUAN TRỌNG**
   - Vector non-pipeline simulator
   - Lệnh chạy song song (num_lanes), nhưng tuần tự (không overlap)
   - Công thức: `cycles = latency + ceil(VL / num_lanes) - 1`

3. **[pipeline_simulator.py](pipeline_simulator.py)** ⭐⭐ **PHỨC TẠP NHẤT**
   - Vector pipeline simulator
   - Hazard detection: RAW, WAR, structural
   - Cho phép lệnh overlap nếu không conflict
   - 100+ dòng, nhưng logic cốt lõi: dependency tracking + issue cycle

**Bài tập:** So sánh 3 mode
```bash
python main.py --program examples/saxpy.txt --mode scalar --export
python main.py --program examples/saxpy.txt --mode vector --export
python main.py --program examples/saxpy.txt --mode pipeline --export
```

Xem timeline CSV để hiểu sự khác biệt.

---

### **Tier 3: Nâng cao (1-2 giờ)**

1. **[memory.py](memory.py)** - Mô hình bộ nhớ
   - Cache, memory hierarchy
   - Load/store operations

2. **[cache.py](cache.py)** - Mô phỏng cache
   - LRU replacement
   - Cache hit/miss logic

3. **[ooo_simulator.py](ooo_simulator.py)** - Out-of-order execution
   - Data-flow model
   - Instruction window, reorder buffer

4. **[convolution_simulator.py](convolution_simulator.py)** - Chuyên biệt
   - Optimized cho CNN workload
   - Im2col pattern

5. **[performance_analyzer.py](performance_analyzer.py)** - Phân tích
   - Tính speedup, throughput, utilization
   - Công thức: `Speedup = scalar_cycles / vector_cycles`

6. **[visualization.py](visualization.py)** + **[report_generator.py](report_generator.py)**
   - Xuất kết quả CSV, biểu đồ, Markdown report

---

### **Tier 4: Interface & Testing (30 phút)**

1. **[app.py](app.py)** - Streamlit web interface
   - Chạy: `streamlit run app.py`

2. **[run_experiments.py](run_experiments.py)** - Thí nghiệm tự động
   - Chạy: `python run_experiments.py`
   - Tạo 7 thí nghiệm, so sánh workload

3. **[tests/](tests/)** - Unit tests & integration tests
   - Chạy: `python -m unittest discover -s tests -v`

---

## 📊 Các Thuật Toán Chính (Công Thức Ghi Nhớ)

### **1️⃣ Parse (parser.py)**
```
Đầu vào: file.txt
         ↓
[For each line]
  • If "DATA ..." → Extract array
  • If "SCALAR ..." → Extract scalar
  • If "OPCODE ..." → Parse instruction
         ↓
Đầu ra: (data_dict, scalar_dict, instruction_list)
```

---

### **2️⃣ Scalar Baseline (scalar_simulator.py)**
```
cycles = 0
for each instruction:
    if VLOAD:
        cycles += load_latency
    elif VADD:
        cycles += add_latency × num_elements
    elif VREDUCE_SUM:
        cycles += reduction_latency × num_elements  (sequential)
        
Example: 4-element VADD, add_latency=2
  cycles = 2 × 4 = 8 cycles
  (phần tử 0, 1, 2, 3 xử lý tuần tự)
```

---

### **3️⃣ Vector Non-Pipeline (vector_simulator.py) ⭐**
```
timeline = []
for each instruction i:
    
    ┌─ ALU (VADD, VMUL, ...) ─┐
    │ iterations = ceil(VL / num_lanes)
    │ duration = latency + iterations - 1
    │
    │ Example: VADD, VL=8, lanes=4, latency=2
    │   iterations = 2
    │   duration = 2 + 2 - 1 = 3 cycles
    │
    │ Lanes:        Lane 0-3          Lane 4-7
    │ Cycle 0:      Add start
    │ Cycle 1:      Add complete (lat=2)
    │ Cycle 2:      Next iteration starts
    │ Cycle 3:      Next iteration complete
    └──────────────────────────────┘
    
    ┌─ Memory (VLOAD, VSTORE) ─┐
    │ duration = lat + ceil(VL / bandwidth) - 1
    │
    │ Example: VLOAD, VL=8, BW=4, lat=3
    │   iterations = 2
    │   duration = 3 + 2 - 1 = 4 cycles
    └─────────────────────────┘
    
    ┌─ Reduction (VREDUCE_SUM) ─┐
    │ duration = latency × ceil(log2(VL))
    │
    │ Example: VREDUCE_SUM, VL=8, lat=3
    │   duration = 3 × 3 = 9 cycles (parallel tree)
    │
    │ vs Scalar: 3 × 8 = 24 cycles (sequential)
    └──────────────────────────┘
    
    timeline.append((instruction, start_cycle, end_cycle))
    current_cycle = end_cycle  ← ⚠️ NO OVERLAP!
    
Nhận xét: Lệnh i+1 ALWAYS chờ lệnh i xong
```

---

### **4️⃣ Vector Pipeline (pipeline_simulator.py) ⭐⭐**
```
KEY: Lệnh CÓ THỂ overlap nếu KHÔNG có hazard

for each instruction i:
    
    ┌─ Step 1: Compute RAW (Read-After-Write) delay ─┐
    │ RAW: Lệnh i đọc register → phải chờ writer     │
    │ earliest_raw = max(1)                           │
    │ for each src_reg in src:                        │
    │   if src_reg written by previous instr:         │
    │     earliest_raw = writer.end_cycle + 1         │
    └───────────────────────────────────────────────┘
    
    ┌─ Step 2: Compute Structural delay ────────────┐
    │ Structural: Cạnh tranh unit (ALU, Memory, ..)  │
    │ earliest_struct = max(1)                        │
    │ for each required_unit:                        │
    │   if unit busy:                                │
    │     earliest_struct = unit.free_at + 1         │
    └───────────────────────────────────────────────┘
    
    actual_issue = max(earliest_raw, earliest_struct)
    duration = calculate_duration(instruction)
    actual_end = actual_issue + duration
    
    timeline.append((i, actual_issue, actual_end))
    
Ví dụ (2 load units):
  I0: VLOAD V1    [0-3]
  I1: VLOAD V2    [0-3]  ← overlap! (diff units)
  I2: VADD V3, V1, V2 [4-6]  ← chờ V2 (RAW hazard)
  
vs (1 load unit):
  I0: VLOAD V1    [0-3]
  I1: VLOAD V2    [4-7]  ← phải chờ (same unit)
  I2: VADD V3     [8-10]
```

---

### **5️⃣ Chaining / Data Forwarding**
```
Normal (no chaining):
  I0: VLOAD V1     [0-3]
  I1: VADD V2, V1  [4-6]  ← wait V1 ready

With chaining (enable_chaining = true):
  I0: VLOAD V1     [0-3]
  I1: VADD V2, V1  [3-5]  ← forward from mem unit
  
Effect: Giảm pipeline stall
```

---

### **6️⃣ Reduction (Tree-based Parallel)**
```
Scalar: result = a0 + a1 + a2 + a3
  s0 = a0
  s1 = s0 + a1    [latency]
  s2 = s1 + a2    [latency]
  s3 = s2 + a3    [latency]
  Total: 3 × latency (sequential)

Vector (parallel reduction tree):
          a0 + a1    a2 + a3       [latency]
            ↓          ↓
        (a0+a1) + (a2+a3)          [latency]
            ↓
        Final result                [latency]
  
  Total: 3 × latency (log2(4) = 2 levels)
  
Công thức: cycles = ceil(log2(vector_length)) × reduction_latency
           = 3 × latency (for 4-element vector)
```

---

### **7️⃣ Gather/Scatter**
```
VGATHER V1, base_array, indices:
  for each lane (parallel):
    V1[lane] = base_array[indices[lane]]
  
  cycles = load_latency + ceil(VL / memory_bandwidth)

Rủi ro: Bank conflict nếu 2+ lane access same address
        → serialization → cycles tăng

VSCATTER output_array, indices, V1:
  for each lane (parallel):
    output_array[indices[lane]] = V1[lane]
    
  cycles = store_latency + ceil(VL / memory_bandwidth)
```

---

### **8️⃣ Masking (Predicate Execution)**
```
VADD_MASKED V3, V1, V2, V9:
  for each lane:
    if V9[lane] == 1:  ← active
      V3[lane] = V1[lane] + V2[lane]
    else:              ← inactive
      V3[lane] = unchanged (merge mode)
      
  cycles: SAME (không tăng)
  lane_utilization: giảm (inactive lane skip)
  
  Example: V9 = [1,0,1,0]
    lanes active = 2/4 = 50% utilization
```

---

## 📈 Comparison Table

```
┌─────────────┬──────────────┬─────────────┬──────────────┐
│ Metric      │ Scalar       │ Vector      │ Pipeline     │
├─────────────┼──────────────┼─────────────┼──────────────┤
│ Parallelism │ None         │ Lane-level  │ Lane + Inst  │
│ Overlap     │ No           │ No          │ Yes (if OK)  │
│ Hazard      │ N/A          │ None        │ RAW, struct  │
│ Cycles      │ ~100-1000s   │ 10-100s     │ 5-50s        │
│ Speedup     │ 1x           │ 2-4x        │ 5-20x        │
│ Complexity  │ ⭐           │ ⭐⭐        │ ⭐⭐⭐      │
└─────────────┴──────────────┴─────────────┴──────────────┘
```

---

## 🔥 Top 5 Điểm Cần Nhớ

1. **Scalar = Sequential, Baseline**
   - Mỗi phần tử xử lý 1 cái
   - Công thức: `cycles = latency × num_elements`

2. **Vector = Parallel Lanes, Sequential Instructions**
   - Tất cả lane xử lý cùng lúc
   - Nhưng lệnh i+1 PHẢI chờ i xong

3. **Pipeline = Overlap Instructions**
   - Lệnh i+1 có thể bắt đầu trong khi i chưa xong
   - Nhưng phải check hazard trước

4. **Reduction = Tree Not Sequential**
   - Log tree parallel, không loop sequential
   - Speedup lớn cho large vectors

5. **Cycles Formula**
   ```
   Scalar:     latency × N
   Vector:     latency + ceil(N/lanes) - 1
   Pipeline:   hazard_delay + latency
   Reduction:  ceil(log2(N)) × latency
   ```

---

## 🚀 Bài Tập Thực Hành (Theo Độ Khó)

### **Easy (15 phút)**
```bash
# Chạy vector_add
python main.py --program examples/vector_add.txt --mode all --export

# Xem kết quả
cat outputs/performance.csv
cat outputs/report_vector_add.md
```

### **Medium (30 phút)**
```bash
# So sánh scalar vs vector vs pipeline
python main.py --program examples/saxpy.txt --mode all --export

# Chạy với cấu hình khác
python main.py --program examples/saxpy.txt --vector-length 32 --num-lanes 8

# Xem timeline chi tiết
cat outputs/timeline_vector.csv
cat outputs/timeline_pipeline.csv
```

### **Hard (1 giờ)**
```bash
# Chạy toàn bộ thí nghiệm
python run_experiments.py

# Phân tích kết quả
ls experiment_results/
cat experiment_results/exp*.csv

# Xem biểu đồ
# Open experiment_results/figures/*.png
```

### **Advanced (2+ giờ)**
```bash
# Chạy với cache simulation
python app.py  # Streamlit interface

# Enable cache & OOO simulator trong UI
# Upload custom program

# Hoặc modify config.json
# {"enable_chaining": true, "enable_masking": true}

python main.py --program examples/mask_operation.txt --mode all --config config.json
```

---

## 📖 File Reference Sheet

| File | Lines | Purpose | Tier |
|------|-------|---------|------|
| config.py | 50 | Hardware config dataclass | 1 |
| instruction.py | 150 | Instruction definition | 1 |
| parser.py | 300+ | Parse program file | 1 |
| main.py | 200 | CLI entry point | 1 |
| scalar_simulator.py | 200 | Baseline simulator | 2 |
| vector_simulator.py | 400+ | Vector non-pipeline | 2 |
| pipeline_simulator.py | 500+ | Vector pipeline w/ hazard | 2 |
| memory.py | 300+ | Memory model | 3 |
| ooo_simulator.py | 400+ | Out-of-order exec | 3 |
| performance_analyzer.py | 200 | Speedup/throughput | 3 |
| report_generator.py | 250 | Export CSV/MD | 3 |
| app.py | 400+ | Streamlit interface | 4 |
| run_experiments.py | 500+ | Automated experiments | 4 |

---

## 🎓 Học Tập Đề Xuất

**Week 1:** Tier 1 (config, instruction, parser, main)  
**Week 2:** Tier 2 (scalar, vector, pipeline simulators)  
**Week 3:** Tier 3 (memory, cache, OOO, analysis)  
**Week 4:** Tier 4 (interface, tests, experiments)

Mỗi tuần đọc + chạy ví dụ = hiểu sâu

