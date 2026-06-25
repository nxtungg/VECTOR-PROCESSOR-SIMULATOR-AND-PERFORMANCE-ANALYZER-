# Hướng dẫn Thứ tự Đọc Code và Các Thuật toán

## Phần 1: Hiểu Tổng Quan Project

**Mục tiêu:** Nắm bắt được cấu trúc tổng thể và luồng chạy chính.

### 1.1. Tài liệu chính
- **README.md** - Tổng quan project, cách chạy, cấu trúc thư mục
- **config.json** - Cấu hình phần cứng (vector_length, num_lanes, latencies, ...)

### 1.2. Entry point
- **main.py** - CLI chính, điểm bắt đầu của toàn project
  - Dòng 35-133: `run_program()` - hàm chính chạy mô phỏng
  - Dòng 45-56: Load config và khởi tạo các simulator
  - Dòng 74-108: Chạy vector non-pipeline và pipeline simulator
  
---

## Phần 2: Hiểu Cấu Trúc Dữ Liệu

**Mục tiêu:** Nắm các kiểu dữ liệu cơ bản và cách biểu diễn.

### 2.1. Định nghĩa lệnh
- **instruction.py** - Định nghĩa `Instruction` dataclass
  - Dòng 5-25: `InstructionType`, `FunctionalUnit` enum
  - Dòng 27-56: Class `Instruction` với fields: opcode, dst, src, mask, mask_mode
  - Dòng 57-80: `inst_type` property phân loại lệnh
  
### 2.2. Cấu hình phần cứng
- **config.py** - Class `VectorProcessorConfig`
  - Vector length, số lanes, latencies, memory bandwidth
  - Các tham số phần cứng được lưu và validate tại đây

### 2.3. Bộ nhớ
- **memory.py** - Mô hình bộ nhớ
  - Bộ nhớ chính, cache hierarchy
  - Các phép toán load/store

---

## Phần 3: Input Parsing (Đọc Chương Trình Đầu Vào)

**Mục tiêu:** Hiểu cách parser chuyển file `.txt` thành instruction list.

### 3.1. Parser chính
- **parser.py** - Class `ProgramParser`
  - Dòng 11-73: Khởi tạo, định nghĩa opcode hợp lệ
  - Dòng 75-?: Hàm `parse_file()` - đọc file, trích data, scalar, instruction
  - Xử lý định dạng: `DATA A = [...]`, `SCALAR a = ...`, `VLOAD V1, A`

### 3.2. Định dạng chương trình
Xem file ví dụ:
- `examples/vector_add.txt` - Phép cộng vector đơn giản
- `examples/saxpy.txt` - SAXPY: Y = aX + Y
- `examples/dot_product.txt` - Tích vô hướng (reduction)

---

## Phần 4: Kiến Trúc Mô Phỏng

**Mục tiêu:** Hiểu ba mô hình mô phỏng chính.

### 4.1. Baseline Scalar Simulator
- **scalar_simulator.py**
  - Đơn giản nhất: tuần tự từng phần tử
  - Công thức: cycles = số phần tử × latency (data-dependent)
  - Dùng làm baseline để tính speedup

**Lưu ý:** Không có parallelism, mỗi phần tử xử lý lần lượt.

### 4.2. Vector Non-Pipeline Simulator
- **vector_simulator.py** - QUAN TRỌNG
  - Xử lý toàn bộ vector cùng lúc (vectorized)
  - **Không có pipeline**: phải chờ hết tất cả lane xong mới chạy lệnh tiếp
  - Tính toán: max(lanes_latency) × ceil(vector_length / num_lanes)
  - Có hỗ trợ chaining (data forwarding)

**Điểm chính:**
- Dòng 80-90: Hàm `run()` - thực thi danh sách instruction
- Khối lệnh ALU, MEMORY, REDUCTION được xử lý tuần tự (không pipeline)

### 4.3. Pipeline Simulator
- **pipeline_simulator.py** - PHỨC TẠP NHẤT
  - Cho phép lệnh chồng chéo (overlap)
  - Xử lý data hazard (RAW, WAR, WAW)
  - Xử lý structural hazard (cạnh tranh tài nguyên)
  - Xử lý control hazard (branching nếu có)

**Điểm chính:**
- Dòng 100-150: Khối pipeline logic chính
- Tracking dependency graph giữa các lệnh
- Tính toán "earliest issue cycle" cho mỗi lệnh

### 4.4. Out-of-Order Simulator (Mở rộng)
- **ooo_simulator.py** - Cho phép lệnh sai thứ tự
  - Thực thi khi operand sẵn sàng (data-flow model)
  - Có instruction window/reorder buffer

---

## Phần 5: Chi Tiết Các Thuật Toán Chính

### **Thuật toán 1: Parse Chương Trình**
**File:** parser.py → `parse_file()`

```
1. Đọc file từng dòng
2. Loại comment (#...) và whitespace
3. Nếu dòng = "DATA ...", trích mảng vào dict data
4. Nếu dòng = "SCALAR ...", trích giá trị vào dict scalars
5. Nếu dòng = opcode, parse thành Instruction object
6. Validate: check opcode hợp lệ, số operand đúng
7. Return: (data_dict, scalar_dict, instruction_list)
```

### **Thuật toán 2: Scalar Simulator - Baseline**
**File:** scalar_simulator.py → `estimate_cycles()`

```
Input: instruction_list, data
Output: total_cycles (số chu kỳ ước lượng)

for each instruction in instruction_list:
    if VLOAD:
        cycles += load_latency
    elif VSTORE:
        cycles += store_latency
    elif VADD/VSUB:
        cycles += add_latency × num_elements
    elif VMUL:
        cycles += mul_latency × num_elements
    elif VREDUCE_SUM:
        cycles += reduction_latency × log(num_elements)
    else:
        cycles += operation_latency × num_elements

return total_cycles
```

### **Thuật toán 3: Vector Non-Pipeline Simulator**
**File:** vector_simulator.py → `run()`

```
Input: instruction_list, vector_length, num_lanes
Output: memory_state, timeline, cycles

timeline = []  # ghi nhận mỗi lệnh chạy ở cycle nào

current_cycle = 0
for each instruction in instruction_list:
    
    if is_ALU_operation (VADD, VMUL, ...):
        # Xử lý tất cả lanes song song
        num_iterations = ceil(vector_length / num_lanes)
        duration = operation_latency + num_iterations - 1
        # Chaining có thể giảm duration
        
    elif is_MEMORY (VLOAD, VSTORE):
        num_iterations = ceil(vector_length / memory_bandwidth)
        duration = load/store_latency + num_iterations - 1
        
    elif is_REDUCTION (VREDUCE_SUM, ...):
        # Parallel reduction tree
        duration = reduction_latency × ceil(log2(num_lanes))
        
    end_cycle = current_cycle + duration
    timeline.append({instruction, start: current_cycle, end: end_cycle})
    current_cycle = end_cycle

return memory_state, timeline, current_cycle
```

**Điểm chính:** Không có overlapping giữa lệnh, luôn phải chờ hết lệnh trước.

### **Thuật toán 4: Pipeline Simulator - Với Hazard Detection**
**File:** pipeline_simulator.py → `run()`

```
Input: instruction_list, vector_length, num_lanes
Output: memory_state, timeline, cycles

timeline = []
register_file = {}
dependency_graph = {}

for i, instruction in enumerate(instruction_list):
    
    # Tính dependent_instructions (RAW hazard)
    # Nếu dst của lệnh trước = src của lệnh hiện tại
    earliest_issue = max(
        1 + end_cycle[j] for j in dependent_instructions
    )
    
    # Tính structural hazard (cạnh tranh ALU, MEMORY)
    # Nếu unit bận, phải chờ
    earliest_available = max(
        1 + available_cycle[unit] for unit in used_units
    )
    
    actual_issue = max(earliest_issue, earliest_available)
    
    # Tính duration (như non-pipeline)
    duration = calculate_duration(instruction)
    
    actual_end = actual_issue + duration
    
    timeline.append({instruction, start: actual_issue, end: actual_end})
    register_file[dst] = actual_end  # ghi nhớ khi nào data sẵn sàng
    available_cycle[unit] = actual_end
    
return memory_state, timeline, actual_cycle
```

**Điểm khác biệt:** Lệnh có thể overlap nếu không conflict.

### **Thuật toán 5: Chaining / Data Forwarding**
**File:** vector_simulator.py hoặc pipeline_simulator.py

```
Khi enable_chaining = true:
    Nếu lệnh i+1 dùng kết quả từ lệnh i:
        Không cần chờ hết lệnh i
        Có thể bắt đầu sớm hơn nếu data forwarding có sẵn
        
    Công thức chaining:
        actual_issue[i+1] = end_cycle[i] - min_forward_delay
        (min_forward_delay phụ thuộc vào cấu trúc pipeline)
```

### **Thuật toán 6: Reduction (Tích vô hướng, Sum, Max, Min)**
**File:** vector_simulator.py, pipeline_simulator.py

```
Ví dụ VREDUCE_SUM([a0, a1, a2, a3]):

Cách scalar (baseline):
    result = a0 + a1 + a2 + a3
    cycles = 3 × add_latency (tuần tự)

Cách vector parallel:
    Round 1: (a0+a1), (a2+a3)           // 2 phép toán song song
    Round 2: (a0+a1) + (a2+a3)          // 1 phép toán
    
    Cycles = ceil(log2(4)) × add_latency
           = 2 × add_latency
```

### **Thuật toán 7: Gather/Scatter**
**File:** parser.py, vector_simulator.py

```
VGATHER V1, base_array, indices:
    for each lane in parallel:
        V1[lane] = base_array[indices[lane]]
    cycles = load_latency + ceil(vector_length / memory_bandwidth)

VSCATTER output_array, indices, V1:
    for each lane in parallel:
        output_array[indices[lane]] = V1[lane]
    cycles = store_latency + ceil(vector_length / memory_bandwidth)

Rủi ro: Conflict nếu 2 lane access cùng địa chỉ → serialization
```

### **Thuật toán 8: Masking (Predicate Execution)**
**File:** instruction.py, vector_simulator.py

```
VADD_MASKED V3, V1, V2, V9:
    # V9 là mask register (0 = inactive, 1 = active)
    for each lane:
        if V9[lane] == 1:
            V3[lane] = V1[lane] + V2[lane]
        else:
            V3[lane] = V3[lane]  (merge mode) hoặc 0 (zero mode)
    
    cycles = add_latency (không tăng, nhưng lane inactive bị skip)
    lane_utilization giảm nếu nhiều lane inactive
```

---

## Phần 6: Phân Tích Hiệu Năng

### 6.1. Performance Analyzer
- **performance_analyzer.py**
  - Tính: speedup, throughput, lane utilization, execution time
  - So sánh scalar vs vector vs pipeline

**Công thức chính:**
```
Speedup = scalar_cycles / vector_cycles
Throughput (elem/cycle) = vector_length / vector_cycles
Lane Utilization = (actual_lane_work) / (num_lanes × cycles)
Execution Time (ns) = cycles / (clock_frequency_mhz × 1e6)
```

### 6.2. Report Generator
- **report_generator.py**
  - Xuất timeline CSV
  - Tạo Markdown report

---

## Phần 7: Giao Diện Web

- **app.py** - Streamlit interface
  - Upload file chương trình
  - Chỉnh tham số config (vector length, lanes, ...)
  - Bật cache, OOO simulator
  - Xem kết quả timeline, register file, performance chart

---

## Phần 8: Các Ví Dụ Workload

**Thứ tự đề xuất để hiểu:**

1. **vector_add.txt** - Đơn giản nhất
   ```
   C = A + B
   - 1 VLOAD, 1 VLOAD, 1 VADD, 1 VSTORE
   - Không reduction, không gather/scatter
   ```

2. **saxpy.txt** - Thêm scalar phép toán
   ```
   Y = aX + Y
   - Multiply scalar, cộng vector
   ```

3. **dot_product.txt** - Thêm reduction
   ```
   result = A · B
   - VDOT → Giới thiệu parallel reduction
   ```

4. **matrix_vector.txt** - Vòng lặp lồng
   ```
   Y = A × X (matrix-vector multiply)
   ```

5. **gather_scatter.txt** - Truy xuất ngẫu nhiên
   ```
   VGATHER, VSCATTER
   - Điểm cạnh tranh bộ nhớ
   ```

6. **convolution_1d.txt, convolution_2d.txt** - Phức tạp
   - Convolution 1D/2D
   - Im2col optimization

---

## Phần 9: Các File Test

- **tests/test_parser.py** - Kiểm tra parser
- **tests/test_vector_simulator.py** - Kiểm tra logic mô phỏng
- **tests/test_performance.py** - Kiểm tra tính toán cycles
- **tests/test_examples.py** - Chạy các ví dụ workload

---

## Phần 10: Thứ Tự Học Tập Được Đề Xuất

### **Tuần 1: Cơ bản**
1. Đọc README.md
2. Đọc main.py (tổng quan luồng chạy)
3. Đọc instruction.py (hiểu Instruction dataclass)
4. Đọc config.py (cấu hình phần cứng)
5. Đọc parser.py (cách parse file)
6. Chạy ví dụ: `python main.py --program examples/vector_add.txt --mode all`

### **Tuần 2: Simulator Logic**
1. Đọc scalar_simulator.py (baseline, dễ nhất)
2. Đọc vector_simulator.py (non-pipeline, quan trọng)
3. Hiểu timeline output (cycle tracking)
4. Chạy ví dụ compare scalar vs vector: `python main.py --program examples/saxpy.txt --mode vector --export`

### **Tuần 3: Pipeline**
1. Đọc pipeline_simulator.py (hazard detection, complexity)
2. Hiểu dependency graph
3. Hiểu structural hazard, RAW hazard
4. Chạy ví dụ: `python main.py --program examples/dot_product.txt --mode pipeline --export`

### **Tuần 4: Nâng cao**
1. Đọc cache.py, memory.py (mô phỏng bộ nhớ)
2. Đọc ooo_simulator.py (out-of-order execution)
3. Đọc convolution_simulator.py (chuyên biệt cho CNN)
4. Đọc performance_analyzer.py (công thức speedup)

### **Tuần 5: Interface & Testing**
1. Chạy app.py (Streamlit)
2. Chạy run_experiments.py (thí nghiệm tự động)
3. Đọc visualization.py, report_generator.py (output)
4. Chạy tests: `python -m unittest discover -s tests -v`

---

## Phần 11: Sơ Đồ Luồng Chính

```
main.py
  ↓
Load config.json
  ↓
parser.parse_file(program.txt)
  ├→ data dict
  ├→ scalars dict
  └→ instructions list
  ↓
Run three simulators in parallel:
  ├→ scalar_simulator.estimate_cycles()
  │    └→ baseline cycles
  ├→ vector_simulator.run() [if mode=vector/all]
  │    └→ timeline, cycles, register file
  └→ pipeline_simulator.run() [if mode=pipeline/all]
       └→ timeline, cycles, register file
  ↓
performance_analyzer.summarize()
  └→ speedup, throughput, utilization
  ↓
report_generator.print_timeline()
report_generator.save_performance()
  ↓
Output: CSV, Markdown report
```

---

## Phần 12: Chú ý Khi Đọc Code

1. **Timeline:** Mỗi lệnh ghi nhận (start_cycle, end_cycle)
2. **Register file:** Ghi nhớ khi nào giá trị data sẵn sàng (để hazard detection)
3. **Chaining:** Để enable, set `enable_chaining: true` trong config.json
4. **Masking:** Mỗi lệnh có thể có mask register (từ instruction.py → mask field)
5. **Memory bandwidth:** Giới hạn số phần tử load/store mỗi cycle
6. **Latency vs throughput:**
   - Latency = bao lâu lệnh hoàn thành từ khi bắt đầu
   - Throughput = độc lập với lane parallel

---

## Tóm Tắt: Thuật Toán Chính (Quick Reference)

| Thuật toán | File | Công thức chính |
|---|---|---|
| **Parse** | parser.py | Dòng → Instruction object |
| **Scalar baseline** | scalar_simulator.py | cycles = sum(latencies × num_elements) |
| **Vector non-pipeline** | vector_simulator.py | cycles = max(duration_per_op) |
| **Vector pipeline** | pipeline_simulator.py | cycles += max(hazard_delay, structural_delay) |
| **Reduction** | vector_simulator.py | cycles ≈ latency × ceil(log2(num_lanes)) |
| **Gather/Scatter** | vector_simulator.py | cycles ≈ latency + ceil(length/bandwidth) |
| **Masking** | instruction.py | Predicate per-lane execution |
| **Chaining** | vector_simulator.py | Giảm start_cycle lệnh tiếp theo |
| **Out-of-order** | ooo_simulator.py | Data-flow model: issue khi operand ready |

---

## Lưu Ý Quan Trọng

- **Không có branching:** Simulator không hỗ trợ branch (mọi instruction đều execute)
- **Deterministic:** Không có randomness trong mô phỏng
- **Python, không HDL:** Đây là behavioral simulation, không RTL synthesis
- **Register file là array:** Không mô phỏng port conflict chi tiết

