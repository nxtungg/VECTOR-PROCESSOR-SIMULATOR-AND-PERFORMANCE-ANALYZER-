# 📚 Hướng Dẫn Đầu Tiên - Index

Dự án này có 4 file hướng dẫn chi tiết dành cho bạn. Hãy bắt đầu từ đây!

---

## 🎯 Bạn nên đọc gì?

### ❓ "Tôi chưa hiểu gì về project"
👉 **Bắt đầu:** [QUICK_START.md](QUICK_START.md)
- Tóm tắt 30 phút về 3 simulator chính
- Reference table các file quan trọng
- Bài tập từ dễ → khó
- Công thức nhắc nhở nhanh

**Rồi đọc tiếp:** [READING_GUIDE.md](READING_GUIDE.md) phần 1-2 để chi tiết

---

### 📖 "Tôi muốn hiểu tất cả các file code"
👉 **Đọc theo thứ tự:**
1. [QUICK_START.md](QUICK_START.md) - Tier 1 (30 phút)
2. [READING_GUIDE.md](READING_GUIDE.md) - Chi tiết từng file
3. [ALGORITHM_FLOWCHART.md](ALGORITHM_FLOWCHART.md) - Sơ đồ visual
4. [FORMULAS.md](FORMULAS.md) - Công thức

**Dành:** 2-3 ngày, mix reading + running examples

---

### 🧮 "Tôi chỉ quan tâm công thức & thuật toán"
👉 **Đọc trực tiếp:** [FORMULAS.md](FORMULAS.md)
- Tất cả công thức tính cycles
- Hazard detection details
- Performance metrics

**Tham khảo:** [ALGORITHM_FLOWCHART.md](ALGORITHM_FLOWCHART.md) phần 5-7

---

### 🔍 "Tôi cần debug/modify code"
👉 **Bạn cần:**
1. [READING_GUIDE.md](READING_GUIDE.md) - Tìm line numbers
2. [ALGORITHM_FLOWCHART.md](ALGORITHM_FLOWCHART.md) - Luồng execution
3. [FORMULAS.md](FORMULAS.md) - Verify công thức

**Chạy test:** `python -m unittest discover -s tests -v`

---

### 💡 "Tôi muốn hiểu sâu Vector/Pipeline Simulator"
👉 **Focus:** [ALGORITHM_FLOWCHART.md](ALGORITHM_FLOWCHART.md)
- Phần 4: Vector non-pipeline flow
- Phần 5: Pipeline flow chi tiết
- Phần 6: Hazard detection

**Thực hành:** Chạy `python main.py` với các config khác nhau

---

---

## 📄 Chi tiết từng file

### [QUICK_START.md](QUICK_START.md) ⭐ BẮT ĐẦU ĐÂY
**Dành cho:** Ai muốn nhanh gọn  
**Độ dài:** ~400 dòng  
**Nội dung:**
- Tier 1-4 file (ordered by complexity)
- Công thức 8 thuật toán chính (dễ nhớ)
- Bảng so sánh scalar vs vector vs pipeline
- 4 bài tập (easy → advanced)
- Reference sheet

**Thời gian:** 30-60 phút đọc + 2 giờ thực hành

---

### [READING_GUIDE.md](READING_GUIDE.md) 📖 CHI TIẾT NHẤT
**Dành cho:** Ai muốn hiểu toàn bộ project  
**Độ dài:** ~1500 dòng  
**Nội dung:**
- 12 phần từ cơ bản → nâng cao
- Mỗi file: line numbers + giải thích
- Thuật toán chi tiết (pseudo code)
- 8 giải pháp vấn đề chính
- Thứ tự học tập (5 tuần)
- File reference table

**Thời gian:** 2-3 ngày (mix reading + hands-on)

---

### [ALGORITHM_FLOWCHART.md](ALGORITHM_FLOWCHART.md) 📊 HÌNH MINH HỌA
**Dành cho:** Ai muốn thấy flow rõ ràng  
**Độ dài:** ~1000 dòng  
**Nội dung:**
- 14 sơ đồ flowchart visual (ASCII art)
- Parser → Simulator logic chi tiết
- Hazard detection 3 loại (RAW, WAR, struct)
- Reduction tree diagram
- Gather/scatter pattern
- Timeline example (before/after)
- Comparison matrix

**Thời gian:** 45-90 phút đọc (visual learning)

---

### [FORMULAS.md](FORMULAS.md) 🧮 CÔNG THỨC
**Dành cho:** Ai muốn reference nhanh  
**Độ dài:** ~800 dòng  
**Nội dung:**
- 15 phần công thức
- Scalar/vector/pipeline cycles chi tiết
- Reduction speedup tính toán
- Hazard detection code
- Performance metrics (speedup, throughput, utilization)
- 3 config templates (small → supercomputer)
- Cheatsheet cuối cùng

**Thời gian:** 30-60 phút reference

---

---

## 🚀 Bước Đầu Tiên (5 Phút)

1. **Chạy một ví dụ đơn giản:**
   ```bash
   python main.py --program examples/vector_add.txt --mode all
   ```

2. **Xem output:**
   - Scalar cycles (baseline)
   - Vector cycles
   - Speedup
   - Throughput

3. **Đọc [QUICK_START.md](QUICK_START.md) Tier 1 (15 phút)**
   - Hiểu 3 simulator chính
   - Đọc config.py, instruction.py, parser.py, main.py

4. **Chạy ví dụ phức tạp hơn:**
   ```bash
   python main.py --program examples/saxpy.txt --mode all --export
   cat outputs/performance.csv
   cat outputs/report_saxpy.md
   ```

---

## 📊 Roadmap Học (Recommended)

### **Ngày 1 (2 giờ)**
- [ ] Đọc [QUICK_START.md](QUICK_START.md) Tier 1
- [ ] Chạy 3 ví dụ: vector_add, saxpy, dot_product
- [ ] Xem output timeline, performance.csv

### **Ngày 2 (2-3 giờ)**
- [ ] Đọc [QUICK_START.md](QUICK_START.md) Tier 2
- [ ] Đọc [READING_GUIDE.md](READING_GUIDE.md) Phần 4-5
- [ ] Chạy gather_scatter, convolution ví dụ
- [ ] Modify config.json, re-run

### **Ngày 3 (2-3 giờ)**
- [ ] Đọc [ALGORITHM_FLOWCHART.md](ALGORITHM_FLOWCHART.md) (visual)
- [ ] Đọc [FORMULAS.md](FORMULAS.md) Phần 1-6
- [ ] Chạy một vài test: `python -m unittest discover -s tests`
- [ ] Chạy Streamlit: `streamlit run app.py`

### **Ngày 4+ (nâng cao)**
- [ ] Đọc [READING_GUIDE.md](READING_GUIDE.md) Tier 3-4
- [ ] Đọc memory.py, cache.py, ooo_simulator.py
- [ ] Chạy thí nghiệm: `python run_experiments.py`
- [ ] Modify code & implement feature mới

---

## 🔑 Key Concepts Quick Review

| Concept | Công Thức | File |
|---------|-----------|------|
| **Scalar** | cycles = L × N | scalar_simulator.py |
| **Vector** | cycles = L + ceil(VL/lanes) - 1 | vector_simulator.py |
| **Pipeline** | cycles += hazard_delay | pipeline_simulator.py |
| **Reduction** | cycles = L × ceil(log2(N)) | vector_simulator.py |
| **Speedup** | scalar_cycles / vector_cycles | performance_analyzer.py |

👉 Chi tiết: [FORMULAS.md](FORMULAS.md) Phần 1-6

---

## ❓ FAQ & Troubleshoot

### "Code quá dài, bắt đầu từ đâu?"
**Trả lời:** [QUICK_START.md](QUICK_START.md) Tier 1
- Đọc 4 file chính: config.py, instruction.py, parser.py, main.py
- Rồi chạy ví dụ, hiểu output

### "Tôi không hiểu pipeline simulator"
**Trả lời:** [ALGORITHM_FLOWCHART.md](ALGORITHM_FLOWCHART.md) Phần 5 + [FORMULAS.md](FORMULAS.md) Phần 4
- Vẽ timeline manually
- Trace code line by line

### "Kết quả không match công thức?"
**Trả lời:** [FORMULAS.md](FORMULAS.md) Phần 2-4
- Kiểm tra assumptions (VL, lanes, BW)
- Enable/disable chaining, masking
- Add debug print trong simulator

### "Tôi muốn modify simulator"
**Trả lời:** [READING_GUIDE.md](READING_GUIDE.md) + [ALGORITHM_FLOWCHART.md](ALGORITHM_FLOWCHART.md)
- Tìm exact line numbers
- Hiểu logic before/after
- Thêm test case verify

---

## 📞 File Navigation

```
Project Root/
├── GUIDES_INDEX.md ...................... Bạn đang đọc đây!
├── QUICK_START.md ...................... Bắt đầu từ đây (30 phút)
├── READING_GUIDE.md .................... Chi tiết mọi file (2-3 ngày)
├── ALGORITHM_FLOWCHART.md .............. Sơ đồ visual (1-2 giờ)
├── FORMULAS.md ......................... Công thức (reference)
│
├── config.py ........................... ⭐ Tier 1 start
├── instruction.py ...................... ⭐ Tier 1 start
├── parser.py ........................... ⭐ Tier 1 start
├── main.py ............................. ⭐ Tier 1 start
│
├── scalar_simulator.py ................. Tier 2
├── vector_simulator.py ................. Tier 2 ⭐⭐
├── pipeline_simulator.py ............... Tier 2 ⭐⭐⭐
│
├── memory.py ........................... Tier 3
├── cache.py ............................ Tier 3
├── ooo_simulator.py .................... Tier 3
├── performance_analyzer.py ............. Tier 3
│
├── app.py .............................. Tier 4
├── run_experiments.py .................. Tier 4
├── tests/ .............................. Tier 4
│
├── examples/ ........................... Workload mẫu
│  ├── vector_add.txt ................... Easy
│  ├── saxpy.txt ........................ Medium
│  ├── dot_product.txt .................. Medium
│  ├── gather_scatter.txt ............... Hard
│  └── convolution_2d.txt ............... Hard+
│
└── README.md ........................... Official documentation
```

---

## ⏱️ Thời Gian Dự Kiến

| Activity | Time | Files |
|----------|------|-------|
| Read QUICK_START | 0.5h | QUICK_START.md |
| Run 3 examples | 0.5h | main.py, examples/ |
| Read READING_GUIDE Tier 1-2 | 2h | READING_GUIDE.md |
| Read ALGORITHM_FLOWCHART | 1h | ALGORITHM_FLOWCHART.md |
| Read FORMULAS | 1h | FORMULAS.md |
| Run experiments | 0.5h | run_experiments.py |
| Modify & debug | 2-4h | All files |
| **Total | 7-9h | All |

---

## 🎓 Next Steps After Reading

1. **Run all examples:**
   ```bash
   for f in examples/*.txt; do
     python main.py --program "$f" --mode all --export
   done
   ```

2. **Experiment with configs:**
   ```bash
   # Double vector length
   python main.py --program examples/saxpy.txt --vector-length 32
   
   # Double number of lanes
   python main.py --program examples/saxpy.txt --num-lanes 8
   ```

3. **Enable advanced features:**
   - Edit config.json: `enable_chaining: true`
   - Run cache simulation: `streamlit run app.py`
   - Try masking: `examples/mask_operation.txt`

4. **Write custom workload:**
   - Create `examples/my_workload.txt`
   - Test it: `python main.py --program examples/my_workload.txt --mode all --export`

5. **Modify simulator:**
   - Add feature to pipeline_simulator.py
   - Add test case to tests/
   - Run: `python -m unittest discover -s tests -v`

---

## ✅ Checklist - Hiểu Project

- [ ] Chạy được ví dụ cơ bản
- [ ] Hiểu 3 simulator (scalar, vector, pipeline)
- [ ] Biết công thức cycles chính
- [ ] Đọc được timeline output
- [ ] Hiểu hazard detection
- [ ] Chạy được thí nghiệm
- [ ] Chạy được test
- [ ] Sửa được code nhỏ
- [ ] Thêm được feature mới

---

## 📖 Chọn Hướng Dẫn Của Bạn

```
             ┌─────────────────────┐
             │  Bạn là ai?         │
             └──────────┬──────────┘
                        │
            ┌───────────┼───────────┐
            │           │           │
            ▼           ▼           ▼
       [Student]   [Developer]  [Researcher]
            │           │           │
            ▼           ▼           ▼
      QUICK_START  READING_GUIDE  FORMULAS
            ▼           ▼           ▼
      Examples      Modify Code   Theory
            ▼           ▼           ▼
      All Guides   Flowchart     Experiment
```

---

## 🎬 Bây giờ hãy bắt đầu!

### **Nhanh nhất (30 phút):**
```bash
# Read this
cat QUICK_START.md | head -200

# Run example
python main.py --program examples/vector_add.txt --mode all

# Understand output
cat outputs/performance.csv
```

### **Đầy đủ (2-3 giờ):**
```bash
# Read guides
cat QUICK_START.md
cat READING_GUIDE.md | head -500
cat ALGORITHM_FLOWCHART.md | head -500

# Run + analyze
python main.py --program examples/saxpy.txt --mode all --export
```

### **Sâu sắc (4+ giờ):**
```bash
# Read all
cat QUICK_START.md
cat READING_GUIDE.md
cat ALGORITHM_FLOWCHART.md
cat FORMULAS.md

# Experiment
python run_experiments.py
streamlit run app.py

# Code understanding
python -m unittest discover -s tests -v
```

---

**👉 Lựa chọn của bạn?** Hãy chọn một file ở trên và bắt đầu!

