# Vector Processor Simulator and Performance Analyzer

Project mô phỏng bộ xử lý vector bằng Python, dùng để so sánh cách thực thi scalar, vector non-pipeline và vector pipeline. Project có giao diện dòng lệnh, giao diện web Streamlit, bộ thí nghiệm tự động, biểu đồ và test tự động.

## 1. Yêu cầu môi trường

- Python 3.10 trở lên
- pip
- Windows, Linux hoặc macOS

Kiểm tra phiên bản Python:

```bash
python --version
```

Nếu máy có nhiều phiên bản Python, có thể cần dùng:

```bash
py --version
```

## 2. Cài đặt project

Mở terminal tại thư mục gốc của project:

```bash
cd "e:\VECTOR PROCESSOR SIMULATOR AND PERFORMANCE ANALYZER"
```

Tạo môi trường ảo:

```bash
python -m venv venv
```

Kích hoạt môi trường ảo trên Windows PowerShell:

```bash
.\venv\Scripts\Activate.ps1
```

Kích hoạt môi trường ảo trên Windows CMD:

```bat
venv\Scripts\activate.bat
```

Kích hoạt môi trường ảo trên Linux/macOS:

```bash
source venv/bin/activate
```

Cài thư viện cần thiết:

```bash
pip install -r requirements.txt
```

Nếu không dùng được `requirements.txt`, có thể cài các thư viện chính:

```bash
pip install streamlit pandas numpy matplotlib plotly Jinja2
```

## 3. Chạy nhanh bằng CLI

Lệnh cơ bản:

```bash
python main.py --program examples/vector_add.txt
```

Chạy cả vector non-pipeline và pipeline:

```bash
python main.py --program examples/vector_add.txt --mode all
```

Chỉ chạy vector non-pipeline:

```bash
python main.py --program examples/vector_add.txt --mode vector
```

Chỉ chạy pipeline:

```bash
python main.py --program examples/vector_add.txt --mode pipeline
```

Chạy và xuất kết quả ra thư mục `outputs/`:

```bash
python main.py --program examples/vector_add.txt --mode all --export
```

Ví dụ với các workload khác:

```bash
python main.py --program examples/saxpy.txt --mode all --export
python main.py --program examples/dot_product.txt --mode pipeline
python main.py --program examples/gather_scatter.txt --mode all
python main.py --program examples/convolution_2d.txt --mode pipeline
```

## 4. Tùy chỉnh cấu hình khi chạy CLI

Project mặc định đọc cấu hình từ `config.json` nếu file này tồn tại.

Chạy với file cấu hình cụ thể:

```bash
python main.py --program examples/vector_add.txt --config config.json
```

Ghi đè một vài tham số trực tiếp từ dòng lệnh:

```bash
python main.py --program examples/vector_add.txt --vector-length 32 --num-lanes 8 --memory-bandwidth 4
```

Các tham số dòng lệnh quan trọng:

| Tham số | Ý nghĩa |
|---|---|
| `--program` | Đường dẫn file chương trình vector cần chạy |
| `--mode` | Chọn `vector`, `pipeline` hoặc `all` |
| `--export` | Xuất timeline/performance/report vào `outputs/` |
| `--config` | Đường dẫn file cấu hình JSON |
| `--vector-length` | Ghi đè độ dài vector |
| `--num-lanes` | Ghi đè số lane xử lý song song |
| `--memory-bandwidth` | Ghi đè băng thông bộ nhớ, tính theo element/cycle |

Xem trợ giúp CLI:

```bash
python main.py --help
```

## 5. Chạy giao diện web Streamlit

Khởi động app:

```bash
streamlit run app.py
```

Sau khi chạy, Streamlit sẽ mở trình duyệt hoặc hiển thị URL dạng:

```text
http://localhost:8501
```

Trong giao diện web, bạn có thể:

- Chọn workload mẫu như Vector Add, SAXPY, Dot Product, Matrix-Vector, Gather/Scatter, Mask Operation.
- Chỉnh `Vector Length`, `Number of Lanes`, `Memory Bandwidth`, `Clock Frequency`.
- Bật mô phỏng cache.
- Bật mô phỏng out-of-order execution.
- Upload file chương trình `.txt` hoặc `.asm`.
- Xem cycles, speedup, throughput, lane utilization, timeline, register file và kết quả memory.
- Tải timeline dạng CSV.

## 6. Chạy bộ thí nghiệm tự động

Chạy toàn bộ thí nghiệm:

```bash
python run_experiments.py
```

Kết quả được ghi vào:

```text
experiment_results/
```

Các file kết quả thường gặp:

```text
experiment_results/exp1_vector_length.csv
experiment_results/exp2_num_lanes.csv
experiment_results/exp3_memory_bandwidth.csv
experiment_results/exp4_workload_comparison.csv
experiment_results/exp5_stride_access.csv
experiment_results/exp6_cache_size.csv
experiment_results/exp7_mask_utilization.csv
experiment_results/figures/*.png
```

Ý nghĩa các thí nghiệm:

| Thí nghiệm | Nội dung |
|---|---|
| EXP1 | Ảnh hưởng của vector length |
| EXP2 | Ảnh hưởng của số lane |
| EXP3 | Ảnh hưởng của memory bandwidth |
| EXP4 | So sánh nhiều workload |
| EXP5 | Ảnh hưởng của stride access |
| EXP6 | Ảnh hưởng của cache size |
| EXP7 | Mask utilization |

## 7. Chạy test

Chạy bộ test nhanh:

```bash
python test_all.py
```

Chạy toàn bộ unit/integration test:

```bash
python -m unittest discover -s tests -v
```

Chạy một file test cụ thể:

```bash
python -m unittest tests.test_parser -v
python -m unittest tests.test_vector_register_file -v
python -m unittest tests.test_report_export -v
```

## 8. Cấu hình phần cứng

File cấu hình mặc định là `config.json`.

Ví dụ:

```json
{
  "vector_length": 16,
  "num_lanes": 4,
  "num_vector_registers": 16,
  "load_latency": 3,
  "store_latency": 3,
  "add_latency": 2,
  "mul_latency": 4,
  "div_latency": 8,
  "reduction_latency": 3,
  "startup_latency": 1,
  "memory_bandwidth": 4,
  "clock_frequency_mhz": 1000,
  "num_memory_units": 1,
  "num_alu_units": 1,
  "enable_chaining": true,
  "enable_masking": false
}
```

Ý nghĩa các tham số chính:

| Tham số | Ý nghĩa |
|---|---|
| `vector_length` | Số phần tử tối đa trong một vector |
| `num_lanes` | Số phần tử có thể xử lý song song mỗi cycle |
| `num_vector_registers` | Số thanh ghi vector |
| `load_latency`, `store_latency` | Độ trễ đọc/ghi bộ nhớ |
| `add_latency`, `mul_latency`, `div_latency` | Độ trễ các phép toán |
| `reduction_latency` | Độ trễ phép reduction |
| `startup_latency` | Độ trễ khởi động pipeline |
| `memory_bandwidth` | Số phần tử đọc/ghi được mỗi cycle |
| `clock_frequency_mhz` | Tần số clock để ước lượng thời gian |
| `num_memory_units` | Số đơn vị bộ nhớ trong mô hình pipeline |
| `num_alu_units` | Số đơn vị ALU |
| `enable_chaining` | Cho phép chaining giữa các phép toán |
| `enable_masking` | Cho phép mô phỏng masking |

## 9. Định dạng chương trình đầu vào

Các chương trình mẫu nằm trong thư mục `examples/`.

Ví dụ:

```text
DATA A = [1, 2, 3, 4]
DATA B = [10, 20, 30, 40]
SCALAR a = 2

VLOAD V1, A
VLOAD V2, B
VMULS V3, V1, a
VADD V4, V3, V2
VSTORE C, V4
```

Quy ước:

- `DATA <name> = [...]` khai báo mảng dữ liệu.
- `SCALAR <name> = <value>` khai báo hằng vô hướng.
- Comment bắt đầu bằng `#`.
- Lệnh vector viết mỗi lệnh một dòng.
- File chương trình thường có đuôi `.txt`.

## 10. Opcode được hỗ trợ

| Nhóm | Lệnh |
|---|---|
| Memory | `VLOAD`, `VSTORE`, `VLOAD_STRIDE`, `VGATHER`, `VSCATTER` |
| Arithmetic | `VADD`, `VSUB`, `VMUL`, `VDIV`, `VMAX`, `VMIN` |
| Scalar-vector | `VADDS`, `VSUBS`, `VMULS`, `VDIVS` |
| Reduction | `VREDUCE_SUM`, `VREDUCE_MAX`, `VREDUCE_MIN`, `VDOT` |
| Logic/Mask | `VAND`, `VOR`, `VXOR`, `VMASK`, `VMASK_COND`, `VADD_MASKED`, `VSUB_MASKED`, `VMOV_MASKED` |

## 11. Workload mẫu

| File | Mô tả |
|---|---|
| `examples/vector_add.txt` | C = A + B |
| `examples/saxpy.txt` | Y = aX + Y |
| `examples/dot_product.txt` | Tích vô hướng |
| `examples/matrix_vector.txt` | Nhân ma trận-vector |
| `examples/image_filter_1d.txt` | Bộ lọc 1D |
| `examples/gather_scatter.txt` | Truy xuất bộ nhớ không liên tục |
| `examples/stride_access.txt` | Truy xuất stride |
| `examples/mask_operation.txt` | Lệnh mask |
| `examples/logical_ops.txt` | AND/OR/XOR vector |
| `examples/convolution_1d.txt` | Convolution 1D |
| `examples/convolution_2d.txt` | Convolution 2D |
| `examples/conv2d_im2col.txt` | Convolution 2D theo im2col |
| `examples/reduction_max_min.txt` | Max/min reduction |
| `examples/complex_saxpy.txt` | Biến thể SAXPY mở rộng |

## 12. Kết quả đầu ra

Khi chạy CLI với `--export`, project tạo thư mục `outputs/` nếu chưa có.

Các output chính:

```text
outputs/timeline_vector.csv
outputs/timeline_pipeline.csv
outputs/performance.csv
outputs/report_<program>.md
```

Khi chạy thí nghiệm tự động, project ghi CSV và biểu đồ vào:

```text
experiment_results/
experiment_results/figures/
```

## 13. Cấu trúc thư mục

```text
app.py                    # Giao diện Streamlit
main.py                   # CLI chính
config.py                 # Dataclass cấu hình phần cứng
config.json               # Cấu hình mặc định
instruction.py            # Định nghĩa instruction/opcode
parser.py                 # Parser chương trình vector
memory.py                 # Mô hình bộ nhớ
vector_register_file.py   # Vector register file
scalar_simulator.py       # Scalar baseline
vector_simulator.py       # Vector non-pipeline simulator
pipeline_simulator.py     # Pipeline simulator
ooo_simulator.py          # Out-of-order simulator mở rộng
cache.py                  # Mô hình cache
memory_hierarchy.py       # Mô hình memory hierarchy
performance_analyzer.py   # Tính cycles, speedup, throughput, utilization
visualization.py          # Tạo biểu đồ
report_generator.py       # Xuất CSV và Markdown report
run_experiments.py        # Chạy các thí nghiệm tự động
examples/                 # Chương trình mẫu
tests/                    # Unit test và integration test
outputs/                  # Kết quả chạy CLI export
experiment_results/       # Kết quả thí nghiệm
docs/                     # Tài liệu báo cáo/slide/script nếu có
```

## 14. Luồng chạy gợi ý khi demo

1. Kích hoạt môi trường ảo.

```bash
.\venv\Scripts\Activate.ps1
```

2. Cài dependencies.

```bash
pip install -r requirements.txt
```

3. Chạy một workload đơn giản.

```bash
python main.py --program examples/vector_add.txt --mode all --export
```

4. Chạy workload có truy xuất bộ nhớ phức tạp hơn.

```bash
python main.py --program examples/gather_scatter.txt --mode all --export
```

5. Chạy giao diện web.

```bash
streamlit run app.py
```

6. Chạy test.

```bash
python -m unittest discover -s tests -v
```

## 15. Lỗi thường gặp

Nếu PowerShell không cho kích hoạt môi trường ảo:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

Nếu thiếu thư viện:

```bash
pip install -r requirements.txt
```

Nếu lệnh `python` không tồn tại trên Windows:

```bash
py main.py --program examples/vector_add.txt
```

Nếu Streamlit không chạy:

```bash
python -m streamlit run app.py
```

Nếu output tiếng Việt bị lỗi trên Windows terminal, đặt UTF-8:

```powershell
chcp 65001
```

## 16. Các file nên đọc khi phát triển

- `parser.py`: hiểu cú pháp chương trình đầu vào.
- `instruction.py`: xem danh sách opcode.
- `config.py`: xem các tham số phần cứng.
- `vector_simulator.py`: logic chạy vector non-pipeline.
- `pipeline_simulator.py`: logic pipeline và hazard.
- `performance_analyzer.py`: công thức speedup/throughput/utilization.
- `report_generator.py`: logic xuất timeline, performance và report.
- `tests/`: ví dụ cách kiểm thử từng module.
