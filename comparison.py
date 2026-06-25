"""So sánh vector processor với các kiến trúc thực tế.

LỚP PHÂN TÍCH HẬU XỬ LÝ (post-processing analysis), KHÔNG phải simulator:
- Module này chỉ DIỄN GIẢI và SO SÁNH kiến trúc dựa trên tài liệu/đặc tả của các
ISA thực tế (AVX-512, NEON, RISC-V V, NVIDIA GPU) và trên kết quả speedup đã đo
từ nơi khác. Nó KHÔNG tự chạy mô phỏng và KHÔNG sinh dữ liệu benchmark runtime.
- Các con số "đặc điểm kiến trúc" là thông tin tham chiếu (reference), dùng để
định khung khi viết báo cáo, không phải số đo từ simulator.

DIỄN GIẢI HIỆU NĂNG (thống nhất để tránh mâu thuẫn khi viết report):
- "ideal speedup"    = giới hạn lý thuyết của SIMD/vector = số lanes (mỗi lane xử
                    lý 1 phần tử/chu kỳ một cách lý tưởng).
- "measured speedup" = scalar_cycles / vector_cycles đo được từ simulator.
- "efficiency (raw)" = measured / ideal. KHÔNG cắt trần ở 1.0 — giữ nguyên để
                    không mất thông tin khi measured > ideal.
- "efficiency_norm"  = min(raw, 1.0), CHỈ phục vụ trực quan hoá (thanh % ≤ 100%);
                    không dùng để kết luận khoa học.

VÌ SAO measured speedup CÓ THỂ VƯỢT ideal (raw efficiency > 100%)?
- Không vi phạm lý thuyết SIMD. Baseline scalar gánh thêm chi phí mà phiên bản
vector tiết kiệm/che giấu được:
    • overlap giữa compute và memory (load/store chồng lấn với tính toán),
    • pipeline / startup được khấu hao trên cả vector thay vì lặp lại mỗi phần tử,
    • giảm overhead vòng lặp & điều khiển so với scalar.
- Đây là hiệu ứng hệ thống (compute–memory overlap, pipeline effect), nên nêu rõ
trong báo cáo để tránh hiểu nhầm là "nhanh hơn số lanes là sai".
"""


class ArchitectureComparison:
    """So sánh các kiến trúc SIMD/Vector (phân tích tham chiếu, không phải đo đạc).

    | Đặc điểm | Vector Simulator | Intel AVX-512 | ARM NEON | RISC-V V | NVIDIA GPU |
    |----------|------------------|---------------|----------|----------|------------|
    | Vector Length | 8-64 (config) | 16 (512-bit) | 4 (128-bit) | 8-1024 | 32 (warp) |
    | Lanes | 1-16 | 16 | 4 | 8-32 | 32 |
    | Masking | ✅ | ✅ (k掩码) | ❌ | ✅ | ✅ (predicate) |
    | Gather/Scatter | ✅ | ✅ | ❌ | ✅ | ✅ |
    | Reduction | ✅ | ✅ | ❌ | ✅ | ✅ |
    | Chaining | ✅ | ✅ (register renaming) | ❌ | ✅ | ✅ (warp) |
    """

    @staticmethod
    def get_avx_comparison():
        """So sánh với Intel AVX-512 (thông tin tham chiếu từ đặc tả ISA)."""
        return {
            "vector_length": "512-bit = 16 floats",
            "lanes": 16,
            "instructions": "VADDPS, VMULPS, VGATHERDPS",
            "masking": "k0-k7 mask registers",
            "applications": "HPC, AI inference, video encoding"
        }

    @staticmethod
    def get_neon_comparison():
        """So sánh với ARM NEON (thông tin tham chiếu từ đặc tả ISA)."""
        return {
            "vector_length": "128-bit = 4 floats",
            "lanes": 4,
            "instructions": "VADD.F32, VMUL.F32",
            "masking": "Không hỗ trợ",
            "applications": "Mobile, embedded, IoT"
        }

    @staticmethod
    def get_riscv_v_comparison():
        """So sánh với RISC-V Vector Extension (thông tin tham chiếu từ đặc tả ISA)."""
        return {
            "vector_length": "Tùy chỉnh (VLEN)",
            "lanes": "Tối đa 32 (ELEN)",
            "instructions": "vadd.vv, vmul.vx, vcompress.vm",
            "masking": "v0.t mask register",
            "features": "Segment loads/stores, unit-stride, indexed"
        }

    @staticmethod
    def get_gpu_comparison():
        """So sánh với NVIDIA GPU CUDA (thông tin tham chiếu từ kiến trúc warp)."""
        return {
            "warp_size": 32,  # lanes
            "threads": "32 threads cùng lệnh",
            "masking": "Predicate registers",
            "memory": "Shared memory, global memory",
            "gather_scatter": "Via PTX instructions"
        }

    @staticmethod
    def analyze_efficiency():
        """Phân tích hiệu suất vector so với giới hạn lý thuyết (hậu xử lý).

        Đây là phân tích trên các speedup ĐÃ ĐO (không chạy lại simulator). Mỗi
        workload trả về:
        - measured_speedup : scalar_cycles / vector_cycles (đã đo).
        - ideal_speedup    : giới hạn lý thuyết = số lanes.
        - efficiency       : raw = measured / ideal (KHÔNG cắt trần — giữ thông
                            tin khi measured > ideal).
        - efficiency_norm  : min(raw, 1.0), CHỈ để trực quan hoá (% ≤ 100%).
        - exceeds_ideal    : True nếu measured > ideal (overlap/pipeline effect,
                            KHÔNG phải vi phạm lý thuyết SIMD — xem docstring module).

        Lưu ý nhất quán: "ideal" ở đây là số lanes; nếu báo cáo dùng định nghĩa
        ideal khác (vd có tính memory bound) thì phải nêu rõ để khỏi mâu thuẫn.
        """
        # Speedup đã đo từ thí nghiệm (input cho phân tích, không phải đo tại đây).
        raw = {
            "vector_add": {"speedup": 4.89, "ideal": 4},
            "saxpy": {"speedup": 5.71, "ideal": 4},
            "dot_product": {"speedup": 4.33, "ideal": 4},
            "matrix_vector": {"speedup": 5.50, "ideal": 4},
        }

        results = {}
        for name, v in raw.items():
            measured = v["speedup"]
            ideal = v["ideal"]
            raw_eff = measured / ideal if ideal else 0.0
            results[name] = {
                "measured_speedup": measured,
                "ideal_speedup": ideal,
                # Tương thích ngược: giữ key "speedup" như trước.
                "speedup": measured,
                # Raw efficiency: KHÔNG cap ở 1.0 (giữ trọn thông tin).
                "efficiency": raw_eff,
                # Chỉ dùng cho visualization (thanh %), không kết luận khoa học.
                "efficiency_norm": min(raw_eff, 1.0),
                "exceeds_ideal": measured > ideal,
            }

        return results
