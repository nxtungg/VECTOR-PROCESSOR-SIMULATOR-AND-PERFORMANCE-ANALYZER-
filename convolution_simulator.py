"""Mô phỏng convolution cho AI/Image Processing.

KIẾN TRÚC THỐNG NHẤT — convolution KHÔNG còn tự tính cycles. Nó đóng vai trò
WORKLOAD / INSTRUCTION GENERATOR: sinh ra vector instruction stream rồi đưa vào
backend chung (VectorSimulator / ScalarSimulator) để lấy số chu kỳ. Nhờ đó mọi
con số nhất quán với các simulator khác và không bị double-count.

Ba lớp mô hình được TÁCH BẠCH, không trộn vào một công thức:

L1 — Analytical (roofline)  : ước lượng FLOPs/bytes/intensity. CHỈ tham khảo,
                                KHÔNG dùng để tính speedup. → _roofline_model()
L2 — Vector execution       : cycles THẬT từ backend (lanes + latency +
                                grouping + tree-reduction). → VectorSimulator /
                                ScalarSimulator (cùng instruction stream).
L3 — Memory/cache           : penalty cache RIÊNG, không đẩy ngược vào engine.
                                → _simulate_cache() qua MemoryHierarchy.

total_cycles = compute_cycles (L2) + memory_penalty (L3)

Lưu ý: im2col và tiling ở đây là LOGICAL TRANSFORMATION (cách map convolution
lên vector ALU / GEMM), KHÔNG phải mô phỏng bộ nhớ vật lý.
"""

import math
from typing import List, Dict, Any, Optional, Tuple

from config import VectorProcessorConfig
from instruction import Instruction, InstructionTemplates


class ConvolutionSimulator:
    """Workload generator + phân tích hiệu năng convolution trên vector processor.

    Không tự mô phỏng timing: execution luôn đi qua VectorSimulator (vector) và
    ScalarSimulator (baseline) làm backend duy nhất.
    """

    # Giới hạn kích thước ảnh khi phân tích để tránh sinh stream quá lớn
    # (mỗi output pixel = 3 lệnh; ảnh 64×64/3×3 ≈ 11k lệnh đã chạy backend thật).
    MAX_ANALYZE_IMAGE = 96

    # ==================================================================
    # ===== Functional reference (tính KẾT QUẢ, không phải timing) ======
    # ==================================================================

    @staticmethod
    def conv1d_vectorized(signal: List[float], kernel: List[float]) -> List[float]:
        """1D convolution (tham chiếu chức năng): Y[i] = sum_j X[i+j] * K[j]."""
        if not signal or not kernel:
            return []

        kernel_size = len(kernel)
        output_size = len(signal) - kernel_size + 1
        if output_size <= 0:
            return []

        result = []
        for i in range(output_size):
            conv_sum = 0
            for j in range(kernel_size):
                conv_sum += signal[i + j] * kernel[j]
            result.append(conv_sum)
        return result

    @staticmethod
    def conv2d_vectorized(image: List[List[float]],
                    kernel: List[List[float]]) -> List[List[float]]:
        """2D convolution (tham chiếu chức năng) — dùng để kiểm chứng kết quả."""
        if not image or not kernel:
            return []

        h, w = len(image), len(image[0])
        kh, kw = len(kernel), len(kernel[0])
        if h < kh or w < kw:
            return []

        oh, ow = h - kh + 1, w - kw + 1
        result = [[0.0] * ow for _ in range(oh)]
        for i in range(oh):
            for j in range(ow):
                conv_sum = 0.0
                for ki in range(kh):
                    for kj in range(kw):
                        conv_sum += image[i + ki][j + kj] * kernel[ki][kj]
                result[i][j] = conv_sum
        return result

    # ==================================================================
    # ===== L1 — Analytical model (roofline) — CHỈ THAM KHẢO ===========
    # ==================================================================

    @staticmethod
    def _roofline_model(image_size: int, kernel_size: int,
                        config: VectorProcessorConfig) -> Dict[str, Any]:
        """Ước lượng analytical theo roofline (KHÔNG dùng để tính speedup).

        Tách hẳn khỏi execution model: chỉ trả về các bound lý thuyết để so sánh
        và để bảo vệ "vùng compute-bound vs memory-bound" của workload.
        """
        oh = ow = image_size - kernel_size + 1
        num_pixels = oh * ow
        kernel_area = kernel_size ** 2

        # FLOPs = mỗi pixel: kernel_area phép nhân + (kernel_area-1) phép cộng
        flops = num_pixels * (2 * kernel_area - 1)
        # Bytes di chuyển (im2col logical): đọc patch + kernel + ghi output, 4 byte/phần tử
        elems_moved = num_pixels * kernel_area + kernel_area + num_pixels
        bytes_moved = elems_moved * 4

        arithmetic_intensity = flops / bytes_moved if bytes_moved else 0.0

        # Bound lý thuyết (đơn vị cycle, mô hình thô):
        #   compute-bound: 2*num_lanes FLOP/cycle (1 mul + 1 add mỗi lane)
        #   memory-bound : memory_bandwidth phần tử/cycle
        compute_bound_cycles = flops / (2 * config.num_lanes) if config.num_lanes else 0.0
        mem_bound_cycles = elems_moved / config.memory_bandwidth if config.memory_bandwidth else 0.0
        roofline_cycles = max(compute_bound_cycles, mem_bound_cycles)

        return {
            "note": "Analytical estimate (roofline) — tham khảo, không tính speedup",
            "flops": flops,
            "bytes_moved": bytes_moved,
            "arithmetic_intensity": round(arithmetic_intensity, 4),
            "compute_bound_cycles": round(compute_bound_cycles, 2),
            "mem_bound_cycles": round(mem_bound_cycles, 2),
            "roofline_cycles": round(roofline_cycles, 2),
            "bound": "compute" if compute_bound_cycles >= mem_bound_cycles else "memory",
        }

    # ==================================================================
    # ===== Workload generator — sinh vector instruction stream ========
    # ==================================================================

    @staticmethod
    def _build_im2col_workload(image_size: int, kernel_size: int
                            ) -> Tuple[Dict[str, List[float]], List[Instruction]]:
        """Sinh instruction stream im2col GEMM cho convolution (workload backend).

        Mapping: mỗi output pixel là một dot product giữa patch (đã im2col) và
        kernel đã flatten:
            VLOAD V0, KERNEL                (một lần)
            với mỗi patch:  VLOAD V1, PATCH → VMUL V2, V1, V0 → VREDUCE_SUM out, V2

        TIMING chỉ phụ thuộc ĐỘ DÀI vector (= kernel_area), không phụ thuộc giá
        trị phần tử. Do đó ta tái sử dụng MỘT buffer ``PATCH`` và một scalar
        ``out`` cho mọi pixel — số chu kỳ vẫn chính xác mà tránh tạo hàng nghìn
        mảng dữ liệu. (Kết quả số học không quan trọng ở đây; muốn kiểm chứng giá
        trị, dùng conv2d_im2col / conv2d_vectorized.)
        """
        kernel_area = kernel_size ** 2
        oh = ow = image_size - kernel_size + 1
        num_pixels = max(0, oh) * max(0, ow)

        # Dữ liệu tổng hợp (ramp) — chỉ cần đúng độ dài = kernel_area
        data: Dict[str, List[float]] = {
            "KERNEL": [float(i + 1) for i in range(kernel_area)],
            "PATCH": [float(i + 1) for i in range(kernel_area)],
        }

        instructions: List[Instruction] = [InstructionTemplates.vload("V0", "KERNEL")]
        for _ in range(num_pixels):
            instructions.append(InstructionTemplates.vload("V1", "PATCH"))
            instructions.append(InstructionTemplates.vmul("V2", "V1", "V0"))
            instructions.append(InstructionTemplates.vreduce_sum("out", "V2"))

        for idx, inst in enumerate(instructions, 1):
            inst.line_number = idx
        return data, instructions

    # ==================================================================
    # ===== L2 — Vector execution model (cycles THẬT từ backend) =======
    # ==================================================================

    @staticmethod
    def analyze_convolution_performance(image_size: int, kernel_size: int,
                                        config: VectorProcessorConfig) -> Dict[str, Any]:
        """Phân tích hiệu năng convolution — cycles lấy TỪ BACKEND, không tự tính.

        - vector_cycles: chạy instruction stream qua VectorSimulator (L2).
        - scalar_cycles: chạy CÙNG stream qua ScalarSimulator (cùng abstraction
        → speedup trung thực, không ảo).
        - roofline: ước lượng analytical riêng (L1), chỉ tham khảo.
        """
        if kernel_size > image_size:
            return {
                "error": "Kernel size must be <= image size",
                "image_size": image_size,
                "kernel_size": kernel_size,
            }

        if image_size > ConvolutionSimulator.MAX_ANALYZE_IMAGE:
            image_size = ConvolutionSimulator.MAX_ANALYZE_IMAGE

        oh = ow = image_size - kernel_size + 1
        num_pixels = oh * ow
        kernel_area = kernel_size ** 2
        ops_per_pixel = kernel_area
        total_mul_ops = num_pixels * ops_per_pixel
        total_add_ops = num_pixels * (ops_per_pixel - 1)
        total_ops = total_mul_ops + total_add_ops

        # ----- L2: sinh workload + chạy backend -----
        from vector_simulator import VectorSimulator
        from scalar_simulator import ScalarSimulator

        data, instructions = ConvolutionSimulator._build_im2col_workload(
            image_size, kernel_size)

        # Vector backend = nguồn sự thật duy nhất cho vector_cycles
        vec_sim = VectorSimulator(config, dict(data))
        _, _, vector_cycles = vec_sim.run(instructions)

        # Scalar baseline = cùng instruction stream, cùng cấp abstraction
        scalar_sim = ScalarSimulator(config)
        scalar_cycles = scalar_sim.estimate_cycles(instructions, data)

        speedup = scalar_cycles / vector_cycles if vector_cycles > 0 else 0
        efficiency = (speedup / config.num_lanes) if vector_cycles > 0 else 0

        return {
            "image_size": image_size,
            "kernel_size": kernel_size,
            "num_pixels": num_pixels,
            "ops_per_pixel": ops_per_pixel,
            "total_mul_ops": total_mul_ops,
            "total_add_ops": total_add_ops,
            "total_ops": total_ops,
            "num_elements": num_pixels,
            "num_instructions": len(instructions),
            "scalar_cycles": scalar_cycles,
            "vector_cycles": vector_cycles,
            "speedup": speedup,
            "efficiency": min(efficiency, 1.0),
            # L1 — analytical, tách riêng, không ảnh hưởng speedup
            "roofline": ConvolutionSimulator._roofline_model(
                image_size, kernel_size, config),
        }

    @staticmethod
    def compare_with_cnn_layers(image_size: int,
                                config: VectorProcessorConfig) -> Dict[str, Any]:
        """So sánh hiệu năng với các CNN layer phổ biến (3×3, 5×5, 7×7).

        Lưu ý kích thước: mỗi layer chạy backend thật nên image_size lớn sẽ sinh
        stream rất lớn — dùng kích thước vừa phải (xem MAX_ANALYZE_IMAGE).
        """
        kernel_sizes = [3, 5, 7]
        results = {}
        for k in kernel_sizes:
            if k <= image_size:
                results[f"Conv{k}x{k}"] = \
                    ConvolutionSimulator.analyze_convolution_performance(
                        image_size, k, config)
        return results

    # ==================================================================
    # ===== Logical transformation: im2col, tiling =====================
    # ==================================================================

    @staticmethod
    def im2col(image: List[List[float]], kh: int, kw: int,
            stride: int = 1, padding: int = 0):
        """Biến đổi im2col: trải mỗi patch của ảnh thành một hàng.

        ĐÂY LÀ LOGICAL TRANSFORMATION (cách map conv → GEMM), KHÔNG phải mô phỏng
        bộ nhớ vật lý. Là kỹ thuật lõi để chuyển convolution thành phép nhân ma
        trận (như cuDNN, Caffe).

        Returns:
            (cols, oh, ow): cols là ma trận (n_patches × kh*kw), mỗi hàng là một
            patch đã flatten row-major; oh, ow là kích thước output.
        """
        if not image or not image[0]:
            return [], 0, 0

        h, w = len(image), len(image[0])
        if padding > 0:
            padded = [[0.0] * (w + 2 * padding) for _ in range(h + 2 * padding)]
            for i in range(h):
                for j in range(w):
                    padded[i + padding][j + padding] = image[i][j]
            image = padded
            h, w = len(image), len(image[0])

        oh = (h - kh) // stride + 1
        ow = (w - kw) // stride + 1
        if oh <= 0 or ow <= 0:
            return [], 0, 0

        cols: List[List[float]] = []
        for i in range(oh):
            for j in range(ow):
                patch: List[float] = []
                base_i, base_j = i * stride, j * stride
                for ki in range(kh):
                    row = image[base_i + ki]
                    for kj in range(kw):
                        patch.append(row[base_j + kj])
                cols.append(patch)
        return cols, oh, ow

    @staticmethod
    def conv2d_im2col(image: List[List[float]],
                    kernel: List[List[float]]) -> List[List[float]]:
        """2D convolution qua im2col + GEMM (tham chiếu chức năng).

        Cho kết quả bằng conv2d_vectorized nhưng theo cách map lên vector
        processor: mỗi output là dot product giữa hàng patch và kernel flatten
        (hợp với VMUL + VREDUCE_SUM).
        """
        if not image or not kernel:
            return []
        kh, kw = len(kernel), len(kernel[0])
        cols, oh, ow = ConvolutionSimulator.im2col(image, kh, kw)
        if not cols:
            return []
        kflat = [kernel[a][b] for a in range(kh) for b in range(kw)]
        flat = [sum(p * k for p, k in zip(patch, kflat)) for patch in cols]
        return [flat[r * ow:(r + 1) * ow] for r in range(oh)]

    @staticmethod
    def conv2d_tiled(image: List[List[float]], kernel: List[List[float]],
                    tile_size: int = 2) -> List[List[float]]:
        """2D convolution theo tiling (tham chiếu chức năng).

        Tiling là logical transformation cải thiện locality (tái sử dụng dữ liệu
        ảnh trong cache). Kết quả bằng conv2d_vectorized.
        """
        if not image or not kernel:
            return []
        h, w = len(image), len(image[0])
        kh, kw = len(kernel), len(kernel[0])
        if h < kh or w < kw:
            return []
        oh, ow = h - kh + 1, w - kw + 1
        result = [[0.0] * ow for _ in range(oh)]
        t = max(1, tile_size)
        for ti in range(0, oh, t):
            for tj in range(0, ow, t):
                for i in range(ti, min(ti + t, oh)):
                    for j in range(tj, min(tj + t, ow)):
                        s = 0.0
                        for ki in range(kh):
                            row = image[i + ki]
                            for kj in range(kw):
                                s += row[j + kj] * kernel[ki][kj]
                        result[i][j] = s
        return result

    # ==================================================================
    # ===== L3 — Memory/cache model (penalty RIÊNG, tách compute) ======
    # ==================================================================

    @staticmethod
    def _access_addresses(image_size: int, kernel_size: int,
                        strategy: str, tile_size: int = 4):
        """Sinh chuỗi địa chỉ phần tử ảnh được truy xuất theo từng chiến lược.

        Địa chỉ phẳng row-major: addr = r * image_size + c. Chỉ cần MẪU truy xuất
        (không dựng ảnh thật) để mô phỏng locality cho lớp cache.
        """
        W = image_size
        oh = ow = image_size - kernel_size + 1
        addrs: List[int] = []
        if strategy == "tiled":
            t = max(1, tile_size)
            for ti in range(0, oh, t):
                for tj in range(0, ow, t):
                    for i in range(ti, min(ti + t, oh)):
                        for j in range(tj, min(tj + t, ow)):
                            for ki in range(kernel_size):
                                for kj in range(kernel_size):
                                    addrs.append((i + ki) * W + (j + kj))
        elif strategy == "im2col":
            # Phase 1 — lowering: đọc ảnh (mẫu naive) để dựng ma trận col.
            for i in range(oh):
                for j in range(ow):
                    for ki in range(kernel_size):
                        for kj in range(kernel_size):
                            addrs.append((i + ki) * W + (j + kj))
            # Phase 2 — GEMM: đọc ma trận col TUẦN TỰ (vùng địa chỉ riêng) →
            # locality cao, chỉ compulsory miss.
            col_base = image_size * image_size + W
            n = oh * ow * kernel_size * kernel_size
            for off in range(n):
                addrs.append(col_base + off)
        else:  # "naive"
            for i in range(oh):
                for j in range(ow):
                    for ki in range(kernel_size):
                        for kj in range(kernel_size):
                            addrs.append((i + ki) * W + (j + kj))
        return addrs

    @staticmethod
    def _simulate_cache(addresses, cache_lines: int = 64, line_size: int = 8):
        """Chạy chuỗi địa chỉ qua một L1 cache, trả (hit_rate, penalty_cycles)."""
        from memory_hierarchy import MemoryHierarchy, LevelConfig
        bytes_per_elem = 4
        lines = max(1, cache_lines)
        assoc = 4 if lines % 4 == 0 else (2 if lines % 2 == 0 else 1)
        l1 = LevelConfig(size_bytes=lines * line_size * bytes_per_elem,
                         line_size_bytes=line_size * bytes_per_elem,
                        associativity=assoc, penalty=0)
        h = MemoryHierarchy(l1=l1, mem_penalty=10)
        for a in addresses:
            h.access(a, is_write=False)
        total = h.l1_hits + h.l1_misses
        hit_rate = h.l1_hits / total if total else 0.0
        return hit_rate, h.total_penalty

    @staticmethod
    def compare_conv_strategies(image_size: int, kernel_size: int,
                                config: VectorProcessorConfig,
                                tile_size: int = 4,
                                cache_lines: int = 64,
                                line_size: int = 8) -> Dict[str, Any]:
        """So sánh 3 chiến lược conv: naive, im2col, tiled.

        TÁCH BẠCH compute (L2, từ backend) và memory penalty (L3, từ cache):
            total_cycles = compute_cycles + memory_penalty
        Cache là lớp phân tích phụ, KHÔNG ảnh hưởng execution engine.
        """
        if kernel_size > image_size:
            return {"error": "Kernel size must be <= image size"}

        oh = ow = image_size - kernel_size + 1
        num_pixels = oh * ow
        kernel_area = kernel_size ** 2

        # L2 — compute cycles lấy TỪ BACKEND (một nguồn sự thật), không tự tính
        base = ConvolutionSimulator.analyze_convolution_performance(
            image_size, kernel_size, config)
        compute_cycles = base["vector_cycles"]

        def build(strategy: str):
            # L3 — cache penalty riêng
            addrs = ConvolutionSimulator._access_addresses(
                image_size, kernel_size, strategy, tile_size)
            hit_rate, penalty = ConvolutionSimulator._simulate_cache(
                addrs, cache_lines, line_size)
            total_cycles = compute_cycles + penalty
            return {
                "strategy": strategy,
                "num_pixels": num_pixels,
                "kernel_area": kernel_area,
                "memory_accesses": len(addrs),
                "cache_hit_rate": round(hit_rate, 4),
                "cache_penalty_cycles": penalty,
                "memory_penalty": penalty,       # alias theo thuật ngữ đề bài
                "compute_cycles": compute_cycles,  # L2
                "total_cycles": total_cycles,      # L2 + L3
            }

        naive = build("naive")
        im2col = build("im2col")
        im2col["note"] = "Lowering tốn thêm bộ nhớ; GEMM đọc tuần tự, locality cao"
        tiled = build("tiled")
        tiled["tile_size"] = tile_size

        naive_total = naive["total_cycles"] or 1
        for s in (naive, im2col, tiled):
            s["speedup_vs_naive"] = round(naive_total / s["total_cycles"], 3) \
                if s["total_cycles"] else 0

        return {
            "image_size": image_size,
            "kernel_size": kernel_size,
            "compute_cycles": compute_cycles,
            "naive": naive,
            "im2col": im2col,
            "tiled": tiled,
        }

    # ==================================================================
    # ===== Sinh chương trình assembly minh họa (cho học/teaching) =====
    # ==================================================================

    @staticmethod
    def generate_vectorized_kernel_program(image: List[List[float]],
                                        kernel: List[List[float]]) -> str:
        """Sinh chương trình assembly vector thực thi conv qua im2col + GEMM.

        Mỗi output pixel = VMUL(patch, kernel) rồi VREDUCE_SUM. Chương trình sinh
        ra chạy được trực tiếp qua ProgramParser + các simulator (backend chung).
        """
        kh, kw = len(kernel), len(kernel[0])
        cols, oh, ow = ConvolutionSimulator.im2col(image, kh, kw)
        kflat = [kernel[a][b] for a in range(kh) for b in range(kw)]
        lines: List[str] = [
            f"# Vectorized convolution (im2col + GEMM)",
            f"# Output {oh}x{ow}, kernel {kh}x{kw} flatten = {len(kflat)} phần tử",
            f"DATA KERNEL = {kflat}",
        ]
        out_values: List[float] = []
        # Sinh lệnh cho tối đa vài patch đầu (minh hoạ mapping), tính đủ output.
        demo = min(len(cols), 8)
        for p in range(demo):
            lines.append(f"DATA PATCH{p} = {cols[p]}")
        lines.append("")
        lines.append("VLOAD V0, KERNEL")
        for p in range(demo):
            lines.append(f"VLOAD V1, PATCH{p}")
            lines.append(f"VMUL V2, V1, V0")
            lines.append(f"VREDUCE_SUM out{p}, V2")
        for patch in cols:
            out_values.append(sum(x * k for x, k in zip(patch, kflat)))
        lines.append("")
        lines.append(f"# Kết quả mong đợi (flatten): {out_values}")
        return "\n".join(lines)


# ===== TEST =====
if __name__ == "__main__":
    print("=" * 60)
    print("TEST CONVOLUTION SIMULATOR (kiến trúc thống nhất)")
    print("=" * 60)

    config = VectorProcessorConfig(
        vector_length=64,
        num_lanes=4,
        memory_bandwidth=8,
        add_latency=2,
        mul_latency=4,
        startup_latency=1
    )

    # Functional reference (kiểm chứng kết quả)
    print("\n--- 1D Convolution (functional) ---")
    signal: List[float] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    kernel: List[float] = [1, 2, 1]
    print(f"Result: {ConvolutionSimulator.conv1d_vectorized(signal, kernel)}")

    print("\n--- 2D Convolution (functional) ---")
    image: List[List[float]] = [
        [1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]
    ]
    kernel_2d: List[List[float]] = [[1, 0, -1], [1, 0, -1], [1, 0, -1]]
    result_2d = ConvolutionSimulator.conv2d_vectorized(image, kernel_2d)
    print(f"Result shape: {len(result_2d)}x{len(result_2d[0]) if result_2d else 0}")

    # L2 — Performance qua backend thật
    print("\n--- Performance Analysis (backend-driven, L2) ---")
    r = ConvolutionSimulator.analyze_convolution_performance(64, 3, config)
    print(f"Image 64x64, Kernel 3x3:")
    print(f"  Instructions:  {r['num_instructions']:,}")
    print(f"  Total ops:     {r['total_ops']:,}")
    print(f"  Scalar cycles: {r['scalar_cycles']:,}")
    print(f"  Vector cycles: {r['vector_cycles']:,}")
    print(f"  Speedup:       {r['speedup']:.2f}x")
    print(f"  Efficiency:    {r['efficiency']:.2%}")

    # L1 — Roofline (tham khảo, tách riêng)
    print("\n--- Roofline (analytical, L1 — tham khảo) ---")
    rf = r["roofline"]
    print(f"  Arithmetic intensity: {rf['arithmetic_intensity']} FLOP/byte")
    print(f"  Bound: {rf['bound']}  (compute={rf['compute_bound_cycles']}, "
        f"mem={rf['mem_bound_cycles']})")

    # L3 — Cache penalty tách khỏi compute
    print("\n--- Strategy comparison (L2 compute + L3 cache) ---")
    cmp = ConvolutionSimulator.compare_conv_strategies(16, 3, config,
                                                    tile_size=4, cache_lines=8)
    for name in ("naive", "im2col", "tiled"):
        s = cmp[name]
        print(f"  {name:7}: compute={s['compute_cycles']:>5}  "
            f"penalty={s['memory_penalty']:>5}  total={s['total_cycles']:>5}  "
            f"hit={s['cache_hit_rate']:.2%}  speedup={s['speedup_vs_naive']}x")

    print("\n--- CNN Layer Comparison (Image 32x32) ---")
    for layer, m in ConvolutionSimulator.compare_with_cnn_layers(32, config).items():
        print(f"  {layer}: speedup={m['speedup']:.2f}x  vector_cycles={m['vector_cycles']:,}")

    print("\n✅ All tests passed!")
