import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from config import VectorProcessorConfig
from parser import ProgramParser
from scalar_simulator import ScalarSimulator
from vector_simulator import VectorSimulator
from pipeline_simulator import PipelineSimulator
from ooo_simulator import OOOSimulator
from performance_analyzer import PerformanceAnalyzer
from comparison import ArchitectureComparison

st.set_page_config(page_title="Vector Processor Simulator", layout="wide")

st.title("🖥️ Vector Processor Simulator and Performance Analyzer")
st.markdown("---")

# Sidebar - Cấu hình
st.sidebar.header("⚙️ Configuration")

vector_length = st.sidebar.slider("Vector Length", 4, 128, 8, step=4)
num_lanes = st.sidebar.slider("Number of Lanes", 1, 16, 4, step=1)
memory_bandwidth = st.sidebar.slider("Memory Bandwidth (elements/cycle)", 1, 16, 4, step=1)
clock_frequency = st.sidebar.number_input("Clock Frequency (MHz)", 500, 3000, 1000, step=100)
num_vector_registers = st.sidebar.slider("Vector Registers", 4, 32, 16, step=4)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📁 Program Selection")

# Danh sách workloads
workloads = {
    "Vector Add": "examples/vector_add.txt",
    "SAXPY": "examples/saxpy.txt",
    "Dot Product": "examples/dot_product.txt",
    "Matrix-Vector": "examples/matrix_vector.txt",
    "Stride Access": "examples/stride_access.txt",
    "Gather/Scatter": "examples/gather_scatter.txt",
    "Mask Operation": "examples/mask_operation.txt",
    "2D Convolution": "examples/convolution_2d.txt",
}

selected_workload = st.sidebar.selectbox("Workload", list(workloads.keys()))
program_file = workloads[selected_workload]
uploaded_program = st.sidebar.file_uploader("Upload vector program", type=["txt", "asm"])

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔧 Advanced Simulation")
enable_cache = st.sidebar.checkbox("Enable Cache Simulation")
cache_size = 32
cache_line_size = 8
cache_associativity = 4
cache_levels = 1
cache_policy = "lru"
if enable_cache:
    cache_size = st.sidebar.selectbox("Cache Lines", [8, 16, 32, 64], index=2)
    cache_line_size = st.sidebar.selectbox("Line Size (elements)", [4, 8, 16], index=1)
    cache_associativity = st.sidebar.selectbox("Associativity", [1, 2, 4, 8], index=2)
    cache_levels = st.sidebar.selectbox("Cache Levels", [1, 2], index=0)
    cache_policy = st.sidebar.selectbox("Replacement Policy", ["lru", "fifo", "random"])

enable_ooo = st.sidebar.checkbox("Enable Out-of-Order Execution")
ooo_rs_size = 8
if enable_ooo:
    ooo_rs_size = st.sidebar.slider("Reservation Stations", 4, 16, 8)

# Main content - hiển thị cấu hình
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Vector Length", f"{vector_length}")
with col2:
    st.metric("Number of Lanes", f"{num_lanes}")
with col3:
    st.metric("Memory Bandwidth", f"{memory_bandwidth} elem/cycle")

st.markdown("---")

# ✅ Tạo placeholder cho kết quả
result_placeholder = st.empty()

if st.button("🚀 Run Simulation", type="primary", use_container_width=True):
    with st.spinner("Running simulation..."):
        # Cấu hình
        config = VectorProcessorConfig(
            vector_length=vector_length,
            num_lanes=num_lanes,
            num_vector_registers=num_vector_registers,
            memory_bandwidth=memory_bandwidth,
            clock_frequency_mhz=clock_frequency
        )
        
        # Parse chương trình
        parser = ProgramParser()
        if uploaded_program is not None:
            program_text = uploaded_program.getvalue().decode("utf-8")
            data, scalars, instructions = parser.parse_text(
                program_text, uploaded_program.name
            )
        else:
            with open(program_file, "r", encoding="utf-8") as source:
                program_text = source.read()
            data, scalars, instructions = parser.parse_text(program_text, program_file)
        
        # Chạy scalar
        scalar_sim = ScalarSimulator(config)
        scalar_cycles = scalar_sim.estimate_cycles(instructions, data)
        
        # Chạy vector
        vector_sim = VectorSimulator(config, data, scalars)
        if enable_cache:
            vector_sim.memory.enable_cache_simulation(
                cache_size=cache_size, line_size=cache_line_size,
                associativity=cache_associativity, levels=cache_levels,
                policy=cache_policy,
            )
        memory, vector_timeline, vector_cycles = vector_sim.run(instructions)
        
        # Chạy pipeline
        pipeline_sim = PipelineSimulator(config, data, scalars)
        if enable_cache:
            pipeline_sim.memory.enable_cache_simulation(
                cache_size=cache_size, line_size=cache_line_size,
                associativity=cache_associativity, levels=cache_levels,
                policy=cache_policy,
            )
        memory2, pipeline_timeline, pipeline_cycles = pipeline_sim.run(instructions)

        ooo_cycles = None
        if enable_ooo:
            ooo_sim = OOOSimulator(config, data, scalars, num_rs=ooo_rs_size)
            _, _, ooo_cycles = ooo_sim.run(instructions)
        
        # ✅ Tất cả hiển thị đều nằm TRONG đây
        with result_placeholder.container():
            st.success("✅ Simulation completed!")
            
            # Metrics
            st.subheader("📊 Performance Metrics")
            metric_cols = st.columns(5 if enable_ooo else 4)
            with metric_cols[0]:
                st.metric("Scalar Cycles", f"{scalar_cycles:,}")
            with metric_cols[1]:
                st.metric("Vector Cycles", f"{vector_cycles:,}")
            with metric_cols[2]:
                st.metric("Pipeline Cycles", f"{pipeline_cycles:,}")
            with metric_cols[3]:
                compare_cycles = ooo_cycles if enable_ooo and ooo_cycles is not None else pipeline_cycles
                speedup = scalar_cycles / compare_cycles if compare_cycles else 0
                st.metric("Speedup", f"{speedup:.2f}x", delta=f"{speedup:.2f}x faster")
            if enable_ooo and ooo_cycles is not None:
                with metric_cols[4]:
                    st.metric("OoO Cycles", f"{ooo_cycles:,}")

            if enable_cache:
                cache_stats = pipeline_sim.memory.get_stats()
                st.info(
                    f"Cache: {cache_stats.get('cache_hits', 0)} hits, "
                    f"{cache_stats.get('cache_misses', 0)} misses "
                    f"({cache_stats.get('cache_hit_rate_percent', 0)}% hit rate)"
                )
                fig_cache, ax_cache = plt.subplots(figsize=(6, 3.5))
                ax_cache.bar(
                    ["Hits", "Misses"],
                    [cache_stats.get("cache_hits", 0), cache_stats.get("cache_misses", 0)],
                    color=["#2ecc71", "#e74c3c"],
                )
                ax_cache.set_title("Cache Hit/Miss")
                ax_cache.set_ylabel("Accesses")
                st.pyplot(fig_cache)
            
            # Biểu đồ speedup
            st.subheader("📈 Speedup Comparison")
            fig, ax = plt.subplots(figsize=(8, 5))
            modes = ['Vector', 'Pipeline']
            speedups = [scalar_cycles/vector_cycles, scalar_cycles/pipeline_cycles]
            colors = ['#3498db', '#e74c3c']
            bars = ax.bar(modes, speedups, color=colors, edgecolor='black')
            
            for bar, sp in zip(bars, speedups):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{sp:.2f}x', ha='center', va='bottom', fontsize=12, fontweight='bold')
            
            ax.set_ylabel('Speedup (vs Scalar)', fontsize=12)
            ax.set_title('Performance Speedup', fontsize=14, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig)
            
            # Biểu đồ cycles
            st.subheader("📊 Cycles Comparison")
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            modes2 = ['Scalar', 'Vector', 'Pipeline']
            cycles_data = [scalar_cycles, vector_cycles, pipeline_cycles]
            colors2 = ['#2ecc71', '#3498db', '#e74c3c']
            bars2 = ax2.bar(modes2, cycles_data, color=colors2, edgecolor='black')
            
            for bar, cyc in zip(bars2, cycles_data):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(cycles_data)*0.02,
                        f'{cyc:,}', ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            ax2.set_ylabel('Cycles', fontsize=12)
            ax2.set_title('Execution Cycles Comparison', fontsize=14, fontweight='bold')
            ax2.grid(axis='y', alpha=0.3)
            st.pyplot(fig2)

            num_elements = max((len(v) for v in data.values()), default=0)
            throughput = num_elements / pipeline_cycles if pipeline_cycles else 0.0
            batches = max(1, -(-num_elements // num_lanes))
            lane_util = num_elements / (num_lanes * batches) if num_elements else 0.0
            bandwidth_util = pipeline_sim.memory.get_bandwidth_utilization(pipeline_cycles)
            util_cols = st.columns(3)
            util_cols[0].metric("Throughput", f"{throughput:.3f} elem/cycle")
            util_cols[1].metric("Lane Utilization", f"{lane_util:.1%}")
            util_cols[2].metric("Bandwidth Utilization", f"{bandwidth_util:.1%}")
            
            # Timeline
            st.subheader("⏱️ Execution Timeline")
            df = pd.DataFrame(pipeline_timeline)
            st.dataframe(df, use_container_width=True)
            
            # ✅ Show program code (trong if block)
            with st.expander("📄 View Program Code"):
                st.code(program_text, language='text')

            st.subheader("Vector Register File")
            register_dump = pipeline_sim.vrf.dump_used()
            if register_dump:
                st.dataframe(pd.DataFrame.from_dict(register_dump, orient="index"))
            else:
                st.info("No vector registers were written.")
            
            # ✅ Download results (trong if block)
            st.subheader("💾 Download Results")
            csv_data = pd.DataFrame(pipeline_timeline).to_csv(index=False)
            st.download_button(
                label="📥 Download Timeline CSV",
                data=csv_data,
                file_name="timeline.csv",
                mime="text/csv"
            )
            
            # Kết quả tính toán
            st.subheader("📝 Result")
            result_arrays = {k: v for k, v in memory.items() if isinstance(v, list)}
            st.json(result_arrays)

# Phần tham khảo: không phải kết quả của lần chạy simulator hiện tại.
st.markdown("---")
with st.expander("Reference Analysis: architecture comparison and static efficiency notes", expanded=False):
    st.subheader("Architecture Comparison")
    st.caption(
        "Reference-only ISA summary for context. These values are not produced by the "
        "current simulation run."
    )

    arch_rows = {
        "Intel AVX-512": ArchitectureComparison.get_avx_comparison(),
        "ARM NEON": ArchitectureComparison.get_neon_comparison(),
        "RISC-V V": ArchitectureComparison.get_riscv_v_comparison(),
        "NVIDIA GPU": ArchitectureComparison.get_gpu_comparison(),
    }
    arch_df = pd.DataFrame(arch_rows).T
    st.dataframe(arch_df, use_container_width=True)

    st.markdown("#### Static Efficiency Analysis")
    st.caption(
        "This table uses the static/reference dataset in `ArchitectureComparison`, not "
        "the workload you just ran. For current-run efficiency, use the metrics and "
        "charts above."
    )

    efficiency = ArchitectureComparison.analyze_efficiency()
    eff_table = []
    for name, v in efficiency.items():
        eff_table.append({
            "Workload": name,
            "Measured Speedup": f"{v['measured_speedup']:.2f}x",
            "Ideal Speedup": f"{v['ideal_speedup']}x",
            "Efficiency (raw)": f"{v['efficiency']:.1%}",
            "Exceeds Ideal": "Yes" if v["exceeds_ideal"] else "No",
        })
    st.dataframe(pd.DataFrame(eff_table), use_container_width=True, hide_index=True)

    st.caption(
        "`efficiency_norm` is capped at 100% only for visualization; raw efficiency "
        "above remains uncapped."
    )
    for name, v in efficiency.items():
        suffix = " (measured exceeds ideal)" if v["exceeds_ideal"] else ""
        st.write(f"**{name}** - raw efficiency: {v['efficiency']:.1%}{suffix}")
        st.progress(min(v["efficiency_norm"], 1.0))

# Phần footer (ở ngoài if button)
st.markdown("---")
st.markdown("### 📚 Available Workloads")
st.markdown("""
- **Vector Add**: C = A + B
- **SAXPY**: Y = a*X + Y  
- **Dot Product**: s = sum(A[i] * B[i])
- **Matrix-Vector**: Y = A * X
- **Stride Access**: Load every Nth element
- **Gather/Scatter**: Indexed memory access
- **Mask Operation**: Conditional vector ops
- **2D Convolution**: Multiline data + kernel multiply
""")

st.caption("Vector Processor Simulator - Final Project")
