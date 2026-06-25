import argparse
import os
from pathlib import Path
from config import VectorProcessorConfig
from parser import ProgramParser
from scalar_simulator import ScalarSimulator
from vector_simulator import VectorSimulator
from pipeline_simulator import PipelineSimulator
from performance_analyzer import PerformanceAnalyzer
from report_generator import ReportGenerator

def infer_num_elements(data, default_value):
    """Lấy số lượng phần tử từ dữ liệu đầu vào"""
    if not data:
        return default_value
    return len(next(iter(data.values())))

def _load_config(config_path=None, config_dict=None):
    """Đọc config từ JSON, file mặc định hoặc override dict."""
    path = config_path
    if path is None and os.path.exists("config.json"):
        path = "config.json"
    if path and os.path.exists(path):
        config = VectorProcessorConfig.load_from_file(path)
    else:
        config = VectorProcessorConfig()
    if config_dict:
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
    config.validate()
    return config


def run_program(program_path, mode="all", export=False, config_dict=None, config_path=None):
    """
    Chạy chương trình mô phỏng vector processor
    
    Args:
        program_path: Đường dẫn đến file chương trình .txt
        mode: "vector", "pipeline", hoặc "all"
        export: Có xuất file CSV hay không
        config_dict: Dictionary cấu hình (nếu muốn override)
    """
    config = _load_config(config_path, config_dict)
    
    # Parse chương trình
    parser = ProgramParser()
    data, scalars, instructions = parser.parse_file(program_path)
    
    # Chạy scalar simulator để có baseline
    scalar_sim = ScalarSimulator(config)
    scalar_cycles = scalar_sim.estimate_cycles(instructions, data)
    
    # Khởi tạo analyzer và report generator
    analyzer = PerformanceAnalyzer(config)
    reporter = ReportGenerator()
    perf_rows = []
    cycles = 0
    vector_result = {}
    pipeline_result = {}
    # In thông tin đầu vào
    print("=" * 60)
    print("INPUT DATA")
    print("=" * 60)
    print(f"Data arrays: {list(data.keys())}")
    print(f"Scalars: {scalars}")
    print("\nInstructions:")
    for i, inst in enumerate(instructions, 1):
        print(f"  {i}. {inst.raw_text}")
    print()
    
    # Chạy vector non-pipeline mode
    if mode in ["vector", "all"]:
        print("=" * 60)
        print("VECTOR NON-PIPELINE MODE")
        print("=" * 60)
        sim = VectorSimulator(config, data, scalars)
        memory, timeline, cycles = sim.run(instructions)
        
        print(f"\nResult memory: {memory}")
        reporter.print_timeline(timeline)
        
        num_elements = infer_num_elements(data, config.vector_length)
        row = analyzer.summarize("vector", scalar_cycles, cycles, num_elements)
        perf_rows.append(row)
        vector_result = {"cycles": cycles, "throughput": row.get("throughput_elem_per_cycle", 0), "timeline": timeline}
        
        if export:
            reporter.save_timeline(timeline, "timeline_vector.csv")
    
    # Chạy pipeline mode
    if mode in ["pipeline", "all"]:
        print("\n" + "=" * 60)
        print("VECTOR PIPELINE MODE")
        print("=" * 60)
        sim = PipelineSimulator(config, data, scalars)
        memory, timeline, cycles = sim.run(instructions)
        
        print(f"\nResult memory: {memory}")
        reporter.print_timeline(timeline)
        
        num_elements = infer_num_elements(data, config.vector_length)
        row = analyzer.summarize("pipeline", scalar_cycles, cycles, num_elements)
        perf_rows.append(row)
        pipeline_result = {"cycles": cycles, "throughput": row.get("throughput_elem_per_cycle", 0), "timeline": timeline}
        
        if export:
            reporter.save_timeline(timeline, "timeline_pipeline.csv")
    
    # In kết quả hiệu năng
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"\nScalar estimated cycles: {scalar_cycles}")
    print(f"Scalar execution time: {analyzer.execution_time_ns(scalar_cycles):.2f} ns")
    reporter.print_performance(perf_rows)
    
    # Xuất file CSV
    if export:
        reporter.save_performance(perf_rows)
        md = reporter.generate_markdown_report(
            program_name=Path(program_path).name,
            config=config.to_dict(),
            scalar_cycles=scalar_cycles,
            vector_result=vector_result,
            pipeline_result=pipeline_result,
        )
        report_file = reporter.output_dir / f"report_{Path(program_path).stem}.md"
        report_file.write_text(md, encoding="utf-8")
        print(f"\n✓ CSV and report saved to '{reporter.output_dir}/'")

    return cycles if perf_rows else 0

def main():
    parser = argparse.ArgumentParser(
        description="Vector Processor Simulator and Performance Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py --program examples/vector_add.txt
    python main.py --program examples/vector_add.txt --mode pipeline
    python main.py --program examples/saxpy.txt --mode all --export
    python main.py --program examples/dot_product.txt --mode vector
    python main.py --program examples/vector_add.txt --config config.json --export
        """
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to JSON config file (default: config.json if exists)"
    )
    parser.add_argument(
        "--program", 
        required=True, 
        help="Path to vector program file (e.g., examples/vector_add.txt)"
    )
    parser.add_argument(
        "--mode", 
        default="all", 
        choices=["vector", "pipeline", "all"],
        help="Simulation mode: vector (non-pipeline), pipeline, or all (default)"
    )
    parser.add_argument(
        "--export", 
        action="store_true",
        help="Export results to CSV files in outputs/ directory"
    )
    parser.add_argument(
        "--vector-length", 
        type=int,
        help="Override vector length in config"
    )
    parser.add_argument(
        "--num-lanes", 
        type=int,
        help="Override number of lanes in config"
    )
    parser.add_argument(
        "--memory-bandwidth",
        type=int,
        help="Override memory bandwidth (elements/cycle)"
    )
    
    args = parser.parse_args()
    
    config_dict = {}
    if args.vector_length:
        config_dict["vector_length"] = args.vector_length
    if args.num_lanes:
        config_dict["num_lanes"] = args.num_lanes
    if args.memory_bandwidth:
        config_dict["memory_bandwidth"] = args.memory_bandwidth
    
    run_program(args.program, args.mode, args.export, config_dict, args.config)

if __name__ == "__main__":
    import sys
    import io
    try:
        # Try to reconfigure stdout encoding for UTF-8 support
        reconfigure = getattr(sys.stdout, "reconfigure", None)
        stdout_buffer = getattr(sys.stdout, "buffer", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")
        elif stdout_buffer is not None:
            sys.stdout = io.TextIOWrapper(
                stdout_buffer, encoding="utf-8", errors="replace"
            )
    except (AttributeError, TypeError):
        # Fallback: continue with default encoding
        pass
    main()
