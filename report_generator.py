import csv
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass, field


@dataclass
class ReportConfig:
    """Cấu hình cho báo cáo"""
    output_dir: str = "outputs"
    report_title: str = "Vector Processor Simulation Report"
    include_timeline: bool = True
    include_performance: bool = True
    include_memory_stats: bool = True
    include_register_stats: bool = True
    format_markdown: bool = True
    format_csv: bool = True


class ReportGenerator:
    """
    Tạo báo cáo cho vector processor simulator
    
    Hỗ trợ:
    - Xuất báo cáo dạng CSV, Markdown
    - Tạo timeline chi tiết
    - Tổng hợp hiệu năng
    - Tạo bảng so sánh
    - Xuất dữ liệu cho biểu đồ
    """
    
    def __init__(self, output_dir: str = "outputs", config: Optional[ReportConfig] = None):
        """
        Khởi tạo report generator
        
        Args:
            output_dir: Thư mục đầu ra
            config: Cấu hình báo cáo
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.config = config or ReportConfig(output_dir=output_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # ===== CSV Export Methods =====
    
    def save_timeline(self, timeline: List[Dict], filename: str = "timeline.csv") -> Optional[Path]:
        """
        Lưu timeline ra file CSV
        
        Args:
            timeline: Danh sách các sự kiện timeline
            filename: Tên file output
        
        Returns:
            Path: Đường dẫn file đã lưu
        """
        if not timeline:
            return None
        
        filepath = self.output_dir / filename
        
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            # Lấy tất cả keys từ timeline
            fieldnames = list(timeline[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(timeline)
        
        print(f"✓ Timeline saved to {filepath}")
        return filepath
    
    def save_performance(self, rows: List[Dict], filename: str = "performance.csv") -> Optional[Path]:
        """
        Lưu performance data ra file CSV
        
        Args:
            rows: Danh sách các dòng performance
            filename: Tên file output
        
        Returns:
            Path: Đường dẫn file đã lưu
        """
        if not rows:
            return None
        
        filepath = self.output_dir / filename
        
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            fieldnames = list(rows[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"✓ Performance data saved to {filepath}")
        return filepath
    
    # ===== Markdown Report Methods =====

    
    def generate_markdown_report(self, 
                                program_name: str,
                                config: Dict,
                                scalar_cycles: int,
                                vector_result: Dict,
                                pipeline_result: Dict,
                                analysis: Optional[Dict] = None) -> str:
        """
        Tạo báo cáo Markdown
        
        Args:
            program_name: Tên chương trình
            config: Cấu hình hệ thống
            scalar_cycles: Số chu kỳ scalar
            vector_result: Kết quả vector simulation
            pipeline_result: Kết quả pipeline simulation
            analysis: Phân tích hiệu năng
        
        Returns:
            str: Nội dung Markdown
        """
        lines = []
        
        # Header
        lines.append(f"# {self.config.report_title}")
        lines.append(f"\n**Program:** `{program_name}`")
        lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("\n---\n")
        
        # System Configuration
        lines.append("## 1. System Configuration\n")
        lines.append("| Parameter | Value |")
        lines.append("|-----------|-------|")
        for key, value in config.items():
            lines.append(f"| {key} | {value} |")
        
        # Execution Results
        lines.append("\n## 2. Execution Results\n")
        lines.append("### Cycles Comparison\n")
        lines.append("| Mode | Cycles | Time (ns) | Speedup | Throughput |")
        lines.append("|------|--------|-----------|---------|------------|")
        
        # Scalar
        scalar_time = scalar_cycles * 1000 / config.get('clock_frequency_mhz', 1000)
        lines.append(f"| Scalar | {scalar_cycles} | {scalar_time:.2f} | 1.00x | - |")
        
        # Vector
        if vector_result:
            vec_cycles = vector_result.get('cycles', 0)
            vec_time = vec_cycles * 1000 / config.get('clock_frequency_mhz', 1000)
            vec_speedup = scalar_cycles / vec_cycles if vec_cycles > 0 else 0
            vec_throughput = vector_result.get('throughput', 0)
            lines.append(f"| Vector (Non-pipeline) | {vec_cycles} | {vec_time:.2f} | {vec_speedup:.2f}x | {vec_throughput:.4f} |")
        
        # Pipeline
        if pipeline_result:
            pipe_cycles = pipeline_result.get('cycles', 0)
            pipe_time = pipe_cycles * 1000 / config.get('clock_frequency_mhz', 1000)
            pipe_speedup = scalar_cycles / pipe_cycles if pipe_cycles > 0 else 0
            pipe_throughput = pipeline_result.get('throughput', 0)
            lines.append(f"| Pipeline | {pipe_cycles} | {pipe_time:.2f} | {pipe_speedup:.2f}x | {pipe_throughput:.4f} |")
        
        # Performance Analysis
        if analysis:
            lines.append("\n## 3. Performance Analysis\n")
            
            # Speedup
            lines.append("### Speedup Analysis\n")
            metrics = analysis.get('metrics', {})
            lines.append(f"- **Vector Speedup:** {metrics.get('vector_speedup', 0):.2f}x")
            lines.append(f"- **Pipeline Speedup:** {metrics.get('pipeline_speedup', 0):.2f}x")
            lines.append(f"- **Vector Efficiency:** {metrics.get('vector_efficiency', 0):.1%}")
            lines.append(f"- **Pipeline Efficiency:** {metrics.get('pipeline_efficiency', 0):.1%}")
            
            # Bottlenecks
            bottlenecks = analysis.get('bottlenecks', [])
            if bottlenecks:
                lines.append("\n### Bottlenecks Detected\n")
                for b in bottlenecks:
                    severity_icon = "🔴" if b.get('severity') == "high" else "🟡"
                    lines.append(f"- {severity_icon} **{b.get('type')}**: {b.get('description')}")
                    lines.append(f"  - *Suggestion:* {b.get('suggestion')}")
            
            # Recommendations
            recommendations = analysis.get('recommendations', [])
            if recommendations:
                lines.append("\n### Recommendations\n")
                for rec in recommendations:
                    lines.append(f"- 💡 {rec}")
        
        # Timeline (if available)
        if self.config.include_timeline:
            timeline = vector_result.get('timeline', []) or pipeline_result.get('timeline', [])
            if timeline:
                lines.append("\n## 4. Execution Timeline\n")
                lines.append("```")
                lines.append(f"{'Instruction':<35} {'Start':<8} {'End':<8} {'Duration':<10} {'Unit'}")
                lines.append("-" * 70)
                for event in timeline[:20]:  # Limit to 20 events
                    lines.append(f"{event.get('instruction', ''):<35} {event.get('start', 0):<8} "
                            f"{event.get('end', 0):<8} {event.get('duration', 0):<10} {event.get('unit', '')}")
                if len(timeline) > 20:
                    lines.append(f"... and {len(timeline) - 20} more events")
                lines.append("```")
        
        # Memory Statistics
        if self.config.include_memory_stats and vector_result:
            mem_stats = vector_result.get('memory_stats', {})
            if mem_stats:
                lines.append("\n## 5. Memory Statistics\n")
                lines.append(f"- **Total Loads:** {mem_stats.get('total_loads', 0)}")
                lines.append(f"- **Total Stores:** {mem_stats.get('total_stores', 0)}")
                lines.append(f"- **Elements Loaded:** {mem_stats.get('total_elements_loaded', 0)}")
                lines.append(f"- **Elements Stored:** {mem_stats.get('total_elements_stored', 0)}")
                lines.append(f"- **Memory Bandwidth:** {mem_stats.get('bandwidth', 0)} elem/cycle")
        
        # Footer
        lines.append("\n---")
        lines.append(f"\n*Report generated by Vector Processor Simulator*")
        
        return "\n".join(lines)
    
    def save_markdown_report(self, program_name: str, config: Dict,
                            scalar_cycles: int, vector_result: Dict,
                            pipeline_result: Dict, analysis: Optional[Dict] = None,
                            filename: Optional[str] = None) -> Path:
        """
        Lưu báo cáo Markdown ra file
        
        Args:
            program_name: Tên chương trình
            config: Cấu hình hệ thống
            scalar_cycles: Số chu kỳ scalar
            vector_result: Kết quả vector simulation
            pipeline_result: Kết quả pipeline simulation
            analysis: Phân tích hiệu năng
            filename: Tên file output
        
        Returns:
            Path: Đường dẫn file đã lưu
        """
        if filename is None:
            # Tạo tên file từ program name
            base_name = Path(program_name).stem
            filename = f"report_{base_name}_{self.timestamp}.md"
        
        filepath = self.output_dir / filename
        content = self.generate_markdown_report(
            program_name, config, scalar_cycles, 
            vector_result, pipeline_result, analysis
        )
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        
        print(f"✓ Markdown report saved to {filepath}")
        return filepath
    

    # ===== Utility Methods =====
    
    def save_all_formats(self, program_name: str, config: Dict,
                        scalar_cycles: int, vector_result: Dict,
                        pipeline_result: Dict, analysis: Optional[Dict] = None) -> Dict[str, Path]:
        """
        Lưu báo cáo ở định dạng CSV và Markdown
        
        Returns:
            Dict[str, Path]: Dictionary chứa đường dẫn các file
        """
        files = {}
        
        # CSV
        if self.config.format_csv:
            if vector_result.get('timeline'):
                files['timeline_csv'] = self.save_timeline(
                    vector_result['timeline'],
                    f"timeline_{Path(program_name).stem}.csv"
                )
            if pipeline_result.get('timeline'):
                files['pipeline_timeline_csv'] = self.save_timeline(
                    pipeline_result['timeline'],
                    f"pipeline_timeline_{Path(program_name).stem}.csv"
                )

            # Build performance rows từ vector và pipeline results
            perf_rows = []
            base = Path(program_name).stem
            clock = config.get('clock_frequency_mhz', 1000)
            if vector_result:
                vec_cycles = vector_result.get('cycles', 0)
                perf_rows.append({
                    "mode": "vector",
                    "cycles": vec_cycles,
                    "execution_time_ns": vec_cycles * 1000 / clock,
                    "speedup": scalar_cycles / vec_cycles if vec_cycles > 0 else 0,
                    "throughput_elem_per_cycle": vector_result.get('throughput', 0),
                    "lane_utilization": vector_result.get('lane_utilization', 0),
                })
            if pipeline_result:
                pipe_cycles = pipeline_result.get('cycles', 0)
                perf_rows.append({
                    "mode": "pipeline",
                    "cycles": pipe_cycles,
                    "execution_time_ns": pipe_cycles * 1000 / clock,
                    "speedup": scalar_cycles / pipe_cycles if pipe_cycles > 0 else 0,
                    "throughput_elem_per_cycle": pipeline_result.get('throughput', 0),
                    "lane_utilization": pipeline_result.get('lane_utilization', 0),
                })
            if perf_rows:
                files['performance_csv'] = self.save_performance(
                    perf_rows, f"performance_{base}.csv"
                )
        
        # Markdown
        if self.config.format_markdown:
            files['markdown'] = self.save_markdown_report(
                program_name, config, scalar_cycles,
                vector_result, pipeline_result, analysis
            )
        
        return files


    
    def print_timeline(self, timeline: List[Dict], limit: Optional[int] = None):
        """
        In timeline ra console
        
        Args:
            timeline: Danh sách timeline events
            limit: Giới hạn số dòng hiển thị
        """
        if not timeline:
            print("No timeline data available")
            return
        
        print("\n" + "=" * 80)
        print("EXECUTION TIMELINE")
        print("=" * 80)
        print(f"{'Instruction':<40} {'Start':<8} {'End':<8} {'Duration':<10} {'Unit'}")
        print("-" * 80)
        
        events = timeline[:limit] if limit else timeline
        for event in events:
            inst = event.get('instruction', '')[:38]
            print(f"{inst:<40} {event.get('start', 0):<8} "
                f"{event.get('end', 0):<8} {event.get('duration', 0):<10} {event.get('unit', '')}")
        
        if limit and len(timeline) > limit:
            print(f"\n... and {len(timeline) - limit} more events")
    
    def print_performance(self, rows: List[Dict]):
        """
        In performance data ra console
        
        Args:
            rows: Danh sách performance rows
        """
        if not rows:
            print("No performance data available")
            return
        
        print("\n" + "=" * 80)
        print("PERFORMANCE SUMMARY")
        print("=" * 80)
        
        for row in rows:
            print(f"\nMode: {row.get('mode', 'Unknown').upper()}")
            print(f"  Cycles:           {row.get('cycles', 0):,}")
            print(f"  Execution Time:   {row.get('execution_time_ns', 0):.2f} ns")
            print(f"  Speedup:          {row.get('speedup', 0):.2f}x")
            print(f"  Throughput:       {row.get('throughput_elem_per_cycle', 0):.4f} elem/cycle")
            print(f"  Lane Utilization: {row.get('lane_utilization', 0):.1%}")


# ===== TEST =====
if __name__ == "__main__":
    # Tạo dữ liệu test
    timeline = [
        {"instruction": "VLOAD V1, A", "start": 0, "end": 5, "duration": 5, "unit": "Memory"},
        {"instruction": "VLOAD V2, B", "start": 5, "end": 10, "duration": 5, "unit": "Memory"},
        {"instruction": "VADD V3, V1, V2", "start": 10, "end": 15, "duration": 5, "unit": "Vector ALU"},
        {"instruction": "VSTORE C, V3", "start": 15, "end": 20, "duration": 5, "unit": "Memory"},
    ]
    
    performance = [
        {"mode": "vector", "cycles": 20, "execution_time_ns": 20.0, 
        "speedup": 4.4, "throughput_elem_per_cycle": 0.4, "lane_utilization": 1.0},
        {"mode": "pipeline", "cycles": 16, "execution_time_ns": 16.0,
        "speedup": 5.5, "throughput_elem_per_cycle": 0.5, "lane_utilization": 1.0},
    ]
    
    config = {
        "vector_length": 8,
        "num_lanes": 4,
        "memory_bandwidth": 4,
        "clock_frequency_mhz": 1000
    }
    
    vector_result = {
        "cycles": 20,
        "throughput": 0.4,
        "timeline": timeline,
        "memory_stats": {"total_loads": 2, "total_stores": 1, "bandwidth": 4}
    }
    
    pipeline_result = {
        "cycles": 16,
        "throughput": 0.5,
        "timeline": timeline
    }
    
    # Khởi tạo report generator
    reporter = ReportGenerator()
    
    # Test print methods
    reporter.print_timeline(timeline)
    reporter.print_performance(performance)
    
    # Test save methods
    reporter.save_timeline(timeline, "test_timeline.csv")
    reporter.save_performance(performance, "test_performance.csv")
    
    # Test markdown report
    reporter.save_markdown_report("vector_add.txt", config, 88, 
                                vector_result, pipeline_result)
    
    # Test all formats
    files = reporter.save_all_formats("vector_add.txt", config, 88,
                                    vector_result, pipeline_result)
    
    print("\n✓ All tests passed!")
    print(f"\nGenerated files in '{reporter.output_dir}/':")
    for name, path in files.items():
        print(f"  - {path.name}")
