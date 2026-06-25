import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class PerformanceMetrics:
    """Các chỉ số hiệu năng"""
    # Thời gian và cycles
    scalar_cycles: int = 0
    vector_cycles: int = 0
    pipeline_cycles: int = 0
    
    # Speedup
    vector_speedup: float = 0.0
    pipeline_speedup: float = 0.0
    
    # Throughput
    scalar_throughput: float = 0.0
    vector_throughput: float = 0.0
    pipeline_throughput: float = 0.0
    
    # Utilization
    lane_utilization: float = 0.0
    memory_bandwidth_utilization: float = 0.0
    
    # Efficiency
    vector_efficiency: float = 0.0
    pipeline_efficiency: float = 0.0
    
    # Additional metrics
    total_elements: int = 0
    clock_frequency_mhz: int = 1000
    
    @property
    def scalar_time_ns(self) -> float:
        return self.scalar_cycles * 1000 / self.clock_frequency_mhz
    
    @property
    def vector_time_ns(self) -> float:
        return self.vector_cycles * 1000 / self.clock_frequency_mhz
    
    @property
    def pipeline_time_ns(self) -> float:
        return self.pipeline_cycles * 1000 / self.clock_frequency_mhz
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "scalar_cycles": self.scalar_cycles,
            "vector_cycles": self.vector_cycles,
            "pipeline_cycles": self.pipeline_cycles,
            "scalar_time_ns": self.scalar_time_ns,
            "vector_time_ns": self.vector_time_ns,
            "pipeline_time_ns": self.pipeline_time_ns,
            "vector_speedup": self.vector_speedup,
            "pipeline_speedup": self.pipeline_speedup,
            "vector_throughput": self.vector_throughput,
            "pipeline_throughput": self.pipeline_throughput,
            "lane_utilization": self.lane_utilization,
            "memory_bandwidth_utilization": self.memory_bandwidth_utilization,
            "vector_efficiency": self.vector_efficiency,
            "pipeline_efficiency": self.pipeline_efficiency
        }


class PerformanceAnalyzer:
    """
    Phân tích hiệu năng cho vector processor
    
    Hỗ trợ:
    - Tính speedup, throughput, utilization
    - So sánh giữa scalar, vector, pipeline
    - Phân tích bottleneck
    - Xuất báo cáo chi tiết
    - Tạo dữ liệu cho biểu đồ
    """
    
    def __init__(self, config):
        """
        Khởi tạo performance analyzer
        
        Args:
            config: VectorProcessorConfig object
        """
        self.config = config
        self.metrics = PerformanceMetrics(
            clock_frequency_mhz=config.clock_frequency_mhz
        )
    
    # ===== Các phương thức tính toán cơ bản =====
    
    def calculate_speedup(self, scalar_cycles: int, vector_cycles: int) -> float:
        """
        Tính speedup so với scalar
        
        Args:
            scalar_cycles: Số chu kỳ scalar
            vector_cycles: Số chu kỳ vector
        
        Returns:
            float: Speedup factor
        """
        if vector_cycles <= 0:
            return 0.0
        return scalar_cycles / vector_cycles
    
    def calculate_throughput(self, num_elements: int, cycles: int) -> float:
        """
        Tính throughput (elements per cycle)
        
        Args:
            num_elements: Số phần tử xử lý
            cycles: Số chu kỳ
        
        Returns:
            float: Throughput
        """
        if cycles <= 0:
            return 0.0
        return num_elements / cycles
    
    def calculate_lane_utilization(self, num_elements: int) -> float:
        """
        Tính hiệu suất sử dụng lane
        
        Args:
            num_elements: Số phần tử xử lý
        
        Returns:
            float: Lane utilization (0-1)
        """
        groups = math.ceil(num_elements / self.config.num_lanes)
        available_slots = groups * self.config.num_lanes
        if available_slots <= 0:
            return 0.0
        return num_elements / available_slots
    
    def calculate_memory_bandwidth_utilization(self, total_elements: int, 
                                                 total_cycles: int) -> float:
        """
        Tính hiệu suất sử dụng băng thông bộ nhớ
        
        Args:
            total_elements: Tổng số phần tử truy xuất
            total_cycles: Tổng số chu kỳ
        
        Returns:
            float: Bandwidth utilization (0-1)
        """
        max_transfers = total_cycles * self.config.memory_bandwidth
        if max_transfers <= 0:
            return 0.0
        return min(max(total_elements / max_transfers, 0.0), 1.0)
    
    def calculate_efficiency(self, speedup: float, num_lanes: int) -> float:
        """
        Tính hiệu suất sử dụng lanes
        
        Args:
            speedup: Speedup đạt được
            num_lanes: Số lanes
        
        Returns:
            float: Efficiency (0-1)
        """
        if num_lanes <= 0:
            return 0.0
        return min(max(speedup / num_lanes, 0.0), 1.0)
    
    def calculate_ipc(self, total_instructions: int, total_cycles: int) -> float:
        """
        Tính IPC (Instructions Per Cycle)
        
        Args:
            total_instructions: Tổng số lệnh
            total_cycles: Tổng số chu kỳ
        
        Returns:
            float: IPC
        """
        if total_cycles <= 0:
            return 0.0
        return total_instructions / total_cycles
    
    # ===== Phương thức phân tích =====
    
    def analyze(self, scalar_cycles: int, vector_cycles: int, 
                pipeline_cycles: int, num_elements: int) -> PerformanceMetrics:
        """
        Phân tích hiệu năng đầy đủ
        
        Args:
            scalar_cycles: Số chu kỳ scalar
            vector_cycles: Số chu kỳ vector (non-pipeline)
            pipeline_cycles: Số chu kỳ pipeline
            num_elements: Số phần tử xử lý
        
        Returns:
            PerformanceMetrics: Các chỉ số hiệu năng
        """
        metrics = PerformanceMetrics(
            scalar_cycles=scalar_cycles,
            vector_cycles=vector_cycles,
            pipeline_cycles=pipeline_cycles,
            total_elements=num_elements,
            clock_frequency_mhz=self.config.clock_frequency_mhz
        )
        
        # Speedup
        metrics.vector_speedup = self.calculate_speedup(scalar_cycles, vector_cycles)
        metrics.pipeline_speedup = self.calculate_speedup(scalar_cycles, pipeline_cycles)
        
        # Throughput
        metrics.scalar_throughput = self.calculate_throughput(num_elements, scalar_cycles)
        metrics.vector_throughput = self.calculate_throughput(num_elements, vector_cycles)
        metrics.pipeline_throughput = self.calculate_throughput(num_elements, pipeline_cycles)
        
        # Utilization
        metrics.lane_utilization = self.calculate_lane_utilization(num_elements)
        
        # Efficiency
        metrics.vector_efficiency = self.calculate_efficiency(metrics.vector_speedup, self.config.num_lanes)
        metrics.pipeline_efficiency = self.calculate_efficiency(metrics.pipeline_speedup, self.config.num_lanes)
        
        self.metrics = metrics
        return metrics
    
    def analyze_with_stats(self, scalar_cycles: int, vector_result: Dict,
                           pipeline_result: Dict, num_elements: int) -> Dict[str, Any]:
        """
        Phân tích hiệu năng với thống kê chi tiết
        
        Args:
            scalar_cycles: Số chu kỳ scalar
            vector_result: Kết quả từ vector simulator
            pipeline_result: Kết quả từ pipeline simulator
            num_elements: Số phần tử xử lý
        
        Returns:
            Dict: Báo cáo phân tích chi tiết
        """
        metrics = self.analyze(
            scalar_cycles,
            vector_result.get('cycles', 0),
            pipeline_result.get('cycles', 0),
            num_elements
        )
        
        # Phân tích bottleneck
        bottlenecks = self._identify_bottlenecks(metrics, vector_result, pipeline_result)
        
        # Phân tích scalability
        scalability = self._analyze_scalability(metrics)
        
        return {
            "metrics": metrics.to_dict(),
            "bottlenecks": bottlenecks,
            "scalability": scalability,
            "recommendations": self._generate_recommendations(bottlenecks),
            "comparison": self._compare_modes(metrics)
        }
    
    def _identify_bottlenecks(self, metrics: PerformanceMetrics, 
                               vector_result: Dict, 
                               pipeline_result: Dict) -> List[Dict]:
        """Xác định các bottleneck trong hệ thống"""
        bottlenecks = []
        
        # Kiểm tra lane utilization
        if metrics.lane_utilization < 0.8:
            bottlenecks.append({
                "type": "lane_utilization",
                "severity": "high" if metrics.lane_utilization < 0.5 else "medium",
                "description": f"Low lane utilization: {metrics.lane_utilization:.1%}",
                "suggestion": "Increase vector length or use more elements"
            })
        
        # Kiểm tra memory bandwidth
        if 'memory_stats' in vector_result:
            mem_stats = vector_result['memory_stats']
            total_elements = mem_stats.get('total_elements_loaded', 0) + mem_stats.get('total_elements_stored', 0)
            total_cycles = metrics.vector_cycles
            bw_util = self.calculate_memory_bandwidth_utilization(total_elements, total_cycles)
            metrics.memory_bandwidth_utilization = bw_util
            
            if bw_util > 0.9:
                bottlenecks.append({
                    "type": "memory_bandwidth",
                    "severity": "high",
                    "description": f"Memory bandwidth saturated: {bw_util:.1%}",
                    "suggestion": "Increase memory bandwidth or reduce stride"
                })
        
        # Kiểm tra pipeline efficiency
        pipeline_speedup = metrics.pipeline_speedup
        ideal_speedup = self.config.num_lanes
        efficiency = pipeline_speedup / ideal_speedup if ideal_speedup > 0 else 0
        
        if efficiency < 0.7:
            bottlenecks.append({
                "type": "pipeline_efficiency",
                "severity": "medium",
                "description": f"Low pipeline efficiency: {efficiency:.1%}",
                "suggestion": "Reduce dependencies or enable chaining"
            })
        
        # Kiểm tra startup latency effect
        if metrics.vector_speedup < ideal_speedup * 0.5:
            bottlenecks.append({
                "type": "startup_latency",
                "severity": "medium",
                "description": "Startup latency dominating execution",
                "suggestion": "Increase vector length for better amortization"
            })
        
        return bottlenecks
    
    def _analyze_scalability(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Phân tích khả năng mở rộng"""
        ideal_speedup = self.config.num_lanes
        
        return {
            "ideal_speedup": ideal_speedup,
            "achieved_speedup": metrics.pipeline_speedup,
            "scalability_efficiency": metrics.pipeline_efficiency,
            "scalability_grade": self._get_scalability_grade(metrics.pipeline_efficiency),
            "max_sustainable_lanes": self._estimate_max_lanes(metrics)
        }
    
    def _get_scalability_grade(self, efficiency: float) -> str:
        """Đánh giá khả năng mở rộng"""
        if efficiency >= 0.9:
            return "Excellent"
        elif efficiency >= 0.7:
            return "Good"
        elif efficiency >= 0.5:
            return "Fair"
        elif efficiency >= 0.3:
            return "Poor"
        else:
            return "Very Poor"
    
    def _estimate_max_lanes(self, metrics: PerformanceMetrics) -> int:
        """Ước lượng số lanes tối đa hiệu quả"""
        # Dựa trên memory bandwidth
        bw_limited_lanes = self.config.memory_bandwidth
        # Dựa trên efficiency
        current_lanes = self.config.num_lanes
        if metrics.pipeline_efficiency < 0.5:
            return current_lanes
        return min(current_lanes * 2, bw_limited_lanes * 2)
    
    def _compare_modes(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """So sánh giữa các chế độ"""
        return {
            "scalar_vs_vector": {
                "improvement": f"{metrics.vector_speedup:.2f}x",
                "time_saved_ns": metrics.scalar_time_ns - metrics.vector_time_ns,
                "percentage_reduction": (1 - metrics.vector_cycles / metrics.scalar_cycles) * 100 if metrics.scalar_cycles > 0 else 0
            },
            "scalar_vs_pipeline": {
                "improvement": f"{metrics.pipeline_speedup:.2f}x",
                "time_saved_ns": metrics.scalar_time_ns - metrics.pipeline_time_ns,
                "percentage_reduction": (1 - metrics.pipeline_cycles / metrics.scalar_cycles) * 100 if metrics.scalar_cycles > 0 else 0
            },
            "vector_vs_pipeline": {
                "improvement": f"{metrics.pipeline_speedup / metrics.vector_speedup:.2f}x",
                "additional_speedup": metrics.pipeline_speedup - metrics.vector_speedup
            }
        }
    
    def _generate_recommendations(self, bottlenecks: List[Dict]) -> List[str]:
        """Tạo recommendations dựa trên bottlenecks"""
        recommendations = []
        
        for b in bottlenecks:
            if b['suggestion'] not in recommendations:
                recommendations.append(b['suggestion'])
        
        if not recommendations:
            recommendations.append("System is well-balanced. Consider increasing vector length for better performance.")
        
        return recommendations
    
    # ===== Phương thức báo cáo =====
    
    def print_report(self, metrics: PerformanceMetrics):
        """In báo cáo hiệu năng"""
        print("\n" + "=" * 70)
        print("PERFORMANCE ANALYSIS REPORT")
        print("=" * 70)
        
        # Thông tin cấu hình
        print(f"\nConfiguration:")
        print(f"  Vector Length:     {self.config.vector_length}")
        print(f"  Number of Lanes:   {self.config.num_lanes}")
        print(f"  Memory Bandwidth:  {self.config.memory_bandwidth} elem/cycle")
        print(f"  Clock Frequency:   {self.config.clock_frequency_mhz} MHz")
        
        # Kết quả cycles
        print(f"\nExecution Cycles:")
        print(f"  Scalar:            {metrics.scalar_cycles:,} cycles")
        print(f"  Vector:            {metrics.vector_cycles:,} cycles")
        print(f"  Pipeline:          {metrics.pipeline_cycles:,} cycles")
        
        # Thời gian thực
        print(f"\nExecution Time:")
        print(f"  Scalar:            {metrics.scalar_time_ns:.2f} ns")
        print(f"  Vector:            {metrics.vector_time_ns:.2f} ns")
        print(f"  Pipeline:          {metrics.pipeline_time_ns:.2f} ns")
        
        # Speedup
        print(f"\nSpeedup (vs Scalar):")
        print(f"  Vector:            {metrics.vector_speedup:.2f}x")
        print(f"  Pipeline:          {metrics.pipeline_speedup:.2f}x")
        
        # Throughput
        print(f"\nThroughput (elements/cycle):")
        print(f"  Scalar:            {metrics.scalar_throughput:.4f}")
        print(f"  Vector:            {metrics.vector_throughput:.4f}")
        print(f"  Pipeline:          {metrics.pipeline_throughput:.4f}")
        
        # Utilization
        print(f"\nUtilization:")
        print(f"  Lane Utilization:  {metrics.lane_utilization:.1%}")
        print(f"  Vector Efficiency: {metrics.vector_efficiency:.1%}")
        print(f"  Pipeline Efficiency: {metrics.pipeline_efficiency:.1%}")
        
        if metrics.memory_bandwidth_utilization > 0:
            print(f"  Memory BW Util:    {metrics.memory_bandwidth_utilization:.1%}")
        
        print("=" * 70)
    
    def print_detailed_analysis(self, analysis: Dict[str, Any]):
        """In phân tích chi tiết"""
        print("\n" + "=" * 70)
        print("DETAILED PERFORMANCE ANALYSIS")
        print("=" * 70)
        
        # Bottlenecks
        if analysis['bottlenecks']:
            print("\n⚠ Bottlenecks Detected:")
            for b in analysis['bottlenecks']:
                severity_icon = "🔴" if b['severity'] == "high" else "🟡"
                print(f"  {severity_icon} {b['type']}: {b['description']}")
                print(f"     → Suggestion: {b['suggestion']}")
        
        # Scalability
        print(f"\n📈 Scalability Analysis:")
        print(f"  Ideal Speedup:     {analysis['scalability']['ideal_speedup']}x")
        print(f"  Achieved Speedup:  {analysis['scalability']['achieved_speedup']:.2f}x")
        print(f"  Scalability Grade: {analysis['scalability']['scalability_grade']}")
        print(f"  Max Sustainable Lanes: {analysis['scalability']['max_sustainable_lanes']}")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        for rec in analysis['recommendations']:
            print(f"  • {rec}")
        
        # Mode comparison
        print(f"\n📊 Mode Comparison:")
        comp = analysis['comparison']
        print(f"  Vector vs Scalar:  {comp['scalar_vs_vector']['improvement']} faster "
              f"({comp['scalar_vs_vector']['percentage_reduction']:.0f}% reduction)")
        print(f"  Pipeline vs Scalar: {comp['scalar_vs_pipeline']['improvement']} faster "
              f"({comp['scalar_vs_pipeline']['percentage_reduction']:.0f}% reduction)")
        print(f"  Pipeline vs Vector: {comp['vector_vs_pipeline']['improvement']} faster")
        
        print("=" * 70)
    
    # ===== Xuất dữ liệu cho biểu đồ =====
    
    def export_for_plotting(self, metrics: PerformanceMetrics) -> Dict[str, List]:
        """Xuất dữ liệu để vẽ biểu đồ"""
        return {
            "modes": ["Scalar", "Vector", "Pipeline"],
            "cycles": [metrics.scalar_cycles, metrics.vector_cycles, metrics.pipeline_cycles],
            "time_ns": [metrics.scalar_time_ns, metrics.vector_time_ns, metrics.pipeline_time_ns],
            "speedup": [1.0, metrics.vector_speedup, metrics.pipeline_speedup],
            "throughput": [metrics.scalar_throughput, metrics.vector_throughput, metrics.pipeline_throughput]
        }
    
    def export_comparison_data(self, experiments: List[Tuple[str, int, int, int]]) -> Dict:
        """
        Xuất dữ liệu so sánh cho nhiều thí nghiệm
        
        Args:
            experiments: List of (name, scalar_cycles, vector_cycles, pipeline_cycles)
        
        Returns:
            Dict: Dữ liệu cho biểu đồ
        """
        data = {
            "names": [],
            "scalar": [],
            "vector": [],
            "pipeline": [],
            "vector_speedup": [],
            "pipeline_speedup": []
        }
        
        for name, scalar, vector, pipeline in experiments:
            data["names"].append(name)
            data["scalar"].append(scalar)
            data["vector"].append(vector)
            data["pipeline"].append(pipeline)
            data["vector_speedup"].append(self.calculate_speedup(scalar, vector))
            data["pipeline_speedup"].append(self.calculate_speedup(scalar, pipeline))
        
        return data
    
    # ===== Phương thức tiện ích =====
    
    def summarize(self, mode: str, scalar_cycles: int, 
                vector_cycles: int, num_elements: int) -> Dict[str, Any]:
        
        """
        Tóm tắt hiệu năng cho một chế độ
        """
        # Tính toán các chỉ số chung
        speedup = self.calculate_speedup(scalar_cycles, vector_cycles)
        throughput = self.calculate_throughput(num_elements, vector_cycles)
        lane_util = self.calculate_lane_utilization(num_elements)
        exec_time = self.execution_time_ns(vector_cycles)
    
        # Trả về dictionary (không cần if-else vì cả 2 mode đều giống cấu trúc)
        return {
            "mode": mode,
            "cycles": vector_cycles,
            "execution_time_ns": exec_time,
            "speedup": speedup,
            "throughput_elem_per_cycle": throughput,
            "lane_utilization": lane_util,
            "efficiency": self.calculate_efficiency(speedup, self.config.num_lanes)
        }

    def execution_time_ns(self, cycles: int) -> float:
        """
        Tính thời gian thực thi từ số cycles
        
        Args:
            cycles: Số chu kỳ
        
        Returns:
            float: Thời gian (nanoseconds)
        """
        return cycles * 1000 / self.config.clock_frequency_mhz


# ===== TEST =====
if __name__ == "__main__":
    from config import VectorProcessorConfig
    
    # Tạo config
    config = VectorProcessorConfig(
        vector_length=8,
        num_lanes=4,
        memory_bandwidth=4,
        clock_frequency_mhz=1000
    )
    
    # Khởi tạo analyzer
    analyzer = PerformanceAnalyzer(config)
    
    # Dữ liệu test
    scalar_cycles = 880
    vector_cycles = 200
    pipeline_cycles = 160
    num_elements = 64
    
    # Phân tích
    metrics = analyzer.analyze(scalar_cycles, vector_cycles, pipeline_cycles, num_elements)
    
    # In báo cáo
    analyzer.print_report(metrics)
    
    # Phân tích chi tiết
    vector_result = {'cycles': vector_cycles, 'memory_stats': {'total_elements_loaded': 64, 'total_elements_stored': 64}}
    pipeline_result = {'cycles': pipeline_cycles}
    
    analysis = analyzer.analyze_with_stats(scalar_cycles, vector_result, pipeline_result, num_elements)
    analyzer.print_detailed_analysis(analysis)
    
    # Test summarize (interface cũ)
    summary = analyzer.summarize("vector", scalar_cycles, vector_cycles, num_elements)
    print("\n" + "=" * 70)
    print("SUMMARIZE (Legacy Interface)")
    print("=" * 70)
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print("\n✓ All tests passed!")
