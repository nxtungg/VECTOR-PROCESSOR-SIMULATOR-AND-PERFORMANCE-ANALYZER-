import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import matplotlib
from matplotlib.patches import Rectangle
import pandas as pd

# Setup font cho tiếng Việt (nếu cần)
try:
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

# Style settings
plt.style.use('seaborn-v0_8-darkgrid') if 'seaborn-v0_8-darkgrid' in plt.style.available else plt.style.use('default')


class PerformanceVisualizer:
    """
    Tạo biểu đồ trực quan hóa hiệu năng cho vector processor
    
    Hỗ trợ:
    - Biểu đồ so sánh cycles/speedup
    - Biểu đồ ảnh hưởng của vector length
    - Biểu đồ ảnh hưởng của số lanes
    - Biểu đồ ảnh hưởng của memory bandwidth
    - Biểu đồ so sánh workload
    - Biểu đồ timeline
    - Biểu đồ utilization
    """
    
    def __init__(self, output_dir: str = "figures"):
        """
        Khởi tạo visualizer
        
        Args:
            output_dir: Thư mục lưu biểu đồ
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Màu sắc cho biểu đồ
        self.colors = {
            'scalar': '#2ecc71',      # Xanh lá
            'vector': '#3498db',       # Xanh dương
            'pipeline': '#e74c3c',     # Đỏ
            'ideal': '#9b59b6',        # Tím
            'memory': '#f39c12',       # Cam
            'alu': '#1abc9c',          # Xanh ngọc
            'reduction': '#e67e22'     # Da cam
        }
        
        # Markers cho biểu đồ
        self.markers = {
            'scalar': 'o',
            'vector': 's',
            'pipeline': '^',
            'ideal': 'D'
        }
    
    # ===== Basic Charts =====
    
    def plot_cycles_comparison(self, scalar_cycles: int, vector_cycles: int, 
                                pipeline_cycles: int, save_name: str = "cycles_comparison.png"):
        """
        Vẽ biểu đồ so sánh số cycles giữa các chế độ
        
        Args:
            scalar_cycles: Số cycles scalar
            vector_cycles: Số cycles vector
            pipeline_cycles: Số cycles pipeline
            save_name: Tên file lưu
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        modes = ['Scalar', 'Vector\n(Non-pipeline)', 'Pipeline']
        cycles = [scalar_cycles, vector_cycles, pipeline_cycles]
        colors = [self.colors['scalar'], self.colors['vector'], self.colors['pipeline']]
        
        bars = ax.bar(modes, cycles, color=colors, edgecolor='black', linewidth=1.5)
        
        # Thêm giá trị trên đầu cột
        for bar, cycle in zip(bars, cycles):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(cycles)*0.02,
                f'{cycle:,}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_ylabel('Cycles', fontsize=12, fontweight='bold')
        ax.set_title('Execution Cycles Comparison', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {save_name}")
    
    def plot_speedup_comparison(self, vector_speedup: float, pipeline_speedup: float,
                                ideal_speedup: int, save_name: str = "speedup_comparison.png"):
        """
        Vẽ biểu đồ so sánh speedup
        
        Args:
            vector_speedup: Speedup của vector mode
            pipeline_speedup: Speedup của pipeline mode
            ideal_speedup: Speedup lý tưởng (số lanes)
            save_name: Tên file lưu
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        modes = ['Vector', 'Pipeline', 'Ideal']
        speedups = [vector_speedup, pipeline_speedup, ideal_speedup]
        colors = [self.colors['vector'], self.colors['pipeline'], self.colors['ideal']]
        
        bars = ax.bar(modes, speedups, color=colors, edgecolor='black', linewidth=1.5)
        
        # Thêm giá trị
        for bar, speedup in zip(bars, speedups):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(speedups)*0.02,
                f'{speedup:.2f}x', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_ylabel('Speedup (vs Scalar)', fontsize=12, fontweight='bold')
        ax.set_title('Performance Speedup Comparison', fontsize=14, fontweight='bold')
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {save_name}")
    
    def plot_throughput_comparison(self, vector_throughput: float, pipeline_throughput: float,
                                    save_name: str = "throughput_comparison.png"):
        """
        Vẽ biểu đồ so sánh throughput
        
        Args:
            vector_throughput: Throughput của vector mode
            pipeline_throughput: Throughput của pipeline mode
            save_name: Tên file lưu
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        modes = ['Vector\n(Non-pipeline)', 'Pipeline']
        throughputs = [vector_throughput, pipeline_throughput]
        colors = [self.colors['vector'], self.colors['pipeline']]
        
        bars = ax.bar(modes, throughputs, color=colors, edgecolor='black', linewidth=1.5)
        
        for bar, tp in zip(bars, throughputs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(throughputs)*0.02,
                f'{tp:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_ylabel('Throughput (Elements/Cycle)', fontsize=12, fontweight='bold')
        ax.set_title('Throughput Comparison', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {save_name}")
    
    # ===== Experiment Charts =====
    
    def plot_vector_length_effect(self, vector_lengths: List[int], 
                                cycles: List[int], speedups: List[float],
                                save_name: str = "vector_length_effect.png"):
        """
        Vẽ biểu đồ ảnh hưởng của vector length
        
        Args:
            vector_lengths: Danh sách vector lengths
            cycles: Danh sách cycles tương ứng
            speedups: Danh sách speedups tương ứng
            save_name: Tên file lưu
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot cycles
        ax1.plot(vector_lengths, cycles, marker=self.markers['pipeline'], 
                linewidth=2, markersize=8, color=self.colors['pipeline'])
        ax1.set_xlabel('Vector Length', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Cycles', fontsize=12, fontweight='bold')
        ax1.set_title('Cycles vs Vector Length', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Plot speedup
        ax2.plot(vector_lengths, speedups, marker=self.markers['pipeline'],
                linewidth=2, markersize=8, color=self.colors['vector'])
        ax2.axhline(y=4, color='gray', linestyle='--', alpha=0.7, label='Ideal (4 lanes)')
        ax2.set_xlabel('Vector Length', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Speedup', fontsize=12, fontweight='bold')
        ax2.set_title('Speedup vs Vector Length', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Effect of Vector Length on Performance', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {save_name}")
    
    def plot_lanes_effect(self, lanes: List[int], cycles: List[float], 
                        speedups: List[float], ideal_speedups: List[int],
                        save_name: str = "lanes_effect.png"):
        """
        Vẽ biểu đồ ảnh hưởng của số lanes
        
        Args:
            lanes: Danh sách số lanes
            cycles: Danh sách cycles
            speedups: Danh sách speedups thực tế
            ideal_speedups: Danh sách speedups lý tưởng
            save_name: Tên file lưu
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(lanes, speedups, marker=self.markers['pipeline'], linewidth=2,
            markersize=8, color=self.colors['vector'], label='Actual Speedup')
        ax.plot(lanes, ideal_speedups, marker=self.markers['ideal'], linewidth=2,
            markersize=8, color=self.colors['ideal'], linestyle='--', label='Ideal Speedup')
        
        ax.set_xlabel('Number of Lanes', fontsize=12, fontweight='bold')
        ax.set_ylabel('Speedup', fontsize=12, fontweight='bold')
        ax.set_title('Effect of Number of Lanes on Speedup', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Thêm annotation
        for i, (l, s, ideal) in enumerate(zip(lanes, speedups, ideal_speedups)):
            efficiency = s / ideal * 100 if ideal > 0 else 0
            ax.annotate(f'{efficiency:.0f}%', (l, s), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {save_name}")
    
    def plot_memory_bandwidth_effect(self, bandwidths: List[int], 
                                    cycles: List[int], speedups: List[float],
                                    save_name: str = "bandwidth_effect.png"):
        """
        Vẽ biểu đồ ảnh hưởng của memory bandwidth
        
        Args:
            bandwidths: Danh sách bandwidth values
            cycles: Danh sách cycles
            speedups: Danh sách speedups
            save_name: Tên file lưu
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot cycles
        ax1.plot(bandwidths, cycles, marker=self.markers['pipeline'],
                linewidth=2, markersize=8, color=self.colors['memory'])
        ax1.set_xlabel('Memory Bandwidth (Elements/Cycle)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Cycles', fontsize=12, fontweight='bold')
        ax1.set_title('Cycles vs Memory Bandwidth', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Plot speedup
        ax2.plot(bandwidths, speedups, marker=self.markers['pipeline'],
                linewidth=2, markersize=8, color=self.colors['memory'])
        ax2.set_xlabel('Memory Bandwidth (Elements/Cycle)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Speedup', fontsize=12, fontweight='bold')
        ax2.set_title('Speedup vs Memory Bandwidth', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Effect of Memory Bandwidth on Performance', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {save_name}")
    
    def plot_workload_comparison(self, workloads: List[str], scalar_cycles: List[int],
                                vector_cycles: List[int], pipeline_cycles: List[int],
                                save_name: str = "workload_comparison.png"):
        """
        Vẽ biểu đồ so sánh các workload
        
        Args:
            workloads: Danh sách tên workloads
            scalar_cycles: Danh sách scalar cycles
            vector_cycles: Danh sách vector cycles
            pipeline_cycles: Danh sách pipeline cycles
            save_name: Tên file lưu
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(workloads))
        width = 0.25
        
        bars1 = ax.bar(x - width, scalar_cycles, width, label='Scalar', 
                    color=self.colors['scalar'], edgecolor='black')
        bars2 = ax.bar(x, vector_cycles, width, label='Vector', 
                    color=self.colors['vector'], edgecolor='black')
        bars3 = ax.bar(x + width, pipeline_cycles, width, label='Pipeline', 
                    color=self.colors['pipeline'], edgecolor='black')
        
        ax.set_xlabel('Workload', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cycles', fontsize=12, fontweight='bold')
        ax.set_title('Workload Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(workloads)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {save_name}")
    
    def plot_speedup_heatmap(self, vector_lengths: List[int], lanes: List[int],
                            speedup_matrix: List[List[float]], 
                            save_name: str = "speedup_heatmap.png"):
        """
        Vẽ heatmap speedup theo vector length và lanes
        
        Args:
            vector_lengths: Danh sách vector lengths
            lanes: Danh sách lanes
            speedup_matrix: Ma trận speedup
            save_name: Tên file lưu
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(speedup_matrix, cmap='YlOrRd', aspect='auto', origin='lower')
        
        ax.set_xticks(np.arange(len(vector_lengths)))
        ax.set_yticks(np.arange(len(lanes)))
        ax.set_xticklabels([str(vl) for vl in vector_lengths])
        ax.set_yticklabels([str(l) for l in lanes])
        
        ax.set_xlabel('Vector Length', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Lanes', fontsize=12, fontweight='bold')
        ax.set_title('Speedup Heatmap', fontsize=14, fontweight='bold')
        
        # Thêm colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Speedup', fontsize=12)
        
        # Thêm giá trị vào ô
        for i in range(len(lanes)):
            for j in range(len(vector_lengths)):
                text = ax.text(j, i, f'{speedup_matrix[i][j]:.1f}x',
                            ha="center", va="center", color="black", fontsize=8)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {save_name}")
    
    def plot_stride_effect(self, strides: List[int], load_cycles: List[int],
                            effective_bandwidth: List[float],
                            save_name: str = "stride_effect.png"):
        """
        Vẽ biểu đồ ảnh hưởng của stride access
        
        Args:
            strides: Danh sách stride values
            load_cycles: Danh sách cycles tương ứng
            effective_bandwidth: Danh sách effective bandwidth
            save_name: Tên file lưu
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot cycles
        ax1.plot(strides, load_cycles, marker='o', linewidth=2, markersize=8,
                color=self.colors['memory'])
        ax1.set_xlabel('Stride', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Load Cycles', fontsize=12, fontweight='bold')
        ax1.set_title('Effect of Stride on Load Cycles', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Plot effective bandwidth
        ax2.plot(strides, effective_bandwidth, marker='s', linewidth=2, markersize=8,
                color=self.colors['memory'])
        ax2.set_xlabel('Stride', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Effective Bandwidth (Elements/Cycle)', fontsize=12, fontweight='bold')
        ax2.set_title('Effective Bandwidth vs Stride', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Effect of Stride on Memory Performance', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {save_name}")
    
    # ===== Utilization Charts =====
    
    def plot_utilization_pie(self, memory_util: float, alu_util: float, 
                            reduction_util: float, save_name: str = "utilization_pie.png"):
        """
        Vẽ biểu đồ tròn utilization
        
        Args:
            memory_util: Memory unit utilization (%)
            alu_util: ALU unit utilization (%)
            reduction_util: Reduction unit utilization (%)
            save_name: Tên file lưu
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        labels = ['Memory Unit', 'ALU Unit', 'Reduction Unit']
        sizes = [memory_util, alu_util, reduction_util]
        colors = [self.colors['memory'], self.colors['alu'], self.colors['reduction']]
        explode = (0.05, 0.05, 0.05)
        
        result = ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                                        autopct='%1.1f%%', shadow=True, startangle=90)
        wedges = result[0]
        ax.set_title('Functional Unit Utilization', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {save_name}")
    
    def plot_lane_utilization(self, vector_lengths: List[int], 
                            utilizations: List[float],
                            save_name: str = "lane_utilization.png"):
        """
        Vẽ biểu đồ lane utilization theo vector length
        
        Args:
            vector_lengths: Danh sách vector lengths
            utilizations: Danh sách utilizations
            save_name: Tên file lưu
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(vector_lengths, utilizations, marker='o', linewidth=2, markersize=8,
            color=self.colors['alu'])
        ax.fill_between(vector_lengths, utilizations, alpha=0.3, color=self.colors['alu'])
        
        ax.set_xlabel('Vector Length', fontsize=12, fontweight='bold')
        ax.set_ylabel('Lane Utilization', fontsize=12, fontweight='bold')
        ax.set_title('Lane Utilization vs Vector Length', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
        
        # Thêm đường 100%
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='100% Utilization')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {save_name}")
    
    def plot_cache_size_effect(self, cache_lines: List[int], hit_rates: List[float],
                               cycles: List[int],
                               save_name: str = "cache_size_effect.png"):
        """Plot cache hit rate and execution cycles versus L1 capacity."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(cache_lines, hit_rates, marker="o", color=self.colors["memory"])
        ax1.set_xlabel("L1 Cache Lines")
        ax1.set_ylabel("Cache Hit Rate")
        ax1.set_ylim(0, 1.05)
        ax1.grid(True, alpha=0.3)
        ax2.plot(cache_lines, cycles, marker="s", color=self.colors["pipeline"])
        ax2.set_xlabel("L1 Cache Lines")
        ax2.set_ylabel("Cycles")
        ax2.grid(True, alpha=0.3)
        fig.suptitle("Effect of Cache Size")
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=150, bbox_inches="tight")
        plt.close()

    def plot_mask_utilization(self, active_ratios: List[float],
                              lane_utilizations: List[float],
                              save_name: str = "mask_utilization.png"):
        """Plot useful SIMD lane utilization for different mask densities."""
        fig, ax = plt.subplots(figsize=(9, 5))
        percentages = [x * 100 for x in active_ratios]
        ax.plot(percentages, lane_utilizations, marker="o", linewidth=2,
                color=self.colors["alu"])
        ax.set_xlabel("Active Mask Elements (%)")
        ax.set_ylabel("Useful Lane Utilization")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.set_title("Effect of Vector Mask Density")
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=150, bbox_inches="tight")
        plt.close()

    # ===== Timeline Chart =====
    
    def plot_timeline(self, timeline: List[Dict], save_name: str = "timeline.png"):
        """
        Vẽ biểu đồ timeline thực thi
        
        Args:
            timeline: Danh sách timeline events
            save_name: Tên file lưu
        """
        if not timeline:
            print("No timeline data to plot")
            return
        
        fig, ax = plt.subplots(figsize=(14, max(6, len(timeline) * 0.5)))
        
        y_pos = np.arange(len(timeline))
        units = [event.get('unit', 'Unknown') for event in timeline]
        starts = [event.get('start', 0) for event in timeline]
        durations = [event.get('duration', 1) for event in timeline]
        
        # Màu sắc theo unit
        unit_colors = {
            'Memory': self.colors['memory'],
            'Vector ALU': self.colors['alu'],
            'Reduction Unit': self.colors['reduction'],
            'Dot Product Unit': self.colors['reduction']
        }
        colors = [unit_colors.get(u, '#95a5a6') for u in units]
        
        # Vẽ thanh
        for i, (start, duration, color) in enumerate(zip(starts, durations, colors)):
            ax.barh(i, duration, left=start, height=0.6, color=color, edgecolor='black')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{event.get('instruction', '')[:40]}" for event in timeline], fontsize=9)
        ax.set_xlabel('Cycle', fontsize=12, fontweight='bold')
        ax.set_title('Execution Timeline', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=self.colors['memory'], label='Memory'),
                        Patch(facecolor=self.colors['alu'], label='Vector ALU'),
                        Patch(facecolor=self.colors['reduction'], label='Reduction Unit')]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {save_name}")
    
    # ===== Combined Report Charts =====
    
    def generate_full_report(self, data: Dict[str, Any], save_prefix: str = "report"):
        """
        Tạo tất cả các biểu đồ cho báo cáo
        
        Args:
            data: Dictionary chứa tất cả dữ liệu
            save_prefix: Prefix cho tên file
        """
        # 1. Cycles comparison
        if 'scalar_cycles' in data and 'vector_cycles' in data and 'pipeline_cycles' in data:
            self.plot_cycles_comparison(
                data['scalar_cycles'], data['vector_cycles'], data['pipeline_cycles'],
                f"{save_prefix}_cycles_comparison.png"
            )
        
        # 2. Speedup comparison
        if 'vector_speedup' in data and 'pipeline_speedup' in data and 'ideal_speedup' in data:
            self.plot_speedup_comparison(
                data['vector_speedup'], data['pipeline_speedup'], data['ideal_speedup'],
                f"{save_prefix}_speedup_comparison.png"
            )
        
        # 3. Vector length effect
        if 'vector_lengths' in data and 'cycles_vs_length' in data and 'speedups_vs_length' in data:
            self.plot_vector_length_effect(
                data['vector_lengths'], data['cycles_vs_length'], data['speedups_vs_length'],
                f"{save_prefix}_vector_length_effect.png"
            )
        
        # 4. Lanes effect
        if 'lanes' in data and 'speedups_vs_lanes' in data and 'ideal_speedups_vs_lanes' in data:
            self.plot_lanes_effect(
                data['lanes'], data.get('cycles_vs_lanes', data['speedups_vs_lanes']),
                data['speedups_vs_lanes'], data['ideal_speedups_vs_lanes'],
                f"{save_prefix}_lanes_effect.png"
            )
        
        # 5. Memory bandwidth effect
        if 'bandwidths' in data and 'cycles_vs_bandwidth' in data and 'speedups_vs_bandwidth' in data:
            self.plot_memory_bandwidth_effect(
                data['bandwidths'], data['cycles_vs_bandwidth'], data['speedups_vs_bandwidth'],
                f"{save_prefix}_bandwidth_effect.png"
            )
        
        # 6. Workload comparison
        if 'workloads' in data and 'scalar_cycles_workloads' in data:
            self.plot_workload_comparison(
                data['workloads'], data['scalar_cycles_workloads'],
                data.get('vector_cycles_workloads', data['scalar_cycles_workloads']),
                data.get('pipeline_cycles_workloads', data['scalar_cycles_workloads']),
                f"{save_prefix}_workload_comparison.png"
            )
        
        # 7. Timeline
        if 'timeline' in data and data['timeline']:
            self.plot_timeline(data['timeline'], f"{save_prefix}_timeline.png")
        
        print(f"\n✓ All charts saved to {self.output_dir}/")


# ===== TEST =====
if __name__ == "__main__":
    # Tạo dữ liệu test
    visualizer = PerformanceVisualizer()
    
    # Test basic charts
    visualizer.plot_cycles_comparison(880, 200, 160, "test_cycles.png")
    visualizer.plot_speedup_comparison(4.4, 5.5, 8, "test_speedup.png")
    visualizer.plot_throughput_comparison(0.4, 0.5, "test_throughput.png")
    
    # Test vector length effect
    lengths = [4, 8, 16, 32, 64]
    cycles = [88, 160, 300, 580, 1100]
    speedups = [1.1, 2.2, 4.0, 5.5, 6.8]
    visualizer.plot_vector_length_effect(lengths, cycles, speedups, "test_vl_effect.png")
    
    # Test lanes effect
    lanes_list = [1, 2, 4, 8, 16]
    speedups_actual = [1.0, 1.9, 3.5, 5.0, 6.2]
    speedups_ideal = [1, 2, 4, 8, 16]
    visualizer.plot_lanes_effect(lanes_list, speedups_actual, speedups_actual, speedups_ideal, "test_lanes_effect.png")
    
    # Test memory bandwidth effect
    bandwidths = [1, 2, 4, 8, 16]
    bw_cycles = [500, 350, 200, 150, 130]
    bw_speedups = [1.0, 1.4, 2.5, 3.3, 3.8]
    visualizer.plot_memory_bandwidth_effect(bandwidths, bw_cycles, bw_speedups, "test_bw_effect.png")
    
    # Test workload comparison
    workloads = ['Vector Add', 'SAXPY', 'Dot Product', 'Matrix-Vector', 'Image Filter']
    scalar_cycles = [88, 176, 132, 528, 264]
    vector_cycles = [20, 40, 30, 120, 60]
    pipeline_cycles = [16, 32, 25, 100, 50]
    visualizer.plot_workload_comparison(workloads, scalar_cycles, vector_cycles, pipeline_cycles, "test_workloads.png")
    
    # Test lane utilization
    utilizations = [0.5, 0.75, 0.9, 0.95, 0.98]
    visualizer.plot_lane_utilization(lengths, utilizations, "test_lane_util.png")
    
    # Test timeline
    timeline = [
        {"instruction": "VLOAD V1, A", "start": 0, "duration": 5, "unit": "Memory"},
        {"instruction": "VLOAD V2, B", "start": 5, "duration": 5, "unit": "Memory"},
        {"instruction": "VADD V3, V1, V2", "start": 10, "duration": 5, "unit": "Vector ALU"},
        {"instruction": "VSTORE C, V3", "start": 15, "duration": 5, "unit": "Memory"},
        {"instruction": "VMUL V4, V1, V2", "start": 20, "duration": 7, "unit": "Vector ALU"},
        {"instruction": "VREDUCE_SUM S, V4", "start": 27, "duration": 4, "unit": "Reduction Unit"},
    ]
    visualizer.plot_timeline(timeline, "test_timeline.png")
    
    # Test utilization pie
    visualizer.plot_utilization_pie(45, 60, 25, "test_util_pie.png")
    
    print("\n✓ All test charts generated successfully!")
