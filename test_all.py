#!/usr/bin/env python3
"""Test script để kiểm tra toàn bộ project"""

import sys
import os
import io

# Thêm thư mục hiện tại vào path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _ensure_utf8_stdout():
    """Tránh UnicodeEncodeError trên Windows (cp1252/cp1258) khi in emoji."""
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        try:
            reconfigure(encoding="utf-8", errors="replace")
            return
        except Exception:
            pass
    buffer = getattr(sys.stdout, "buffer", None)
    if buffer is not None:
        sys.stdout = io.TextIOWrapper(buffer, encoding="utf-8", errors="replace")

def test_imports():
    print("\n📦 Testing imports...")
    modules = [
        'config', 'instruction', 'parser', 'memory', 
        'vector_register_file', 'scalar_simulator', 
        'vector_simulator', 'pipeline_simulator',
        'performance_analyzer', 'report_generator', 
        'visualization', 'convolution_simulator',
        'cache', 'ooo_simulator', 'comparison'
    ]
    for mod in modules:
        try:
            __import__(mod)
            print(f"  ✅ {mod}")
        except Exception as e:
            print(f"  ❌ {mod}: {e}")
            return False
    return True

def test_config():
    print("\n⚙️ Testing Config...")
    try:
        from config import VectorProcessorConfig
        c = VectorProcessorConfig(vector_length=16, num_lanes=8)
        assert c.vector_length == 16
        assert c.num_lanes == 8
        print("  ✅ Config OK")
        return True
    except Exception as e:
        print(f"  ❌ Config: {e}")
        return False

def test_parser():
    print("\n📝 Testing Parser...")
    try:
        from parser import ProgramParser
        p = ProgramParser()
        data, scalars, insts = p.parse_file('examples/vector_add.txt')
        assert len(insts) == 4
        assert 'A' in data
        print("  ✅ Parser OK")
        return True
    except Exception as e:
        print(f"  ❌ Parser: {e}")
        return False

def test_vector_simulator():
    print("\n🖥️ Testing Vector Simulator...")
    try:
        from config import VectorProcessorConfig
        from parser import ProgramParser
        from vector_simulator import VectorSimulator
        
        config = VectorProcessorConfig()
        parser = ProgramParser()
        data, scalars, insts = parser.parse_file('examples/vector_add.txt')
        sim = VectorSimulator(config, data, scalars)
        mem, timeline, cycles = sim.run(insts)
        
        assert 'C' in mem
        assert mem['C'][0] == 11
        assert cycles > 0
        print(f"  ✅ Vector Simulator: {cycles} cycles")
        return True
    except Exception as e:
        print(f"  ❌ Vector Simulator: {e}")
        return False

def test_pipeline_simulator():
    print("\n⏱️ Testing Pipeline Simulator...")
    try:
        from config import VectorProcessorConfig
        from parser import ProgramParser
        from pipeline_simulator import PipelineSimulator
        
        config = VectorProcessorConfig()
        parser = ProgramParser()
        data, scalars, insts = parser.parse_file('examples/vector_add.txt')
        sim = PipelineSimulator(config, data, scalars)
        mem, timeline, cycles = sim.run(insts)
        
        assert cycles > 0
        print(f"  ✅ Pipeline Simulator: {cycles} cycles")
        return True
    except Exception as e:
        print(f"  ❌ Pipeline Simulator: {e}")
        return False

def test_performance():
    print("\n📊 Testing Performance Analyzer...")
    try:
        from config import VectorProcessorConfig
        from performance_analyzer import PerformanceAnalyzer
        
        config = VectorProcessorConfig()
        pa = PerformanceAnalyzer(config)
        speedup = pa.calculate_speedup(100, 20)
        assert speedup == 5.0
        print("  ✅ Performance Analyzer OK")
        return True
    except Exception as e:
        print(f"  ❌ Performance Analyzer: {e}")
        return False

def test_multiline_parser():
    print("\n📄 Testing Multiline Parser...")
    try:
        from parser import ProgramParser
        p = ProgramParser()
        data, scalars, insts = p.parse_file('examples/convolution_2d.txt')
        assert len(data['IMAGE']) == 9
        assert len(data['KERNEL']) == 9
        print("  ✅ Multiline Parser OK")
        return True
    except Exception as e:
        print(f"  ❌ Multiline Parser: {e}")
        return False

def test_convolution():
    print("\n🧠 Testing Convolution Simulator...")
    try:
        from config import VectorProcessorConfig
        from convolution_simulator import ConvolutionSimulator
        
        config = VectorProcessorConfig()
        result = ConvolutionSimulator.analyze_convolution_performance(64, 3, config)
        assert result['speedup'] > 0
        print(f"  ✅ Convolution Simulator: speedup={result['speedup']:.2f}x")
        return True
    except Exception as e:
        print(f"  ❌ Convolution Simulator: {e}")
        return False

def main():
    _ensure_utf8_stdout()
    print("=" * 60)
    print("🧪 VECTOR PROCESSOR - FULL TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Config", test_config),
        ("Parser", test_parser),
        ("Multiline Parser", test_multiline_parser),
        ("Vector Simulator", test_vector_simulator),
        ("Pipeline Simulator", test_pipeline_simulator),
        ("Performance Analyzer", test_performance),
        ("Convolution Simulator", test_convolution),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        if test_func():
            passed += 1
        else:
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"📈 RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED! Project is ready for submission.")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")

    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
