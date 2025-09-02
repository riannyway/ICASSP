#!/usr/bin/env python3
"""
语音转录文本纠错系统测试脚本
用于验证系统各个组件是否正常工作
"""

import os
import sys
import tempfile
from config import Config
from glm_client import GLMClient
from text_processor import TextProcessor
from error_detector import ErrorDetector

def test_config():
    """测试配置加载"""
    print("🔧 测试配置加载...")
    try:
        api_key = Config.GLM_API_KEY
        base_url = Config.GLM_BASE_URL
        print(f"✅ API密钥: {api_key[:8]}...{api_key[-4:] if len(api_key) > 12 else api_key}")
        print(f"✅ API地址: {base_url}")
        return True
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return False

def test_glm_client():
    """测试GLM客户端"""
    print("\n🤖 测试GLM客户端...")
    try:
        client = GLMClient()
        
        # 测试简单文本
        result = client.detect_and_correct_text_errors("这是一个测试文本，没有错误。")
        
        if 'error' in result:
            print(f"❌ API调用失败: {result['error']}")
            return False
        else:
            print("✅ API调用成功")
            print(f"   原文: {result.get('original_text', '')}")
            print(f"   修正: {result.get('corrected_text', '')}")
            print(f"   有错误: {result.get('has_errors', False)}")
            return True
    except Exception as e:
        print(f"❌ GLM客户端测试失败: {e}")
        return False

def test_text_processor():
    """测试文本处理器"""
    print("\n📝 测试文本处理器...")
    try:
        processor = TextProcessor()
        
        # 创建测试文件
        test_content = """发言人1 04:49
这是第一段测试内容，包含一些可能的错误文字。

发言人2 05:13
我觉的这个方案不错，我们应该仔细考虑一下。

发言人1 05:30
好的，那我们在会议上讨论一下吧。
"""
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
            f.write(test_content)
            test_file = f.name
        
        # 解析文件
        segments = processor.parse_transcription_file(test_file)
        
        print(f"✅ 成功解析 {len(segments)} 个段落")
        for i, segment in enumerate(segments[:3]):
            print(f"   段落{i+1}: {segment['speaker']} ({segment['timestamp']}) - {segment['text'][:30]}...")
        
        # 清理测试文件
        os.unlink(test_file)
        return True
        
    except Exception as e:
        print(f"❌ 文本处理器测试失败: {e}")
        return False

def test_error_detector():
    """测试错误检测器"""
    print("\n🔍 测试错误检测器...")
    try:
        detector = ErrorDetector()
        
        # 创建测试文件
        test_content = """发言人1 04:49
老师说我们应该好好学习，天天向上，但是我觉的有些困难。

发言人2 05:13
我同意你的看法，学习确实需要持之以恒的努力。
"""
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
            f.write(test_content)
            test_file = f.name
        
        # 处理文件
        print("   正在处理测试文件...")
        report_path, corrected_path = detector.detect_and_correct_file(test_file)
        
        if os.path.exists(report_path) and os.path.exists(corrected_path):
            print("✅ 错误检测器工作正常")
            print(f"   报告文件: {report_path}")
            print(f"   修正文件: {corrected_path}")
            
            # 清理测试文件
            os.unlink(test_file)
            return True
        else:
            print("❌ 输出文件未生成")
            return False
        
    except Exception as e:
        print(f"❌ 错误检测器测试失败: {e}")
        return False

def test_file_formats():
    """测试不同的文件格式支持"""
    print("\n📄 测试不同文件格式支持...")
    processor = TextProcessor()
    
    formats = {
        "发言人+时间戳": """发言人1 04:49
这是测试内容

发言人2 05:13
这是另一段内容""",
        
        "时间戳+内容": """[00:04:49] 这是测试内容
[00:05:13] 这是另一段内容""",
        
        "时间戳+发言人+内容": """[00:04:49-00:05:13] 张三: 这是测试内容
[00:05:13-00:05:30] 李四: 这是另一段内容"""
    }
    
    all_passed = True
    for format_name, content in formats.items():
        try:
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
                f.write(content)
                test_file = f.name
            
            segments = processor.parse_transcription_file(test_file)
            
            if len(segments) > 0:
                print(f"✅ {format_name}: 解析出 {len(segments)} 个段落")
            else:
                print(f"❌ {format_name}: 解析失败")
                all_passed = False
                
            os.unlink(test_file)
            
        except Exception as e:
            print(f"❌ {format_name}: 测试失败 - {e}")
            all_passed = False
    
    return all_passed

def main():
    """主测试函数"""
    print("🚀 开始系统测试...\n")
    
    tests = [
        ("配置加载", test_config),
        ("GLM客户端", test_glm_client),
        ("文本处理器", test_text_processor),
        ("文件格式支持", test_file_formats),
        ("错误检测器", test_error_detector)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} 测试失败")
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")
    
    print(f"\n📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！系统正常工作。")
        return True
    else:
        print("⚠️  部分测试失败，请检查配置和网络连接。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)