#!/usr/bin/env python3
"""
优化版批量处理转录文件的便捷脚本
大幅减少token消耗和处理时间 - 真正的批量处理
"""

import os
import sys
import glob
from pathlib import Path

def show_menu():
    """显示操作菜单"""
    print("\n" + "="*60)
    print("🚀 语音转录文本批量纠错工具 (大批次优化版)")
    print("="*60)
    print()
    print("请选择处理方式:")
    print()
    print("1. 📁 处理当前目录下所有txt文件 (推荐批量模式)")
    print("2. 📂 处理指定目录(包含子目录)")
    print("3. 🔍 使用通配符模式 (如 *.txt)")
    print("4. 📄 处理单个文件")
    print("5. 🔧 自定义高级选项")
    print("6. 🧪 测试API连接")
    print("7. 📊 预估处理成本 (已优化)")
    print("0. 退出")
    print()

def get_processing_mode():
    """获取处理模式选择"""
    print("处理模式:")
    print("1. 📊 完整模式 (推荐) - 生成详细报告和修正文件")
    print("2. ⚡ 快速模式 - 只生成修正文件，速度更快")
    
    while True:
        mode = input("选择模式 (1/2，回车默认完整模式): ").strip()
        if not mode:
            mode = "1"
        
        if mode in ["1", "2"]:
            return mode
        else:
            print("❌ 请输入 1 或 2")

def estimate_processing_cost(files):
    """
    预估处理成本和时间 - 基于大批次优化
    """
    if not files:
        return
    
    print(f"\n📊 处理成本预估 (大批次优化版)")
    print("-" * 50)
    
    total_files = len(files)
    
    # 统计文件大小和段落数
    total_size = 0
    estimated_segments = 0
    sample_files = files[:min(10, len(files))]  # 采样分析
    
    for file in sample_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
                file_size = len(content)
                total_size += file_size
                
                # 改进的段落数估算
                lines = content.strip().split('\n')
                segments = 0
                for line in lines:
                    line = line.strip()
                    if (line and 
                        not line.startswith('=') and 
                        not line.startswith('-') and
                        not line.startswith('chat-') and
                        len(line) > 10):
                        if ('发言人' in line and ':' in line) or '[' in line or len(line) > 20:
                            segments += 1
                
                estimated_segments += segments
        except:
            continue
    
    # 推算总体
    if len(sample_files) > 0:
        avg_segments = estimated_segments / len(sample_files)
        total_estimated_segments = int(avg_segments * total_files)
    else:
        total_estimated_segments = 0
    
    if total_estimated_segments == 0:
        print("❌ 无法估算段落数，请检查文件格式")
        return
    
    # 基于大批次优化的全新预估模型
    batch_size = 30  # 每批处理30个段落
    
    # 优化后的处理分布估算
    quick_fixes_estimated = int(total_estimated_segments * 0.08)    # 快速修正8%
    pre_filter_estimated = int(total_estimated_segments * 0.50)     # 预过滤跳过50%
    api_segments = total_estimated_segments - quick_fixes_estimated - pre_filter_estimated  # 需要API处理的段落
    
    # 批次计算
    api_batches = max(1, (api_segments + batch_size - 1) // batch_size)  # 向上取整
    
    # Token估算 - 大批次优化模型
    base_prompt_tokens = 180  # 优化后的prompt token数
    tokens_per_segment = 25   # 每个段落的平均token数（输入）
    response_tokens_per_segment = 30  # 每个段落的响应token数（更精准）
    
    # 大批次处理的token计算
    total_input_tokens = 0
    total_output_tokens = 0
    
    for batch_idx in range(api_batches):
        segments_in_batch = min(batch_size, api_segments - batch_idx * batch_size)
        
        # 每批的输入token = 基础prompt + 所有段落文本
        batch_input_tokens = base_prompt_tokens + (segments_in_batch * tokens_per_segment)
        batch_output_tokens = segments_in_batch * response_tokens_per_segment
        
        total_input_tokens += batch_input_tokens
        total_output_tokens += batch_output_tokens
    
    total_tokens_estimated = total_input_tokens + total_output_tokens
    
    # 时间估算 - 优化版
    api_batch_time = api_batches * 3.0   # 每个大批次约3秒（包含网络延迟）
    quick_fix_time = quick_fixes_estimated * 0.005  # 快速修正更快
    pre_filter_time = pre_filter_estimated * 0.001   # 预过滤极快
    total_time = api_batch_time + quick_fix_time + pre_filter_time
    
    print(f"文件总数: {total_files}")
    print(f"预估段落数: {total_estimated_segments:,}")
    print()
    print("大批次优化处理分布:")
    print(f"  API批次数: {api_batches} (每批{batch_size}个段落)")
    print(f"  API处理段落: {api_segments:,} ({api_segments/total_estimated_segments*100:.1f}%)")
    print(f"  快速修正: {quick_fixes_estimated:,} ({quick_fixes_estimated/total_estimated_segments*100:.1f}%)")
    print(f"  预过滤跳过: {pre_filter_estimated:,} ({pre_filter_estimated/total_estimated_segments*100:.1f}%)")
    print()
    print(f"Token消耗分析:")
    print(f"  输入Token: {total_input_tokens:,}")
    print(f"  输出Token: {total_output_tokens:,}")
    print(f"  总Token: {total_tokens_estimated:,}")
    print(f"预估处理时间: {total_time/60:.1f} 分钟")
    
    # 成本估算（基于GLM-4.5的定价）
    input_cost_per_1k = 0.0005  # GLM-4.5输入定价
    output_cost_per_1k = 0.002  # GLM-4.5输出定价
    estimated_cost = (total_input_tokens / 1000) * input_cost_per_1k + (total_output_tokens / 1000) * output_cost_per_1k
    print(f"预估成本: ¥{estimated_cost:.3f}")
    
    # 与之前版本对比
    original_api_calls = int(total_estimated_segments * 0.42)  # 原版API调用率更高
    original_tokens = original_api_calls * 1000  # 原版每次调用约1000 tokens
    original_cost = (original_tokens / 1000) * 0.002
    
    print()
    print("🔥 大批次优化效果对比:")
    print(f"  原版API调用: {original_api_calls} 次")
    print(f"  优化版API调用: {api_batches} 次")
    print(f"  调用次数减少: {(1 - api_batches/max(original_api_calls, 1))*100:.1f}%")
    print(f"  Token减少: {(1 - total_tokens_estimated/max(original_tokens, 1))*100:.1f}%")
    print(f"  成本减少: {(1 - estimated_cost/max(original_cost, 0.001))*100:.1f}%")
    print(f"  时间减少: ~90-95%")
    
    # 给出建议
    if total_files > 20:
        print(f"\n💡 建议:")
        print(f"  • 大批次处理已高度优化，建议直接运行")
        print(f"  • 推荐使用完整模式以获得详细报告")
        print(f"  • 可先处理小批量文件验证效果")
    
    if total_estimated_segments > 1000:
        print(f"  • 段落较多，但大批次处理已大幅优化时间和成本")
        print(f"  • 预计处理时间：{total_time/60:.1f}分钟，成本：¥{estimated_cost:.3f}")

def get_file_count(pattern, recursive=False):
    """获取匹配文件的数量"""
    if os.path.isfile(pattern):
        return 1
    elif os.path.isdir(pattern):
        search_pattern = "**/*.txt" if recursive else "*.txt"
        search_path = os.path.join(pattern, search_pattern)
        files = glob.glob(search_path, recursive=recursive)
        # 过滤掉已处理的文件
        files = [f for f in files if not (f.endswith('_corrected.txt') or f.endswith('_report.txt'))]
        return len(files)
    else:
        files = glob.glob(pattern)
        files = [f for f in files if f.endswith('.txt') and not (f.endswith('_corrected.txt') or f.endswith('_report.txt'))]
        return len(files)

def get_files_list(pattern, recursive=False):
    """获取文件列表"""
    files = []
    
    if os.path.isfile(pattern):
        files.append(pattern)
    elif os.path.isdir(pattern):
        search_pattern = "**/*.txt" if recursive else "*.txt"
        search_path = os.path.join(pattern, search_pattern)
        files.extend(glob.glob(search_path, recursive=recursive))
    else:
        files.extend(glob.glob(pattern))
    
    # 过滤
    valid_files = [f for f in files if f.endswith('.txt') and not (f.endswith('_corrected.txt') or f.endswith('_report.txt'))]
    return valid_files

def run_processing(command):
    """执行处理命令"""
    print(f"\n🚀 执行命令: {command}")
    print("-" * 60)
    
    # 添加确认
    confirm = input("是否开始处理？(y/N): ").lower().strip()
    if confirm not in ['y', 'yes']:
        print("❌ 取消操作")
        return
        
    # 执行命令
    exit_code = os.system(command)
    
    if exit_code == 0:
        print("\n🎉 处理完成！")
    else:
        print(f"\n❌ 处理失败，退出码: {exit_code}")

def main():
    """主函数"""
    while True:
        show_menu()
        
        try:
            choice = input("请选择操作 (0-7): ").strip()
            
            if choice == '0':
                print("👋 再见！")
                break
                
            elif choice == '1':
                # 处理当前目录 - 大批次优化模式
                file_count = get_file_count(".")
                if file_count == 0:
                    print("❌ 当前目录下没有找到txt文件")
                    continue
                
                print(f"📊 找到 {file_count} 个文件")
                
                # 处理模式选择
                mode = get_processing_mode()
                
                print("大批次处理已高度优化，无需额外并行配置")
                
                if mode == '2':
                    # 快速模式 - 只生成修正文件
                    command = f"python main.py . --only-correct --continue-on-error"
                else:
                    # 完整模式 - 生成报告和修正文件
                    command = f"python main.py . --correct --continue-on-error"
                
                run_processing(command)
                
            elif choice == '2':
                # 处理指定目录
                directory = input("请输入目录路径: ").strip()
                if not directory:
                    print("❌ 请输入有效的目录路径")
                    continue
                    
                if not os.path.isdir(directory):
                    print("❌ 目录不存在")
                    continue
                
                file_count = get_file_count(directory, recursive=True)
                if file_count == 0:
                    print("❌ 指定目录下没有找到txt文件")
                    continue
                
                print(f"📊 找到 {file_count} 个文件 (包含子目录)")
                
                # 处理模式
                mode = get_processing_mode()
                
                if mode == '2':
                    command = f'python main.py "{directory}" --only-correct --recursive --continue-on-error'
                else:
                    command = f'python main.py "{directory}" --correct --recursive --continue-on-error'
                
                run_processing(command)
                
            elif choice == '3':
                # 通配符模式
                pattern = input("请输入通配符模式 (如 *.txt 或 chat-*.txt): ").strip()
                if not pattern:
                    print("❌ 请输入有效的通配符模式")
                    continue
                
                file_count = get_file_count(pattern)
                if file_count == 0:
                    print("❌ 没有找到匹配的文件")
                    continue
                
                print(f"📊 找到 {file_count} 个匹配文件")
                
                # 预览文件
                preview = input("是否预览匹配的文件？(y/N): ").lower().strip()
                if preview in ['y', 'yes']:
                    os.system(f'python main.py "{pattern}" --dry-run')
                
                # 处理模式
                mode = get_processing_mode()
                
                if mode == '2':
                    command = f'python main.py "{pattern}" --only-correct --continue-on-error'
                else:
                    command = f'python main.py "{pattern}" --correct --continue-on-error'
                
                run_processing(command)
                
            elif choice == '4':
                # 单个文件
                file_path = input("请输入文件路径: ").strip()
                if not file_path:
                    print("❌ 请输入有效的文件路径")
                    continue
                    
                if not os.path.isfile(file_path):
                    print("❌ 文件不存在")
                    continue
                
                # 处理模式
                mode = get_processing_mode()
                
                if mode == '2':
                    command = f'python main.py "{file_path}" --only-correct'
                else:
                    command = f'python main.py "{file_path}" --correct'
                
                run_processing(command)
                
            elif choice == '5':
                # 自定义高级选项
                print("\n🔧 自定义高级选项")
                print("推荐的大批次优化命令模板:")
                print("python main.py \"*.txt\" --correct --continue-on-error  # 完整模式")
                print("python main.py \"*.txt\" --only-correct --continue-on-error  # 快速模式")
                print("python main.py transcripts/ --recursive --correct  # 目录完整模式")
                print()
                
                custom_command = input("请输入命令: ").strip()
                if not custom_command:
                    print("❌ 请输入有效的命令")
                    continue
                
                if not custom_command.startswith('python main.py'):
                    print("❌ 命令必须以 'python main.py' 开头")
                    continue
                
                run_processing(custom_command)
                
            elif choice == '6':
                # 测试连接
                print("\n🔍 测试API连接...")
                test_file = "test.txt"
                if not os.path.exists(test_file):
                    with open("temp_test.txt", "w", encoding="utf-8") as f:
                        f.write("发言人1 00:01\n这是测试内容，我觉的应该没有错误。")
                    test_file = "temp_test.txt"
                
                os.system(f'python main.py "{test_file}" --test-connection')
                
                if test_file == "temp_test.txt":
                    try:
                        os.remove("temp_test.txt")
                    except:
                        pass
                    
            elif choice == '7':
                # 预估处理成本
                pattern = input("请输入要分析的文件模式 (如 *.txt 或目录路径): ").strip()
                if not pattern:
                    pattern = "."
                
                files = get_files_list(pattern, recursive=True)
                if files:
                    estimate_processing_cost(files)
                else:
                    print("❌ 未找到匹配的文件")
                
            else:
                print("❌ 无效的选择，请输入 0-7 之间的数字")
                
        except KeyboardInterrupt:
            print("\n⚠️  用户中断操作")
            break
        except Exception as e:
            print(f"❌ 操作失败: {e}")
        
        input("\n按回车键继续...")

if __name__ == "__main__":
    # 检查main.py是否存在
    if not os.path.exists("main.py"):
        print("❌ 错误: 未找到 main.py 文件")
        print("请确保在包含 main.py 的目录下运行此脚本")
        sys.exit(1)
    
    main()