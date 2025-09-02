#!/usr/bin/env python3
"""
批量处理转录文件的便捷脚本
简化常用的批量处理操作
"""

import os
import sys
import glob
from pathlib import Path

def show_menu():
    """显示操作菜单"""
    print("\n" + "="*60)
    print("🚀 语音转录文本批量纠错工具")
    print("="*60)
    print("请选择处理方式:")
    print()
    print("1. 📁 处理当前目录下所有txt文件")
    print("2. 📂 处理指定目录(包含子目录)")
    print("3. 🔍 使用通配符模式 (如 *.txt)")
    print("4. 📄 处理单个文件")
    print("5. 🔧 自定义高级选项")
    print("6. 🧪 测试API连接")
    print("0. 退出")
    print()

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
            choice = input("请选择操作 (0-6): ").strip()
            
            if choice == '0':
                print("👋 再见！")
                break
                
            elif choice == '1':
                # 处理当前目录
                file_count = get_file_count(".")
                if file_count == 0:
                    print("❌ 当前目录下没有找到txt文件")
                    continue
                
                print(f"📊 找到 {file_count} 个文件")
                
                # 选择处理模式
                print("处理模式:")
                print("1. 生成报告和修正文件")
                print("2. 只生成修正文件(更快)")
                mode = input("选择模式 (1/2): ").strip()
                
                # 选择线程数
                threads = input(f"并行线程数 (建议2-8，回车使用4): ").strip()
                threads = threads if threads else "4"
                
                if mode == '2':
                    command = f"python main.py . --only-correct --parallel {threads} --continue-on-error"
                else:
                    command = f"python main.py . --correct --parallel {threads} --continue-on-error"
                
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
                
                # 选择处理模式
                print("处理模式:")
                print("1. 生成报告和修正文件")
                print("2. 只生成修正文件(更快)")
                mode = input("选择模式 (1/2): ").strip()
                
                # 选择线程数
                threads = input(f"并行线程数 (建议2-8，回车使用4): ").strip()
                threads = threads if threads else "4"
                
                if mode == '2':
                    command = f'python main.py "{directory}" --only-correct --recursive --parallel {threads} --continue-on-error'
                else:
                    command = f'python main.py "{directory}" --correct --recursive --parallel {threads} --continue-on-error'
                
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
                
                # 选择处理模式
                print("处理模式:")
                print("1. 生成报告和修正文件")
                print("2. 只生成修正文件(更快)")
                mode = input("选择模式 (1/2): ").strip()
                
                # 选择线程数
                threads = input(f"并行线程数 (建议2-8，回车使用4): ").strip()
                threads = threads if threads else "4"
                
                if mode == '2':
                    command = f'python main.py "{pattern}" --only-correct --parallel {threads} --continue-on-error'
                else:
                    command = f'python main.py "{pattern}" --correct --parallel {threads} --continue-on-error'
                
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
                
                print("处理模式:")
                print("1. 生成报告和修正文件")
                print("2. 只生成修正文件")
                mode = input("选择模式 (1/2): ").strip()
                
                if mode == '2':
                    command = f'python main.py "{file_path}" --only-correct'
                else:
                    command = f'python main.py "{file_path}" --correct'
                
                run_processing(command)
                
            elif choice == '5':
                # 自定义高级选项
                print("\n🔧 自定义高级选项")
                print("请手动输入完整的命令，例如:")
                print("python main.py \"*.txt\" --correct --parallel 8 --continue-on-error")
                print("python main.py transcripts/ --recursive --only-correct --parallel 4")
                print()
                
                custom_command = input("请输入命令: ").strip()
                if not custom_command:
                    print("❌ 请输入有效的命令")
                    continue
                
                # 检查命令格式
                if not custom_command.startswith('python main.py'):
                    print("❌ 命令必须以 'python main.py' 开头")
                    continue
                
                run_processing(custom_command)
                
            elif choice == '6':
                # 测试连接
                print("\n🔍 测试API连接...")
                test_file = "1.txt"  # 假设存在测试文件
                if not os.path.exists(test_file):
                    # 创建一个临时测试文件
                    with open("temp_test.txt", "w", encoding="utf-8") as f:
                        f.write("发言人1 00:01\n这是测试内容。")
                    test_file = "temp_test.txt"
                
                os.system(f'python main.py "{test_file}" --test-connection')
                
                if test_file == "temp_test.txt":
                    os.remove("temp_test.txt")
                
            else:
                print("❌ 无效的选择，请输入 0-6 之间的数字")
                
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