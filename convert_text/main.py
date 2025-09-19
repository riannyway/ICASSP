import argparse
import os
import sys
import glob
import time
import random
from pathlib import Path
from error_detector import ErrorDetector
from config import Config

def find_transcript_files(input_path: str, recursive: bool = False) -> list:
    """
    查找转录文件，支持文件、目录和通配符
    """
    files = []
    
    if os.path.isfile(input_path):
        # 单个文件
        files.append(input_path)
    elif os.path.isdir(input_path):
        # 目录处理
        pattern = "**/*.txt" if recursive else "*.txt"
        search_path = os.path.join(input_path, pattern)
        files.extend(glob.glob(search_path, recursive=recursive))
    else:
        # 通配符匹配
        files.extend(glob.glob(input_path))
    
    # 过滤掉非文本文件和输出文件
    valid_files = []
    for file in files:
        if (file.endswith('.txt') and 
            not file.endswith('_corrected.txt') and 
            not file.endswith('_report.txt') and
            os.path.isfile(file)):
            valid_files.append(file)
    
    return valid_files

def configure_high_api_usage(detector: ErrorDetector, mode: str = 'high'):
    """
    配置高API使用量模式
    """
    if not hasattr(detector, 'glm_client'):
        print("⚠️ 检测器没有GLM客户端，跳过高API配置")
        return
    
    # 高API使用配置预设
    api_configs = {
        'maximum': {
            'api_usage_ratio': 0.95,
            'max_tokens': 500,
            'use_detailed_prompts': True,
            'min_text_length': 2,
            'description': '最大API使用模式 - 预期30K+ tokens'
        },
        'high': {
            'api_usage_ratio': 0.80,
            'max_tokens': 400,
            'use_detailed_prompts': True,
            'min_text_length': 5,
            'description': '高API使用模式 - 预期20K+ tokens'
        },
        'medium': {
            'api_usage_ratio': 0.50,
            'max_tokens': 300,
            'use_detailed_prompts': False,
            'min_text_length': 8,
            'description': '中等API使用模式 - 预期10K+ tokens'
        }
    }
    
    config = api_configs.get(mode, api_configs['high'])
    
    # 应用配置到GLM客户端
    detector.glm_client.high_api_mode = True
    detector.glm_client.api_usage_ratio = config['api_usage_ratio']
    detector.glm_client.max_tokens = config['max_tokens']
    detector.glm_client.use_detailed_prompts = config['use_detailed_prompts']
    detector.glm_client.min_text_length = config['min_text_length']
    
    print(f"✅ 已启用 {mode.upper()} 模式: {config['description']}")
    print(f"   API使用率: {config['api_usage_ratio']*100:.0f}%")
    print(f"   每次最大token: {config['max_tokens']}")
    print(f"   详细prompt: {config['use_detailed_prompts']}")

def process_single_file(detector: ErrorDetector, file_path: str, args) -> dict:
    """
    处理单个文件 - 修复版本
    """
    try:
        print(f"\n📄 处理文件: {file_path}")
        start_time = time.time()
        
        if args.only_correct:
            # 只生成修正文件
            corrected_path = detector.detect_and_correct_file_only_correct(file_path)
            report_path = None
        else:
            # 生成报告和修正文件
            report_path, corrected_path = detector.detect_and_correct_file(file_path)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        return {
            'file': file_path,
            'status': 'success',
            'report_path': report_path,
            'corrected_path': corrected_path,
            'processing_time': processing_time,
            'error': None
        }
        
    except Exception as e:
        return {
            'file': file_path,
            'status': 'error',
            'report_path': None,
            'corrected_path': None,
            'processing_time': 0,
            'error': str(e)
        }

def generate_batch_summary(results: list, output_dir: str) -> str:
    """
    生成批量处理总结报告
    """
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(output_dir, f"batch_summary_{timestamp}.txt")
    
    total_files = len(results)
    successful_files = len([r for r in results if r['status'] == 'success'])
    failed_files = total_files - successful_files
    total_time = sum(r['processing_time'] for r in results)
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("批量处理总结报告 (修复优化版)\n")
        f.write("=" * 70 + "\n")
        f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总文件数: {total_files}\n")
        f.write(f"成功处理: {successful_files}\n")
        f.write(f"处理失败: {failed_files}\n")
        f.write(f"成功率: {successful_files/total_files*100:.1f}%\n")
        f.write(f"总耗时: {total_time:.1f}秒\n")
        f.write(f"平均耗时: {total_time/total_files:.1f}秒/文件\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("详细处理结果\n")
        f.write("=" * 70 + "\n\n")
        
        # 成功处理的文件
        if successful_files > 0:
            f.write("✅ 成功处理的文件:\n")
            f.write("-" * 50 + "\n")
            for result in results:
                if result['status'] == 'success':
                    f.write(f"文件: {result['file']}\n")
                    f.write(f"耗时: {result['processing_time']:.1f}秒\n")
                    if result['report_path']:
                        f.write(f"报告: {result['report_path']}\n")
                    f.write(f"修正: {result['corrected_path']}\n\n")
        
        # 失败处理的文件
        if failed_files > 0:
            f.write("\n❌ 处理失败的文件:\n")
            f.write("-" * 50 + "\n")
            for result in results:
                if result['status'] == 'error':
                    f.write(f"文件: {result['file']}\n")
                    f.write(f"错误: {result['error']}\n\n")
    
    return summary_path

def main():
    parser = argparse.ArgumentParser(description='语音转录文本错误检测与纠错工具 - 支持批量处理')
    parser.add_argument('input', help='输入文件/目录路径或通配符模式 (如 "*.txt" 或 "transcripts/")')
    parser.add_argument('--api-key', help='GLM API密钥（可选，优先使用环境变量）')
    parser.add_argument('--recursive', '-r', action='store_true', help='递归处理子目录中的文件')
    parser.add_argument('--correct', action='store_true', help='同时生成纠错版本')
    parser.add_argument('--only-correct', action='store_true', help='只生成纠错版本，不生成检测报告')
    parser.add_argument('--test-connection', action='store_true', help='测试API连接')
    parser.add_argument('--parallel', type=int, metavar='N', help='并行处理的线程数 (默认串行处理)')
    parser.add_argument('--continue-on-error', action='store_true', help='遇到错误时继续处理其他文件')
    parser.add_argument('--dry-run', action='store_true', help='预览模式：只显示要处理的文件，不实际处理')
    
    # 新增：API使用模式参数
    parser.add_argument('--api-mode', choices=['low', 'medium', 'high', 'maximum'], 
                       default='high', help='API使用模式 (默认: high)')
    
    args = parser.parse_args()
    
    try:
        # 初始化错误检测器
        api_key = args.api_key or Config.GLM_API_KEY
        detector = ErrorDetector(api_key)
        
        # 配置高API使用模式
        print(f"\n🚀 配置API使用模式...")
        configure_high_api_usage(detector, args.api_mode)
        
        # 测试连接
        if args.test_connection:
            print("\n🔍 测试API连接...")
            if detector.glm_client.test_connection():
                print("✅ API连接正常")
            else:
                print("❌ API连接失败，请检查密钥和网络")
                sys.exit(1)
            return
        
        # 查找要处理的文件
        print(f"\n🔍 查找转录文件: {args.input}")
        files = find_transcript_files(args.input, args.recursive)
        
        if not files:
            print(f"❌ 未找到匹配的转录文件: {args.input}")
            sys.exit(1)
        
        print(f"📋 找到 {len(files)} 个文件待处理")
        
        # 显示文件列表
        for i, file in enumerate(files, 1):
            print(f"  {i}. {file}")
        
        # 预览模式
        if args.dry_run:
            print("\n🔍 预览模式：以上文件将被处理")
            print("使用 --dry-run 参数只是预览，未实际处理文件")
            return
        
        # 确认处理
        if len(files) > 1:
            print(f"\n📊 批量处理统计:")
            print(f"   文件数量: {len(files)}")
            print(f"   处理模式: {'只生成修正文件' if args.only_correct else '生成报告和修正文件'}")
            print(f"   API模式: {args.api_mode.upper()}")
            print(f"   API密钥: {api_key[:8]}...{api_key[-4:]}")
            
            confirm = input("\n是否开始批量处理？(y/N): ").lower().strip()
            if confirm not in ['y', 'yes']:
                print("❌ 用户取消操作")
                return
        
        # 开始批量处理
        if len(files) > 1:
            print(f"\n🚀 开始批量处理 {len(files)} 个文件...")
        else:
            print(f"\n🚀 开始处理文件...")
        print("=" * 60)
        
        results = []
        
        # 并行处理 (如果指定)
        if args.parallel and args.parallel > 1:
            print(f"📄 使用 {args.parallel} 个线程并行处理...")
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            with ThreadPoolExecutor(max_workers=args.parallel) as executor:
                # 提交所有任务
                future_to_file = {
                    executor.submit(process_single_file, detector, file, args): file 
                    for file in files
                }
                
                # 收集结果
                for i, future in enumerate(as_completed(future_to_file), 1):
                    file = future_to_file[future]
                    print(f"📊 总进度: {i}/{len(files)} 完成")
                    
                    try:
                        result = future.result()
                        results.append(result)
                        
                        if result['status'] == 'success':
                            print(f"✅ {result['file']} - 处理成功")
                        else:
                            print(f"❌ {result['file']} - 处理失败: {result['error']}")
                            if not args.continue_on_error:
                                print("⚠️ 中止处理 (使用 --continue-on-error 继续处理其他文件)")
                                break
                                
                    except Exception as e:
                        print(f"❌ {file} - 处理异常: {str(e)}")
                        results.append({
                            'file': file, 'status': 'error', 'error': str(e),
                            'report_path': None, 'corrected_path': None, 'processing_time': 0
                        })
        else:
            # 串行处理
            for i, file in enumerate(files, 1):
                if len(files) > 1:
                    print(f"\n📊 进度: {i}/{len(files)}")
                
                result = process_single_file(detector, file, args)
                results.append(result)
                
                if result['status'] == 'success':
                    print(f"✅ 处理完成 ({result['processing_time']:.1f}秒)")
                    if result['report_path']:
                        print(f"📊 检测报告: {result['report_path']}")
                    print(f"📝 修正文件: {result['corrected_path']}")
                else:
                    print(f"❌ 处理失败: {result['error']}")
                    if not args.continue_on_error:
                        print("⚠️ 中止处理 (使用 --continue-on-error 继续处理其他文件)")
                        break
        
        # 生成批量处理总结
        if len(files) > 1:
            summary_path = generate_batch_summary(results, Config.OUTPUT_DIR)
            print(f"\n📈 批量处理总结: {summary_path}")
        
        # 最终统计
        successful = len([r for r in results if r['status'] == 'success'])
        failed = len(results) - successful
        total_time = sum(r['processing_time'] for r in results)
        
        print("\n" + "=" * 60)
        if len(files) > 1:
            print("🎉 批量处理完成！")
        else:
            print("🎉 处理完成！")
        print("=" * 60)
        print(f"📊 处理统计:")
        print(f"   总文件数: {len(results)}")
        print(f"   成功处理: {successful}")
        print(f"   处理失败: {failed}")
        print(f"   成功率: {successful/len(results)*100:.1f}%")
        print(f"   总耗时: {total_time:.1f}秒")
        if successful > 0:
            print(f"   平均耗时: {total_time/successful:.1f}秒/文件")
        
        # 显示输出目录
        if successful > 0:
            print(f"\n📁 输出文件保存在: {Config.OUTPUT_DIR}")
        
        # 根据结果设置退出码
        if failed > 0:
            sys.exit(1)
            
    except ValueError as e:
        print(f"❌ 配置错误: {str(e)}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断操作")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 处理过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()