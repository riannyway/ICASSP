import os
import json
import pandas as pd
import time
from datetime import datetime
from typing import List, Dict
from glm_client import GLMClient
from text_processor import TextProcessor
from config import Config

class ErrorDetector:
    def __init__(self, api_key: str = None):
        self.glm_client = GLMClient(api_key)
        self.text_processor = TextProcessor()
        self.results = []
        
        # 创建必要的目录
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        os.makedirs(Config.LOG_DIR, exist_ok=True)

    def detect_and_correct_file(self, input_file: str) -> tuple:
        """
        检测错误并自动生成修正版本文件
        返回：(检测报告路径, 修正文件路径)
        """
        print(f"开始处理文件: {input_file}")
        
        # 1. 解析转录文件
        segments = self.text_processor.parse_transcription_file(input_file)
        print(f"解析得到 {len(segments)} 个文本段落")
        
        # 过滤掉空文本段落
        valid_segments = [seg for seg in segments if seg.get('text', '').strip()]
        print(f"有效段落数: {len(valid_segments)}")
        
        # 2. 批量检测和自动修正 - 使用优化版批量处理
        print("开始错误检测和自动修正...")
        print("使用批量处理模式，大幅减少API调用次数和token消耗...")
        
        start_time = time.time()
        results = self.glm_client.batch_detect_and_correct_segments(valid_segments)
        end_time = time.time()
        
        processing_time = end_time - start_time
        print(f"批量处理完成，耗时: {processing_time:.1f}秒")
        
        # 3. 生成检测报告
        report_path = self._generate_correction_report(results, input_file)
        
        # 4. 生成修正版本文件
        corrected_path = self._generate_corrected_file(results, input_file)
        
        # 5. 生成统计摘要
        self._print_correction_summary(results)
        
        print(f"处理完成！")
        print(f"📊 检测报告: {report_path}")
        print(f"📝 修正文件: {corrected_path}")
        
        return report_path, corrected_path

    def _generate_correction_report(self, results: List[Dict], input_file: str) -> str:
        """
        生成包含修正信息的详细报告
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.splitext(os.path.basename(input_file))[0]
        report_path = os.path.join(Config.OUTPUT_DIR, f"{filename}_correction_report_{timestamp}.txt")
        
        total_segments = len(results)
        corrected_segments = sum(1 for r in results if r.get('has_errors', False))
        api_errors = sum(1 for r in results if 'error' in r)
        
        # 统计处理方式
        batch_api_count = len([r for r in results if r.get('method') == 'batch_api'])
        quick_fix_count = len([r for r in results if r.get('method') == 'quick_fix'])
        pre_filter_count = len([r for r in results if r.get('method') == 'pre_filter'])
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("语音转录文本自动修正报告 (批量优化版)\n")
            f.write("=" * 70 + "\n")
            f.write(f"输入文件: {input_file}\n")
            f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总段落数: {total_segments}\n")
            f.write(f"修正段落数: {corrected_segments}\n")
            f.write(f"修正率: {corrected_segments/total_segments*100:.2f}%\n")
            if api_errors > 0:
                f.write(f"API错误数: {api_errors}\n")
            
            # 新增：处理方式统计
            f.write(f"\n处理方式分布:\n")
            f.write(f"  批量API处理: {batch_api_count} ({batch_api_count/total_segments*100:.1f}%)\n")
            f.write(f"  快速修正: {quick_fix_count} ({quick_fix_count/total_segments*100:.1f}%)\n")
            f.write(f"  预过滤跳过: {pre_filter_count} ({pre_filter_count/total_segments*100:.1f}%)\n")
            
            f.write("\n" + "=" * 70 + "\n")
            f.write("详细修正列表\n")
            f.write("=" * 70 + "\n\n")
            
            for i, result in enumerate(results, 1):
                # 写入基本信息
                f.write(f"【段落 {i}】\n")
                f.write(f"时间: {result.get('timestamp', 'Unknown')}\n")
                f.write(f"发言人: {result.get('speaker', 'Unknown')}\n")
                f.write(f"处理方式: {result.get('method', 'Unknown')}\n")
                
                # 检查是否有API错误
                if 'error' in result:
                    f.write(f"❌ API调用错误: {result['error']}\n")
                    f.write(f"原文: {result.get('text', '')}\n")
                
                # 检查是否有修正
                elif result.get('has_errors', False):
                    f.write("🔧 已修正\n")
                    f.write(f"原文: {result.get('original_text', result.get('text', ''))}\n")
                    f.write(f"修正: {result.get('corrected_text', '')}\n")
                    f.write(f"置信度: {result.get('confidence', 0):.2f}\n")
                    
                    # 显示具体错误详情
                    errors = result.get('errors', [])
                    if errors:
                        f.write("错误详情:\n")
                        for j, error in enumerate(errors, 1):
                            f.write(f"  {j}. {error.get('type', 'Unknown')}: ")
                            f.write(f"'{error.get('original', '')}' → '{error.get('corrected', '')}'\n")
                            if error.get('reason'):
                                f.write(f"     原因: {error.get('reason')}\n")
                
                else:
                    f.write("✅ 无需修正\n")
                    f.write(f"文本: {result.get('text', '')}\n")
                
                f.write("\n" + "-" * 50 + "\n\n")
        
        return report_path

    def _generate_corrected_file(self, results: List[Dict], input_file: str) -> str:
        """
        生成修正后的对话文件 - 修复版本，保持原始格式
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.splitext(os.path.basename(input_file))[0]
        corrected_path = os.path.join(Config.OUTPUT_DIR, f"{filename}_corrected_{timestamp}.txt")
        
        with open(corrected_path, 'w', encoding='utf-8') as f:
            f.write(f"{filename} - 自动修正版 (批量处理优化)\n\n")
            f.write(f"修正时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"原始文件: {input_file}\n")
            f.write("=" * 50 + "\n\n")
            
            for result in results:
                speaker = result.get('speaker', 'Unknown')
                timestamp = result.get('timestamp', 'Unknown')
                
                # 使用修正后的文本，如果没有修正则使用原文
                if 'error' in result:
                    # API调用出错，使用原文并标记错误
                    display_text = result.get('text', '【处理出错】')
                    f.write(f"{speaker} {timestamp}\n")
                    f.write(f"❌ {display_text}\n\n")
                
                elif result.get('has_errors', False) and result.get('corrected_text'):
                    # 有修正，使用修正后的文本
                    corrected_text = result.get('corrected_text', result.get('text', ''))
                    # 如果修正文本包含多行，保持原有的换行格式
                    if '\n' in corrected_text:
                        f.write(f"{speaker} {timestamp}\n")
                        f.write(f"{corrected_text}\n\n")
                    else:
                        f.write(f"{speaker} {timestamp}\n")
                        f.write(f"{corrected_text}\n\n")
                
                else:
                    # 无需修正，使用原文
                    original_text = result.get('text', '')
                    if original_text.strip():  # 只有非空文本才输出
                        f.write(f"{speaker} {timestamp}\n")
                        f.write(f"{original_text}\n\n")
        
        return corrected_path

    def _print_correction_summary(self, results: List[Dict]):
        """
        打印修正统计摘要 - 包含优化效果
        """
        total = len(results)
        corrected = sum(1 for r in results if r.get('has_errors', False))
        errors = sum(1 for r in results if 'error' in r)
        unchanged = total - corrected - errors
        
        # 统计处理方式
        batch_api_count = len([r for r in results if r.get('method') == 'batch_api'])
        quick_fix_count = len([r for r in results if r.get('method') == 'quick_fix'])
        pre_filter_count = len([r for r in results if r.get('method') == 'pre_filter'])
        
        print("\n" + "=" * 50)
        print("📊 修正统计摘要")
        print("=" * 50)
        print(f"总段落数: {total}")
        print(f"已修正: {corrected} ({corrected/total*100:.1f}%)")
        print(f"无需修正: {unchanged} ({unchanged/total*100:.1f}%)")
        if errors > 0:
            print(f"处理失败: {errors} ({errors/total*100:.1f}%)")
        
        print(f"\n批量处理优化效果:")
        print(f"批量API处理: {batch_api_count} ({batch_api_count/total*100:.1f}%)")
        print(f"快速修正: {quick_fix_count} ({quick_fix_count/total*100:.1f}%)")
        print(f"预过滤跳过: {pre_filter_count} ({pre_filter_count/total*100:.1f}%)")
        print("=" * 50)

    def detect_and_correct_file_only_correct(self, input_file: str) -> str:
        """
        只生成修正文件，不生成检测报告 - 用于 --only-correct 模式
        """
        print(f"开始处理文件: {input_file}")
        
        # 1. 解析转录文件
        segments = self.text_processor.parse_transcription_file(input_file)
        print(f"解析得到 {len(segments)} 个文本段落")
        
        # 过滤掉空文本段落
        valid_segments = [seg for seg in segments if seg.get('text', '').strip()]
        print(f"有效段落数: {len(valid_segments)}")
        
        # 2. 批量检测和自动修正
        print("开始错误检测和自动修正（仅生成修正文件）...")
        
        start_time = time.time()
        results = self.glm_client.batch_detect_and_correct_segments(valid_segments)
        end_time = time.time()
        
        processing_time = end_time - start_time
        print(f"批量处理完成，耗时: {processing_time:.1f}秒")
        
        # 3. 生成修正版本文件
        corrected_path = self._generate_corrected_file(results, input_file)
        
        # 4. 生成统计摘要
        self._print_correction_summary(results)
        
        print(f"处理完成！")
        print(f"📝 修正文件: {corrected_path}")
        
        return corrected_path