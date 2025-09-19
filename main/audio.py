from modelscope import snapshot_download, AutoProcessor, Qwen2AudioForConditionalGeneration
import torch
import os
import json
from typing import Dict, List, Optional
import re
import librosa
import numpy as np
from io import BytesIO
import glob
from tqdm import tqdm
from datetime import datetime
import argparse
import traceback

class AudioAnalyzer:
    """音频分析器类"""
    
    def __init__(self, model_dir: Optional[str] = None, cache_dir: str = '/autodl-tmp/models'):
        """
        初始化音频分析器
        
        Args:
            model_dir: 模型目录路径，如果为None则自动下载
            cache_dir: 模型缓存目录
        """
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # 下载或加载模型
        if model_dir is None:
            model_dir = snapshot_download('Qwen/Qwen2-Audio-7B-Instruct', cache_dir=cache_dir)
        
        self.processor = AutoProcessor.from_pretrained(
            model_dir,
            trust_remote_code=True
        )
        
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_dir,
            device_map="auto",
            trust_remote_code=True,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True
        ).eval()
    
    def analyze_full_audio(self, audio_path: str) -> Dict:
        """
        分析单个音频文件，返回按说话人区分的分析结果
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            包含分析结果的字典
        """
        if not os.path.exists(audio_path):
            return {"error": f"音频文件 {audio_path} 不存在"}
        
        try:
            # 加载音频文件
            print(f"正在加载音频文件: {os.path.basename(audio_path)}...")
            audio_data, sample_rate = librosa.load(
                audio_path, 
                sr=self.processor.feature_extractor.sampling_rate
            )
            
            # 检查音频长度
            duration = len(audio_data) / sample_rate
            
            # 构建详细的分析提示词
            prompt_text = """请对这段音频进行全面深入的分析，要求如下：

特别注意的是，如果你不能理解我传递给你的音频，请说"无法理解音频内容"，而不是编造内容。

## 分析任务：
1. **说话人识别与区分**
   - 识别音频中共有几位说话人
   - 为每位说话人分配标识（如：说话人1、说话人2等）
   - 描述每位说话人的声音特征（性别、年龄段估计、音色特点）

2. **按说话人分析语音特征**
   对每位说话人，请分析以下内容：
   
   a) **语速分析**
      - 平均语速（快/中/慢）
      - 语速变化情况（是否有明显的加速或减速）
      - 停顿模式和节奏特点
   
   b) **语调分析**
      - 基本语调类型（平调/升调/降调主导）
      - 语调变化范围（单调/适中/丰富）
      - 情感表达强度
   
   c) **音高变化**
      - 基础音高水平（高/中/低）
      - 音高变化幅度
      - 音高波动模式

3. **内容转录**
   - 按说话人区分转录内容
   - 标注每段话的说话人

4. **情绪状态分析**
   - 分析每位说话人的主要情绪状态
   - 情绪变化轨迹

5. **交互分析**（如有多人）
   - 说话人之间的互动模式
   - 话轮转换特点

## 输出格式要求：
请按以下结构化格式输出：

【音频概览】
- 总时长估计：
- 说话人数量：
- 音频质量：

【说话人分析】
[说话人1]
- 声音特征：
- 语速：
- 语调：
- 音高：
- 主要情绪：
- 转录内容：

[说话人2]
（同上格式）

【交互特征】
- 话轮模式：
- 互动特点：

请确保分析详尽且按说话人清晰区分。请帮我分析这段音频。"""

            conversation = [
                {"role": "user", "content": [
                    {"type": "audio", "audio_data": audio_data},  
                    {"type": "text", "text": prompt_text}
                ]}
            ]

            # 准备音频数据列表
            audios = []
            for message in conversation:
                if isinstance(message["content"], list):
                    for ele in message["content"]:
                        if ele["type"] == "audio":
                            audios.append(ele["audio_data"]) 
            
            # 准备文本输入
            text_input = self.processor.apply_chat_template(
                conversation, 
                add_generation_prompt=True, 
                tokenize=False
            )
            
            inputs = self.processor(
                text=text_input,
                audio=audios, 
                return_tensors="pt",
                sampling_rate=sample_rate,
                padding=True
            )
            
            inputs = inputs.to("cuda")

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=4096,
                    temperature=0.3,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    repetition_penalty=1.05
                )
            
            # 解码生成的文本
            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            response = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            
            # 解析响应为结构化数据
            parsed_result = self._parse_response(response)
            parsed_result["raw_response"] = response
            parsed_result["audio_info"] = {
                "duration": duration,
                "sample_rate": sample_rate,
                "file_path": audio_path
            }
            
            return parsed_result
            
        except Exception as e:
            traceback.print_exc()
            return {
                "error": f"处理音频时出错: {str(e)}",
                "raw_response": "",
                "audio_info": {"file_path": audio_path}
            }

    def batch_analyze(self, audio_paths: List[str], output_dir: str = None) -> List[Dict]:
        """
        批量分析多个音频文件
        
        Args:
            audio_paths: 音频文件路径列表
            output_dir: 结果输出目录，如果为None则不保存文件
            
        Returns:
            包含所有分析结果的列表
        """
        results = []
        
        # 创建输出目录
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 使用进度条显示处理进度
        for audio_path in tqdm(audio_paths, desc="处理音频文件"):
            print(f"\n{'='*60}")
            print(f"正在处理: {os.path.basename(audio_path)}")
            print(f"{'='*60}")
            
            # 分析单个音频文件
            result = self.analyze_full_audio(audio_path)
            results.append(result)
            
            # 保存结果
            if output_dir:
                base_name = os.path.splitext(os.path.basename(audio_path))[0]
                
                # 保存文本报告
                txt_path = os.path.join(output_dir, f"{base_name}_analysis.txt")
                formatted_output = self.format_output(result)
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(formatted_output)
                print(f"✅ 文本报告已保存到: {txt_path}")
                
                # 保存JSON数据
                json_path = os.path.join(output_dir, f"{base_name}_analysis.json")
                json_data = {k: v for k, v in result.items() if k != "raw_response"}
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, ensure_ascii=False, indent=2)
                print(f"✅ JSON数据已保存到: {json_path}")
        
        # 保存汇总报告
        if output_dir and len(results) > 1:
            summary_path = os.path.join(output_dir, "batch_analysis_summary.txt")
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write("="*60)
                f.write("\n批量音频分析汇总报告\n")
                f.write("="*60)
                f.write(f"\n处理时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                f.write(f"\n处理文件总数: {len(audio_paths)}")
                f.write(f"\n成功分析: {len([r for r in results if 'error' not in r])}")
                f.write(f"\n分析失败: {len([r for r in results if 'error' in r])}")
                f.write("\n\n文件列表:\n")
                for i, (path, result) in enumerate(zip(audio_paths, results), 1):
                    status = "✅ 成功" if 'error' not in result else "❌ 失败"
                    f.write(f"{i}. {os.path.basename(path)} - {status}\n")
            print(f"✅ 汇总报告已保存到: {summary_path}")
        
        return results

    def _parse_response(self, response: str) -> Dict:
        """
        解析模型响应，提取结构化信息
        
        Args:
            response: 模型的原始响应文本
            
        Returns:
            解析后的结构化数据
        """
        result = {
            "overview": {},
            "speakers": [],
            "interaction": {},
            "raw_text": response
        }
        
        try:
            if "无法理解音频内容" in response:
                result["error"] = "模型无法理解音频内容，可能是音频质量问题或格式不支持"
                return result
            
            overview_match = re.search(r'【音频概览】(.*?)(?=【|$)', response, re.DOTALL)
            if overview_match:
                overview_text = overview_match.group(1)
                result["overview"] = self._extract_key_values(overview_text)
            
            # 提取每个说话人的信息
            speaker_pattern = r'\[说话人(\d+)\](.*?)(?=\[说话人|\【|$)'
            speaker_matches = re.finditer(speaker_pattern, response, re.DOTALL)
            
            for match in speaker_matches:
                speaker_id = match.group(1)
                speaker_text = match.group(2)
                
                speaker_info = {
                    "id": f"说话人{speaker_id}",
                    "features": self._extract_speaker_features(speaker_text)
                }
                result["speakers"].append(speaker_info)
            
            interaction_match = re.search(r'【交互特征】(.*?)(?=【|$)', response, re.DOTALL)
            if interaction_match:
                interaction_text = interaction_match.group(1)
                result["interaction"] = self._extract_key_values(interaction_text)
            
        except Exception as e:
            print(f"解析响应时出错: {e}")
            result["parse_error"] = str(e)
        
        return result
    
    def _extract_key_values(self, text: str) -> Dict:
        """提取键值对信息"""
        result = {}
        lines = text.strip().split('\n')
        for line in lines:
            if '：' in line or ':' in line:
                parts = re.split('[：:]', line, 1)
                if len(parts) == 2:
                    key = parts[0].strip().replace('-', '').strip()
                    value = parts[1].strip()
                    if key and value:
                        result[key] = value
        return result
    
    def _extract_speaker_features(self, text: str) -> Dict:
        """提取说话人特征"""
        features = {}
        patterns = {
            "声音特征": r'声音特征[：:](.*?)(?=\n[-•]|\n[^\n]*[：:]|$)',
            "语速": r'语速[：:](.*?)(?=\n[-•]|\n[^\n]*[：:]|$)',
            "语调": r'语调[：:](.*?)(?=\n[-•]|\n[^\n]*[：:]|$)',
            "音高": r'音高[：:](.*?)(?=\n[-•]|\n[^\n]*[：:]|$)',
            "情绪": r'(?:主要)?情绪[：:](.*?)(?=\n[-•]|\n[^\n]*[：:]|$)',
            "内容": r'转录内容[：:](.*?)(?=\n[-•]|\n[^\n]*[：:]|$)'
        }
        
        for key, pattern  in patterns.items():
            match = re.search(pattern, text, re.DOTALL)
            if match:
                    features[key] = match.group(1).strip()
        
            return features
    
    def format_output(self, analysis_result: Dict) -> str:
        """
        格式化输出分析结果
        
        Args:
            analysis_result: 分析结果字典
            
        Returns:
            格式化的字符串输出
        """
        if "error" in analysis_result:
            return f"错误: {analysis_result['error']}"
        
        output = []
        output.append("="*60)
        output.append("音频分析报告")
        output.append("="*60)
        
        # 输出音频信息
        if analysis_result.get("audio_info"):
            info = analysis_result["audio_info"]
            output.append(f"\n【音频文件信息】")
            output.append(f"  文件路径: {info.get('file_path', 'N/A')}")
            output.append(f"  音频时长: {info.get('duration', 0):.2f}秒")
            output.append(f"  采样率: {info.get('sample_rate', 0)}Hz")
        
        # 输出概览
        if analysis_result.get("overview"):
            output.append("\n【音频概览】")
            for key, value in analysis_result["overview"].items():
                output.append(f"  {key}: {value}")
        
        # 输出说话人分析
        if analysis_result.get("speakers"):
            output.append("\n【说话人详细分析】")
            for speaker in analysis_result["speakers"]:
                output.append(f"\n{'-'*40}")
                output.append(f"📢 {speaker['id']}")
                output.append(f"{'-'*40}")
                
                features = speaker.get("features", {})
                for key, value in features.items():
                    if value:
                        output.append(f"  ▪ {key}: {value}")
        
        # 输出交互分析
        if analysis_result.get("interaction") and analysis_result["interaction"]:
            output.append("\n【交互特征分析】")
            for key, value in analysis_result["interaction"].items():
                output.append(f"  {key}: {value}")
        
        output.append("\n" + "="*60)
        
        return "\n".join(output)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='音频分析工具')
    parser.add_argument('--audio_path', type=str, required=True, help='音频文件路径或包含音频文件的目录')
    parser.add_argument('--model_dir', type=str, default='', help='模型目录路径，如果为空则自动下载')
    parser.add_argument('--output_dir', type=str, default='./audio_analysis_results', help='结果输出目录，默认为./audio_analysis_results')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("="*60)
    
    # 初始化分析器
    try:
        analyzer = AudioAnalyzer(model_dir=args.model_dir)
    except Exception as e:
        print(f"❌ 初始化分析器失败: {e}")
        return
    
    # 收集要处理的音频文件
    audio_files = []
    
    if os.path.isfile(args.audio_path):
        # 单个文件模式
        audio_files = [args.audio_path]
    elif os.path.isdir(args.audio_path):
        # 目录模式 - 收集所有音频文件
        supported_formats = ['.mp3']
        for ext in supported_formats:
            audio_files.extend(glob.glob(os.path.join(args.audio_path, f'*{ext}')))
            audio_files.extend(glob.glob(os.path.join(args.audio_path, f'*{ext.upper()}')))
        
        if not audio_files:
            print(f"❌ 在目录 {args.audio_path} 中未找到支持的音频文件")
            return
        
        print(f" 找到 {len(audio_files)} 个音频文件")
    else:
        print(f"❌ 路径 {args.audio_path} 既不是文件也不是目录")
        return
    
    if len(audio_files) > 1:
        print("文件列表:")
        for i, file in enumerate(audio_files, 1):
            print(f"  {i}. {os.path.basename(file)}")
    
    confirm = input("\n确认开始处理? (y/n): ")
    if confirm.lower() != 'y':
        print("处理已取消")
        return
    print("-" * 60)
    
    results = analyzer.batch_analyze(audio_files, args.output_dir)
    
    # 输出汇总信息
    print("\n" + "="*60)
    print("批处理完成!")
    print("="*60)
    print(f"处理文件总数: {len(audio_files)}")
    print(f"成功分析: {len([r for r in results if 'error' not in r])}")
    print(f"分析失败: {len([r for r in results if 'error' in r])}")
    
    if len([r for r in results if 'error' in r]) > 0:
        print("\n失败文件列表:")
        for path, result in zip(audio_files, results):
            if 'error' in result:
                print(f"  ❌ {os.path.basename(path)}: {result['error']}")
    
    print(f"\n所有结果已保存到目录: {args.output_dir}")


if __name__ == "__main__":
    main()