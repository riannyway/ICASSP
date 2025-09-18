import os
import json
import argparse
from typing import Dict, Any

def merge_json_files(audio_analysis_path: str, emotion_timeline_path: str, output_path: str) -> None:
    """
    合并音频分析结果和情绪时间序列数据
    
    Args:
        audio_analysis_path: 音频分析结果JSON文件路径
        emotion_timeline_path: 情绪时间序列JSON文件路径
        output_path: 合并后的输出文件路径
    """
    # 读取音频分析结果
    with open(audio_analysis_path, 'r', encoding='utf-8') as f:
        audio_data = json.load(f)
    
    # 读取情绪时间序列数据
    with open(emotion_timeline_path, 'r', encoding='utf-8') as f:
        emotion_data = json.load(f)
    
    # 合并数据
    merged_data = {**audio_data, **emotion_data}
    
    # 写入合并后的文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
    
    print(f"成功合并文件: {output_path}")

def process_batch(audio_dir: str, emotion_dir: str, output_dir: str) -> None:
    """
    批量处理音频分析结果和情绪时间序列数据
    
    Args:
        audio_dir: 包含音频分析结果的目录
        emotion_dir: 包含情绪时间序列的目录
        output_dir: 输出目录
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取音频分析结果文件列表
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith('_audio.json')]
    
    if not audio_files:
        print(f"在目录 {audio_dir} 中未找到音频分析文件")
        return
    
    for audio_file in audio_files:
        # 提取基础文件名
        base_name = audio_file.replace('_audio.json', '')
        
        # 构建文件路径
        audio_path = os.path.join(audio_dir, audio_file)
        emotion_path = os.path.join(emotion_dir, f"{base_name}_emotion.json")
        output_path = os.path.join(output_dir, f"{base_name}_merged.json")
        
        # 检查情绪时间序列文件是否存在
        if not os.path.exists(emotion_path):
            continue
        
        # 合并文件
        try:
            merge_json_files(audio_path, emotion_path, output_path)
        except Exception as e:
            print(f"处理文件 {audio_file} 时出错: {str(e)}")
            continue
    
    print("done!")

def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='combined_audio_video')
    parser.add_argument('--audio_dir', type=str, required=True, 
                        help='(format: *_audio.json)')
    parser.add_argument('--emotion_dir', type=str, required=True, 
                        help='(emo_format: *_emotion.json)')
    parser.add_argument('--output_dir', type=str, required=True, 
                        help='--output_dir (format: *_merged.json)')
    
    args = parser.parse_args()
    
    # 执行批处理
    process_batch(args.audio_dir, args.emotion_dir, args.output_dir)

if __name__ == '__main__':
    main()
