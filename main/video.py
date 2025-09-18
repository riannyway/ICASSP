import os
import argparse
import re
from humanomni import model_init, mm_infer
from humanomni.utils import disable_torch_init
from modelscope import BertTokenizer

os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def extract_speaker_data(file_path):
    speaker_data = []
    current_speaker = None
    current_timestamp = None
    current_text = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # 匹配发言人行
            speaker_match = re.match(r'发言人\s*(\d+)\s*(\d{2}:\d{2})', line)
            if speaker_match:
                # 保存前一个发言人的数据
                if current_speaker is not None:
                    speaker_data.append({
                        'speaker': current_speaker,
                        'timestamp': current_timestamp,
                        'text': ' '.join(current_text)
                    })
                
                # 开始新的发言人记录
                current_speaker = speaker_match.group(1)
                current_timestamp = speaker_match.group(2)
                current_text = []
            elif line and not line.startswith('[') and not line.startswith('http'):
                # 添加非空行且非元数据的文本
                current_text.append(line)
    
        # 添加最后一个发言人
        if current_speaker is not None:
            speaker_data.append({
                'speaker': current_speaker,
                'timestamp': current_timestamp,
                'text': ' '.join(current_text)
            })
    
    return speaker_data

def format_prompt(speaker_data):
    """将提取的数据格式化为prompt"""
    prompt_parts = []
    for entry in speaker_data:
        prompt_parts.append(
            f"At {entry['timestamp']}, Speaker {entry['speaker']} said: {entry['text']}"
        )
    
    base_instruct = "please analysis each speakers emotion,and records. Output the thinkong process in  and final emotion in <answer> </answer> tags."
    return "Here is a conversation transcript with timestamps and speakers:\n" + "\n".join(prompt_parts) + "\n\n" + base_instruct

import json

def main():
    modal = "video_audio"
    model_path = "/gpfs/work/aac/yulongli19/.cache/modelscope/hub/models/iic/R1-Omni-0.5B"
    video_path = ".cache/modelscope/hub/datasets/cr17784624325/datatsets/chat-1236/chat-1236.mp4"
    transcript_file = ".cache/modelscope/hub/datasets/cr17784624325/datatsets/chat-1236/chat-1236.txt" 
    
    # 提取并格式化prompt
    speaker_data = extract_speaker_data(transcript_file)
    instruct = format_prompt(speaker_data)
    
    # 初始化BERT分词器
    bert_model = ".cache/modelscope/hub/models/AI-ModelScope/bert-base-uncased"
    bert_tokenizer = BertTokenizer.from_pretrained(bert_model)

    # 禁用Torch初始化
    disable_torch_init()

    # 初始化模型、处理器和分词器
    model, processor, tokenizer = model_init(model_path)

    # 处理视频输入
    video_tensor = processor['video'](video_path)
    
    if modal == 'video_audio' or modal == 'audio':
        audio = processor['audio'](video_path)[0]
    else:
        audio = None

    # 执行推理
    output = mm_infer(
        video_tensor, 
        instruct, 
        model=model, 
        tokenizer=tokenizer, 
        modal=modal, 
        question=instruct, 
        bert_tokeni=bert_tokenizer, 
        do_sample=False, 
        audio=audio
    )
    
    # 保存输出为JSON文件
    output_file = "output.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=4)
    
    print(f"Output saved to {output_file}")
