import requests
import json
import time
from typing import Dict, List, Optional
from config import Config

class GLMClient:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or Config.GLM_API_KEY
        self.base_url = Config.GLM_BASE_URL
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        })

    def detect_and_correct_text_errors(self, text_segment: str, context: str = "") -> Dict:
        """
        检测文本段落中的错误并自动修正
        """
        prompt = f"""
请分析以下语音转录文本，检测错误并提供修正版本：

转录文本："{text_segment}"
上下文："{context}"

请检测以下类型的错误并修正：
1. 同音字错误（如：的/得/地，在/再，等等）
2. 语法错误（主谓不一致，时态错误等）
3. 标点符号错误
4. 词汇搭配错误
5. 语义不通顺的地方
6. 专业术语错误
7. 人名、地名可能的错误

请以JSON格式返回结果：
{{
    "original_text": "原始文本",
    "corrected_text": "修正后的文本",
    "has_errors": true/false,
    "confidence": 0.0-1.0,
    "errors": [
        {{
            "type": "错误类型",
            "original": "原始错误部分",
            "corrected": "修正后部分",
            "position": "错误位置描述",
            "confidence": 0.0-1.0,
            "reason": "修正原因说明"
        }}
    ],
    "suggestions": "整体修正建议"
}}
"""
        
        try:
            response = self.session.post(
                f'{self.base_url}chat/completions',
                json={
                    "model": "glm-4-flash",  # 使用免费模型
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": Config.TEMPERATURE,
                    "max_tokens": Config.MAX_TOKENS
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                # 尝试解析JSON结果
                try:
                    # 提取JSON部分（如果被包装在其他文本中）
                    json_start = content.find('{')
                    json_end = content.rfind('}') + 1
                    if json_start != -1 and json_end != 0:
                        json_content = content[json_start:json_end]
                        parsed_result = json.loads(json_content)
                    else:
                        parsed_result = json.loads(content)
                    
                    # 确保包含原始文本
                    if 'original_text' not in parsed_result:
                        parsed_result['original_text'] = text_segment
                    return parsed_result
                    
                except json.JSONDecodeError as e:
                    # 如果无法解析JSON，返回基本结果
                    return {
                        'original_text': text_segment,
                        'corrected_text': content,  # 将整个回复作为修正文本
                        'has_errors': True,
                        'confidence': 0.5,
                        'errors': [],
                        'suggestions': '无法解析API返回的JSON格式',
                        'parse_error': str(e)
                    }
            else:
                return {
                    'original_text': text_segment,
                    'error': f'API调用失败: {response.status_code} - {response.text}',
                    'has_errors': False,
                    'confidence': 0.0
                }
                
        except requests.exceptions.Timeout:
            return {
                'original_text': text_segment,
                'error': 'API调用超时',
                'has_errors': False,
                'confidence': 0.0
            }
        except requests.exceptions.RequestException as e:
            return {
                'original_text': text_segment,
                'error': f'网络请求错误: {str(e)}',
                'has_errors': False,
                'confidence': 0.0
            }
        except Exception as e:
            return {
                'original_text': text_segment,
                'error': f'未知错误: {str(e)}',
                'has_errors': False,
                'confidence': 0.0
            }

    def batch_detect_and_correct_segments(self, segments: List[Dict], max_retries: int = 3) -> List[Dict]:
        """
        批量检测和修正文本段落，包含重试机制
        """
        results = []
        total_segments = len(segments)
        
        for i, segment in enumerate(segments):
            print(f"处理进度: {i+1}/{total_segments}")
            
            text = segment.get('text', '')
            if not text.strip():
                # 空文本直接跳过
                result = segment.copy()
                result.update({
                    'has_errors': False,
                    'confidence': 1.0,
                    'corrected_text': text
                })
                results.append(result)
                continue
            
            # 构建上下文（前后段落）
            context_parts = []
            if i > 0 and segments[i-1].get('text'):
                context_parts.append(f"前文: {segments[i-1]['text']}")
            if i < len(segments) - 1 and segments[i+1].get('text'):
                context_parts.append(f"后文: {segments[i+1]['text']}")
            context = " | ".join(context_parts)
            
            # 重试机制
            for attempt in range(max_retries):
                result = self.detect_and_correct_text_errors(text, context)
                
                if 'error' not in result:
                    break
                    
                if attempt < max_retries - 1:
                    print(f"重试 {attempt + 1}/{max_retries}")
                    time.sleep(1)  # 等待1秒再重试
                else:
                    print(f"段落 {i+1} 处理失败: {result.get('error', 'Unknown error')}")
            
            # 合并原始段落信息和检测结果
            final_result = segment.copy()
            final_result.update(result)
            results.append(final_result)
            
            # 避免API调用过于频繁
            time.sleep(0.1)
        
        return results

    def test_connection(self) -> bool:
        """
        测试API连接是否正常
        """
        try:
            result = self.detect_and_correct_text_errors("测试连接")
            return 'error' not in result
        except Exception:
            return False