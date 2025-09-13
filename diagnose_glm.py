#!/usr/bin/env python3
"""
GLM API诊断脚本 - 找出空响应的根本原因
"""

import requests
import json
import time
from config import Config

def test_basic_api_call():
    """测试最基础的API调用"""
    print("🔍 测试基础API调用...")
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {Config.GLM_API_KEY}'
    }
    
    # 最简单的请求
    payload = {
        "model": "glm-4.5",
        "messages": [{"role": "user", "content": "你好"}],
        "max_tokens": 50,
        "temperature": 0.1
    }
    
    try:
        response = requests.post(
            f'{Config.GLM_BASE_URL}chat/completions',
            headers=headers,
            json=payload,
            timeout=10
        )
        
        print(f"状态码: {response.status_code}")
        print(f"响应头: {dict(response.headers)}")
        print(f"响应内容长度: {len(response.text)}")
        print(f"响应内容: {response.text[:500]}")
        
        if response.status_code == 200:
            try:
                response_json = response.json()
                print(f"JSON结构: {json.dumps(response_json, ensure_ascii=False, indent=2)[:500]}")
                return True
            except json.JSONDecodeError as e:
                print(f"❌ JSON解析失败: {e}")
                return False
        else:
            print(f"❌ HTTP状态码错误")
            return False
            
    except Exception as e:
        print(f"❌ 请求异常: {e}")
        return False

def test_different_models():
    """测试不同的模型名称"""
    print("\n🔍 测试不同模型...")
    
    models = ["glm-4.5", "glm-4", "glm-4-0520", "glm-4-plus"]
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {Config.GLM_API_KEY}'
    }
    
    for model in models:
        print(f"\n测试模型: {model}")
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "你好"}],
            "max_tokens": 20,
            "temperature": 0.1
        }
        
        try:
            response = requests.post(
                f'{Config.GLM_BASE_URL}chat/completions',
                headers=headers,
                json=payload,
                timeout=10
            )
            
            print(f"  状态码: {response.status_code}")
            print(f"  内容长度: {len(response.text)}")
            
            if len(response.text) > 0:
                print(f"  ✅ {model} 工作正常")
                return model
            else:
                print(f"  ❌ {model} 返回空响应")
                
        except Exception as e:
            print(f"  ❌ {model} 请求失败: {e}")
    
    return None

def test_content_filtering():
    """测试内容过滤问题"""
    print("\n🔍 测试内容过滤...")
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {Config.GLM_API_KEY}'
    }
    
    test_contents = [
        "你好",
        "今天天气不错",
        "我觉的这个方案不错",  # 包含错误的文本
        "修正以下文本中的错误：我觉的很好",  # 修正任务
        "请检查这段话：在说一遍",  # 另一个修正任务
    ]
    
    for i, content in enumerate(test_contents):
        print(f"\n测试内容 {i+1}: {content}")
        
        payload = {
            "model": "glm-4.5",
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 50,
            "temperature": 0.1
        }
        
        try:
            response = requests.post(
                f'{Config.GLM_BASE_URL}chat/completions',
                headers=headers,
                json=payload,
                timeout=10
            )
            
            print(f"  状态码: {response.status_code}, 长度: {len(response.text)}")
            
            if len(response.text) > 0:
                print(f"  ✅ 内容正常")
            else:
                print(f"  ❌ 可能被过滤")
                
        except Exception as e:
            print(f"  ❌ 请求失败: {e}")

def test_api_quota():
    """测试API配额和速率限制"""
    print("\n🔍 测试API配额...")
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {Config.GLM_API_KEY}'
    }
    
    payload = {
        "model": "glm-4.5",
        "messages": [{"role": "user", "content": "测试"}],
        "max_tokens": 10,
        "temperature": 0.1
    }
    
    # 快速连续请求
    for i in range(5):
        print(f"  快速请求 {i+1}/5...")
        
        try:
            response = requests.post(
                f'{Config.GLM_BASE_URL}chat/completions',
                headers=headers,
                json=payload,
                timeout=10
            )
            
            print(f"    状态码: {response.status_code}, 长度: {len(response.text)}")
            
            if response.status_code == 429:
                print(f"    ⚠️ 速率限制触发")
                break
            elif response.status_code != 200:
                print(f"    ❌ 其他错误: {response.status_code}")
                break
                
        except Exception as e:
            print(f"    ❌ 请求异常: {e}")
        
        time.sleep(0.5)

def test_auth_and_key():
    """测试认证和密钥"""
    print("\n🔍 测试认证...")
    
    # 测试无效密钥
    invalid_headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer invalid_key_test'
    }
    
    payload = {
        "model": "glm-4.5",
        "messages": [{"role": "user", "content": "你好"}],
        "max_tokens": 10
    }
    
    try:
        response = requests.post(
            f'{Config.GLM_BASE_URL}chat/completions',
            headers=invalid_headers,
            json=payload,
            timeout=10
        )
        
        print(f"无效密钥测试 - 状态码: {response.status_code}")
        print(f"无效密钥测试 - 响应: {response.text[:200]}")
        
    except Exception as e:
        print(f"无效密钥测试异常: {e}")
    
    # 测试当前密钥格式
    print(f"\n当前密钥格式检查:")
    print(f"密钥长度: {len(Config.GLM_API_KEY)}")
    print(f"密钥前缀: {Config.GLM_API_KEY[:10]}...")
    print(f"密钥后缀: ...{Config.GLM_API_KEY[-10:]}")

def test_simplified_correction():
    """测试简化的纠错请求"""
    print("\n🔍 测试简化纠错...")
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {Config.GLM_API_KEY}'
    }
    
    # 极简纠错请求
    simple_prompts = [
        "修正：我觉的很好",
        "纠错：在说一遍",
        "检查：因该好好学习"
    ]
    
    for prompt in simple_prompts:
        print(f"\n测试提示: {prompt}")
        
        payload = {
            "model": "glm-4.5",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 30,
            "temperature": 0.1
        }
        
        try:
            response = requests.post(
                f'{Config.GLM_BASE_URL}chat/completions',
                headers=headers,
                json=payload,
                timeout=15
            )
            
            print(f"  状态码: {response.status_code}")
            print(f"  内容长度: {len(response.text)}")
            
            if len(response.text) > 0:
                try:
                    response_json = response.json()
                    if 'choices' in response_json and response_json['choices']:
                        content = response_json['choices'][0]['message']['content']
                        print(f"  ✅ 响应: {content}")
                    else:
                        print(f"  ❌ 响应格式异常")
                except:
                    print(f"  ❌ JSON解析失败")
            else:
                print(f"  ❌ 空响应")
                
        except Exception as e:
            print(f"  ❌ 请求异常: {e}")

def main():
    """主诊断流程"""
    print("🚀 GLM API深度诊断开始...\n")
    
    print(f"配置信息:")
    print(f"API Base URL: {Config.GLM_BASE_URL}")
    print(f"API Key: {Config.GLM_API_KEY[:8]}...{Config.GLM_API_KEY[-4:]}")
    print(f"默认模型: glm-4.5")
    print("=" * 60)
    
    tests = [
        ("基础API调用", test_basic_api_call),
        ("不同模型测试", test_different_models), 
        ("内容过滤测试", test_content_filtering),
        ("API配额测试", test_api_quota),
        ("认证测试", test_auth_and_key),
        ("简化纠错测试", test_simplified_correction)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} 执行异常: {e}")
            results[test_name] = False
        
        time.sleep(2)  # 避免过快请求
    
    # 总结诊断结果
    print(f"\n{'='*60}")
    print("📊 诊断总结")
    print("=" * 60)
    
    for test_name, result in results.items():
        status = "✅ 正常" if result else "❌ 异常"
        print(f"{test_name}: {status}")
    
    # 给出建议
    print(f"\n💡 建议:")
    if not results.get("基础API调用", False):
        print("- 基础API调用失败，检查网络连接和API密钥")
    
    working_model = results.get("不同模型测试")
    if working_model:
        print(f"- 建议使用模型: {working_model}")
    
    if not results.get("内容过滤测试", False):
        print("- 可能存在内容过滤问题，尝试更简单的提示")
    
    if not results.get("API配额测试", False):
        print("- 可能遇到速率限制，增加请求间隔")

if __name__ == "__main__":
    main()