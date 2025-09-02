import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class Config:
    # GLM API配置
    GLM_API_KEY = os.getenv('GLM_API_KEY')
    if not GLM_API_KEY:
        raise ValueError("请在.env文件中设置GLM_API_KEY环境变量")
    
    GLM_BASE_URL = 'https://open.bigmodel.cn/api/paas/v4/'
    
    # 错误检测配置
    CONFIDENCE_THRESHOLD = 0.7  # 错误识别置信度阈值
    MAX_TOKENS = 2000          # 每次API调用的最大token数
    TEMPERATURE = 0.1          # 温度参数，越低越准确
    
    # 文件路径配置
    INPUT_DIR = './input'       # 输入文件目录
    OUTPUT_DIR = './output'     # 输出文件目录
    LOG_DIR = './logs'         # 日志目录
    
    # API调用配置
    MAX_RETRIES = 3            # 最大重试次数
    REQUEST_TIMEOUT = 30       # 请求超时时间（秒）
    RATE_LIMIT_DELAY = 0.1     # API调用间隔时间（秒）