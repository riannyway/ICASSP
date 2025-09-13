import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # 智谱AI GLM-4.5 配置
    GLM_API_KEY = os.getenv('GLM_API_KEY')
    if not GLM_API_KEY:
        raise ValueError("请在.env文件中设置GLM_API_KEY环境变量")
    
    GLM_BASE_URL = 'https://open.bigmodel.cn/api/paas/v4/'
    GLM_MODEL = 'glm-4.5'  # 使用GLM-4.5旗舰模型
    
    # 错误检测配置
    CONFIDENCE_THRESHOLD = 0.7
    MAX_TOKENS = 2000
    TEMPERATURE = 0.1
    
    # 文件路径配置
    INPUT_DIR = './input'
    OUTPUT_DIR = './output'
    LOG_DIR = './logs'
    
    # API调用配置
    MAX_RETRIES = 3
    REQUEST_TIMEOUT = 30
    RATE_LIMIT_DELAY = 0.1