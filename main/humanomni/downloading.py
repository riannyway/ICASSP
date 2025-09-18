#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('iic/R1-Omni-0.5B',cache_dir='autodl-tmp')
