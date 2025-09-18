## 文档

部署了两个大模型，分别是 Qwen-Audio-7B-Instruct 和R1-Omni-0.5B, Qwen的模型文件没有做任何修改，直接modelscope官方的即可

R1-Omni-0.5B需要额外部署三个模型，一个是Whisper-Large-V3，一个是 siglip-base-patch16-224，还有bert-uncased，并且在config.json中的第23和31行替换掉：
```json
 "mm_audio_tower": "/path/to/local/models/whisper-large-v3",
 "mm_vision_tower": "/path/to/local/models/siglip-base-patch16-224"
```
另外，在Video代码中替换掉bert-uncased的实际路径
要准备R1-Omni-0.5B，下载路径：
```shell
https://www.modelscope.cn/models/iic/R1-Omni-0.5B
```

Audio代码用于生成音频参数的分析，需要准备mp3音频文件:
```python
    AUDIO_PATH = "/root/autodl-fs/9.12.MP3"  # 目录
    MODEL_DIR =""  #改称bert模型路径
    OUTPUT_DIR = "./audio_analysis_results"  # 输出目录
```

Video代码用于生成视频分析，需要将MP4格式的文件和txt文件一并传入（用于提取说话人和timestamp），需要将human的文件夹与代码放在同一个根目录下：
```shell
python video.py --root_dir {folder} --output_dir {} --modal {video or video_audio or audio}
```
combined用于将两者的输出结合起来，用于提供给后续生成因果链的prompt之一
```shell
python combined.py --audio_dir --emotion_dir --output_dir
```
在get_emo_sw中，需要提供txt文本，还有上述拼接完成的json字符串,生成好的json字符串放入到input_dir中，其他的输入不变
