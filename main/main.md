## 文档

部署了两个大模型，分别是 Qwen-Audio-7B-Instruct 和R1-Omni-0.5B, Qwen的模型文件没有做任何修改,然后直接用modelscope官方的即可
主目录下的代码为文本转录以及订正，调用了Fun-ASR和ChatGLM的API:
```shell
python main.py --api_key --recursive --correct --test-connection --parallel --continue-on-error --dry-run
```
完整参数如下：
```python
    parser.add_argument('input', help='输入文件/目录路径或通配符模式 (如 "*.txt" 或 "transcripts/")')
    parser.add_argument('--api-key', help='GLM API密钥（可选，优先使用环境变量）')
    parser.add_argument('--recursive', '-r', action='store_true', help='递归处理子目录中的文件')
    parser.add_argument('--correct', action='store_true', help='同时生成纠错版本')
    parser.add_argument('--only-correct', action='store_true', help='只生成纠错版本，不生成检测报告')
    parser.add_argument('--test-connection', action='store_true', help='测试API连接')
    parser.add_argument('--parallel', type=int, metavar='N', help='并行处理的线程数 (默认串行处理)')
    parser.add_argument('--continue-on-error', action='store_true', help='遇到错误时继续处理其他文件')
    parser.add_argument('--dry-run', action='store_true', help='预览模式：只显示要处理的文件，不实际处理')
```
转文本的环境在主目录的requirements.txt中

R1-Omni需要额外部署四个模型，一个是Whisper-Large-V3，一个是 siglip-base-patch16-224，一个是R1-Omni-0.5B，还有bert-uncased。部署完后需要在R1-Omni-0.5B的config.json中的第23和31行进行替换：
下载路径分别为：
```shell
https://www.modelscope.cn/models/AI-ModelScope/bert-base-uncased
https://hugging-face.cn/docs/transformers/model_doc/siglip
https://huggingface.co/openai/whisper-large-v3
```
下载完后更改对应的模型路径（第23、31行）
```json
 "mm_audio_tower": "/path/to/local/models/whisper-large-v3",
 "mm_vision_tower": "/path/to/local/models/siglip-base-patch16-224"
```
如果出现报错可能是没有替换bert-uncased，如果你已经部署bert，你需要在humanOmni/humanOmni_arch.py的第83行替换掉bert路径：
```python
bert_model = "/gpfs/work/aac/yulongli19/.cache/modelscope/hub/models/AI-ModelScope/bert-base-uncased" #替换为你的bert模型路径
```

在Video（在运行文件）代码中替换成你的bert-uncased实际路径
要准备R1-Omni-0.5B，下载路径：
```shell
https://www.modelscope.cn/models/iic/R1-Omni-0.5B
```
还需要下载human（但在main文件夹中已经提供）
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
所有流程运行完之后才能进行因果链生成和评估
在get_emo_sw中，需要提供txt文本，还有上述拼接完成的json字符串,生成好的json字符串放入到input_dir中，其他的输入不变

评估代码（get_emo_score)不变，与原本的相同
```shell
python get_emo_score.py --gt_dir --input_dir --output_dir --batch --event_shreshold
```
