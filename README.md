# MCF: Text LLMS For Multimodal Emotional Causality

<div align="center">

[![arXiv](https://img.shields.io/badge/📚%20Arxiv-Coming%20soon-ff0000)](#)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-MCF-blueviolet)](https://modelscope.cn/datasets/zRzRzRzRzRzRzR/MCF)
</div>

## Data

Dataset task definition and annotation example of the MCF framework. The framework contains two core subtasks:
five-tuple
element extraction (identifying Target, Holder, Aspect, Opinion, Sentiment, and Rationale) and sentiment chain
analysis (constructing causal
relationship chains between emotional events).

![dataset.png](resources/dataset.png)

## MCF pipeline

The MCF (Multimodal Causality Framework) architecture. MCF employs a three-stage pipeline: Recognition extracts
multimodal features through adaptive fidelity control, Memory aggregates events to compress 200+ dialogue turns into 50-80 semantic
units, and Attribution performs cross-modal alignment and progressive reasoning to identify emotional causal chains. The framework
transforms multimodal dialogue sequences into structured representations processable by text-based LLMs while preserving long-distance causal
dependencies.

![structure.png](resources/structure.png)

# Result

Impact of MCF on LLMs Performance Across Different Evaluation Metrics. The scores are reported in percentage (%). ↓and +
indicate performance decrease and increase compared to GPT-o1(text-only), respectively. Bold values represent the best
performance in each
section. Causemotion uses only the text modal. LLM refers to using only the LLM itself.

![bench.png](resources/bench.png)

## 快速开始

确保这些模型均已经安装：
`google-bert/bert-base-uncased`
`google/siglip-base-patch16-224`
`openai/whisper-large-v3`
`StarJiaxing/R1-Omni-0.5B`
`https://modelscope.cn/models/qwen/Qwen2-Audio-7B-Instruct`

模型文件路径替换：
上述模型下载完后，你需要在R1-Omni-0.5B/config.json中更改对应的模型路径（第23、31行）：
```json
 "mm_audio_tower": "/path/to/local/models/whisper-large-v3",
 "mm_vision_tower": "/path/to/local/models/siglip-base-patch16-224"
```
并分别在vido.py、humanomni_arch.py中的第122行以及第83行替换bert实际路径：
```python
#vido.py
bert_model = ".cache/modelscope/hub/models/AI-ModelScope/bert-base-uncased" #替换为你的实际路径
#humanomni_arch.py
bert_model = "/gpfs/work/aac/yulongli19/.cache/modelscope/hub/models/AI-ModelScope/bert-base-uncased" #替换为你的bert模型路径
 ```

数据集：
你需要准备MP4视频文件、txt转录文本文件，以及音频文件，推荐使用mp3格式：

|-video
    |-chat_1.mp4

|-chat
    |-chat_1.txt

|-audio
    |-chat_1.mp3


## 运行
1.准备环节：
我们提供了convert_text进行文本转录:
```shell
cd convert_text
python main.py --input {chat.mp4} --api-key --parallel 8 --api-mode high
```
audio_convert.py进行音频提取：
```shell
cd main
python audio_convert.py --input_dir {mp4} --output_dir {mp3}
```
在三种文件（mp3，mp4，txt）都准备好后，首先进行音频提取：
```shell
cd main
python audio.py --audio_path --model_path --output_path
```
然后进行视频提取：
```shell
python video.py --root_dir {data} --output_dir --modal {audio or video_audio}
```
拼接提取到的内容：
```shell
python combined.py --audio_dir {audio} --input_dir {video_info} --output_dir {output} 
```
所有的提取做完之后，进行因果链的生成
```shell
python get_emo_sw.py --input_dir --other_text {combined_text} --output_dir --config_path --llm_model --batch --window_sizes --step_sizes
```
评分代码不变（get_emo_score）：
```shell
python get_emo_score.py --gt_dir --input_dir --output_dir --batch --event_shreshold
```
