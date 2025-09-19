import argparse
from moviepy import VideoFileClip
import os

def convert_mp4_to_mp3(mp4_file, output_mp3=None):
    """
    将 MP4 文件转换为 MP3 文件
    
    参数:
        mp4_file (str): 输入的 MP4 文件路径
        output_mp3 (str, 可选): 输出的 MP3 文件路径。如果未提供，则默认与 MP4 同目录，仅更改扩展名为 .mp3
    """
    if not os.path.isfile(mp4_file):
        print(f"错误：文件 '{mp4_file}' 不存在。")
        return

    if not mp4_file.lower().endswith('.mp4'):
        print(f"错误：'{mp4_file}' 不是 MP4 文件。")
        return

    if output_mp3 is None:
        base_name = os.path.splitext(mp4_file)[0]
        output_mp3 = base_name + '.mp3'
    else:
        if not output_mp3.lower().endswith('.mp3'):
            output_mp3 += '.mp3'

    try:
        video = VideoFileClip(mp4_file)
        audio = video.audio
        audio.write_audiofile(output_mp3)
        audio.close()
        video.close()
    except Exception as e:
        print(f"转换失败：{e}")

def main():
    parser = argparse.ArgumentParser(description="将 MP4 视频文件转换为 MP3 音频文件")
    parser.add_argument("input", help="输入的 MP4 文件路径")
    parser.add_argument("-o", "--output", help="输出的 MP3 文件路径（可选，默认同目录下更名）")

    args = parser.parse_args()

    convert_mp4_to_mp3(args.input, args.output)

if __name__ == "__main__":
    main()