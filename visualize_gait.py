import os
import gradio as gr
import pickle
import numpy as np
from PIL import Image

# 基础目录
BASE_DIR = "/root/autodl-tmp/"

def list_directories(path):
    """列出指定路径下的所有子目录"""
    full_path = os.path.expanduser(path)
    return [d for d in os.listdir(full_path) if os.path.isdir(os.path.join(full_path, d))]

def list_pkl_files(path):
    """列出指定路径下的PKL文件"""
    full_path = os.path.expanduser(path)
    return [f for f in os.listdir(full_path) if f.endswith('.pkl')]

def update_subject(dataset):
    """更新受试者选项"""
    path = os.path.join(BASE_DIR, dataset)
    return gr.Dropdown(choices=list_directories(path))

def update_type(dataset, subject):
    """更新类型选项"""
    path = os.path.join(BASE_DIR, dataset, subject)
    return gr.Dropdown(choices=list_directories(path))

def update_view(dataset, subject, type_):
    """更新视角选项"""
    path = os.path.join(BASE_DIR, dataset, subject, type_)
    return gr.Dropdown(choices=list_directories(path))

def load_and_display(dataset, subject, type_, view):
    """加载并显示PKL文件中的图像"""
    path = os.path.join(BASE_DIR, dataset, subject, type_, view)
    pkl_files = list_pkl_files(path)
    
    if not pkl_files:
        return None
    
    # 加载第一个pkl文件
    with open(os.path.join(path, pkl_files[0]), 'rb') as f:
        frames = pickle.load(f)
    
    # 转换为PIL图像列表
    images = []
    for frame in frames:
        if isinstance(frame, np.ndarray):
            # 假设图像数据是numpy数组，需要转换为0-255范围
            if frame.dtype == np.float32:
                frame = (frame * 255).astype(np.uint8)
            img = Image.fromarray(frame)
            images.append(img)
    
    return images

with gr.Blocks() as demo:
    gr.Markdown("## 步态数据集可视化工具")
    
    with gr.Row():
        dataset_dd = gr.Dropdown(
            label="选择数据集",
            choices=list_directories(BASE_DIR),
            interactive=True
        )
        subject_dd = gr.Dropdown(label="选择受试者", interactive=True)
        type_dd = gr.Dropdown(label="选择类型", interactive=True)
        view_dd = gr.Dropdown(label="选择视角", interactive=True)
    
    gallery = gr.Gallery(label="帧序列预览", columns=5)
    
    # 设置级联更新
    dataset_dd.change(
        update_subject,
        inputs=dataset_dd,
        outputs=subject_dd
    ).then(
        lambda: [gr.Dropdown(value=None), gr.Dropdown(value=None)],
        outputs=[type_dd, view_dd]
    )
    
    subject_dd.change(
        update_type,
        inputs=[dataset_dd, subject_dd],
        outputs=type_dd
    ).then(
        lambda: gr.Dropdown(value=None),
        outputs=view_dd
    )
    
    type_dd.change(
        update_view,
        inputs=[dataset_dd, subject_dd, type_dd],
        outputs=view_dd
    )
    
    view_dd.change(
        load_and_display,
        inputs=[dataset_dd, subject_dd, type_dd, view_dd],
        outputs=gallery
    )

if __name__ == "__main__":
    demo.launch()
