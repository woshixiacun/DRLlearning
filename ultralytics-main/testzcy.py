# make_yolo_dirs.py
from pathlib import Path
from typing import Union
import os
import random
import argparse
from collections import defaultdict


def find_all_categories(root_dir: str):
    """
    参数
    ----
    root_dir : str
        数据集根目录，其下应包含 images/ 子目录。

    返回
    ----
    list[str]
        去重、按字典序排好序的类别名称列表。
    """
    img_dir = Path(root_dir) / 'images'
    if not img_dir.is_dir():
        raise FileNotFoundError(f'目录不存在：{img_dir}')

    categories = set()   # set可以自动去重
    for img_file in img_dir.glob('*.jpg'):  # 在 img_dir 这个文件夹里，找出所有扩展名为 .jpg 的文件，并返回一个可迭代的路径对象集合。
        # 去掉扩展名
        stem = img_file.stem
        # 从右往左找到第一个下划线，前面的部分就是类别名
        if '_' not in stem:
            continue  # 不符合命名规则，跳过
        category = stem.rsplit('_', 1)[0]  # 参数1：按 '_' 分割, 参数2：最多分割1次（从右往左）。得到['crazing', '1']
        categories.add(category)

    return sorted(categories)


def split_dataset(root_dir: str,
                  train_ratio: float = 0.8,
                  val_ratio: float = 0.1,
                  seed: int = 42):
    """
    将 NEU_DET 数据集按类别 8:1:1 分成 train/val/test，
    并把对应图片路径写入三个 txt 文件。
    """
    random.seed(seed)
    img_dir = Path(root_dir).resolve() / 'images'
    assert img_dir.is_dir(), f'{img_dir} 不存在'

    # 1. 按类别聚合图片
    category_to_imgs = defaultdict(list)
    for img_path in img_dir.glob('*.jpg'):
        stem = img_path.stem
        if '_' not in stem:
            continue
        category = stem.rsplit('_', 1)[0]
        category_to_imgs[category].append(img_path)

    # 2. 划分
    train_paths, val_paths, test_paths = [], [], []
    for category, imgs in category_to_imgs.items():
        imgs = sorted(imgs)
        random.shuffle(imgs)
        n = len(imgs)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_paths.extend(imgs[:n_train])
        val_paths.extend(imgs[n_train:n_train + n_val])
        test_paths.extend(imgs[n_train + n_val:])  
    
     # 准备输出目录
    out_dir = Path(root_dir).resolve() / 'dataset'
    out_dir.mkdir(exist_ok=True)

    # 3. 写入 txt（改为绝对路径，也可换成相对路径）
    def write_txt(paths, txt_name):
        with open(out_dir / txt_name, 'w', encoding='utf-8') as f:
            for p in sorted(paths):
                f.write(str(p) + '\n')

    write_txt(train_paths, 'train.txt')
    write_txt(val_paths, 'val.txt')
    write_txt(test_paths, 'test.txt')

    print(f'划分完成：{out_dir}/train.txt 等共 3 个文件已生成。')


# 如果脚本被直接运行，则默认创建当前目录下的 dataset 文件夹
if __name__ == "__main__":
    # make_yolo_dirs()  #'C:/Users/Clavi/Desktop/DRLlearning/zcy'
    root = 'C:/Users/Clavi/Desktop/DRLlearning/ultralytics-main/NEU_DET'  # 换成你的实际路径
    split_dataset(root)