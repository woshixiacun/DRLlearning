# make_yolo_dirs.py
from pathlib import Path
from typing import Union
import os
import random
import argparse
from collections import defaultdict
import shutil
import xml.etree.ElementTree as ET, os, glob

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
    cat = sorted(categories)
    cls_map = {v: i for i, v in enumerate(cat)}
    print(f"CLASS_MAP--> {cls_map}")
    return cls_map


def split_dataset(root_dir: str,
                  train_ratio: float = 0.8,
                  val_ratio: float = 0.1,
                  seed: int = 42):
    """
    将 NEU_DET 数据集按类别 8:1:1 分成 train/val/test
    并把对应图片路径写入三个 txt 文件。
    """
    random.seed(seed)  #给 Python 内置的随机数引擎固定种子，让后续所有 shuffle、randint 等随机操作每次运行结果都一样，保证数据集划分可复现。
    img_dir = Path(root_dir).resolve() / 'images'
    assert img_dir.is_dir(), f'{img_dir} 不存在'

    # 1. 按类别聚合图片,得到所有类别的路径
    category_to_imgs = defaultdict(list) # 把 category_to_imgs 初始化为“空字典默认值是 空列表”的容器，以后对任意新键第一次 append 时不会报错，会自动先给该键建一个空列表。
    for img_path in img_dir.glob('*.jpg'):  #在 img_dir 目录下通配查找所有 .jpg 文件
        stem = img_path.stem
        if '_' not in stem:
            continue
        category = stem.rsplit('_', 1)[0]
        category_to_imgs[category].append(img_path)

    # 2. 划分
    train_paths, val_paths, test_paths = [], [], []
    for category, imgs in category_to_imgs.items():
        # 路径按字符串顺序统一排个序，保证每次划分前图片顺序一致，结果可复现。
        imgs = sorted(imgs)  
        random.shuffle(imgs)
        n = len(imgs)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_paths.extend(imgs[:n_train])
        val_paths.extend(imgs[n_train:n_train + n_val])
        test_paths.extend(imgs[n_train + n_val:])  
    
     # 准备输出目录
    out_dir = Path(root_dir).resolve()   #把 root_dir 转换成绝对路径（解析掉 .、.. 等符号），避免后续路径出错。
    out_dir.mkdir(exist_ok=True)

    # 3. 写入 txt（改为绝对路径，也可换成相对路径）
    def write_txt(paths, txt_name):
        # 把 out_dir 和文件名拼成完整路径，以 写模式、UTF-8 编码打开文件；with 块结束后会自动关闭文件。
        with open(out_dir / txt_name, 'w', encoding='utf-8') as f:   
            for p in sorted(paths):   # 把路径列表先排序，再逐个取出路径 p，保证写入顺序固定、可复现。
                f.write(str(p) + '\n')

    write_txt(train_paths, 'train.txt')
    write_txt(val_paths, 'val.txt')
    write_txt(test_paths, 'test.txt')

    print(f'划分完成：{out_dir}/train.txt 等共 3 个文件已生成。')


def split_dataset2(root_dir: str,
                  train_ratio: float = 0.8,
                  val_ratio: float = 0.2,
                  seed: int = 42):
    '''
    保存到文件夹
    '''
    # 创建训练集和验证集文件夹
    mydata_pth =  Path(root_dir).resolve() / 'mydata'
    train_images_folder = os.path.join(mydata_pth, 'images', 'train')
    train_labels_folder = os.path.join(mydata_pth, 'labels', 'train')
    val_images_folder = os.path.join(mydata_pth, 'images', 'val')
    val_labels_folder = os.path.join(mydata_pth, 'labels', 'val')

    os.makedirs(train_images_folder, exist_ok=True)
    os.makedirs(train_labels_folder, exist_ok=True)
    os.makedirs(val_images_folder, exist_ok=True)
    os.makedirs(val_labels_folder, exist_ok=True)

    input_image_folder = Path(root_dir).resolve() / 'images'
    assert input_image_folder.is_dir(), f'{input_image_folder} 不存在'
    
    input_label_folder = Path(root_dir).resolve() / 'txt'
    assert input_label_folder.is_dir(), f'{input_label_folder} 不存在'


    # 获取所有图像文件列表
    images = [f for f in os.listdir(input_image_folder) if f.endswith('.jpg') or f.endswith('.png')]

    # 随机打乱图像文件列表
    random.seed(seed)
    random.shuffle(images)

    # 计算验证集的数量
    val_size = int(len(images) * val_ratio)

    # 划分验证集和训练集
    val_images = images[:val_size]
    train_images = images[val_size:]

    # 复制验证集图像和标签
    for image in val_images:
        label = os.path.splitext(image)[0] + '.txt'
        shutil.copy(os.path.join(input_image_folder, image), os.path.join(val_images_folder, image))
        shutil.copy(os.path.join(input_label_folder, label), os.path.join(val_labels_folder, label))


    # 复制训练集图像和标签
    for image in train_images:
        label = os.path.splitext(image)[0] + '.txt'
        shutil.copy(os.path.join(input_image_folder, image), os.path.join(train_images_folder, image))
        shutil.copy(os.path.join(input_label_folder, label), os.path.join(train_labels_folder, label))

def xml_2_txt(root_dir):

    xml_path  = Path(root_dir).resolve() / 'xml'  # 原始 xml 路径
    out_path  = Path(root_dir) / 'txt' # 输出 txt 路径
    os.makedirs(out_path, exist_ok=True)

    # 按自己类别改；key 是 xml 里的 <name> 文本 # {'crazing':0, 'inclusion':1, 'patches':2, 'pitted_surface':3, 'rolled-in_scale':4, 'scratches':5}
    class_map = find_all_categories(root_dir)
    
    for xml_file in xml_path.glob('*.xml'):
        
        # 把硬盘上的 .xml 文件读进来，生成一个“可查询、可修改”的 XML 树对象，并赋值给变量 tree，后面才能按需提取节点、属性或文本。
        tree = ET.parse(xml_file)
        # 拿到 XML 树的“根节点”对象，并赋值给变量 root，之后所有查找、读取、修改操s作都从它开始。对 Pascal-VOC 的标注文件来说，根节点就是 <annotation> 标签。
        root = tree.getroot() 
        w  = int(root.find('size/width').text)
        h  = int(root.find('size/height').text)

        lines = []
        for obj in root.iter('object'):
            cls = obj.find('name').text
            if cls not in class_map:          # 跳过未定义类别
                continue
            
            cid = class_map[cls]
            
            box = obj.find('bndbox')
            x1 = float(box.find('xmin').text)
            y1 = float(box.find('ymin').text)
            x2 = float(box.find('xmax').text)
            y2 = float(box.find('ymax').text)

            xc = ((x1 + x2) / 2) / w
            yc = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h

            lines.append(f'{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}') # cid=0，xc=0.4875，yc=0.49，bw=0.955，bh=0.96
        
        basename = os.path.splitext(os.path.basename(xml_file))[0]
        """
        os.path.basename(xml_path)
            去掉路径，只留文件名。
            例 xml_path = "dataset/labels_xml/000123.xml"
            → 得到 "000123.xml"
        os.path.splitext(...)
            再把文件名切成 (主名, 扩展名) 元组。
            接上例 → ("000123", ".xml")
        """
        with open(f'{out_path}/{basename}.txt', 'w') as f:
            f.write('\n'.join(lines))

# 如果脚本被直接运行，则默认创建当前目录下的 dataset 文件夹
if __name__ == "__main__":
    root_dir = 'C:/Users/Clavi/Desktop/coding/DRLlearning/ultralytics-main/NEU_DET'  # 换成你的实际路径
    xml_2_txt(root_dir)
    split_dataset2(root_dir)
    
    
    # 使用示例
    # input_image_folder = '/mnt/d/Study_File/codezcy/DRLlearning/ultralytics-main/datasets/NEU_DET/images' # 图片路径
    # input_label_folder = '/mnt/d/Study_File/codezcy/DRLlearning/ultralytics-main/datasets/NEU_DET/xml' # 标签路径
    # output_folder = '/mnt/d/Study_File/codezcy/DRLlearning/ultralytics-main/datasets/NEU_DET/mydata'
    # split_dataset(input_image_folder, input_label_folder, output_folder, test_ratio=0.2)

    
    

