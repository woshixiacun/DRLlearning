# make_yolo_dirs.py
from pathlib import Path
from typing import Union


def make_yolo_dirs(root: Union[str, Path] = "dataset") -> Path:
    """
    创建 YOLO 标准目录结构：
    root/
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/

    参数
    ----
    root : str | Path
        数据集根目录路径，可绝对或相对。

    返回
    ----
    Path
        创建好的根目录绝对路径对象。
    """
    root = Path(root).resolve()
    print(f'root is  {root}')
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    print(f"YOLO 目录结构已生成：{root}")
    return root


# 如果脚本被直接运行，则默认创建当前目录下的 dataset 文件夹
if __name__ == "__main__":
    # make_yolo_dirs()  #'C:/Users/Clavi/Desktop/DRLlearning/zcy'
    pass