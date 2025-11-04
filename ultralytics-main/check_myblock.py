from ultralytics import YOLO

import os
import importlib
import inspect

def check_myblock_registration():
    print("🔍 Checking YOLO custom module registration...")

    # 1️⃣ 检查 myblock.py 是否存在
    myblock_path = os.path.join("ultralytics", "nn", "modules", "myblock.py")
    if not os.path.exists(myblock_path):
        print(f"❌ 找不到文件: {myblock_path}")
        print("➡️ 请确保你已将 MyBlock 放在 ultralytics/nn/modules/ 下。")
        return

    print(f"✅ 找到文件: {myblock_path}")

    # 2️⃣ 检查 __init__.py 是否导入 MyBlock
    init_path = os.path.join("ultralytics", "nn", "modules", "__init__.py")
    with open(init_path, "r", encoding="utf-8") as f:
        init_content = f.read()
    if "from .myblock import MyBlock" not in init_content:
        print(f"⚠️ __init__.py 中未导入 MyBlock：{init_path}")
        print("➡️ 请在文件末尾添加： from .myblock import MyBlock")
    else:
        print("✅ MyBlock 已在 __init__.py 中注册。")

    # 3️⃣ 尝试动态导入
    try:
        module = importlib.import_module("ultralytics.nn.modules.myblock")
        print("✅ 成功导入 myblock.py 模块。")
    except Exception as e:
        print("❌ 无法导入 ultralytics.nn.modules.myblock：")
        print(e)
        return

    # 4️⃣ 检查是否定义了类 MyBlock
    if hasattr(module, "MyBlock"):
        cls = getattr(module, "MyBlock")
        print(f"✅ 找到类: {cls.__name__}")
        # 检查是否为 nn.Module 子类
        import torch.nn as nn
        if issubclass(cls, nn.Module):
            print("✅ MyBlock 是 nn.Module 的子类。")
        else:
            print("⚠️ MyBlock 不是 nn.Module 的子类（请检查类定义）。")
    else:
        print("❌ 在 myblock.py 中未找到类 MyBlock。")
        return

    # 5️⃣ 测试 ultralytics 是否能识别 MyBlock
    try:
        from ultralytics.nn.modules import MyBlock
        print("🎉 YOLO 框架成功识别 MyBlock 模块！")
        print("🚀 你现在可以在 yolov8.yaml 中使用它了。")
    except Exception as e:
        print("❌ YOLO 框架未能识别 MyBlock：")
        print(e)
        print("➡️ 请确认 __init__.py 导入语句拼写正确。")

if __name__ == "__main__":
    check_myblock_registration()

    print("another check------------------------")
    model = YOLO("ultralytics/cfg/models/v8/yolov8-zcy.yaml")
    print("✅ 成功加载自定义 MyBlock 模型！")
    exit()
